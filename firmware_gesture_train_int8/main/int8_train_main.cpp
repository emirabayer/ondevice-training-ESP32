/*
 * INT8 + sparse gesture training on the ESP32-S3.
 *
 * The feature extractor (GDS_W1/GDS_B1) was trained on the laptop; here it is
 * quantized to int8 and frozen. Its output is computed over the dataset once
 * and reused, since a frozen layer's features never change. Only the 32->5
 * head is trained on the board, in int8, with Adam and per-layer scale
 * calibration.
 *
 * All buffers come from the 256 KB arena. Prints per-epoch time, arena
 * high-water mark, and test accuracy.
 */

#include <cmath>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "arena.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "qas.hpp"
#include "rng.hpp"
#include "tensor.hpp"

#include "gesture_dataset.h"

using namespace edge;

namespace {

constexpr char TAG[] = "int8_train";

constexpr uint32_t NF = GDS_N_FEATURES;  // 64
constexpr uint32_t NH = GDS_HIDDEN;      // 32
constexpr uint32_t NC = GDS_N_CLASSES;   // 5
constexpr uint32_t NTR = GDS_N_TRAIN;    // 80
constexpr uint32_t NTE = GDS_N_TEST;     // 20

constexpr int EPOCHS = 800;

// The 256 KB arena must be static (.bss), never on the task stack.
Arena g_arena;

float max_abs_2d(const float* a, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) { float v = a[i] < 0 ? -a[i] : a[i]; if (v > m) m = v; }
    return m;
}

// One-shot QAS calibration of a layer's output scale (same pattern as the
// synthetic demo): forward with a provisional unit scale, observe the real
// activation range, retarget output_scale to it.
float calibrate_output_scale(Arena& arena, DenseLayer& layer, const Tensor<int8_t>& X) {
    ArenaScope scope(arena);
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, X.shape[0], layer.out_features(), 1.0f, 0);
    layer.Forward(arena, X, Y);
    float* real = arena.allocate<float>(Y.size());
    DequantizeBuffer(Y, real);
    DynamicScaleTracker tracker(1.0f);
    float cal = tracker.Observe(MaxAbs(real, Y.size()));
    layer.SetOutputScale(cal);
    return cal;
}

// Quantize the normalized float dataset rows [n][NF] into an INT8 tensor.
void quantize_inputs(const float X[][NF], uint32_t n, float in_scale, Tensor<int8_t>& out) {
    for (uint32_t s = 0; s < n; s++)
        for (uint32_t i = 0; i < NF; i++)
            out.at(s, i) = QuantizeInt8(X[s][i], in_scale, 0);
}

// Forward the cached hidden features through the head and return test
// accuracy. This calls head.Forward, which re-caches the head's input, so only
// call it at the end of an epoch (after the optimizer step), never between the
// training forward and backward.
float eval_head(Arena& arena, DenseLayer& head, const Tensor<int8_t>& H, const int32_t* y,
                uint32_t n) {
    ArenaScope scope(arena);
    Tensor<int8_t> logits_q = MakeTensor2D<int8_t>(arena, n, NC, head.output_scale(), 0);
    head.Forward(arena, H, logits_q);
    float* logits = arena.allocate<float>((size_t)n * NC);
    DequantizeBuffer(logits_q, logits);
    float* probs = arena.allocate<float>((size_t)n * NC);
    Softmax(logits, probs, n, NC);
    uint32_t correct = 0;
    for (uint32_t s = 0; s < n; s++) {
        uint32_t pred = 0; float best = probs[s * NC];
        for (uint32_t k = 1; k < NC; k++) if (probs[s * NC + k] > best) { best = probs[s * NC + k]; pred = k; }
        if ((int32_t)pred == y[s]) correct++;
    }
    return (float)correct / (float)n;
}

} // namespace

extern "C" void app_main(void) {
    Rng rng(12345u);
    ESP_LOGI(TAG, "=== INT8 QAS + sparse-update on-device gesture training ===");
    ESP_LOGI(TAG, "model %u->%u->%u (extractor FROZEN, head trainable), train=%u test=%u",
             NF, NH, NC, NTR, NTE);

    // --- Frozen feature extractor: load host-pre-trained W1/b1, quantized ---
    const float in_scale = 1.0f / 127.0f;  // normalized features live in [-1,1]
    const float w1_scale = max_abs_2d(&GDS_W1[0][0], (size_t)NF * NH) / 127.0f;
    DenseLayer fe;
    fe.Init(g_arena, NF, NH, in_scale, w1_scale, /*output_scale placeholder*/ 1.0f,
            /*trainable=*/false, rng);
    fe.LoadFrozenFromFloat(&GDS_W1[0][0], GDS_B1);

    // Persistent quantized inputs.
    Tensor<int8_t> Xtr = MakeTensor2D<int8_t>(g_arena, NTR, NF, in_scale, 0);
    Tensor<int8_t> Xte = MakeTensor2D<int8_t>(g_arena, NTE, NF, in_scale, 0);
    quantize_inputs(GDS_X_TRAIN, NTR, in_scale, Xtr);
    quantize_inputs(GDS_X_TEST, NTE, in_scale, Xte);

    // Calibrate the extractor's output scale on the training inputs.
    calibrate_output_scale(g_arena, fe, Xtr);
    ESP_LOGI(TAG, "frozen extractor: w1_scale=%.6f  out_scale=%.6f", w1_scale, fe.output_scale());

    // --- Cache the frozen hidden features ONCE (sparse-update efficiency) ---
    Tensor<int8_t> Htr = MakeTensor2D<int8_t>(g_arena, NTR, NH, fe.output_scale(), 0);
    Tensor<int8_t> Hte = MakeTensor2D<int8_t>(g_arena, NTE, NH, fe.output_scale(), 0);
    fe.Forward(g_arena, Xtr, Htr); ReLU_INT8(Htr);
    fe.Forward(g_arena, Xte, Hte); ReLU_INT8(Hte);

    // --- Trainable INT8 head ---
    const float head_w_scale = 2.0f / 127.0f;  // headroom for weights up to ~2.0
    DenseLayer head;
    head.Init(g_arena, NH, NC, fe.output_scale(), head_w_scale, /*out placeholder*/ 1.0f,
              /*trainable=*/true, rng);
    calibrate_output_scale(g_arena, head, Htr);
    ESP_LOGI(TAG, "head: w_scale=%.6f  out_scale=%.6f", head_w_scale, head.output_scale());

    AdamOptimizer opt;
    AdamOptimizer::Config cfg; cfg.learning_rate = 0.05f;
    opt.Init(g_arena, head, cfg);
    SparseUpdateSelector<1> sel; sel.Add(head);

    // Memory used by the trained head: a trainable int8 layer stores the int8
    // weight plus a float shadow, gradient and Adam m/v for each one, so
    // 17 bytes per weight and 20 per bias. The frozen extractor keeps only its
    // int8 weights and int32 bias, with no optimizer state.
    const size_t head_state = head.weight_count() * (1 + 4 + 4 + 4 + 4) +
                              head.bias_count() * (4 + 4 + 4 + 4 + 4);
    const size_t frozen_state = fe.weight_count() * 1 + fe.bias_count() * 4;
    ESP_LOGI(TAG, "trainable head state (int8 W + float shadow/grad/Adam): %u bytes (%u params)",
             (unsigned)head_state, (unsigned)(head.weight_count() + head.bias_count()));
    ESP_LOGI(TAG, "frozen extractor (int8, no optimizer state): %u bytes", (unsigned)frozen_state);

    ESP_LOGI(TAG, "initial test accuracy: %.1f%%", eval_head(g_arena, head, Hte, GDS_Y_TEST, NTE) * 100.0f);

    int64_t t0 = esp_timer_get_time();
    for (int e = 1; e <= EPOCHS; e++) {
        int64_t es = esp_timer_get_time();
        float loss;
        {
            ArenaScope scope(g_arena);
            Tensor<int8_t> logits_q = MakeTensor2D<int8_t>(g_arena, NTR, NC, head.output_scale(), 0);
            head.Forward(g_arena, Htr, logits_q);     // caches Htr as head input
            float* logits = g_arena.allocate<float>((size_t)NTR * NC);
            DequantizeBuffer(logits_q, logits);
            float* probs = g_arena.allocate<float>((size_t)NTR * NC);
            Softmax(logits, probs, NTR, NC);
            loss = CrossEntropyLoss(probs, GDS_Y_TRAIN, NTR, NC);
            float* dz = g_arena.allocate<float>((size_t)NTR * NC);
            Softmax_CrossEntropy_Backward(probs, GDS_Y_TRAIN, NTR, NC, dz);
            sel.ZeroAllGradients();
            head.Backward(g_arena, NTR, dz, /*dX_out=*/nullptr);  // frozen upstream -> no dX
            opt.Step(head);
        }
        int64_t ee = esp_timer_get_time();
        if (e == 1 || e % 100 == 0 || e == EPOCHS) {
            float te = eval_head(g_arena, head, Hte, GDS_Y_TEST, NTE);
            float tr = eval_head(g_arena, head, Htr, GDS_Y_TRAIN, NTR);
            ESP_LOGI(TAG, "epoch %4d  loss=%.4f  train=%.1f%%  test=%.1f%%  epoch_time=%lld us",
                     e, loss, tr * 100.0f, te * 100.0f, (long long)(ee - es));
        }
        // Yield periodically so the idle task runs (feeds the Task Watchdog).
        // Excluded from the per-epoch timing measured above.
        if (e % 10 == 0) vTaskDelay(1);
    }
    int64_t t1 = esp_timer_get_time();

    float te = eval_head(g_arena, head, Hte, GDS_Y_TEST, NTE);
    ESP_LOGI(TAG, "=== RESULT (INT8 QAS + sparse) ===");
    ESP_LOGI(TAG, "final test accuracy: %.1f%%", te * 100.0f);
    ESP_LOGI(TAG, "total train time: %lld us (%.2f s), avg %.0f us/epoch",
             (long long)(t1 - t0), (t1 - t0) / 1e6, (double)(t1 - t0) / EPOCHS);
    ESP_LOGI(TAG, "arena capacity: %u bytes (256 KB)", (unsigned)g_arena.capacity());
    ESP_LOGI(TAG, "arena PEAK (high-water): %u bytes (%.1f KB), %.2f%% of budget",
             (unsigned)g_arena.high_water_mark(), g_arena.high_water_mark() / 1024.0,
             100.0 * g_arena.high_water_mark() / g_arena.capacity());
}
