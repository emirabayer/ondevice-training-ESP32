// Bare-metal ESP32-S3 (ESP-IDF v5.x) entry point.
//
// This is the firmware equivalent of src/main.cpp: same architecture
// (frozen random-projection feature extractor -> ReLU -> trainable INT8
// head), same synthetic dataset generation, same training loop, built
// directly against the shared engine sources in ../../src via
// main/CMakeLists.txt (not copied). It differs from the host demo only in
// how it does I/O and timing: ESP_LOGI instead of stdio, plus explicit
// per-epoch wall-clock profiling (esp_timer_get_time()) and a cycle-exact
// benchmark of the INT8 MatMul hot path (esp_cpu_get_cycle_count()), since
// neither of those has a meaningful host-side equivalent.
//
// NOTE on ESP_LOGI's "%f": ESP-IDF defaults to newlib's "nano" formatting
// (CONFIG_NEWLIB_NANO_FORMAT=y), which drops floating point support from
// printf-family functions. If floats render as garbage/empty on the serial
// monitor, disable that option (or enable
// CONFIG_NEWLIB_NANO_FORMAT_FLOAT via `idf.py menuconfig` ->
// Component config -> Newlib), see README.md.

#include <cstdint>

#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "arena.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "qas.hpp"
#include "rng.hpp"
#include "tensor.hpp"

using namespace edge;

namespace {

constexpr char TAG[] = "edge_train";

constexpr uint32_t kInputDim = 12;
constexpr uint32_t kHiddenDim = 24;
constexpr uint32_t kNumClasses = 3;
constexpr uint32_t kSamplesPerClass = 100;
constexpr uint32_t kNumSamples = kNumClasses * kSamplesPerClass;
constexpr uint32_t kNumTrain = 240;
constexpr uint32_t kNumTest = kNumSamples - kNumTrain;
constexpr uint32_t kBatchSize = 20;
constexpr uint32_t kNumEpochs = 60;
constexpr uint32_t kHotPathBenchIterations = 20;

// The 256KB arena as a *static* object: it must never be a stack local --
// app_main()'s task stack (sdkconfig CONFIG_ESP_MAIN_TASK_STACK_SIZE) is a
// few KB by default, nowhere near large enough to hold Arena's 256KB
// internal buffer. Static storage duration places it in .bss instead,
// which puts it in the single fixed buffer the engine needs. On chip
// variants/sdkconfig profiles where internal SRAM is too
// tight for 256KB of .bss once the rest of the app (WiFi/BT stacks, FreeRTOS
// heap, etc.) is linked in, either trim those components or place this in
// external PSRAM by adding `EXT_RAM_BSS_ATTR` here (requires
// CONFIG_SPIRAM=y in sdkconfig).
static Arena g_arena;

// Synthetic dataset storage: also static, also outside the 256KB engine
// arena, standing in for data that would in practice stream from flash or
// a sensor rather than sit permanently in the training engine's own
// working memory.
float g_features[kNumSamples][kInputDim];
int32_t g_labels[kNumSamples];
uint32_t g_shuffled_index[kNumSamples];

void GenerateSyntheticDataset(Rng& rng) {
    float centers[kNumClasses][kInputDim];
    for (uint32_t c = 0; c < kNumClasses; ++c) {
        for (uint32_t d = 0; d < kInputDim; ++d) {
            centers[c][d] = rng.NextUniformSigned() * 3.0f;
        }
    }

    uint32_t idx = 0;
    for (uint32_t c = 0; c < kNumClasses; ++c) {
        for (uint32_t s = 0; s < kSamplesPerClass; ++s) {
            for (uint32_t d = 0; d < kInputDim; ++d) {
                g_features[idx][d] = centers[c][d] + rng.NextUniformSigned() * 0.6f;
            }
            g_labels[idx] = static_cast<int32_t>(c);
            ++idx;
        }
    }

    for (uint32_t i = 0; i < kNumSamples; ++i) g_shuffled_index[i] = i;
    for (uint32_t i = kNumSamples - 1; i > 0; --i) {
        const uint32_t j = rng.NextU32() % (i + 1);
        const uint32_t tmp = g_shuffled_index[i];
        g_shuffled_index[i] = g_shuffled_index[j];
        g_shuffled_index[j] = tmp;
    }
}

float DatasetMaxAbs() {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < kNumSamples; ++i) {
        for (uint32_t d = 0; d < kInputDim; ++d) {
            const float a = g_features[i][d] < 0.0f ? -g_features[i][d] : g_features[i][d];
            if (a > max_abs) max_abs = a;
        }
    }
    return max_abs;
}

void QuantizeBatch(uint32_t offset, uint32_t count, float input_scale, Tensor<int8_t>& out,
                    int32_t* labels_out) {
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t sample = g_shuffled_index[offset + i];
        for (uint32_t d = 0; d < kInputDim; ++d) {
            out.at(i, d) = QuantizeInt8(g_features[sample][d], input_scale, 0);
        }
        labels_out[i] = g_labels[sample];
    }
}

// One-shot QAS calibration, see qas.hpp / DynamicScaleTracker and the
// identical helper in src/main.cpp for the full rationale.
float CalibrateOutputScale(Arena& arena, DenseLayer& layer, const Tensor<int8_t>& X) {
    ArenaScope scope(arena);
    Tensor<int8_t> Y =
        MakeTensor2D<int8_t>(arena, X.shape[0], layer.out_features(), /*scale=*/1.0f, 0);
    layer.Forward(arena, X, Y);

    float* real = arena.allocate<float>(Y.size());
    DequantizeBuffer(Y, real);

    DynamicScaleTracker tracker(/*initial_scale=*/1.0f);
    const float calibrated = tracker.Observe(MaxAbs(real, Y.size()));
    layer.SetOutputScale(calibrated);
    return calibrated;
}

struct EpochStats {
    float mean_loss = 0.0f;
    float accuracy = 0.0f;
};

template <typename Optimizer>
EpochStats RunEpoch(Arena& arena, DenseLayer& feature_extractor, DenseLayer& head, Optimizer* opt,
                     SparseUpdateSelector<1>* selector, float input_scale, uint32_t offset,
                     uint32_t count, bool train) {
    float total_loss = 0.0f;
    uint32_t total_correct = 0;
    uint32_t total_seen = 0;

    for (uint32_t start = 0; start < count; start += kBatchSize) {
        const uint32_t batch = (count - start < kBatchSize) ? (count - start) : kBatchSize;

        ArenaScope scope(arena);
        Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, batch, kInputDim, input_scale, 0);
        int32_t labels[kBatchSize];
        QuantizeBatch(offset + start, batch, input_scale, X, labels);

        Tensor<int8_t> hidden =
            MakeTensor2D<int8_t>(arena, batch, kHiddenDim, feature_extractor.output_scale(), 0);
        feature_extractor.Forward(arena, X, hidden);
        ReLU_INT8(hidden);

        Tensor<int8_t> logits_q =
            MakeTensor2D<int8_t>(arena, batch, kNumClasses, head.output_scale(), 0);
        head.Forward(arena, hidden, logits_q);

        float* logits = arena.allocate<float>(static_cast<size_t>(batch) * kNumClasses);
        DequantizeBuffer(logits_q, logits);
        float* probs = arena.allocate<float>(static_cast<size_t>(batch) * kNumClasses);
        Softmax(logits, probs, batch, kNumClasses);
        const float batch_loss = CrossEntropyLoss(probs, labels, batch, kNumClasses);

        total_loss += batch_loss * static_cast<float>(batch);
        for (uint32_t m = 0; m < batch; ++m) {
            uint32_t predicted = 0;
            float best = probs[m * kNumClasses];
            for (uint32_t c = 1; c < kNumClasses; ++c) {
                if (probs[m * kNumClasses + c] > best) {
                    best = probs[m * kNumClasses + c];
                    predicted = c;
                }
            }
            if (static_cast<int32_t>(predicted) == labels[m]) ++total_correct;
        }
        total_seen += batch;

        if (train) {
            float* dz = arena.allocate<float>(static_cast<size_t>(batch) * kNumClasses);
            Softmax_CrossEntropy_Backward(probs, labels, batch, kNumClasses, dz);

            selector->ZeroAllGradients();
            head.Backward(arena, batch, dz, nullptr);
            opt->Step(head);
        }
    }

    EpochStats stats;
    stats.mean_loss = total_loss / static_cast<float>(total_seen);
    stats.accuracy = static_cast<float>(total_correct) / static_cast<float>(total_seen);
    return stats;
}

// Cycle-exact benchmark of the INT8 MatMul hot path (MatMul_INT8, as invoked
// by DenseLayer::Forward for the feature extractor, the single largest matmul
// in this network, kBatchSize x
// kInputDim x kHiddenDim). Runs several iterations and reports both the
// minimum (least disturbed by interrupts/cache effects) and average cycle
// count using the Xtensa cycle counter via esp_cpu_get_cycle_count().
void BenchmarkMatMulHotPath(Arena& arena, DenseLayer& feature_extractor, float input_scale) {
    ArenaScope scope(arena);
    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, kBatchSize, kInputDim, input_scale, 0);
    int32_t labels[kBatchSize];
    QuantizeBatch(0, kBatchSize, input_scale, X, labels);
    Tensor<int8_t> hidden =
        MakeTensor2D<int8_t>(arena, kBatchSize, kHiddenDim, feature_extractor.output_scale(), 0);

    // Warm up (first invocation can pay a one-time icache/branch-predictor
    // cost that would otherwise skew the very first sample).
    feature_extractor.Forward(arena, X, hidden);

    uint32_t min_cycles = UINT32_MAX;
    uint64_t total_cycles = 0;
    for (uint32_t i = 0; i < kHotPathBenchIterations; ++i) {
        const uint32_t start_cycles = esp_cpu_get_cycle_count();
        feature_extractor.Forward(arena, X, hidden);
        const uint32_t end_cycles = esp_cpu_get_cycle_count();

        const uint32_t elapsed = end_cycles - start_cycles;
        total_cycles += elapsed;
        if (elapsed < min_cycles) min_cycles = elapsed;
    }
    const uint32_t avg_cycles =
        static_cast<uint32_t>(total_cycles / kHotPathBenchIterations);
    const uint32_t macs = kBatchSize * kInputDim * kHiddenDim;

    ESP_LOGI(TAG, "INT8 MatMul hot path: DenseLayer::Forward %ux%ux%u (%u MACs), %u iters:",
             kBatchSize, kInputDim, kHiddenDim, macs, kHotPathBenchIterations);
    ESP_LOGI(TAG, "  min=%u cycles, avg=%u cycles (%.3f cycles/MAC)", min_cycles, avg_cycles,
             static_cast<double>(avg_cycles) / static_cast<double>(macs));
}

} // namespace

extern "C" void app_main(void) {
    Rng rng(20260822u);

    GenerateSyntheticDataset(rng);
    const float input_scale = DatasetMaxAbs() / 127.0f;

    ESP_LOGI(TAG, "=== Edge Quantized Training Engine: ESP32-S3 firmware demo ===");
    ESP_LOGI(TAG, "dataset: %u samples (%u train / %u test), input_dim=%u, classes=%u",
             kNumSamples, kNumTrain, kNumTest, kInputDim, kNumClasses);
    ESP_LOGI(TAG, "input_scale=%.6f (calibrated from data max-abs)", input_scale);

    DenseLayer feature_extractor;
    const float weight_scale_l1 = (1.0f / 4.0f) / 127.0f; // ~ 1/sqrt(12) / 127
    feature_extractor.Init(g_arena, kInputDim, kHiddenDim, input_scale, weight_scale_l1,
                            /*output_scale placeholder, calibrated below=*/1.0f,
                            /*trainable=*/false, rng);

    float calib_output_scale;
    {
        ArenaScope scope(g_arena);
        Tensor<int8_t> calib_X =
            MakeTensor2D<int8_t>(g_arena, kBatchSize, kInputDim, input_scale, 0);
        int32_t calib_labels[kBatchSize];
        QuantizeBatch(0, kBatchSize, input_scale, calib_X, calib_labels);
        calib_output_scale = CalibrateOutputScale(g_arena, feature_extractor, calib_X);
    }
    ESP_LOGI(TAG, "L1 (frozen %ux%u) calibrated output_scale=%.6f", kInputDim, kHiddenDim,
             calib_output_scale);

    BenchmarkMatMulHotPath(g_arena, feature_extractor, input_scale);

    DenseLayer head;
    const float weight_scale_l2 = (1.0f / 5.0f) / 127.0f; // ~ 1/sqrt(24) / 127
    head.Init(g_arena, kHiddenDim, kNumClasses, feature_extractor.output_scale(), weight_scale_l2,
              /*output_scale placeholder, calibrated below=*/1.0f, /*trainable=*/true, rng);

    {
        ArenaScope scope(g_arena);
        Tensor<int8_t> calib_X =
            MakeTensor2D<int8_t>(g_arena, kBatchSize, kInputDim, input_scale, 0);
        int32_t calib_labels[kBatchSize];
        QuantizeBatch(0, kBatchSize, input_scale, calib_X, calib_labels);

        Tensor<int8_t> calib_hidden = MakeTensor2D<int8_t>(
            g_arena, kBatchSize, kHiddenDim, feature_extractor.output_scale(), 0);
        feature_extractor.Forward(g_arena, calib_X, calib_hidden);
        ReLU_INT8(calib_hidden);
        CalibrateOutputScale(g_arena, head, calib_hidden);
    }
    ESP_LOGI(TAG, "L2 (trainable head %ux%u) calibrated output_scale=%.6f", kHiddenDim,
             kNumClasses, head.output_scale());

    AdamOptimizer optimizer;
    AdamOptimizer::Config opt_config;
    opt_config.learning_rate = 0.08f;
    optimizer.Init(g_arena, head, opt_config);

    SparseUpdateSelector<1> selector;
    selector.Add(head);

    ESP_LOGI(TAG, "training: %u epochs, batch_size=%u, optimizer=Adam(lr=%.3f)", kNumEpochs,
             kBatchSize, opt_config.learning_rate);

    const int64_t training_start_us = esp_timer_get_time();
    for (uint32_t epoch = 1; epoch <= kNumEpochs; ++epoch) {
        const int64_t epoch_start_us = esp_timer_get_time();
        const EpochStats train_stats =
            RunEpoch(g_arena, feature_extractor, head, &optimizer, &selector, input_scale, 0,
                      kNumTrain, /*train=*/true);
        const int64_t epoch_end_us = esp_timer_get_time();

        const EpochStats test_stats =
            RunEpoch<AdamOptimizer>(g_arena, feature_extractor, head, nullptr, nullptr,
                                     input_scale, kNumTrain, kNumTest, /*train=*/false);

        ESP_LOGI(TAG,
                 "epoch %2u: %6lld us  train_loss=%.5f train_acc=%.4f  "
                 "test_loss=%.5f test_acc=%.4f",
                 epoch, static_cast<long long>(epoch_end_us - epoch_start_us),
                 train_stats.mean_loss, train_stats.accuracy, test_stats.mean_loss,
                 test_stats.accuracy);
    }
    const int64_t training_end_us = esp_timer_get_time();

    const EpochStats final_test =
        RunEpoch<AdamOptimizer>(g_arena, feature_extractor, head, nullptr, nullptr, input_scale,
                                 kNumTrain, kNumTest, false);

    ESP_LOGI(TAG, "total training wall time: %lld us (%.2f s) over %u epochs",
             static_cast<long long>(training_end_us - training_start_us),
             static_cast<double>(training_end_us - training_start_us) / 1e6, kNumEpochs);
    ESP_LOGI(TAG, "final held-out test accuracy: %.2f%%", final_test.accuracy * 100.0f);

    ESP_LOGI(TAG, "=== Memory report ===");
    ESP_LOGI(TAG, "arena capacity:       %u bytes (%.1f KB)",
             static_cast<unsigned>(g_arena.capacity()),
             static_cast<double>(g_arena.capacity()) / 1024.0);
    ESP_LOGI(TAG, "arena in use (final): %u bytes (%.1f KB)",
             static_cast<unsigned>(g_arena.used()), static_cast<double>(g_arena.used()) / 1024.0);
    ESP_LOGI(TAG, "arena peak (high-water mark): %u bytes (%.1f KB), %.2f%% of the 256KB budget",
             static_cast<unsigned>(g_arena.high_water_mark()),
             static_cast<double>(g_arena.high_water_mark()) / 1024.0,
             100.0 * static_cast<double>(g_arena.high_water_mark()) /
                 static_cast<double>(g_arena.capacity()));
}
