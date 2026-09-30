/*
 * FP32 baseline: trains the whole 64 -> 32 -> 5 network from scratch on the
 * board, in plain float32 with Adam, on the embedded dataset
 * (include/gesture_dataset.h).
 *
 * Every parameter, gradient and Adam value lives in a static float buffer, so
 * the memory footprint is exact. Prints per-epoch time, that footprint, and
 * test accuracy, to compare against the int8 + sparse version. Does not use
 * the int8 engine.
 */

#include <cmath>
#include <cstdint>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"

#include "gesture_dataset.h"

namespace {

constexpr char TAG[] = "fp32_train";

constexpr int NF = GDS_N_FEATURES;   // 64
constexpr int NH = GDS_HIDDEN;       // 32
constexpr int NC = GDS_N_CLASSES;    // 5
constexpr int NTR = GDS_N_TRAIN;     // 80
constexpr int NTE = GDS_N_TEST;      // 20

constexpr int EPOCHS = 800;
constexpr float LR = 0.05f;
// NB: 'EPS' is a reserved Xtensa register macro (specreg.h, via FreeRTOS),
// so the Adam epsilon is named ADAM_EPS to avoid the collision.
constexpr float B1 = 0.9f, B2 = 0.999f, ADAM_EPS = 1e-8f;

// ---- Parameters, gradients, Adam state: all static (no heap) ----
struct Net {
    float W1[NF][NH], b1[NH];
    float W2[NH][NC], b2[NC];
};
struct Grads {
    float W1[NF][NH], b1[NH];
    float W2[NH][NC], b2[NC];
};
Net   g_p;              // parameters
Grads g_g;              // gradients
Net   g_m, g_v;         // Adam moments (reuse Net layout)

// Small per-sample scratch.
float s_h[NH];
float s_logits[NC];
float s_probs[NC];

uint32_t g_rng = 0x1234567u;
float frand_signed() {  // uniform [-1,1)
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((float)(g_rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

void init_params() {
    const float l1 = 1.0f / sqrtf((float)NF);
    const float l2 = 1.0f / sqrtf((float)NH);
    for (int i = 0; i < NF; i++) for (int j = 0; j < NH; j++) g_p.W1[i][j] = frand_signed() * l1;
    for (int j = 0; j < NH; j++) g_p.b1[j] = 0.0f;
    for (int i = 0; i < NH; i++) for (int j = 0; j < NC; j++) g_p.W2[i][j] = frand_signed() * l2;
    for (int j = 0; j < NC; j++) g_p.b2[j] = 0.0f;
    std::memset(&g_m, 0, sizeof(g_m));
    std::memset(&g_v, 0, sizeof(g_v));
}

// Forward for one sample; returns predicted class, fills s_probs.
int forward(const float* x) {
    for (int j = 0; j < NH; j++) {
        float acc = g_p.b1[j];
        for (int i = 0; i < NF; i++) acc += x[i] * g_p.W1[i][j];
        s_h[j] = acc > 0.0f ? acc : 0.0f;   // ReLU
    }
    float maxl = -1e30f;
    for (int k = 0; k < NC; k++) {
        float acc = g_p.b2[k];
        for (int j = 0; j < NH; j++) acc += s_h[j] * g_p.W2[j][k];
        s_logits[k] = acc;
        if (acc > maxl) maxl = acc;
    }
    float sum = 0.0f;
    for (int k = 0; k < NC; k++) { s_probs[k] = expf(s_logits[k] - maxl); sum += s_probs[k]; }
    int pred = 0; float best = -1.0f;
    for (int k = 0; k < NC; k++) { s_probs[k] /= sum; if (s_probs[k] > best) { best = s_probs[k]; pred = k; } }
    return pred;
}

float evaluate(const float X[][NF], const int32_t* y, int n) {
    int correct = 0;
    for (int s = 0; s < n; s++) if (forward(X[s]) == y[s]) correct++;
    return (float)correct / (float)n;
}

// One full-batch training epoch; returns mean loss.
float train_epoch(int t) {
    std::memset(&g_g, 0, sizeof(g_g));
    float loss = 0.0f;
    for (int s = 0; s < NTR; s++) {
        const float* x = GDS_X_TRAIN[s];
        int y = GDS_Y_TRAIN[s];
        forward(x);
        loss += -logf(s_probs[y] > 1e-12f ? s_probs[y] : 1e-12f);
        // dL/dlogits = (p - onehot)/N
        float dz[NC];
        for (int k = 0; k < NC; k++) dz[k] = (s_probs[k] - (k == y ? 1.0f : 0.0f)) / (float)NTR;
        // head grads + backprop to hidden
        float dh[NH];
        for (int j = 0; j < NH; j++) dh[j] = 0.0f;
        for (int k = 0; k < NC; k++) {
            g_g.b2[k] += dz[k];
            for (int j = 0; j < NH; j++) {
                g_g.W2[j][k] += s_h[j] * dz[k];
                dh[j] += dz[k] * g_p.W2[j][k];
            }
        }
        for (int j = 0; j < NH; j++) {
            if (s_h[j] <= 0.0f) { dh[j] = 0.0f; continue; }   // ReLU grad
            g_g.b1[j] += dh[j];
            for (int i = 0; i < NF; i++) g_g.W1[i][j] += x[i] * dh[j];
        }
    }
    // Adam update over every parameter (flat view).
    float* p = (float*)&g_p; float* gr = (float*)&g_g;
    float* m = (float*)&g_m; float* v = (float*)&g_v;
    const int n = sizeof(Net) / sizeof(float);
    const float bc1 = 1.0f - powf(B1, (float)t);
    const float bc2 = 1.0f - powf(B2, (float)t);
    for (int i = 0; i < n; i++) {
        m[i] = B1 * m[i] + (1 - B1) * gr[i];
        v[i] = B2 * v[i] + (1 - B2) * gr[i] * gr[i];
        p[i] -= LR * (m[i] / bc1) / (sqrtf(v[i] / bc2) + ADAM_EPS);
    }
    return loss / (float)NTR;
}

} // namespace

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "=== Naive FP32 on-device gesture training ===");
    ESP_LOGI(TAG, "model %d->%d->%d, train=%d test=%d, epochs=%d", NF, NH, NC, NTR, NTE, EPOCHS);

    const size_t working = sizeof(g_p) + sizeof(g_g) + sizeof(g_m) + sizeof(g_v);
    ESP_LOGI(TAG, "training working memory (params+grads+Adam, static): %u bytes (%.1f KB)",
             (unsigned)working, working / 1024.0);
    ESP_LOGI(TAG, "  breakdown: params %u B + grads %u B + Adam(m,v) %u B; %d trainable params (WHOLE net)",
             (unsigned)sizeof(g_p), (unsigned)sizeof(g_g),
             (unsigned)(sizeof(g_m) + sizeof(g_v)),
             (int)(sizeof(Net) / sizeof(float)));

    init_params();
    ESP_LOGI(TAG, "initial test accuracy: %.1f%%", evaluate(GDS_X_TEST, GDS_Y_TEST, NTE) * 100.0f);

    int64_t t0 = esp_timer_get_time();
    for (int e = 1; e <= EPOCHS; e++) {
        int64_t es = esp_timer_get_time();
        float loss = train_epoch(e);
        int64_t ee = esp_timer_get_time();
        if (e == 1 || e % 100 == 0 || e == EPOCHS) {
            float tr = evaluate(GDS_X_TRAIN, GDS_Y_TRAIN, NTR);
            float te = evaluate(GDS_X_TEST, GDS_Y_TEST, NTE);
            ESP_LOGI(TAG, "epoch %4d  loss=%.4f  train=%.1f%%  test=%.1f%%  epoch_time=%lld us",
                     e, loss, tr * 100.0f, te * 100.0f, (long long)(ee - es));
        }
        // Yield periodically so the idle task runs (feeds the Task Watchdog).
        // Excluded from the per-epoch timing measured above.
        if (e % 10 == 0) vTaskDelay(1);
    }
    int64_t t1 = esp_timer_get_time();

    float te = evaluate(GDS_X_TEST, GDS_Y_TEST, NTE);
    ESP_LOGI(TAG, "=== RESULT (FP32 baseline) ===");
    ESP_LOGI(TAG, "final test accuracy: %.1f%%", te * 100.0f);
    ESP_LOGI(TAG, "total train time: %lld us (%.2f s), avg %.0f us/epoch",
             (long long)(t1 - t0), (t1 - t0) / 1e6, (double)(t1 - t0) / EPOCHS);
    ESP_LOGI(TAG, "working memory: %u bytes (%.1f KB)", (unsigned)working, working / 1024.0);
    ESP_LOGI(TAG, "heap min-free during run: %u bytes (confirms no heap used for training)",
             (unsigned)esp_get_minimum_free_heap_size());
}
