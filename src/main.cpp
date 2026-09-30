// End-to-end training demonstration.
//
// Architecture: a frozen random-projection "feature extractor" DenseLayer
// (L1) followed by ReLU and a trainable INT8 classification head (L2). This
// is the sparse layer update in action: L1 never allocates gradient/shadow
// buffers and is never touched by Backward(), so only L2's tiny parameter
// set actually trains.
//
// Everything that participates in the forward/backward passes, weights,
// biases, gradients, optimizer state, activation scratch, is allocated
// from a single fixed 256KB edge::Arena. The only memory outside that
// arena is the synthetic dataset itself (a handful of KB of plain static
// storage standing in for "data streamed from flash/sensor" on a real
// target, exactly as it would never live in the training engine's own
// working memory) and this file's own bookkeeping locals.

#include <cstdint>
#include <cstdio>

#include "arena.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "qas.hpp"
#include "rng.hpp"
#include "tensor.hpp"

using namespace edge;

namespace {

constexpr uint32_t kInputDim = 12;
constexpr uint32_t kHiddenDim = 24;
constexpr uint32_t kNumClasses = 3;
constexpr uint32_t kSamplesPerClass = 100;
constexpr uint32_t kNumSamples = kNumClasses * kSamplesPerClass;
constexpr uint32_t kNumTrain = 240; // remaining 60 held out as the test split
constexpr uint32_t kNumTest = kNumSamples - kNumTrain;
constexpr uint32_t kBatchSize = 20;
constexpr uint32_t kNumEpochs = 60;

// Synthetic dataset storage: NOT part of the 256KB engine arena (see file
// header). Static, fixed-size, no heap.
float g_features[kNumSamples][kInputDim];
int32_t g_labels[kNumSamples];
uint32_t g_shuffled_index[kNumSamples];

// Three well-separated random cluster centers in kInputDim-space, each
// class's samples drawn uniformly around its center, an easy but genuine
// classification problem, adequate for demonstrating that the quantized
// training pipeline actually converges rather than benchmarking accuracy.
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

// Quantizes `count` samples (via g_shuffled_index starting at `offset`) into
// a caller-allocated INT8 tensor of shape [count, kInputDim].
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

// One-shot post-init calibration of the output scale, applied once before
// training starts rather than continuously,
// since a layer's output_scale also has to stay in lockstep with the next
// layer's input_scale, see qas.hpp / DynamicScaleTracker's doc comment).
// Runs a forward pass with a provisional unit output scale, observes the
// real dequantized activation range it actually produced, and re-targets
// the layer's output_scale to match.
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

// Runs one full pass over [offset, offset+count) samples. If `train` is
// true, also backprops through the head and takes one optimizer step per
// batch; otherwise it's a pure evaluation pass (used for the held-out test
// set).
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
            // dX_out == nullptr: the feature extractor is frozen, so there is
            // no layer upstream of `head` that needs a gradient, which is
            // where the sparse update saves the work.
            head.Backward(arena, batch, dz, nullptr);
            opt->Step(head);
        }
    }

    EpochStats stats;
    stats.mean_loss = total_loss / static_cast<float>(total_seen);
    stats.accuracy = static_cast<float>(total_correct) / static_cast<float>(total_seen);
    return stats;
}

} // namespace

int main() {
    Arena arena;
    Rng rng(20260822u);

    GenerateSyntheticDataset(rng);
    const float input_scale = DatasetMaxAbs() / 127.0f;

    std::printf("=== Edge Quantized Training Engine: convergence demo ===\n");
    std::printf("dataset: %u samples (%u train / %u test), input_dim=%u, classes=%u\n",
                kNumSamples, kNumTrain, kNumTest, kInputDim, kNumClasses);
    std::printf("input_scale=%.6f (calibrated from data max-abs)\n\n", input_scale);

    // L1: frozen "feature extractor". Weight scale sized to the Xavier-ish
    // init range (1/sqrt(fan_in)) so the initial random weights use most of
    // the INT8 range instead of clipping or collapsing to a few levels.
    DenseLayer feature_extractor;
    const float weight_scale_l1 = (1.0f / 4.0f) / 127.0f; // ~ 1/sqrt(12) / 127
    feature_extractor.Init(arena, kInputDim, kHiddenDim, input_scale, weight_scale_l1,
                            /*output_scale placeholder, calibrated below=*/1.0f,
                            /*trainable=*/false, rng);

    // Calibrate L1's output scale using the first training batch, then
    // build L2 with a matching input_scale (DenseLayer's input_scale_ is
    // fixed at Init(), so this ordering, calibrate producer before
    // constructing consumer, matters).
    float calib_input_scale;
    {
        ArenaScope scope(arena);
        Tensor<int8_t> calib_X = MakeTensor2D<int8_t>(arena, kBatchSize, kInputDim, input_scale, 0);
        int32_t calib_labels[kBatchSize];
        QuantizeBatch(0, kBatchSize, input_scale, calib_X, calib_labels);
        calib_input_scale = CalibrateOutputScale(arena, feature_extractor, calib_X);
    }
    std::printf("L1 (frozen %ux%u) calibrated output_scale=%.6f\n", kInputDim, kHiddenDim,
                calib_input_scale);

    // L2: trainable classification head.
    DenseLayer head;
    const float weight_scale_l2 = (1.0f / 5.0f) / 127.0f; // ~ 1/sqrt(24) / 127
    head.Init(arena, kHiddenDim, kNumClasses, feature_extractor.output_scale(), weight_scale_l2,
              /*output_scale placeholder, calibrated below=*/1.0f, /*trainable=*/true, rng);

    {
        ArenaScope scope(arena);
        Tensor<int8_t> calib_X =
            MakeTensor2D<int8_t>(arena, kBatchSize, kInputDim, input_scale, 0);
        int32_t calib_labels[kBatchSize];
        QuantizeBatch(0, kBatchSize, input_scale, calib_X, calib_labels);

        Tensor<int8_t> calib_hidden = MakeTensor2D<int8_t>(arena, kBatchSize, kHiddenDim,
                                                             feature_extractor.output_scale(), 0);
        feature_extractor.Forward(arena, calib_X, calib_hidden);
        ReLU_INT8(calib_hidden);
        CalibrateOutputScale(arena, head, calib_hidden);
    }
    std::printf("L2 (trainable head %ux%u) calibrated output_scale=%.6f\n\n", kHiddenDim,
                kNumClasses, head.output_scale());

    AdamOptimizer optimizer;
    AdamOptimizer::Config opt_config;
    opt_config.learning_rate = 0.08f;
    optimizer.Init(arena, head, opt_config);

    SparseUpdateSelector<1> selector;
    selector.Add(head);

    std::printf("training: %u epochs, batch_size=%u, optimizer=Adam(lr=%.3f)\n\n", kNumEpochs,
                kBatchSize, opt_config.learning_rate);
    std::printf("%-6s %-12s %-10s %-12s %-10s\n", "epoch", "train_loss", "train_acc", "test_loss",
                "test_acc");

    for (uint32_t epoch = 1; epoch <= kNumEpochs; ++epoch) {
        const EpochStats train_stats =
            RunEpoch(arena, feature_extractor, head, &optimizer, &selector, input_scale, 0,
                      kNumTrain, /*train=*/true);
        const EpochStats test_stats =
            RunEpoch<AdamOptimizer>(arena, feature_extractor, head, nullptr, nullptr, input_scale,
                                     kNumTrain, kNumTest, /*train=*/false);

        if (epoch == 1 || epoch % 5 == 0 || epoch == kNumEpochs) {
            std::printf("%-6u %-12.5f %-10.4f %-12.5f %-10.4f\n", epoch, train_stats.mean_loss,
                        train_stats.accuracy, test_stats.mean_loss, test_stats.accuracy);
        }
    }

    const EpochStats final_test = RunEpoch<AdamOptimizer>(
        arena, feature_extractor, head, nullptr, nullptr, input_scale, kNumTrain, kNumTest, false);

    std::printf("\nfinal held-out test accuracy: %.2f%%\n", final_test.accuracy * 100.0f);

    std::printf("\n=== Memory report ===\n");
    std::printf("arena capacity:        %6zu bytes (%.1f KB)\n", arena.capacity(),
                static_cast<double>(arena.capacity()) / 1024.0);
    std::printf("arena in use (final):  %6zu bytes (%.1f KB)\n", arena.used(),
                static_cast<double>(arena.used()) / 1024.0);
    std::printf("arena peak (high-water mark): %6zu bytes (%.1f KB), %.2f%% of the 256KB budget\n",
                arena.high_water_mark(), static_cast<double>(arena.high_water_mark()) / 1024.0,
                100.0 * static_cast<double>(arena.high_water_mark()) /
                    static_cast<double>(arena.capacity()));

    return (final_test.accuracy >= 0.8f) ? 0 : 1;
}
