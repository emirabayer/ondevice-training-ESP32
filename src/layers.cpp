#include "layers.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace edge {

namespace {

[[noreturn]] void Panic(const char* message) {
    std::fprintf(stderr, "[edge::layers] PANIC: %s\n", message);
    std::abort();
}

} // namespace

void MatMul_INT8(Arena& scratch_arena, const Tensor<int8_t>& A, const Tensor<int8_t>& B,
                  const int32_t* bias, Tensor<int8_t>& C) {
    ArenaScope scope(scratch_arena);
    Tensor<int32_t> acc = MakeTensor2D<int32_t>(scratch_arena, A.shape[0], B.shape[1]);

    MatMulInt8Accumulate(A, B, acc);

    if (bias != nullptr) {
        const uint32_t M = acc.shape[0];
        const uint32_t N = acc.shape[1];
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                acc.at(m, n) += bias[n];
            }
        }
    }

    const double real_multiplier =
        static_cast<double>(A.scale) * static_cast<double>(B.scale) / static_cast<double>(C.scale);
    const FixedPointMultiplier multiplier = QuantizeMultiplier(real_multiplier);
    RequantizeAccumulator(acc, multiplier, C);
}

void ReLU_INT8(Tensor<int8_t>& x) {
    const int8_t zp = static_cast<int8_t>(x.zero_point);
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        if (x.data[i] < zp) {
            x.data[i] = zp;
        }
    }
}

void ReLU_Backward(const Tensor<int8_t>& activated, const float* dY, float* dX) {
    const int8_t zp = static_cast<int8_t>(activated.zero_point);
    const size_t n = activated.size();
    for (size_t i = 0; i < n; ++i) {
        dX[i] = (activated.data[i] > zp) ? dY[i] : 0.0f;
    }
}

void Softmax(const float* logits, float* probs, uint32_t batch, uint32_t num_classes) {
    for (uint32_t m = 0; m < batch; ++m) {
        const float* row = logits + static_cast<size_t>(m) * num_classes;
        float* prow = probs + static_cast<size_t>(m) * num_classes;

        float max_logit = row[0];
        for (uint32_t c = 1; c < num_classes; ++c) {
            if (row[c] > max_logit) max_logit = row[c];
        }

        float sum = 0.0f;
        for (uint32_t c = 0; c < num_classes; ++c) {
            prow[c] = std::exp(row[c] - max_logit);
            sum += prow[c];
        }

        const float inv_sum = 1.0f / sum;
        for (uint32_t c = 0; c < num_classes; ++c) {
            prow[c] *= inv_sum;
        }
    }
}

float CrossEntropyLoss(const float* probs, const int32_t* labels, uint32_t batch,
                        uint32_t num_classes) {
    constexpr float kEps = 1e-12f;
    float total = 0.0f;
    for (uint32_t m = 0; m < batch; ++m) {
        const float p = probs[static_cast<size_t>(m) * num_classes + labels[m]];
        total += -std::log(p > kEps ? p : kEps);
    }
    return total / static_cast<float>(batch);
}

void Softmax_CrossEntropy_Backward(const float* probs, const int32_t* labels, uint32_t batch,
                                    uint32_t num_classes, float* dz) {
    const float inv_batch = 1.0f / static_cast<float>(batch);
    for (uint32_t m = 0; m < batch; ++m) {
        for (uint32_t c = 0; c < num_classes; ++c) {
            const float y = (static_cast<int32_t>(c) == labels[m]) ? 1.0f : 0.0f;
            const size_t idx = static_cast<size_t>(m) * num_classes + c;
            dz[idx] = (probs[idx] - y) * inv_batch;
        }
    }
}

// ---------------------------------------------------------------------------
// DenseLayer

Tensor<int8_t> DenseLayer::WeightView() const {
    Tensor<int8_t> w;
    w.data = weight_q_;
    w.rank = 2;
    w.shape[0] = in_features_;
    w.shape[1] = out_features_;
    w.shape[2] = 1;
    w.shape[3] = 1;
    w.strides[0] = out_features_;
    w.strides[1] = 1;
    w.strides[2] = 1;
    w.strides[3] = 1;
    w.scale = weight_scale_;
    w.zero_point = 0;
    return w;
}

void DenseLayer::Init(Arena& persistent_arena, uint32_t in_features, uint32_t out_features,
                       float input_scale, float weight_scale, float output_scale,
                       bool trainable, Rng& rng) {
    if (in_features == 0 || out_features == 0) {
        Panic("DenseLayer::Init called with a zero-sized dimension");
    }

    in_features_ = in_features;
    out_features_ = out_features;
    input_scale_ = input_scale;
    weight_scale_ = weight_scale;
    output_scale_ = output_scale;
    trainable_ = trainable;

    weight_q_ = persistent_arena.allocate<int8_t>(weight_count());
    bias_q_ = persistent_arena.allocate<int32_t>(bias_count());

    // Xavier/Glorot-style uniform init range for the underlying real-valued
    // weights, independent of the quantization scale (a poor choice of
    // weight_scale_ relative to this range will simply clip at init, which
    // is a caller configuration concern, not a bug in the layer).
    const float limit = 1.0f / std::sqrt(static_cast<float>(in_features_));

    if (trainable_) {
        weight_shadow_ = persistent_arena.allocate<float>(weight_count());
        weight_grad_ = persistent_arena.allocate<float>(weight_count());
        bias_shadow_ = persistent_arena.allocate<float>(bias_count());
        bias_grad_ = persistent_arena.allocate<float>(bias_count());

        for (size_t i = 0; i < weight_count(); ++i) {
            weight_shadow_[i] = rng.NextUniformSigned() * limit;
        }
        for (size_t i = 0; i < bias_count(); ++i) {
            bias_shadow_[i] = 0.0f;
        }
        ZeroGradients();
        RequantizeWeights();
        RequantizeBias();
    } else {
        for (size_t i = 0; i < weight_count(); ++i) {
            const float w_real = rng.NextUniformSigned() * limit;
            weight_q_[i] = QuantizeInt8(w_real, weight_scale_, 0);
        }
        for (size_t i = 0; i < bias_count(); ++i) {
            bias_q_[i] = 0;
        }
    }
}

void DenseLayer::Forward(Arena& scratch_arena, const Tensor<int8_t>& X, Tensor<int8_t>& Y) {
    if (X.shape[1] != in_features_ || Y.shape[1] != out_features_ || X.shape[0] != Y.shape[0]) {
        Panic("DenseLayer::Forward called with mismatched tensor shapes");
    }

    const Tensor<int8_t> W = WeightView();
    MatMul_INT8(scratch_arena, X, W, bias_q_, Y);

    if (trainable_) {
        cached_input_ = X;
    }
}

void DenseLayer::Backward(Arena& scratch_arena, uint32_t batch, const float* dY,
                           float* dX_out) const {
    if (!trainable_) {
        Panic("DenseLayer::Backward called on a frozen (non-trainable) layer");
    }
    if (cached_input_.data == nullptr || cached_input_.shape[0] != batch) {
        Panic("DenseLayer::Backward called without a matching cached Forward() input");
    }

    ArenaScope scope(scratch_arena);
    float* X_real = scratch_arena.allocate<float>(static_cast<size_t>(batch) * in_features_);
    DequantizeBuffer(cached_input_, X_real);

    // dW[k,n] += sum_m X_real[m,k] * dY[m,n]
    for (uint32_t k = 0; k < in_features_; ++k) {
        for (uint32_t n = 0; n < out_features_; ++n) {
            float sum = 0.0f;
            for (uint32_t m = 0; m < batch; ++m) {
                sum += X_real[static_cast<size_t>(m) * in_features_ + k] *
                       dY[static_cast<size_t>(m) * out_features_ + n];
            }
            weight_grad_[static_cast<size_t>(k) * out_features_ + n] += sum;
        }
    }

    // db[n] += sum_m dY[m,n]
    for (uint32_t n = 0; n < out_features_; ++n) {
        float sum = 0.0f;
        for (uint32_t m = 0; m < batch; ++m) {
            sum += dY[static_cast<size_t>(m) * out_features_ + n];
        }
        bias_grad_[n] += sum;
    }

    // dX[m,k] = sum_n dY[m,n] * W_real[k,n]
    if (dX_out != nullptr) {
        for (uint32_t m = 0; m < batch; ++m) {
            for (uint32_t k = 0; k < in_features_; ++k) {
                float sum = 0.0f;
                for (uint32_t n = 0; n < out_features_; ++n) {
                    sum += dY[static_cast<size_t>(m) * out_features_ + n] *
                           weight_shadow_[static_cast<size_t>(k) * out_features_ + n];
                }
                dX_out[static_cast<size_t>(m) * in_features_ + k] = sum;
            }
        }
    }
}

void DenseLayer::ZeroGradients() {
    if (!trainable_) return;
    for (size_t i = 0; i < weight_count(); ++i) weight_grad_[i] = 0.0f;
    for (size_t i = 0; i < bias_count(); ++i) bias_grad_[i] = 0.0f;
}

void DenseLayer::RequantizeWeights() {
    const float* src = trainable_ ? weight_shadow_ : nullptr;
    if (src == nullptr) return; // frozen layers quantize once at Init and never change
    for (size_t i = 0; i < weight_count(); ++i) {
        weight_q_[i] = QuantizeInt8(src[i], weight_scale_, 0);
    }
}

void DenseLayer::RequantizeBias() {
    if (!trainable_) return;
    const float scale = bias_scale();
    for (size_t i = 0; i < bias_count(); ++i) {
        const double scaled = static_cast<double>(bias_shadow_[i]) / static_cast<double>(scale);
        double rounded = std::lround(scaled);
        if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min())) {
            rounded = static_cast<double>(std::numeric_limits<int32_t>::min());
        }
        if (rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            rounded = static_cast<double>(std::numeric_limits<int32_t>::max());
        }
        bias_q_[i] = static_cast<int32_t>(rounded);
    }
}

void DenseLayer::LoadFrozenFromFloat(const float* W, const float* b) {
    if (trainable_) {
        Panic("LoadFrozenFromFloat called on a trainable layer (use the shadow + optimizer instead)");
    }
    for (size_t i = 0; i < weight_count(); ++i) {
        weight_q_[i] = QuantizeInt8(W[i], weight_scale_, 0);
    }
    const float scale = bias_scale();
    for (size_t i = 0; i < bias_count(); ++i) {
        double rounded = (b == nullptr) ? 0.0
                          : std::lround(static_cast<double>(b[i]) / static_cast<double>(scale));
        if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min())) {
            rounded = static_cast<double>(std::numeric_limits<int32_t>::min());
        }
        if (rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            rounded = static_cast<double>(std::numeric_limits<int32_t>::max());
        }
        bias_q_[i] = static_cast<int32_t>(rounded);
    }
}

} // namespace edge
