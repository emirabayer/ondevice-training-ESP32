#pragma once

#include <cstdint>

#include "arena.hpp"
#include "rng.hpp"
#include "tensor.hpp"

namespace edge {

// ---------------------------------------------------------------------------
// Standalone INT8 matmul kernel: C = requantize(A@B [+ bias]).
// `bias` is an optional (may be nullptr) INT32 buffer of length B.shape[1],
// added to the accumulator in the bias's own fixed-point domain (scale =
// A.scale * B.scale, the standard quantized-bias convention) before
// requantization. All scratch (the INT32 accumulator) comes from
// `scratch_arena` and is reclaimed before this function returns.
void MatMul_INT8(Arena& scratch_arena, const Tensor<int8_t>& A, const Tensor<int8_t>& B,
                  const int32_t* bias, Tensor<int8_t>& C);

// In-place INT8 ReLU: clamps every element below the tensor's zero_point up
// to the zero_point (i.e. max(x, 0) for the symmetric/zero_point==0 case
// used throughout this engine).
void ReLU_INT8(Tensor<int8_t>& x);

// Backward pass for ReLU_INT8. `activated` is the (already-ReLU'd) output
// tensor cached from the forward pass; dY/dX are real-valued (dequantized
// domain) gradient buffers of activated.size() elements. Straight-through:
// dX[i] = dY[i] if activated[i] was > zero_point (i.e. the unit was active
// on the forward pass), else 0.
void ReLU_Backward(const Tensor<int8_t>& activated, const float* dY, float* dX);

// Row-wise softmax over `batch` rows of `num_classes` real-valued logits.
void Softmax(const float* logits, float* probs, uint32_t batch, uint32_t num_classes);

// Mean cross-entropy loss over the batch, given sparse integer class labels
// (labels[m] in [0, num_classes)) rather than materialized one-hot vectors.
float CrossEntropyLoss(const float* probs, const int32_t* labels, uint32_t batch,
                        uint32_t num_classes);

// Analytical gradient of mean Softmax+CrossEntropy loss w.r.t. the logits:
//   dL/dz_i = (p_i - y_i) / batch
// The 1/batch factor folds in the mean reduction used by CrossEntropyLoss
// above, so downstream weight gradients are scaled correctly without a
// separate normalization pass.
void Softmax_CrossEntropy_Backward(const float* probs, const int32_t* labels, uint32_t batch,
                                    uint32_t num_classes, float* dz);

// ---------------------------------------------------------------------------
// A single INT8-quantized fully-connected layer: Y = quantize(X @ W + b).
//
// Every layer keeps its INT8 weights (`weight_q_`) as the ground truth used
// by Forward(). A *trainable* layer additionally keeps a full-precision
// "shadow" copy of its weights/bias plus float gradient accumulators. A
// frozen layer allocates none of that, which is what makes the sparse layer
// update save memory.
//
// Backward() computes gradients via the straight-through estimator: it
// treats the quantize/dequantize round-trip as identity for gradient
// purposes and differentiates the underlying real-valued linear map
// Y_real = X_real @ W_real + b_real using the cached dequantized input and
// the float shadow weights. This is standard practice for quantization-aware
// training, since round() is piecewise-constant and not usefully
// differentiable on its own.
class DenseLayer {
public:
    DenseLayer() = default;

    // Allocates this layer's parameters from `persistent_arena` (NOT a
    // scratch/scoped arena, these buffers must outlive many training
    // steps) and randomly initializes weights (He/Xavier-style scaled
    // uniform noise) via `rng`. `trainable` gates whether the float shadow
    // weight/bias and gradient accumulator buffers are allocated at all.
    void Init(Arena& persistent_arena, uint32_t in_features, uint32_t out_features,
              float input_scale, float weight_scale, float output_scale, bool trainable,
              Rng& rng);

    // Quantized INT8 forward pass. X: [batch, in_features], Y: [batch,
    // out_features] (caller-allocated, e.g. via MakeTensor2D). If this layer
    // is trainable, the (non-owning) view of X is cached for Backward(): the
    // caller must keep X's backing memory alive until Backward() has run.
    void Forward(Arena& scratch_arena, const Tensor<int8_t>& X, Tensor<int8_t>& Y);

    // Requires this layer to be trainable (Init(..., trainable=true, ...)
    // and Forward() to have already been called this step, so a cached
    // input is available). `dY`: [batch, out_features] real-valued gradient
    // of the loss w.r.t. this layer's real-valued output. `dX_out`: optional
    // [batch, in_features] real-valued gradient buffer to receive dL/dX_real
    // (pass nullptr to skip, e.g. because everything upstream is frozen).
    // Accumulates (adds) into the float weight/bias gradient buffers; call
    // ZeroGradients() between optimizer steps.
    void Backward(Arena& scratch_arena, uint32_t batch, const float* dY, float* dX_out) const;

    void ZeroGradients();

    // Re-derives the INT8 ground truth from the float shadow parameters.
    // Called by the optimizer after it updates the shadow via SGD/Adam.
    void RequantizeWeights();
    void RequantizeBias();

    // Loads externally pre-trained (e.g. host-trained) parameters into this
    // layer's INT8 ground truth, quantized with the layer's configured
    // scales. Intended for a FROZEN feature-extractor layer whose weights
    // come from a pre-trained backbone that stays frozen. `W` is row-major
    // [in_features, out_features]; `b` (length out_features, may be nullptr)
    // is quantized
    // into the INT32 accumulator domain (scale = input_scale * weight_scale).
    void LoadFrozenFromFloat(const float* W, const float* b);

    bool trainable() const noexcept { return trainable_; }
    uint32_t in_features() const noexcept { return in_features_; }
    uint32_t out_features() const noexcept { return out_features_; }
    float input_scale() const noexcept { return input_scale_; }
    float weight_scale() const noexcept { return weight_scale_; }
    float output_scale() const noexcept { return output_scale_; }
    float bias_scale() const noexcept { return input_scale_ * weight_scale_; }

    // Re-targets the scale this layer's output is expected to be quantized
    // at (callers must allocate this layer's next Forward() output tensor
    // with this scale). Used for one-shot post-init QAS calibration
    // (qas.hpp's DynamicScaleTracker); safe to call at any time since
    // output_scale_ does not feed any of this layer's own internal
    // computation, only bias_scale() (input_scale_ * weight_scale_) does.
    void SetOutputScale(float scale) noexcept { output_scale_ = scale; }

    size_t weight_count() const noexcept {
        return static_cast<size_t>(in_features_) * out_features_;
    }
    size_t bias_count() const noexcept { return out_features_; }

    float* weight_shadow() noexcept { return weight_shadow_; }
    const float* weight_shadow() const noexcept { return weight_shadow_; }
    float* weight_grad() noexcept { return weight_grad_; }
    const float* weight_grad() const noexcept { return weight_grad_; }
    float* bias_shadow() noexcept { return bias_shadow_; }
    const float* bias_shadow() const noexcept { return bias_shadow_; }
    float* bias_grad() noexcept { return bias_grad_; }
    const float* bias_grad() const noexcept { return bias_grad_; }

    const int8_t* weight_q() const noexcept { return weight_q_; }
    const int32_t* bias_q() const noexcept { return bias_q_; }

private:
    Tensor<int8_t> WeightView() const;

    uint32_t in_features_ = 0;
    uint32_t out_features_ = 0;
    bool trainable_ = false;

    int8_t* weight_q_ = nullptr; // [in_features, out_features], row-major
    int32_t* bias_q_ = nullptr;  // [out_features], scale == bias_scale()

    // Only allocated when trainable_ == true.
    float* weight_shadow_ = nullptr;
    float* weight_grad_ = nullptr;
    float* bias_shadow_ = nullptr;
    float* bias_grad_ = nullptr;

    float input_scale_ = 1.0f;
    float weight_scale_ = 1.0f;
    float output_scale_ = 1.0f;

    // Non-owning view of the most recent Forward() input; only meaningful
    // (and only populated) when trainable_ == true.
    Tensor<int8_t> cached_input_{};
};

} // namespace edge
