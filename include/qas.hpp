#pragma once

#include <cstdint>

#include "layers.hpp"

namespace edge {

// ---------------------------------------------------------------------------
// Quantization-Aware Scaling.
//
// A layer's INT8 output domain is defined by a fixed per-layer scale S_Y.
// Choosing S_Y too large wastes INT8's ~256 levels on headroom the
// activations never use (excess quantization error); choosing it too small
// clips the activation distribution. DynamicScaleTracker watches the
// real-valued (dequantized) activation magnitudes a layer actually produces
// and adapts S_Y towards them via an exponential moving average (EMA) of the
// observed max-abs value. Tracking it this way avoids underflow during the
// backward pass: a scale that has drifted far above the true activation
// range collapses small (but real) gradients/activations to zero when they
// are requantized to INT8.
class DynamicScaleTracker {
public:
    // `min_scale` floors the tracked scale away from zero (a degenerate
    // scale of ~0 would blow up on the next dequantization / gradient
    // computation). `momentum` in [0,1) controls how much of each new
    // observation is folded in (closer to 1 = slower-adapting/smoother).
    explicit DynamicScaleTracker(float initial_scale, float min_scale = 1e-6f,
                                  float momentum = 0.9f) noexcept
        : scale_(initial_scale), min_scale_(min_scale), momentum_(momentum) {}

    // Folds one observed batch's max-abs real value into the EMA and
    // returns the updated scale. INT8 is symmetric ([-128,127]) so the
    // scale that exactly spans an observed max-abs value of `max_abs_value`
    // is max_abs_value / 127.
    float Observe(float max_abs_value) noexcept {
        const float target_scale = max_abs_value / 127.0f;
        scale_ = momentum_ * scale_ + (1.0f - momentum_) * target_scale;
        if (scale_ < min_scale_) {
            scale_ = min_scale_;
        }
        return scale_;
    }

    float scale() const noexcept { return scale_; }

private:
    float scale_;
    float min_scale_;
    float momentum_;
};

// Scans a real-valued buffer and returns the maximum absolute value, for
// feeding into DynamicScaleTracker::Observe.
inline float MaxAbs(const float* data, size_t n) noexcept {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = data[i] < 0.0f ? -data[i] : data[i];
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

// Computes the requantization multiplier M = Sx*Sw/Sy for a DenseLayer, as a
// plain double. This scale factor stays out of the INT8/INT32 runtime path
// and lives in parameter/scale bookkeeping, not the per-element matmul loop
// (which uses the fixed-point form; see tensor.hpp's FixedPointMultiplier /
// QuantizeMultiplier).
inline double ComputeRequantMultiplier(float input_scale, float weight_scale,
                                        float output_scale) {
    return static_cast<double>(input_scale) * static_cast<double>(weight_scale) /
           static_cast<double>(output_scale);
}

} // namespace edge
