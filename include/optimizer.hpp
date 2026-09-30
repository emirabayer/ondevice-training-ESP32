#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "arena.hpp"
#include "layers.hpp"

namespace edge {

// SGD (with optional momentum) acting on a DenseLayer's full-precision
// shadow parameters. Step() reads weight_grad_/bias_grad_, applies the
// update to weight_shadow_/bias_shadow_, and re-derives the INT8 ground
// truth via RequantizeWeights()/RequantizeBias(), it does NOT zero the
// layer's gradients, since a caller may want to inspect them first; call
// layer.ZeroGradients() (or SparseUpdateSelector::ZeroAllGradients())
// before the next backward pass.
class SGDOptimizer {
public:
    struct Config {
        float learning_rate = 0.01f;
        float momentum = 0.0f; // 0 disables momentum (plain SGD)
    };

    // Allocates per-parameter momentum buffers (sized to `layer`) from
    // `persistent_arena`. `layer` must already be Init()'d and trainable.
    void Init(Arena& persistent_arena, const DenseLayer& layer, Config config);

    void Step(DenseLayer& layer);

private:
    float* weight_velocity_ = nullptr;
    float* bias_velocity_ = nullptr;
    size_t weight_count_ = 0;
    size_t bias_count_ = 0;
    Config config_{};
};

// Adam (Kingma & Ba, 2014) acting on the same full-precision shadow
// parameters.
class AdamOptimizer {
public:
    struct Config {
        float learning_rate = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
    };

    void Init(Arena& persistent_arena, const DenseLayer& layer, Config config);

    void Step(DenseLayer& layer);

private:
    float* weight_m_ = nullptr;
    float* weight_v_ = nullptr;
    float* bias_m_ = nullptr;
    float* bias_v_ = nullptr;
    size_t weight_count_ = 0;
    size_t bias_count_ = 0;
    uint32_t timestep_ = 0;
    Config config_{};
};

// Sparse layer update selector: a fixed-capacity,
// arena-free registry of which layers in the network are trainable. The
// training loop iterates only over this set for Backward()/gradient-zeroing,
// so frozen "early feature extraction" layers are never touched during the
// backward pass, their absence of gradient/shadow buffers (DenseLayer only
// allocates those when trainable==true) is what actually saves arena memory,
// this selector is what keeps the training loop from trying to use them.
template <uint32_t kMaxLayers>
class SparseUpdateSelector {
public:
    void Add(DenseLayer& layer) {
        if (!layer.trainable()) {
            Panic("SparseUpdateSelector::Add called with a frozen (non-trainable) layer");
        }
        if (count_ >= kMaxLayers) {
            Panic("SparseUpdateSelector::Add exceeded its fixed capacity");
        }
        layers_[count_++] = &layer;
    }

    uint32_t count() const noexcept { return count_; }
    DenseLayer& operator[](uint32_t i) const { return *layers_[i]; }

    void ZeroAllGradients() {
        for (uint32_t i = 0; i < count_; ++i) layers_[i]->ZeroGradients();
    }

private:
    [[noreturn]] static void Panic(const char* message);

    DenseLayer* layers_[kMaxLayers] = {};
    uint32_t count_ = 0;
};

template <uint32_t kMaxLayers>
[[noreturn]] void SparseUpdateSelector<kMaxLayers>::Panic(const char* message) {
    std::fprintf(stderr, "[edge::SparseUpdateSelector] PANIC: %s\n", message);
    std::abort();
}

} // namespace edge
