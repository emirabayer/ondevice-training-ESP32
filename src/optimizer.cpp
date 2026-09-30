#include "optimizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace edge {

namespace {

[[noreturn]] void Panic(const char* message) {
    std::fprintf(stderr, "[edge::optimizer] PANIC: %s\n", message);
    std::abort();
}

void RequireTrainable(const DenseLayer& layer, const char* who) {
    if (!layer.trainable()) {
        std::fprintf(stderr, "[edge::optimizer] PANIC: %s called on a frozen layer\n", who);
        std::abort();
    }
}

} // namespace

void SGDOptimizer::Init(Arena& persistent_arena, const DenseLayer& layer, Config config) {
    RequireTrainable(layer, "SGDOptimizer::Init");
    config_ = config;
    weight_count_ = layer.weight_count();
    bias_count_ = layer.bias_count();

    weight_velocity_ = persistent_arena.allocate<float>(weight_count_);
    bias_velocity_ = persistent_arena.allocate<float>(bias_count_);
    for (size_t i = 0; i < weight_count_; ++i) weight_velocity_[i] = 0.0f;
    for (size_t i = 0; i < bias_count_; ++i) bias_velocity_[i] = 0.0f;
}

void SGDOptimizer::Step(DenseLayer& layer) {
    RequireTrainable(layer, "SGDOptimizer::Step");
    if (layer.weight_count() != weight_count_ || layer.bias_count() != bias_count_) {
        Panic("SGDOptimizer::Step called on a layer of a different shape than Init()");
    }

    float* w = layer.weight_shadow();
    const float* gw = layer.weight_grad();
    for (size_t i = 0; i < weight_count_; ++i) {
        weight_velocity_[i] = config_.momentum * weight_velocity_[i] - config_.learning_rate * gw[i];
        w[i] += weight_velocity_[i];
    }

    float* b = layer.bias_shadow();
    const float* gb = layer.bias_grad();
    for (size_t i = 0; i < bias_count_; ++i) {
        bias_velocity_[i] = config_.momentum * bias_velocity_[i] - config_.learning_rate * gb[i];
        b[i] += bias_velocity_[i];
    }

    layer.RequantizeWeights();
    layer.RequantizeBias();
}

void AdamOptimizer::Init(Arena& persistent_arena, const DenseLayer& layer, Config config) {
    RequireTrainable(layer, "AdamOptimizer::Init");
    config_ = config;
    weight_count_ = layer.weight_count();
    bias_count_ = layer.bias_count();
    timestep_ = 0;

    weight_m_ = persistent_arena.allocate<float>(weight_count_);
    weight_v_ = persistent_arena.allocate<float>(weight_count_);
    bias_m_ = persistent_arena.allocate<float>(bias_count_);
    bias_v_ = persistent_arena.allocate<float>(bias_count_);
    for (size_t i = 0; i < weight_count_; ++i) {
        weight_m_[i] = 0.0f;
        weight_v_[i] = 0.0f;
    }
    for (size_t i = 0; i < bias_count_; ++i) {
        bias_m_[i] = 0.0f;
        bias_v_[i] = 0.0f;
    }
}

void AdamOptimizer::Step(DenseLayer& layer) {
    RequireTrainable(layer, "AdamOptimizer::Step");
    if (layer.weight_count() != weight_count_ || layer.bias_count() != bias_count_) {
        Panic("AdamOptimizer::Step called on a layer of a different shape than Init()");
    }

    ++timestep_;
    const float bias_correction1 = 1.0f - std::pow(config_.beta1, static_cast<float>(timestep_));
    const float bias_correction2 = 1.0f - std::pow(config_.beta2, static_cast<float>(timestep_));

    auto adam_update = [&](float* param, const float* grad, float* m, float* v, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            m[i] = config_.beta1 * m[i] + (1.0f - config_.beta1) * grad[i];
            v[i] = config_.beta2 * v[i] + (1.0f - config_.beta2) * grad[i] * grad[i];
            const float m_hat = m[i] / bias_correction1;
            const float v_hat = v[i] / bias_correction2;
            param[i] -= config_.learning_rate * m_hat / (std::sqrt(v_hat) + config_.epsilon);
        }
    };

    adam_update(layer.weight_shadow(), layer.weight_grad(), weight_m_, weight_v_, weight_count_);
    adam_update(layer.bias_shadow(), layer.bias_grad(), bias_m_, bias_v_, bias_count_);

    layer.RequantizeWeights();
    layer.RequantizeBias();
}

} // namespace edge
