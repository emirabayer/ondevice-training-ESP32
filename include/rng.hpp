#pragma once

#include <cstdint>

namespace edge {

// Tiny, dependency-free deterministic PRNG (xorshift32). Used for weight
// initialization and synthetic data generation. Stack-allocated, no heap
// use, a drop-in replacement for <random> engines that keeps the engine's
// dependency surface minimal and fully reproducible across runs/targets.
class Rng {
public:
    explicit Rng(uint32_t seed) noexcept : state_(seed != 0 ? seed : 0x9E3779B9u) {}

    uint32_t NextU32() noexcept {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    // Uniform float in [-1, 1).
    float NextUniformSigned() noexcept {
        const float unit = static_cast<float>(NextU32() >> 8) / static_cast<float>(1u << 24);
        return unit * 2.0f - 1.0f;
    }

    // Uniform float in [0, 1).
    float NextUniform01() noexcept {
        return static_cast<float>(NextU32() >> 8) / static_cast<float>(1u << 24);
    }

private:
    uint32_t state_;
};

} // namespace edge
