#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "arena.hpp"

namespace edge {

constexpr uint32_t kMaxRank = 4;

// Non-owning view over data that lives in an Arena. A Tensor never owns or
// frees its `data` pointer, lifetime is governed entirely by the Arena
// (and, for scratch buffers, by an enclosing ArenaScope).
//
// Quantization metadata (`scale`, `zero_point`) follows uniform affine
// quantization: real value r <-> quantized value q via
//   q = clip(round(r / scale) + zero_point, qmin, qmax)
//   r = (q - zero_point) * scale
// This engine uses symmetric quantization throughout (zero_point == 0) for
// weights and activations; the field is retained for generality / future
// asymmetric support and so downstream math never needs to special-case it.
template <typename T>
struct Tensor {
    T* data = nullptr;
    uint32_t shape[kMaxRank] = {1, 1, 1, 1};
    uint32_t strides[kMaxRank] = {0, 0, 0, 0}; // in elements, not bytes
    uint32_t rank = 0;

    float scale = 1.0f;
    int32_t zero_point = 0;

    size_t size() const noexcept {
        size_t total = 1;
        for (uint32_t i = 0; i < kMaxRank; ++i) {
            total *= (i < rank) ? shape[i] : 1;
        }
        return total;
    }

    // Row-major element access for the common 2D case (matrices / batched
    // feature vectors): element (row, col).
    T& at(uint32_t row, uint32_t col) const noexcept {
        return data[row * strides[0] + col * strides[1]];
    }
};

// Allocates a contiguous row-major 2D tensor [rows, cols] of T from `arena`.
template <typename T>
Tensor<T> MakeTensor2D(Arena& arena, uint32_t rows, uint32_t cols, float scale = 1.0f,
                        int32_t zero_point = 0) {
    Tensor<T> t;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.shape[2] = 1;
    t.shape[3] = 1;
    t.strides[0] = cols;
    t.strides[1] = 1;
    t.strides[2] = 1;
    t.strides[3] = 1;
    t.scale = scale;
    t.zero_point = zero_point;
    t.data = arena.allocate<T>(static_cast<size_t>(rows) * cols, alignof(T));
    return t;
}

// Allocates a contiguous row-major 1D tensor [n] of T from `arena`.
template <typename T>
Tensor<T> MakeTensor1D(Arena& arena, uint32_t n, float scale = 1.0f, int32_t zero_point = 0) {
    Tensor<T> t;
    t.rank = 1;
    t.shape[0] = n;
    t.shape[1] = 1;
    t.shape[2] = 1;
    t.shape[3] = 1;
    t.strides[0] = 1;
    t.strides[1] = 1;
    t.strides[2] = 1;
    t.strides[3] = 1;
    t.scale = scale;
    t.zero_point = zero_point;
    t.data = arena.allocate<T>(n, alignof(T));
    return t;
}

constexpr int32_t kInt8Min = -128;
constexpr int32_t kInt8Max = 127;

// Rounds to nearest, ties away from zero, and clips to the INT8 range.
inline int8_t QuantizeInt8(float real_value, float scale, int32_t zero_point) {
    const float scaled = real_value / scale;
    const long rounded = std::lround(static_cast<double>(scaled)) + zero_point;
    if (rounded < kInt8Min) return static_cast<int8_t>(kInt8Min);
    if (rounded > kInt8Max) return static_cast<int8_t>(kInt8Max);
    return static_cast<int8_t>(rounded);
}

inline float DequantizeInt8(int8_t q, float scale, int32_t zero_point) {
    return static_cast<float>(static_cast<int32_t>(q) - zero_point) * scale;
}

// Quantizes every element of a contiguous real-valued buffer into `out`
// (already-allocated INT8 tensor of matching size) using `out.scale` /
// `out.zero_point`.
inline void QuantizeBuffer(const float* real, Tensor<int8_t>& out) {
    const size_t n = out.size();
    for (size_t i = 0; i < n; ++i) {
        out.data[i] = QuantizeInt8(real[i], out.scale, out.zero_point);
    }
}

// Dequantizes every element of `in` into the caller-owned `real` buffer
// (must hold at least in.size() floats).
inline void DequantizeBuffer(const Tensor<int8_t>& in, float* real) {
    const size_t n = in.size();
    for (size_t i = 0; i < n; ++i) {
        real[i] = DequantizeInt8(in.data[i], in.scale, in.zero_point);
    }
}

// ---------------------------------------------------------------------------
// Fixed-point requantization.
//
// The multiplier M is a positive real number, almost always < 1. It is
// decomposed once (per layer, not per element) into a Q31 fixed-point
// mantissa `multiplier_q31` and a power-of-two `shift`, following the
// standard integer-quantized-inference technique (as used by gemmlowp /
// TFLite): M ~= multiplier_q31 / 2^31 * 2^shift, with multiplier_q31 in
// [2^30, 2^31). Applying it to an accumulator is then pure integer
// arithmetic (a 32x32->64 multiply, a rounding shift by 31, and a rounding
// shift by |shift|), no floating point in the per-element hot path, which
// is what "integer scaling" in the matmul kernel refers to.
struct FixedPointMultiplier {
    int32_t multiplier_q31 = 0;
    int32_t shift = 0;
};

inline FixedPointMultiplier QuantizeMultiplier(double real_multiplier) {
    FixedPointMultiplier result;
    if (real_multiplier == 0.0) {
        return result;
    }

    double m = real_multiplier;
    bool negative = m < 0.0;
    if (negative) m = -m;

    int32_t shift = 0;
    while (m < 0.5) {
        m *= 2.0;
        --shift;
    }
    while (m >= 1.0) {
        m /= 2.0;
        ++shift;
    }
    // m is now in [0.5, 1.0)

    int64_t q = static_cast<int64_t>(std::lround(m * static_cast<double>(1LL << 31)));
    if (q == (1LL << 31)) {
        q /= 2;
        ++shift;
    }

    result.multiplier_q31 = static_cast<int32_t>(negative ? -q : q);
    result.shift = shift;
    return result;
}

// Rounding right shift (round-half-away-from-zero) used to apply the
// power-of-two part of a FixedPointMultiplier.
inline int32_t RoundingRightShift(int64_t value, int32_t shift) {
    if (shift <= 0) {
        // Left shift by -shift. Shifting a negative signed integer left is
        // undefined behavior in C++, so perform the shift on the unsigned
        // two's-complement bit pattern instead (well-defined wraparound)
        // and reinterpret the result back as signed.
        const uint64_t bits = static_cast<uint64_t>(value) << static_cast<uint64_t>(-shift);
        return static_cast<int32_t>(static_cast<int64_t>(bits));
    }
    // C++'s `>>` on a negative signed integer is an arithmetic (floor) shift
    // on every real-world compiler, so adding a flat `+half` before flooring
    // gives correct round-to-nearest (ties toward +infinity) for both signs
    //, no sign-dependent nudge needed (that would double-count the sign
    // and bias the result by up to one full unit, as it did previously).
    const int64_t half = int64_t{1} << (shift - 1);
    return static_cast<int32_t>((value + half) >> shift);
}

// Applies M ~= multiplier_q31 / 2^31 * 2^shift to a single int32 accumulator
// value using only integer arithmetic. The 2^31 (Q31) divide and the 2^shift
// scale are folded into a single combined rounding shift rather than two
// sequential roundings, which would otherwise compound quantization error
// (e.g. round(value/2) then *2 can be off by one from round(value)).
inline int32_t MultiplyByFixedPointMultiplier(int32_t value, const FixedPointMultiplier& m) {
    const int64_t product = static_cast<int64_t>(value) * static_cast<int64_t>(m.multiplier_q31);
    const int32_t total_right_shift = 31 - m.shift;
    return RoundingRightShift(product, total_right_shift);
}

// ---------------------------------------------------------------------------
// INT8 matrix multiplication kernel.
//
// Computes the raw INT32 accumulation C[M,N] = A[M,K] . B[K,N] with
// zero-point correction; no scaling/requantization is
// applied here so that a bias (also INT32) can be added to the accumulator
// before requantizing to INT8 (see RequantizeAccumulator below / DenseLayer
// in layers.hpp). A, B, C must be 2D tensors with A.shape[1] == B.shape[0],
// C.shape == {A.shape[0], B.shape[1]}.
inline void MatMulInt8Accumulate(const Tensor<int8_t>& A, const Tensor<int8_t>& B,
                                  Tensor<int32_t>& C) {
    const uint32_t M = A.shape[0];
    const uint32_t K = A.shape[1];
    const uint32_t N = B.shape[1];

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t a = static_cast<int32_t>(A.at(i, k)) - A.zero_point;
                const int32_t b = static_cast<int32_t>(B.at(k, j)) - B.zero_point;
                acc += a * b;
            }
            C.at(i, j) = acc;
        }
    }
}

// Requantizes an INT32 accumulator tensor (already including any bias) down
// to INT8 using a single per-tensor FixedPointMultiplier, writing into
// `out` (out.scale/out.zero_point describe the target domain and must be
// consistent with the multiplier the caller derived from Sx*Sw/Sy).
inline void RequantizeAccumulator(const Tensor<int32_t>& acc, const FixedPointMultiplier& m,
                                   Tensor<int8_t>& out) {
    const size_t n = acc.size();
    for (size_t i = 0; i < n; ++i) {
        const int32_t scaled = MultiplyByFixedPointMultiplier(acc.data[i], m);
        const int32_t with_zp = scaled + out.zero_point;
        out.data[i] = static_cast<int8_t>(
            with_zp < kInt8Min ? kInt8Min : (with_zp > kInt8Max ? kInt8Max : with_zp));
    }
}

} // namespace edge
