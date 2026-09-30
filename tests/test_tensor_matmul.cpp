#include <cstdint>
#include <cstdlib>
#include <initializer_list>

#include "arena.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"

using namespace edge;

namespace {

void TestQuantizeDequantizeRoundTrip() {
    std::fprintf(stdout, "TestQuantizeDequantizeRoundTrip\n");
    const float scale = 0.05f;
    for (int q = kInt8Min; q <= kInt8Max; ++q) {
        const float real = DequantizeInt8(static_cast<int8_t>(q), scale, 0);
        const int8_t back = QuantizeInt8(real, scale, 0);
        EDGE_EXPECT_EQ(back, static_cast<int8_t>(q));
    }
}

void TestQuantizeClipsToInt8Range() {
    std::fprintf(stdout, "TestQuantizeClipsToInt8Range\n");
    EDGE_EXPECT_EQ(QuantizeInt8(1000.0f, 1.0f, 0), static_cast<int8_t>(kInt8Max));
    EDGE_EXPECT_EQ(QuantizeInt8(-1000.0f, 1.0f, 0), static_cast<int8_t>(kInt8Min));
}

void TestFixedPointMultiplierMatchesDoubleWithinOneUlp() {
    std::fprintf(stdout, "TestFixedPointMultiplierMatchesDoubleWithinOneUlp\n");
    const double reals[] = {0.9999, 0.5, 0.25, 0.1, 0.0313, 0.001, 0.7654321};
    for (double m : reals) {
        const FixedPointMultiplier fp = QuantizeMultiplier(m);
        for (int32_t v : {0, 1, -1, 100, -100, 12345, -12345, 2000000, -2000000}) {
            const int32_t actual = MultiplyByFixedPointMultiplier(v, fp);
            const double reference = static_cast<double>(v) * m;
            const int64_t expected = std::lround(reference);
            EDGE_EXPECT_TRUE(std::abs(static_cast<int64_t>(actual) - expected) <= 1);
        }
    }
}

// Reference implementation: standard double-precision matmul on the
// dequantized real values:
// Y_real = (Sx*Sw) * sum_k X_ik * W_kj, followed by the same round-to-nearest
// requantization to INT8. The INT8 kernel (integer accumulation + fixed
// point multiplier) must match this reference to within +-1 quantization
// step, which is the standard tolerance for comparing a fixed-point kernel
// against its floating point reference (the only source of divergence is
// the finite 31-bit precision of the fixed-point multiplier).
void TestMatMulInt8MatchesDoublePrecisionReference() {
    std::fprintf(stdout, "TestMatMulInt8MatchesDoublePrecisionReference\n");
    Arena arena;

    const uint32_t M = 5, K = 7, N = 4;
    const float scale_x = 0.03f;
    const float scale_w = 0.02f;
    const float scale_y = 0.01f;

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, M, K, scale_x, 0);
    Tensor<int8_t> W = MakeTensor2D<int8_t>(arena, K, N, scale_w, 0);
    Tensor<int32_t> acc = MakeTensor2D<int32_t>(arena, M, N);
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, M, N, scale_y, 0);

    unsigned seed = 42;
    auto next_int8 = [&seed]() -> int8_t {
        seed = seed * 1103515245u + 12345u;
        return static_cast<int8_t>((seed >> 16) % 256 - 128);
    };
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = next_int8();
    for (size_t i = 0; i < W.size(); ++i) W.data[i] = next_int8();

    MatMulInt8Accumulate(X, W, acc);
    const FixedPointMultiplier multiplier =
        QuantizeMultiplier(static_cast<double>(scale_x) * scale_w / scale_y);
    RequantizeAccumulator(acc, multiplier, Y);

    // Double-precision reference over the same integer data.
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            double real_acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                real_acc += static_cast<double>(X.at(i, k)) * static_cast<double>(W.at(k, j));
            }
            const double y_real = real_acc * scale_x * scale_w;
            double y_scaled = y_real / scale_y;
            long expected = std::lround(y_scaled);
            if (expected < kInt8Min) expected = kInt8Min;
            if (expected > kInt8Max) expected = kInt8Max;

            const int actual = Y.at(i, j);
            EDGE_EXPECT_TRUE(std::abs(actual - static_cast<int>(expected)) <= 1);
        }
    }

    // Sanity: arena usage stayed within budget and nothing overlapped
    // (ASan would already have caught overlap via the writes above).
    EDGE_EXPECT_TRUE(arena.used() <= kArenaCapacityBytes);
}

void TestMatMulIdentityWeightPreservesInput() {
    std::fprintf(stdout, "TestMatMulIdentityWeightPreservesInput\n");
    Arena arena;
    const uint32_t N = 6;
    const float scale = 0.1f;

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, 1, N, scale, 0);
    Tensor<int8_t> I = MakeTensor2D<int8_t>(arena, N, N, 1.0f, 0);
    Tensor<int32_t> acc = MakeTensor2D<int32_t>(arena, 1, N);
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, 1, N, scale, 0);

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            I.at(i, j) = (i == j) ? 1 : 0;
        }
        X.at(0, i) = static_cast<int8_t>(i * 5 - 15);
    }

    MatMulInt8Accumulate(X, I, acc);
    const FixedPointMultiplier multiplier =
        QuantizeMultiplier(static_cast<double>(scale) * 1.0 / scale);
    RequantizeAccumulator(acc, multiplier, Y);

    for (uint32_t i = 0; i < N; ++i) {
        EDGE_EXPECT_EQ(Y.at(0, i), X.at(0, i));
    }
}

} // namespace

EDGE_TEST_MAIN_BEGIN()
    TestQuantizeDequantizeRoundTrip();
    TestQuantizeClipsToInt8Range();
    TestFixedPointMultiplierMatchesDoubleWithinOneUlp();
    TestMatMulInt8MatchesDoublePrecisionReference();
    TestMatMulIdentityWeightPreservesInput();
EDGE_TEST_MAIN_END()
