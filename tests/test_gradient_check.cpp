// Backward-pass gradient checking against finite differences
//.
//
// Quantization's round() is piecewise-constant: its true derivative is zero
// almost everywhere and undefined at bin boundaries, so a literal finite
// difference of the *quantized* forward pass would mostly measure zero and
// tell us nothing about whether the backprop math is right. DenseLayer's
// Backward() therefore implements the standard QAT straight-through
// estimator (STE): it differentiates the real-valued linear map
// Y_real = X_real @ W_real + b_real that the quantized forward pass is
// approximating, using the cached dequantized input and the float shadow
// weights. So that is exactly what these tests check
// numerically: they perturb the float shadow parameters (and a float copy
// of the input), recompute that same real-valued forward function, and
// compare the resulting central-difference gradient against DenseLayer's
// analytical gradient. This validates the backprop calculus itself,
// independent of quantization rounding noise.

#include <cmath>
#include <cstdint>

#include "arena.hpp"
#include "layers.hpp"
#include "rng.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"

using namespace edge;

namespace {

// Real-valued (STE-linearized) forward: Y = X @ W + b, matching what
// DenseLayer::Backward's analytical gradient is differentiating.
float DenseForwardLoss(const float* X, const float* W, const float* b, uint32_t batch,
                        uint32_t in_features, uint32_t out_features, const int32_t* labels,
                        float* scratch_logits, float* scratch_probs) {
    for (uint32_t m = 0; m < batch; ++m) {
        for (uint32_t n = 0; n < out_features; ++n) {
            float sum = b[n];
            for (uint32_t k = 0; k < in_features; ++k) {
                sum += X[m * in_features + k] * W[k * out_features + n];
            }
            scratch_logits[m * out_features + n] = sum;
        }
    }
    Softmax(scratch_logits, scratch_probs, batch, out_features);
    return CrossEntropyLoss(scratch_probs, labels, batch, out_features);
}

void TestDenseLayerWeightAndBiasGradients() {
    std::fprintf(stdout, "TestDenseLayerWeightAndBiasGradients\n");
    Arena arena;
    Rng rng(1234);

    const uint32_t batch = 3, in_features = 4, out_features = 3;
    const float scale = 0.05f;

    DenseLayer layer;
    layer.Init(arena, in_features, out_features, scale, scale, scale, /*trainable=*/true, rng);

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, batch, in_features, scale, 0);
    for (size_t i = 0; i < X.size(); ++i) {
        X.data[i] = static_cast<int8_t>(static_cast<int32_t>(i % 41) - 20);
    }
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, batch, out_features, scale, 0);

    int32_t labels[batch] = {0, 2, 1};

    layer.Forward(arena, X, Y);

    float* X_real = arena.allocate<float>(static_cast<size_t>(batch) * in_features);
    DequantizeBuffer(X, X_real);

    float* logits = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    float* probs = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    float* dz = arena.allocate<float>(static_cast<size_t>(batch) * out_features);

    // Analytical gradient at the current shadow parameters.
    DenseForwardLoss(X_real, layer.weight_shadow(), layer.bias_shadow(), batch, in_features,
                      out_features, labels, logits, probs);
    Softmax_CrossEntropy_Backward(probs, labels, batch, out_features, dz);

    layer.ZeroGradients();
    layer.Backward(arena, batch, dz, /*dX_out=*/nullptr);

    const float eps = 1e-3f;
    const float tol = 3e-2f; // finite-diff / float32 tolerance

    // Check every weight element.
    for (size_t i = 0; i < layer.weight_count(); ++i) {
        const float original = layer.weight_shadow()[i];

        layer.weight_shadow()[i] = original + eps;
        const float loss_plus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                   layer.bias_shadow(), batch, in_features,
                                                   out_features, labels, logits, probs);

        layer.weight_shadow()[i] = original - eps;
        const float loss_minus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                    layer.bias_shadow(), batch, in_features,
                                                    out_features, labels, logits, probs);

        layer.weight_shadow()[i] = original;

        const float numeric_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EDGE_EXPECT_NEAR(numeric_grad, layer.weight_grad()[i], tol);
    }

    // Check every bias element.
    for (size_t i = 0; i < layer.bias_count(); ++i) {
        const float original = layer.bias_shadow()[i];

        layer.bias_shadow()[i] = original + eps;
        const float loss_plus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                   layer.bias_shadow(), batch, in_features,
                                                   out_features, labels, logits, probs);

        layer.bias_shadow()[i] = original - eps;
        const float loss_minus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                    layer.bias_shadow(), batch, in_features,
                                                    out_features, labels, logits, probs);

        layer.bias_shadow()[i] = original;

        const float numeric_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EDGE_EXPECT_NEAR(numeric_grad, layer.bias_grad()[i], tol);
    }
}

void TestDenseLayerInputGradient() {
    std::fprintf(stdout, "TestDenseLayerInputGradient\n");
    Arena arena;
    Rng rng(777);

    const uint32_t batch = 2, in_features = 5, out_features = 3;
    const float scale = 0.05f;

    DenseLayer layer;
    layer.Init(arena, in_features, out_features, scale, scale, scale, /*trainable=*/true, rng);

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, batch, in_features, scale, 0);
    for (size_t i = 0; i < X.size(); ++i) {
        X.data[i] = static_cast<int8_t>(static_cast<int32_t>(i % 33) - 16);
    }
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, batch, out_features, scale, 0);
    layer.Forward(arena, X, Y);

    float* X_real = arena.allocate<float>(static_cast<size_t>(batch) * in_features);
    DequantizeBuffer(X, X_real);

    float* logits = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    float* probs = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    float* dz = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    float* dX = arena.allocate<float>(static_cast<size_t>(batch) * in_features);

    int32_t labels[batch] = {1, 0};

    DenseForwardLoss(X_real, layer.weight_shadow(), layer.bias_shadow(), batch, in_features,
                      out_features, labels, logits, probs);
    Softmax_CrossEntropy_Backward(probs, labels, batch, out_features, dz);

    layer.ZeroGradients();
    layer.Backward(arena, batch, dz, dX);

    const float eps = 1e-3f;
    const float tol = 3e-2f;

    for (size_t i = 0; i < static_cast<size_t>(batch) * in_features; ++i) {
        const float original = X_real[i];

        X_real[i] = original + eps;
        const float loss_plus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                   layer.bias_shadow(), batch, in_features,
                                                   out_features, labels, logits, probs);

        X_real[i] = original - eps;
        const float loss_minus = DenseForwardLoss(X_real, layer.weight_shadow(),
                                                    layer.bias_shadow(), batch, in_features,
                                                    out_features, labels, logits, probs);

        X_real[i] = original;

        const float numeric_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EDGE_EXPECT_NEAR(numeric_grad, dX[i], tol);
    }
}

void TestReLUBackwardMatchesFiniteDifference() {
    std::fprintf(stdout, "TestReLUBackwardMatchesFiniteDifference\n");
    Arena arena;
    const float scale = 0.1f;
    const uint32_t n = 8;

    Tensor<int8_t> x = MakeTensor1D<int8_t>(arena, n, scale, 0);
    // Values deliberately kept away from the zero boundary so the ReLU
    // subgradient there doesn't make the finite-difference check ambiguous.
    const int8_t raw[n] = {-40, -12, -3, -1, 2, 5, 20, 60};
    for (uint32_t i = 0; i < n; ++i) x.data[i] = raw[i];
    ReLU_INT8(x);

    float dY[n];
    for (uint32_t i = 0; i < n; ++i) dY[i] = 1.0f + static_cast<float>(i) * 0.25f;
    float dX[n];
    ReLU_Backward(x, dY, dX);

    auto relu_real = [](float v) { return v > 0.0f ? v : 0.0f; };
    const float eps = 1e-3f;
    for (uint32_t i = 0; i < n; ++i) {
        const float real_x = DequantizeInt8(raw[i], scale, 0);
        const float loss_plus = relu_real(real_x + eps) * dY[i];
        const float loss_minus = relu_real(real_x - eps) * dY[i];
        const float numeric_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EDGE_EXPECT_NEAR(numeric_grad, dX[i], 1e-3f);
    }
}

void TestSoftmaxCrossEntropyBackwardMatchesFiniteDifference() {
    std::fprintf(stdout, "TestSoftmaxCrossEntropyBackwardMatchesFiniteDifference\n");
    const uint32_t batch = 2, num_classes = 4;
    float logits[batch * num_classes] = {0.5f, -1.2f, 2.0f, 0.1f, -0.3f, 0.8f, 0.2f, 1.5f};
    int32_t labels[batch] = {2, 3};

    float probs[batch * num_classes];
    Softmax(logits, probs, batch, num_classes);
    float dz[batch * num_classes];
    Softmax_CrossEntropy_Backward(probs, labels, batch, num_classes, dz);

    const float eps = 1e-3f;
    for (uint32_t i = 0; i < batch * num_classes; ++i) {
        const float original = logits[i];

        logits[i] = original + eps;
        Softmax(logits, probs, batch, num_classes);
        const float loss_plus = CrossEntropyLoss(probs, labels, batch, num_classes);

        logits[i] = original - eps;
        Softmax(logits, probs, batch, num_classes);
        const float loss_minus = CrossEntropyLoss(probs, labels, batch, num_classes);

        logits[i] = original;

        const float numeric_grad = (loss_plus - loss_minus) / (2.0f * eps);
        EDGE_EXPECT_NEAR(numeric_grad, dz[i], 1e-3f);
    }
}

} // namespace

EDGE_TEST_MAIN_BEGIN()
    TestDenseLayerWeightAndBiasGradients();
    TestDenseLayerInputGradient();
    TestReLUBackwardMatchesFiniteDifference();
    TestSoftmaxCrossEntropyBackwardMatchesFiniteDifference();
EDGE_TEST_MAIN_END()
