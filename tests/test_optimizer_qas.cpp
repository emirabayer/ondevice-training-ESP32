#include <cstdint>

#include "arena.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "qas.hpp"
#include "rng.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"

using namespace edge;

namespace {

// One full forward/backward/step over a tiny fixed problem; returns the
// loss. Used by both the SGD and Adam tests below to check that a handful
// of optimizer steps actually reduces the loss.
template <typename Optimizer>
float TrainOneStep(Arena& arena, DenseLayer& layer, Optimizer& opt, const Tensor<int8_t>& X,
                    const int32_t* labels, uint32_t batch, uint32_t out_features) {
    ArenaScope scope(arena);
    Tensor<int8_t> Y = MakeTensor2D<int8_t>(arena, batch, out_features, layer.output_scale(), 0);
    layer.Forward(arena, X, Y);

    float* logits = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    DequantizeBuffer(Y, logits);
    float* probs = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    Softmax(logits, probs, batch, out_features);
    const float loss = CrossEntropyLoss(probs, labels, batch, out_features);

    float* dz = arena.allocate<float>(static_cast<size_t>(batch) * out_features);
    Softmax_CrossEntropy_Backward(probs, labels, batch, out_features, dz);

    layer.ZeroGradients();
    layer.Backward(arena, batch, dz, nullptr);
    opt.Step(layer);

    return loss;
}

void TestSGDStepReducesLoss() {
    std::fprintf(stdout, "TestSGDStepReducesLoss\n");
    Arena arena;
    Rng rng(99);

    const uint32_t batch = 4, in_features = 6, out_features = 3;
    const float scale = 0.05f;

    DenseLayer layer;
    layer.Init(arena, in_features, out_features, scale, scale, scale, true, rng);

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, batch, in_features, scale, 0);
    for (size_t i = 0; i < X.size(); ++i) {
        X.data[i] = static_cast<int8_t>(static_cast<int32_t>(i * 7 % 61) - 30);
    }
    int32_t labels[batch] = {0, 1, 2, 1};

    SGDOptimizer opt;
    SGDOptimizer::Config cfg;
    cfg.learning_rate = 0.5f;
    cfg.momentum = 0.0f;
    opt.Init(arena, layer, cfg);

    const float loss_first = TrainOneStep(arena, layer, opt, X, labels, batch, out_features);
    float loss_last = loss_first;
    for (int step = 0; step < 49; ++step) {
        loss_last = TrainOneStep(arena, layer, opt, X, labels, batch, out_features);
    }

    std::fprintf(stdout, "  loss[0]=%f loss[49]=%f\n", loss_first, loss_last);
    EDGE_EXPECT_TRUE(loss_last < loss_first * 0.5f);
}

void TestAdamStepReducesLoss() {
    std::fprintf(stdout, "TestAdamStepReducesLoss\n");
    Arena arena;
    Rng rng(4242);

    const uint32_t batch = 4, in_features = 6, out_features = 3;
    const float scale = 0.05f;

    DenseLayer layer;
    layer.Init(arena, in_features, out_features, scale, scale, scale, true, rng);

    Tensor<int8_t> X = MakeTensor2D<int8_t>(arena, batch, in_features, scale, 0);
    for (size_t i = 0; i < X.size(); ++i) {
        X.data[i] = static_cast<int8_t>(static_cast<int32_t>(i * 11 % 51) - 25);
    }
    int32_t labels[batch] = {2, 0, 1, 0};

    AdamOptimizer opt;
    AdamOptimizer::Config cfg;
    cfg.learning_rate = 0.05f;
    opt.Init(arena, layer, cfg);

    const float loss_first = TrainOneStep(arena, layer, opt, X, labels, batch, out_features);
    float loss_last = loss_first;
    for (int step = 0; step < 49; ++step) {
        loss_last = TrainOneStep(arena, layer, opt, X, labels, batch, out_features);
    }

    std::fprintf(stdout, "  loss[0]=%f loss[49]=%f\n", loss_first, loss_last);
    EDGE_EXPECT_TRUE(loss_last < loss_first * 0.5f);
}

void TestSparseUpdateSelectorRejectsFrozenLayer() {
    std::fprintf(stdout, "TestSparseUpdateSelectorRejectsFrozenLayer (fork/death test)\n");
    const bool died = edge::test::DiesWhenRun([] {
        Arena arena;
        Rng rng(1);
        DenseLayer frozen;
        frozen.Init(arena, 4, 4, 0.1f, 0.1f, 0.1f, /*trainable=*/false, rng);

        SparseUpdateSelector<4> selector;
        selector.Add(frozen); // must panic: selector only accepts trainable layers
    });
    EDGE_EXPECT_TRUE(died);
}

void TestSparseUpdateSelectorTracksTrainableLayers() {
    std::fprintf(stdout, "TestSparseUpdateSelectorTracksTrainableLayers\n");
    Arena arena;
    Rng rng(2);
    DenseLayer head1, head2;
    head1.Init(arena, 4, 4, 0.1f, 0.1f, 0.1f, /*trainable=*/true, rng);
    head2.Init(arena, 4, 4, 0.1f, 0.1f, 0.1f, /*trainable=*/true, rng);

    SparseUpdateSelector<4> selector;
    selector.Add(head1);
    selector.Add(head2);
    EDGE_EXPECT_EQ(selector.count(), uint32_t{2});

    head1.weight_grad()[0] = 3.0f;
    head2.weight_grad()[0] = 5.0f;
    selector.ZeroAllGradients();
    EDGE_EXPECT_NEAR(selector[0].weight_grad()[0], 0.0f, 1e-9f);
    EDGE_EXPECT_NEAR(selector[1].weight_grad()[0], 0.0f, 1e-9f);
}

void TestDynamicScaleTrackerConvergesTowardsObservedRange() {
    std::fprintf(stdout, "TestDynamicScaleTrackerConvergesTowardsObservedRange\n");
    DynamicScaleTracker tracker(/*initial_scale=*/1.0f, /*min_scale=*/1e-6f, /*momentum=*/0.9f);

    float scale = tracker.scale();
    for (int i = 0; i < 500; ++i) {
        scale = tracker.Observe(6.35f); // target scale = 6.35/127 = 0.05
    }
    EDGE_EXPECT_NEAR(scale, 0.05f, 1e-3f);
}

void TestDynamicScaleTrackerRespectsFloor() {
    std::fprintf(stdout, "TestDynamicScaleTrackerRespectsFloor\n");
    DynamicScaleTracker tracker(/*initial_scale=*/1e-6f, /*min_scale=*/1e-4f, /*momentum=*/0.0f);
    const float scale = tracker.Observe(0.0f); // target scale 0 -> must clamp to the floor
    EDGE_EXPECT_NEAR(scale, 1e-4f, 1e-9f);
}

} // namespace

EDGE_TEST_MAIN_BEGIN()
    TestSGDStepReducesLoss();
    TestAdamStepReducesLoss();
    TestSparseUpdateSelectorRejectsFrozenLayer();
    TestSparseUpdateSelectorTracksTrainableLayers();
    TestDynamicScaleTrackerConvergesTowardsObservedRange();
    TestDynamicScaleTrackerRespectsFloor();
EDGE_TEST_MAIN_END()
