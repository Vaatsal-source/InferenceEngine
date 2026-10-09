// TDD driver for Phase 5. Same structure as every previous phase.
//
// Run just this binary with:  ./build/tests/tinygrad_tests
// Run one group with:         ./build/tests/tinygrad_tests --gtest_filter=LinearForward.*

#include "tinygrad/nn.hpp"

#include <cmath>

#include <gtest/gtest.h>

using tinygrad::GradTensor;
using tinygrad::Linear;
using tinygrad::SGD;
using tinygrad::Sequential;
using tinygrad::Tensor;

namespace {
void expect_tensor_near(const Tensor& t, const std::vector<double>& expected, double tol = 1e-9) {
    auto actual = t.to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "at flat index " << i;
    }
}
}  // namespace

// ---------------------------------------------------------------------
// tanh activation.
// ---------------------------------------------------------------------

TEST(TanhActivation, ForwardMatchesStdTanh) {
    GradTensor x(Tensor({3}, {-1.0, 0.0, 1.0}));
    GradTensor y = tinygrad::tanh(x);
    expect_tensor_near(y.data(), {std::tanh(-1.0), std::tanh(0.0), std::tanh(1.0)});
}

TEST(TanhActivation, BackwardMatchesDerivative) {
    GradTensor x(Tensor({3}, {-1.0, 0.0, 1.0}));
    GradTensor y = tinygrad::tanh(x);
    GradTensor loss = y.sum();
    loss.backward();
    double t0 = std::tanh(-1.0), t1 = std::tanh(0.0), t2 = std::tanh(1.0);
    expect_tensor_near(x.grad(), {1 - t0 * t0, 1 - t1 * t1, 1 - t2 * t2}, 1e-6);
}

// ---------------------------------------------------------------------
// Linear — deterministic constructor, hand-computable forward pass.
// ---------------------------------------------------------------------

TEST(LinearForward, MatchesHandComputedValues) {
    // weight {2,3} (in_features=2, out_features=3):
    //   [[1, 0, 1],
    //    [0, 1, 1]]
    // bias {3} = [0.5, -0.5, 0]
    Linear layer(Tensor({2, 3}, {1, 0, 1, 0, 1, 1}), Tensor({3}, {0.5, -0.5, 0}));

    // x {2,2} (N=2 samples, in_features=2): [[1,2],[3,4]]
    GradTensor x(Tensor({2, 2}, {1, 2, 3, 4}));
    GradTensor y = layer.forward(x);

    EXPECT_EQ(y.data().shape(), (std::vector<size_t>{2, 3}));
    // row0: [1,2] . weight + bias = [1,2,3] + [0.5,-0.5,0] = [1.5, 1.5, 3]
    // row1: [3,4] . weight + bias = [3,4,7] + [0.5,-0.5,0] = [3.5, 3.5, 7]
    expect_tensor_near(y.data(), {1.5, 1.5, 3, 3.5, 3.5, 7});
}

TEST(LinearForward, GradientFlowsToWeightBiasAndInput) {
    Linear layer(Tensor({2, 2}, {1, 0, 0, 1}), Tensor({2}, {0, 0}));
    GradTensor x(Tensor({1, 2}, {3.0, 4.0}));
    GradTensor loss = layer.forward(x).sum();
    loss.backward();

    auto params = layer.parameters();
    ASSERT_EQ(params.size(), 2u);
    for (const auto& p : params) {
        for (double g : p.grad().to_vector()) {
            EXPECT_NE(g, 0.0);
        }
    }
    for (double g : x.grad().to_vector()) {
        EXPECT_NE(g, 0.0);
    }
}

TEST(LinearRandomInit, ProducesCorrectShapesWithinInitRange) {
    Linear layer(4, 3, /*seed=*/42);
    auto params = layer.parameters();
    ASSERT_EQ(params.size(), 2u);

    EXPECT_EQ(params[0].data().shape(), (std::vector<size_t>{4, 3}));  // weight
    double scale = 1.0 / std::sqrt(4.0);
    for (double v : params[0].data().to_vector()) {
        EXPECT_LE(std::abs(v), scale);
    }

    EXPECT_EQ(params[1].data().shape(), (std::vector<size_t>{3}));  // bias
    expect_tensor_near(params[1].data(), {0, 0, 0});                // zero-init
}

// ---------------------------------------------------------------------
// Sequential — composes layers with tanh between (not after the last).
// ---------------------------------------------------------------------

TEST(SequentialForward, ComposesLayersWithTanhBetween) {
    // layer1: identity weight, zero bias (2 -> 2)
    Linear layer1(Tensor({2, 2}, {1, 0, 0, 1}), Tensor({2}, {0, 0}));
    // layer2: sums both features into one output (2 -> 1)
    Linear layer2(Tensor({2, 1}, {1, 1}), Tensor({1}, {0}));
    Sequential model({layer1, layer2});

    GradTensor x(Tensor({1, 2}, {1.0, 1.0}));
    GradTensor out = model.forward(x);

    // layer1(x) = [1,1] (identity) -> tanh -> [tanh(1), tanh(1)]
    // layer2(...) = tanh(1) + tanh(1) = 2*tanh(1), NO tanh after (last layer)
    EXPECT_EQ(out.data().shape(), (std::vector<size_t>{1, 1}));
    expect_tensor_near(out.data(), {2 * std::tanh(1.0)}, 1e-6);
}

TEST(SequentialForward, ParametersIncludeEveryLayer) {
    Linear layer1(2, 3, 1);
    Linear layer2(3, 1, 2);
    Sequential model({layer1, layer2});
    EXPECT_EQ(model.parameters().size(), 4u);  // 2 layers x (weight, bias)
}

// ---------------------------------------------------------------------
// SGD optimizer.
// ---------------------------------------------------------------------

TEST(SGDOptimizer, StepAppliesGradientDescentUpdate) {
    GradTensor p(Tensor({2}, {1.0, 2.0}));
    p.node()->grad = Tensor({2}, {0.1, 0.2});  // pretend a backward() already ran

    SGD opt({p}, /*learning_rate=*/0.1);
    opt.step();

    // p.data <- p.data - lr * grad = {1 - 0.01, 2 - 0.02}
    expect_tensor_near(p.data(), {0.99, 1.98});
}

TEST(SGDOptimizer, ZeroGradResetsGradients) {
    GradTensor p(Tensor({2}, {1.0, 2.0}));
    p.node()->grad = Tensor({2}, {5.0, 5.0});

    SGD opt({p}, 0.1);
    opt.zero_grad();

    expect_tensor_near(p.grad(), {0.0, 0.0});
}

// ---------------------------------------------------------------------
// End-to-end sanity: training should make the loss go down. (The full XOR
// convergence demo lives in examples/xor.cpp, not here — this just checks
// the training loop you'll write there actually learns something.)
// ---------------------------------------------------------------------

TEST(TrainingLoop, LossDecreasesOverEpochs) {
    Sequential model({Linear(2, 8, /*seed=*/42), Linear(8, 1, /*seed=*/43)});
    SGD opt(model.parameters(), 0.5);

    GradTensor x(Tensor({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1}));
    GradTensor neg_y(Tensor({4, 1}, {0, -1, -1, 0}));  // XOR targets, negated

    auto run_epoch = [&]() {
        opt.zero_grad();
        GradTensor pred = model.forward(x);
        GradTensor diff = pred + neg_y;
        GradTensor loss = (diff * diff).sum();
        loss.backward();
        opt.step();
        return loss.data().at({0});
    };

    double first_loss = run_epoch();
    double loss = first_loss;
    for (int epoch = 0; epoch < 200; ++epoch) {
        loss = run_epoch();
    }
    EXPECT_LT(loss, first_loss);
}
