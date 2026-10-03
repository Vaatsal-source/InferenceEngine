// TDD driver for Phase 4. Work top to bottom, same structure as every
// previous phase's test file.
//
// Run just this binary with:  ./build/tests/tinygrad_tests
// Run one group with:         ./build/tests/tinygrad_tests --gtest_filter=Conv2dForward.*

#include "tinygrad/conv.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

using tinygrad::conv2d;
using tinygrad::GradTensor;
using tinygrad::max_pool2d;
using tinygrad::Tensor;

namespace {
void expect_tensor_eq(const Tensor& t, const std::vector<double>& expected) {
    auto actual = t.to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "at flat index " << i;
    }
}

// Rebuilds fresh GradTensors from raw data and runs conv2d().sum() forward
// only — used as the "perturb one input, recompute the whole forward pass"
// step of a finite-difference gradient check (same idea as Phase 1's
// ValueGradCheck, just applied to conv2d).
double conv_sum(const std::vector<double>& input_vals, const std::vector<size_t>& input_shape,
                 const std::vector<double>& weight_vals, const std::vector<size_t>& weight_shape, size_t stride,
                 size_t padding) {
    GradTensor x(Tensor(input_shape, input_vals));
    GradTensor w(Tensor(weight_shape, weight_vals));
    return conv2d(x, w, stride, padding).sum().data().at({0});
}
}  // namespace

// ---------------------------------------------------------------------
// conv2d forward — hand-computed small cases, no backward involved yet.
// ---------------------------------------------------------------------

TEST(Conv2dForward, SingleChannelIdentityKernel) {
    // input (1,1,3,3):        weight (1,1,2,2):
    //   1 2 3                   1 0
    //   4 5 6                   0 1
    //   7 8 9
    // stride=1, padding=0 -> output (1,1,2,2)
    GradTensor x(Tensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    GradTensor w(Tensor({1, 1, 2, 2}, {1, 0, 0, 1}));
    GradTensor out = conv2d(x, w);
    EXPECT_EQ(out.data().shape(), (std::vector<size_t>{1, 1, 2, 2}));
    expect_tensor_eq(out.data(), {6, 8, 12, 14});
}

TEST(Conv2dForward, MultiChannelSum) {
    // input (1,2,2,2): channel0 = [[1,2],[3,4]], channel1 = [[5,6],[7,8]]
    // weight (1,2,2,2): co0ci0 = [[1,0],[0,0]], co0ci1 = [[0,0],[0,1]]
    // stride=1, padding=0 -> output (1,1,1,1) = 1*1 (ci0) + 8*1 (ci1) = 9
    GradTensor x(Tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    GradTensor w(Tensor({1, 2, 2, 2}, {1, 0, 0, 0, 0, 0, 0, 1}));
    GradTensor out = conv2d(x, w);
    expect_tensor_eq(out.data(), {9});
}

TEST(Conv2dForward, ChannelMismatchThrows) {
    GradTensor x(Tensor({1, 2, 3, 3}));   // C_in = 2
    GradTensor w(Tensor({1, 3, 2, 2}));   // C_in = 3 — mismatch
    EXPECT_THROW(conv2d(x, w), std::exception);
}

// ---------------------------------------------------------------------
// Stride and padding.
// ---------------------------------------------------------------------

TEST(Conv2dStridePadding, StrideTwoNoPadding) {
    // input (1,1,4,4) = 1..16 row-major, weight identity {1,0,0,1}, stride=2.
    GradTensor x(Tensor({1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
    GradTensor w(Tensor({1, 1, 2, 2}, {1, 0, 0, 1}));
    GradTensor out = conv2d(x, w, /*stride=*/2, /*padding=*/0);
    EXPECT_EQ(out.data().shape(), (std::vector<size_t>{1, 1, 2, 2}));
    expect_tensor_eq(out.data(), {7, 11, 23, 27});
}

TEST(Conv2dStridePadding, PaddingOne) {
    // input (1,1,2,2) = [[1,2],[3,4]], weight all-ones {1,1,1,1}, padding=1.
    GradTensor x(Tensor({1, 1, 2, 2}, {1, 2, 3, 4}));
    GradTensor w(Tensor({1, 1, 2, 2}, {1, 1, 1, 1}));
    GradTensor out = conv2d(x, w, /*stride=*/1, /*padding=*/1);
    EXPECT_EQ(out.data().shape(), (std::vector<size_t>{1, 1, 3, 3}));
    expect_tensor_eq(out.data(), {1, 3, 2, 4, 10, 6, 3, 7, 4});
}

TEST(Conv2dStridePadding, UnevenWindowThrows) {
    // input H=4, kernel=3, stride=2: (4-3)/2 = 0 remainder 1 -> doesn't tile evenly.
    GradTensor x(Tensor({1, 1, 4, 4}));
    GradTensor w(Tensor({1, 1, 3, 3}));
    EXPECT_THROW(conv2d(x, w, /*stride=*/2, /*padding=*/0), std::exception);
}

// ---------------------------------------------------------------------
// conv2d backward — finite-difference gradient check (same technique as
// Phase 1's ValueGradCheck), since hand-deriving every element by hand
// isn't practical here.
// ---------------------------------------------------------------------

TEST(Conv2dBackward, MatchesNumericalGradientForInputAndWeight) {
    std::vector<size_t> input_shape = {1, 1, 3, 3};
    std::vector<double> input_vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<size_t> weight_shape = {1, 1, 2, 2};
    std::vector<double> weight_vals = {0.5, -1.0, 2.0, 0.3};
    const size_t stride = 1;
    const size_t padding = 0;
    const double h = 1e-6;

    GradTensor x(Tensor(input_shape, input_vals));
    GradTensor w(Tensor(weight_shape, weight_vals));
    GradTensor loss = conv2d(x, w, stride, padding).sum();
    loss.backward();

    auto x_grad = x.grad().to_vector();
    for (size_t i = 0; i < input_vals.size(); ++i) {
        auto plus = input_vals;
        plus[i] += h;
        auto minus = input_vals;
        minus[i] -= h;
        double numeric = (conv_sum(plus, input_shape, weight_vals, weight_shape, stride, padding) -
                           conv_sum(minus, input_shape, weight_vals, weight_shape, stride, padding)) /
                          (2 * h);
        EXPECT_NEAR(x_grad[i], numeric, 1e-4) << "input index " << i;
    }

    auto w_grad = w.grad().to_vector();
    for (size_t i = 0; i < weight_vals.size(); ++i) {
        auto plus = weight_vals;
        plus[i] += h;
        auto minus = weight_vals;
        minus[i] -= h;
        double numeric = (conv_sum(input_vals, input_shape, plus, weight_shape, stride, padding) -
                           conv_sum(input_vals, input_shape, minus, weight_shape, stride, padding)) /
                          (2 * h);
        EXPECT_NEAR(w_grad[i], numeric, 1e-4) << "weight index " << i;
    }
}

// ---------------------------------------------------------------------
// max_pool2d forward.
// ---------------------------------------------------------------------

TEST(MaxPool2dForward, TwoByTwoWindowStrideTwo) {
    // input (1,1,4,4):
    //   1 3 2 4
    //   5 6 8 7
    //   9 2 1 0
    //   3 4 6 5
    GradTensor x(Tensor({1, 1, 4, 4}, {1, 3, 2, 4, 5, 6, 8, 7, 9, 2, 1, 0, 3, 4, 6, 5}));
    GradTensor out = max_pool2d(x, /*kernel_size=*/2, /*stride=*/2);
    EXPECT_EQ(out.data().shape(), (std::vector<size_t>{1, 1, 2, 2}));
    expect_tensor_eq(out.data(), {6, 8, 9, 6});
}

TEST(MaxPool2dForward, UnevenWindowThrows) {
    GradTensor x(Tensor({1, 1, 5, 5}));
    EXPECT_THROW(max_pool2d(x, /*kernel_size=*/2, /*stride=*/2), std::exception);
}

// ---------------------------------------------------------------------
// max_pool2d backward — gradient routes ONLY to the winning (argmax)
// position in each window, everywhere else gets zero.
// ---------------------------------------------------------------------

TEST(MaxPool2dBackward, RoutesGradientToArgmaxOnly) {
    GradTensor x(Tensor({1, 1, 4, 4}, {1, 3, 2, 4, 5, 6, 8, 7, 9, 2, 1, 0, 3, 4, 6, 5}));
    GradTensor out = max_pool2d(x, 2, 2);
    GradTensor loss = out.sum();
    loss.backward();

    // Winning positions (row, col) in the original 4x4, flattened
    // row-major: (1,1)->6, (1,2)->8, (2,0)->9, (3,2)->6 ; everywhere else 0.
    expect_tensor_eq(x.grad(), {0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0});
}
