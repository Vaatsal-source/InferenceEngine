// TDD driver for Phase 3. Work top to bottom — this mirrors Phase 1's
// test_value.cpp structure, just with Tensors instead of scalars.
//
// Run just this binary with:  ./build/tests/tinygrad_tests
// Run one group with:         ./build/tests/tinygrad_tests --gtest_filter=TensorGradBackward.*

#include "tinygrad/tensor_grad.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

using tinygrad::GradTensor;
using tinygrad::Tensor;

namespace {
void expect_tensor_eq(const Tensor& t, const std::vector<double>& expected) {
    auto actual = t.to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "at flat index " << i;
    }
}
}  // namespace

// ---------------------------------------------------------------------
// Construction — data/grad plumbing, no backward() yet.
// ---------------------------------------------------------------------

TEST(TensorGradBasics, StoresDataAndZeroGrad) {
    GradTensor x(Tensor({2}, {3.0, 4.0}));
    expect_tensor_eq(x.data(), {3.0, 4.0});
    expect_tensor_eq(x.grad(), {0.0, 0.0});  // grad starts at all zeros
}

// ---------------------------------------------------------------------
// Backward pass — same-shape case first (no broadcasting involved).
// ---------------------------------------------------------------------

TEST(TensorGradBackward, SimpleAddSameShape) {
    // c = a + b, both {2}. dc/da = 1, dc/db = 1 elementwise.
    GradTensor a(Tensor({2}, {1.0, 2.0}));
    GradTensor b(Tensor({2}, {10.0, 20.0}));
    GradTensor c = a + b;
    expect_tensor_eq(c.data(), {11.0, 22.0});

    GradTensor loss = c.sum();
    loss.backward();
    expect_tensor_eq(a.grad(), {1.0, 1.0});
    expect_tensor_eq(b.grad(), {1.0, 1.0});
}

TEST(TensorGradBackward, SimpleMulSameShape) {
    // c = a * b elementwise. dc/da = b, dc/db = a.
    GradTensor a(Tensor({2}, {2.0, 3.0}));
    GradTensor b(Tensor({2}, {5.0, 7.0}));
    GradTensor c = a * b;
    expect_tensor_eq(c.data(), {10.0, 21.0});

    GradTensor loss = c.sum();
    loss.backward();
    expect_tensor_eq(a.grad(), {5.0, 7.0});
    expect_tensor_eq(b.grad(), {2.0, 3.0});
}

TEST(TensorGradBackward, ReusedTensorAccumulatesGradient) {
    // y = x * x  =>  dy/dx = 2x, same "+=  not =" trap as Phase 1.
    GradTensor x(Tensor({2}, {3.0, 4.0}));
    GradTensor y = x * x;
    GradTensor loss = y.sum();
    loss.backward();
    expect_tensor_eq(x.grad(), {6.0, 8.0});
}

TEST(TensorGradBackward, BackwardOnNonScalarThrows) {
    GradTensor a(Tensor({2}, {1.0, 2.0}));
    GradTensor b(Tensor({2}, {3.0, 4.0}));
    GradTensor c = a + b;  // shape {2}, not a scalar
    EXPECT_THROW(c.backward(), std::exception);
}

// ---------------------------------------------------------------------
// Broadcasting-aware backward — the new idea this phase introduces.
// ---------------------------------------------------------------------

TEST(TensorGradBroadcast, AddRowBroadcastBackward) {
    // a: {2, 3}, b: {3} broadcasts across both rows of a.
    GradTensor a(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
    GradTensor b(Tensor({3}, {10, 20, 30}));
    GradTensor c = a + b;
    expect_tensor_eq(c.data(), {11, 22, 33, 14, 25, 36});

    GradTensor loss = c.sum();
    loss.backward();

    // Every element of `a` contributes 1:1 to the sum, so a.grad() is all 1s.
    expect_tensor_eq(a.grad(), {1, 1, 1, 1, 1, 1});
    // Each element of `b` was reused once per row (2 rows), so its
    // gradient is the SUM over the broadcast axis: 1+1 = 2 for each entry.
    expect_tensor_eq(b.grad(), {2, 2, 2});
}

TEST(TensorGradBroadcast, MulScalarLikeBroadcastBackward) {
    // a: {2}, b: {1} (a "scalar" tensor) broadcasts across a.
    GradTensor a(Tensor({2}, {3.0, 4.0}));
    GradTensor b(Tensor({1}, {10.0}));
    GradTensor c = a * b;
    expect_tensor_eq(c.data(), {30.0, 40.0});

    GradTensor loss = c.sum();
    loss.backward();

    // dc/da = b = 10 for each element.
    expect_tensor_eq(a.grad(), {10.0, 10.0});
    // dc/db = sum(a) = 3 + 4 = 7 (b's single element was reused for both).
    expect_tensor_eq(b.grad(), {7.0});
}

// ---------------------------------------------------------------------
// sum() on its own.
// ---------------------------------------------------------------------

TEST(TensorGradSum, ForwardAndBackward) {
    GradTensor a(Tensor({3}, {1.0, 2.0, 3.0}));
    GradTensor s = a.sum();
    expect_tensor_eq(s.data(), {6.0});

    s.backward();
    // Every element contributed with local derivative 1.
    expect_tensor_eq(a.grad(), {1.0, 1.0, 1.0});
}

// ---------------------------------------------------------------------
// matmul — 2D only, no broadcasting.
// ---------------------------------------------------------------------

TEST(TensorGradMatmul, ForwardMatchesPhase2) {
    GradTensor a(Tensor({2, 2}, {1, 2, 3, 4}));
    GradTensor b(Tensor({2, 2}, {5, 6, 7, 8}));
    GradTensor c = a.matmul(b);
    expect_tensor_eq(c.data(), {19, 22, 43, 50});
}

TEST(TensorGradMatmul, BackwardMatchesHandDerivedGradients) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], L = sum(A @ B)
    // dL/dC is all-ones {2,2} (from sum()). dL/dA = dL/dC @ B^T,
    // dL/dB = A^T @ dL/dC.
    // B^T = [[5, 7], [6, 8]]  =>  dL/dA row0 = [5+6, 7+8] = [11, 15],
    //                              dL/dA row1 = same = [11, 15]
    // A^T = [[1, 3], [2, 4]]  =>  dL/dB row0 = [1+3, 1+3] = [4, 4],
    //                              dL/dB row1 = [2+4, 2+4] = [6, 6]
    GradTensor a(Tensor({2, 2}, {1, 2, 3, 4}));
    GradTensor b(Tensor({2, 2}, {5, 6, 7, 8}));
    GradTensor c = a.matmul(b);
    GradTensor loss = c.sum();
    loss.backward();

    expect_tensor_eq(a.grad(), {11, 15, 11, 15});
    expect_tensor_eq(b.grad(), {4, 4, 6, 6});
}

TEST(TensorGradMatmul, NonRank2Throws) {
    GradTensor a(Tensor({2}, {1.0, 2.0}));
    GradTensor b(Tensor({2, 2}, {1, 2, 3, 4}));
    EXPECT_THROW(a.matmul(b), std::exception);
}
