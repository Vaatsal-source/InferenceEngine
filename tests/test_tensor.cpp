// TDD driver for Phase 2. Work top to bottom: each TEST group is roughly
// ordered by dependency (TensorBasics before TensorViews before
// TensorBroadcast before TensorMatmul).
//
// Run just this binary with:  ./build/tests/tinygrad_tests
// Run one group with:         ./build/tests/tinygrad_tests --gtest_filter=TensorBasics.*

#include "tinygrad/tensor.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

using tinygrad::Tensor;

// ---------------------------------------------------------------------
// Construction, shape/strides bookkeeping, element access.
// ---------------------------------------------------------------------

TEST(TensorBasics, ZeroInitShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.numel(), 6u);
    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3}));
    for (auto v : t.to_vector()) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(TensorBasics, ConstructFromData) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_DOUBLE_EQ(t.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(t.at({0, 2}), 3.0);
    EXPECT_DOUBLE_EQ(t.at({1, 0}), 4.0);
    EXPECT_DOUBLE_EQ(t.at({1, 2}), 6.0);
}

TEST(TensorBasics, ConstructFromDataWrongSizeThrows) {
    EXPECT_THROW(Tensor({2, 3}, {1, 2, 3}), std::exception);
}

TEST(TensorBasics, MutateThroughAt) {
    Tensor t({2, 2});
    t.at({0, 1}) = 42.0;
    EXPECT_DOUBLE_EQ(t.at({0, 1}), 42.0);
}

TEST(TensorBasics, ContiguousStrides) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.strides(), (std::vector<size_t>{12, 4, 1}));
}

TEST(TensorBasics, ToVectorRoundTrips) {
    Tensor t({2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(t.to_vector(), (std::vector<double>{1, 2, 3, 4}));
}

// ---------------------------------------------------------------------
// Views: reshape and transpose share storage instead of copying.
// ---------------------------------------------------------------------

TEST(TensorViews, FreshTensorIsContiguous) {
    Tensor t({2, 3});
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorViews, ReshapePreservesData) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.reshape({3, 2});
    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(r.to_vector(), (std::vector<double>{1, 2, 3, 4, 5, 6}));
}

TEST(TensorViews, ReshapeWrongNumelThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({4, 4}), std::exception);
}

TEST(TensorViews, TransposeSwapsShape) {
    Tensor t({2, 3});
    Tensor tt = t.transpose(0, 1);
    EXPECT_EQ(tt.shape(), (std::vector<size_t>{3, 2}));
}

TEST(TensorViews, TransposeSharesDataCorrectly) {
    // [[1, 2, 3],
    //  [4, 5, 6]]  ->  transpose ->  [[1, 4],
    //                                 [2, 5],
    //                                 [3, 6]]
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tt = t.transpose(0, 1);
    EXPECT_DOUBLE_EQ(tt.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(tt.at({0, 1}), 4.0);
    EXPECT_DOUBLE_EQ(tt.at({1, 0}), 2.0);
    EXPECT_DOUBLE_EQ(tt.at({2, 1}), 6.0);
}

TEST(TensorViews, TransposeIsNotContiguous) {
    Tensor t({2, 3});
    Tensor tt = t.transpose(0, 1);
    EXPECT_FALSE(tt.is_contiguous());
}

TEST(TensorViews, TransposeThenMutateAffectsOriginal) {
    // Views share storage — writing through the view should be visible in
    // the original tensor too.
    Tensor t({2, 2}, {1, 2, 3, 4});
    Tensor tt = t.transpose(0, 1);
    tt.at({0, 1}) = 99.0;  // this is t.at({1, 0})
    EXPECT_DOUBLE_EQ(t.at({1, 0}), 99.0);
}

// ---------------------------------------------------------------------
// Elementwise ops with NumPy-style broadcasting.
// ---------------------------------------------------------------------

TEST(TensorBroadcast, SameShapeAdd) {
    Tensor a({2}, {1, 2});
    Tensor b({2}, {10, 20});
    Tensor c = a + b;
    EXPECT_EQ(c.to_vector(), (std::vector<double>{11, 22}));
}

TEST(TensorBroadcast, ScalarLikeBroadcast) {
    // {2, 3} combined with {1} broadcasts the {1} tensor across every
    // element.
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({1}, {10});
    Tensor c = a + b;
    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(c.to_vector(), (std::vector<double>{11, 12, 13, 14, 15, 16}));
}

TEST(TensorBroadcast, RowBroadcast) {
    // {2, 3} + {3} broadcasts the length-3 row across both rows.
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3}, {10, 20, 30});
    Tensor c = a + b;
    EXPECT_EQ(c.to_vector(), (std::vector<double>{11, 22, 33, 14, 25, 36}));
}

TEST(TensorBroadcast, ColumnBroadcast) {
    // {2, 1} + {1, 3} broadcasts to {2, 3} — the "outer product" style case.
    Tensor a({2, 1}, {1, 2});
    Tensor b({1, 3}, {10, 20, 30});
    Tensor c = a * b;
    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(c.to_vector(), (std::vector<double>{10, 20, 30, 20, 40, 60}));
}

TEST(TensorBroadcast, IncompatibleShapesThrow) {
    Tensor a({2, 3});
    Tensor b({4});
    EXPECT_THROW(a + b, std::exception);
}

TEST(TensorBroadcast, Subtraction) {
    Tensor a({2}, {5, 7});
    Tensor b({2}, {1, 2});
    EXPECT_EQ((a - b).to_vector(), (std::vector<double>{4, 5}));
}

TEST(TensorBroadcast, Division) {
    Tensor a({2}, {10, 9});
    Tensor b({2}, {2, 3});
    EXPECT_EQ((a / b).to_vector(), (std::vector<double>{5, 3}));
}

// ---------------------------------------------------------------------
// 2D matmul.
// ---------------------------------------------------------------------

TEST(TensorMatmul, TwoByTwo) {
    // [[1, 2],   [[5, 6],   [[19, 22],
    //  [3, 4]] x  [7, 8]] =  [43, 50]]
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_EQ(c.to_vector(), (std::vector<double>{19, 22, 43, 50}));
}

TEST(TensorMatmul, NonSquare) {
    // {2, 3} x {3, 2} -> {2, 2}
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 2}));
    // row0: [1,2,3]·[7,9,11]=58, [1,2,3]·[8,10,12]=64
    // row1: [4,5,6]·[7,9,11]=139, [4,5,6]·[8,10,12]=154
    EXPECT_EQ(c.to_vector(), (std::vector<double>{58, 64, 139, 154}));
}

TEST(TensorMatmul, MismatchedInnerDimThrows) {
    Tensor a({2, 3});
    Tensor b({4, 2});
    EXPECT_THROW(a.matmul(b), std::exception);
}

TEST(TensorMatmul, NonRank2Throws) {
    Tensor a({2});
    Tensor b({2, 2});
    EXPECT_THROW(a.matmul(b), std::exception);
}
