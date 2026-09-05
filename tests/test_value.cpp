// TDD driver for Phase 1. Work top to bottom: each TEST group is roughly
// ordered by dependency (ValueBasics before ValueBackward before
// ValueActivations before ValueGradCheck before NeuronSanity).
//
// Run just this binary with:  ./build/tests/tinygrad_tests
// Run one test with:          ./build/tests/tinygrad_tests --gtest_filter=ValueBackward.*

#include "tinygrad/value.hpp"

#include <cmath>

#include <gtest/gtest.h>

using tinygrad::Value;

// ---------------------------------------------------------------------
// Forward-pass sanity checks — no backward() involved yet.
// ---------------------------------------------------------------------

TEST(ValueBasics, StoresData) {
    Value x(3.0);
    EXPECT_DOUBLE_EQ(x.data(), 3.0);
    EXPECT_DOUBLE_EQ(x.grad(), 0.0);  // grads start at zero
}

TEST(ValueBasics, Addition) {
    Value a(2.0), b(3.0);
    Value c = a + b;
    EXPECT_DOUBLE_EQ(c.data(), 5.0);
}

TEST(ValueBasics, Multiplication) {
    Value a(2.0), b(3.0);
    Value c = a * b;
    EXPECT_DOUBLE_EQ(c.data(), 6.0);
}

TEST(ValueBasics, Subtraction) {
    Value a(5.0), b(3.0);
    EXPECT_DOUBLE_EQ((a - b).data(), 2.0);
}

TEST(ValueBasics, Division) {
    Value a(6.0), b(3.0);
    EXPECT_DOUBLE_EQ((a / b).data(), 2.0);
}

TEST(ValueBasics, UnaryNegation) {
    Value a(4.0);
    EXPECT_DOUBLE_EQ((-a).data(), -4.0);
}

TEST(ValueBasics, ScalarOverloads) {
    Value a(2.0);
    EXPECT_DOUBLE_EQ((a + 3.0).data(), 5.0);
    EXPECT_DOUBLE_EQ((3.0 + a).data(), 5.0);
    EXPECT_DOUBLE_EQ((a * 3.0).data(), 6.0);
    EXPECT_DOUBLE_EQ((3.0 * a).data(), 6.0);
}

TEST(ValueBasics, Pow) {
    Value a(2.0);
    EXPECT_DOUBLE_EQ(a.pow(3.0).data(), 8.0);
}

// ---------------------------------------------------------------------
// Backward pass — the actual point of this project.
// ---------------------------------------------------------------------

TEST(ValueBackward, SimpleAdd) {
    // c = a + b  =>  dc/da = 1, dc/db = 1
    Value a(2.0), b(3.0);
    Value c = a + b;
    c.backward();
    EXPECT_DOUBLE_EQ(a.grad(), 1.0);
    EXPECT_DOUBLE_EQ(b.grad(), 1.0);
}

TEST(ValueBackward, SimpleMul) {
    // c = a * b  =>  dc/da = b, dc/db = a
    Value a(2.0), b(3.0);
    Value c = a * b;
    c.backward();
    EXPECT_DOUBLE_EQ(a.grad(), 3.0);
    EXPECT_DOUBLE_EQ(b.grad(), 2.0);
}

TEST(ValueBackward, ReusedValueAccumulatesGradient) {
    // y = x * x  =>  dy/dx = 2x. This is the test that catches the classic
    // bug of using `=` instead of `+=` in backward_fn: if you overwrite
    // instead of accumulate, x's two uses stomp on each other and you get
    // the wrong answer (or the LAST branch's contribution only).
    Value x(3.0);
    Value y = x * x;
    y.backward();
    EXPECT_DOUBLE_EQ(x.grad(), 6.0);  // 2 * 3
}

TEST(ValueBackward, ChainOfOps) {
    // A slightly bigger expression, worked out by hand:
    //   a = 2, b = -3, c = 10
    //   e = a * b        = -6
    //   d = e + c        = 4
    //   f = -2
    //   L = d * f        = -8
    //
    // dL/dd = f = -2
    // dL/de = dL/dd * de/de = -2 * 1 = -2
    // dL/dc = dL/dd * dd/dc = -2 * 1 = -2
    // dL/da = dL/de * de/da = -2 * b = -2 * -3 = 6
    // dL/db = dL/de * de/db = -2 * a = -2 * 2  = -4
    // dL/df = d = 4
    Value a(2.0), b(-3.0), c(10.0), f(-2.0);
    Value e = a * b;
    Value d = e + c;
    Value L = d * f;

    L.backward();

    EXPECT_DOUBLE_EQ(a.grad(), 6.0);
    EXPECT_DOUBLE_EQ(b.grad(), -4.0);
    EXPECT_DOUBLE_EQ(c.grad(), -2.0);
    EXPECT_DOUBLE_EQ(f.grad(), 4.0);
}

// ---------------------------------------------------------------------
// Activations — needed once you build a neuron out of these.
// ---------------------------------------------------------------------

TEST(ValueActivations, ReluPositive) {
    Value x(3.0);
    Value y = x.relu();
    EXPECT_DOUBLE_EQ(y.data(), 3.0);
    y.backward();
    EXPECT_DOUBLE_EQ(x.grad(), 1.0);
}

TEST(ValueActivations, ReluNegativeClampsGradToZero) {
    Value x(-3.0);
    Value y = x.relu();
    EXPECT_DOUBLE_EQ(y.data(), 0.0);
    y.backward();
    EXPECT_DOUBLE_EQ(x.grad(), 0.0);
}

TEST(ValueActivations, TanhForwardValue) {
    Value x(0.0);
    EXPECT_NEAR(x.tanh().data(), 0.0, 1e-9);
}

TEST(ValueActivations, TanhGradientAtZeroIsOne) {
    // d/dx tanh(x) = 1 - tanh(x)^2, which is 1 at x = 0.
    Value x(0.0);
    Value y = x.tanh();
    y.backward();
    EXPECT_NEAR(x.grad(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------
// Numerical gradient checking — the technique you'll reuse for every op
// you add later (conv2d especially). Compares your analytic backward()
// against a finite-difference approximation of the same derivative.
// ---------------------------------------------------------------------

namespace {
// Central difference: f'(x) ~= (f(x+h) - f(x-h)) / 2h
double numerical_grad(const std::function<double(double)>& f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2 * h);
}
}  // namespace

TEST(ValueGradCheck, MatchesNumericalGradientForCompoundExpression) {
    // f(x) = tanh(x * x + 3*x)  — nothing special about this, just
    // exercises mul/add/tanh together.
    auto f = [](double xv) { return std::tanh(xv * xv + 3 * xv); };

    double x0 = 0.7;
    Value x(x0);
    Value y = (x * x + 3.0 * x).tanh();
    y.backward();

    double analytic = x.grad();
    double numeric = numerical_grad(f, x0);
    EXPECT_NEAR(analytic, numeric, 1e-4);
}

// ---------------------------------------------------------------------
// A tiny "neuron" built purely from Value ops — proves the primitives
// compose the way a real network needs them to, before you build any
// Tensor/Layer abstraction on top in a later phase.
// ---------------------------------------------------------------------

TEST(NeuronSanity, TwoInputNeuronForwardAndBackward) {
    // out = tanh(w1*x1 + w2*x2 + b)
    Value x1(1.0), x2(-2.0);
    Value w1(0.5), w2(-1.0), b(0.3);

    Value out = (w1 * x1 + w2 * x2 + b).tanh();
    out.backward();

    // Just check the forward value is right and every input got a
    // non-trivial gradient (i.e. backward actually reached the leaves).
    double expected = std::tanh(0.5 * 1.0 + (-1.0) * (-2.0) + 0.3);
    EXPECT_NEAR(out.data(), expected, 1e-9);

    EXPECT_NE(w1.grad(), 0.0);
    EXPECT_NE(w2.grad(), 0.0);
    EXPECT_NE(x1.grad(), 0.0);
    EXPECT_NE(x2.grad(), 0.0);
    EXPECT_NE(b.grad(), 0.0);
}
