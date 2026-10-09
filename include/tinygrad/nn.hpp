#pragma once
//
// Phase 5: a tiny NN layer on top of everything else — Linear, Sequential,
// an SGD optimizer, and (in examples/) an end-to-end XOR training demo.
//
// The point of this phase is that almost NOTHING new has to be invented:
//   - Linear::forward is just matmul + broadcasting add (Phase 3 + Phase 2).
//   - Sequential is just calling several Linears (and tanh) in a row.
//   - SGD updates parameters using TensorNode's PUBLIC `data`/`grad` fields
//     directly (no new GradTensor API needed at all) — see the comment
//     above SGD below.
// The one genuinely new primitive is `tanh`, because without a
// nonlinearity between Linear layers, stacking them would collapse into a
// single linear transform (two matmuls in a row is still just a matmul) —
// XOR specifically is NOT linearly separable, so this is not optional.

#include "tinygrad/tensor_grad.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace tinygrad {

// Elementwise tanh, GradTensor version — built exactly like every other op
// in tensor_grad.cpp (new TensorNode, prev = {x.node()}, backward_fn using
// d/dx tanh(x) = 1 - tanh(x)^2, same formula as Value::tanh from Phase 1,
// just applied elementwise over however many numbers `x` holds instead of
// one). No broadcasting involved since this is unary.
GradTensor tanh(const GradTensor& x);

// A single fully-connected layer: y = x.matmul(weight) + bias.
//   x:      {N, in_features}
//   weight: {in_features, out_features}   (stored pre-transposed, so
//                                           forward needs no GradTensor
//                                           transpose at all)
//   bias:   {out_features}                 (broadcasts over the N rows)
//   y:      {N, out_features}
class Linear {
public:
    // Random-init constructor (what you'll actually train with). Weights
    // drawn uniformly from [-1/sqrt(in_features), +1/sqrt(in_features)]
    // (a standard, simple init scale); bias starts at all zeros. `seed`
    // defaults to a random one but can be fixed for reproducible tests.
    Linear(size_t in_features, size_t out_features, unsigned seed = std::random_device{}());

    // Deterministic constructor for unit tests — build a layer with known
    // weights/bias so forward/backward can be checked against hand-worked
    // numbers.
    Linear(Tensor weight, Tensor bias);

    GradTensor forward(const GradTensor& x) const;

    // Returns {weight, bias} so an optimizer can find every learnable
    // GradTensor in this layer.
    std::vector<GradTensor> parameters() const;

private:
    GradTensor weight_;
    GradTensor bias_;
};

// A stack of Linear layers with tanh applied after every layer EXCEPT the
// last (so the final output is a raw, unsquashed value — suitable as a
// regression output or a pre-sigmoid/pre-softmax logit).
class Sequential {
public:
    explicit Sequential(std::vector<Linear> layers);

    GradTensor forward(const GradTensor& x) const;

    // Every layer's parameters, concatenated in order.
    std::vector<GradTensor> parameters() const;

private:
    std::vector<Linear> layers_;
};

// Plain stochastic gradient descent. Note there is NO new GradTensor API
// needed to implement this: TensorNode's `data` and `grad` fields are
// public (see tensor_grad.hpp), and GradTensor::node() hands you the
// shared_ptr<TensorNode> directly — so step() can just do
//   p.node()->data = p.node()->data + (p.node()->grad * Tensor({1}, {-learning_rate}));
// reusing Phase 2's broadcasting Tensor::operator* and operator+ exactly
// as-is (multiplying by a {1}-shaped tensor broadcasts against any shape).
class SGD {
public:
    SGD(std::vector<GradTensor> parameters, double learning_rate);

    // p.data <- p.data - learning_rate * p.grad, for every parameter.
    void step();

    // Resets every parameter's grad back to all zeros — call this before
    // each new backward() pass, same reason Value/GradTensor::backward()
    // never auto-zeros: gradients accumulate across multiple backward()
    // calls unless you explicitly reset them.
    void zero_grad();

private:
    std::vector<GradTensor> parameters_;
    double learning_rate_;
};

}  // namespace tinygrad
