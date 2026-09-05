#pragma once
//
// Phase 1: scalar autograd engine (this is your micrograd-in-C++).
//
// The idea: Value wraps a single double and, as you combine Values with
// operators (+, *, tanh, relu, ...), it secretly builds a computation graph
// behind the scenes. Calling backward() on the final Value walks that graph
// in reverse topological order and accumulates dL/dx into every node's
// .grad, via the chain rule.
//
// Why shared_ptr<Node>? Because the SAME Value can be used more than once in
// an expression (e.g. `y = x*x` uses `x` twice), and both uses need to
// accumulate gradient into the *same* underlying node. Value is a thin
// value-typed handle around a shared graph node — copying a Value copies the
// handle, not the node.
//
// This header is fully specified (signatures + the Node graph shape). What's
// NOT implemented yet is the actual math in src/value.cpp — that's the part
// you write. Each operator needs to:
//   1. Compute the forward result (data).
//   2. Build a Node that records its parents (prev) so backward() can walk
//      the graph.
//   3. Attach a backward_fn closure that knows how to push this node's
//      .grad backward onto its parents' .grad, using the chain rule for
//      THAT specific operator.
//
// tests/test_value.cpp already has the target behavior written out — build
// against it and make the tests pass one at a time.

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tinygrad {

// A single node in the computation graph. You generally won't touch Node
// directly outside of value.cpp — Value is the public-facing handle.
struct Node {
    double data = 0.0;
    double grad = 0.0;

    // Parent nodes this one was computed from (empty for leaf/input values).
    std::vector<std::shared_ptr<Node>> prev;

    // Local backward step: given that `grad` has already been set to
    // dL/d(this node), push the appropriate contribution into each parent's
    // `.grad`. Defaults to a no-op (correct for leaf nodes / anything with
    // no parents).
    //
    // IMPORTANT: this must be `+=` onto the parent's grad, not `=` — a node
    // can be a parent of multiple children (e.g. `x` used twice), and
    // gradients from every path that flows through it must accumulate.
    std::function<void()> backward_fn = [] {};

    // Optional: op name for debugging / printing the graph (e.g. "+", "*",
    // "tanh"). Not required for correctness.
    std::string op;
};

class Value {
public:
    // Construct a leaf value (no parents, backward_fn is a no-op).
    explicit Value(double data);

    // Wrap an existing node (used internally by operators to build results).
    explicit Value(std::shared_ptr<Node> node);

    double data() const;
    double grad() const;

    // Direct access to the underlying node, mostly useful for tests/debugging
    // and for implementing backward().
    const std::shared_ptr<Node>& node() const { return node_; }

    // Runs backward from this Value as the root of the graph:
    //   1. Topologically sort all nodes reachable via `prev` (children
    //      before parents... actually parents must be visited AFTER their
    //      children in the reverse pass, so build the order such that a
    //      node appears after everything that depends on it).
    //   2. Seed this node's grad to 1.0 (dL/dL = 1).
    //   3. Walk the topological order in REVERSE, calling backward_fn() on
    //      each node.
    //
    // Does NOT zero out existing grads first — see zero_grad() below.
    void backward();

    // Elementwise ops. Each of these should:
    //   - compute the forward value,
    //   - create a new Node with `prev = {this->node_, other.node_}`
    //     (or just {this->node_} for unary ops),
    //   - set backward_fn to the correct local derivative.
    friend Value operator+(const Value& a, const Value& b);
    friend Value operator-(const Value& a, const Value& b);
    friend Value operator*(const Value& a, const Value& b);
    friend Value operator/(const Value& a, const Value& b);
    Value operator-() const;  // unary negation, i.e. this * -1

    Value pow(double exponent) const;  // this ** exponent (exponent is a
                                        // plain double, not a Value — keeps
                                        // the derivative simple: d/dx x^n =
                                        // n * x^(n-1))
    Value exp() const;
    Value tanh() const;
    Value relu() const;  // max(0, x); derivative is 1 if x > 0 else 0

private:
    std::shared_ptr<Node> node_;
};

// Convenience overloads so you can write `2.0 * x` and `x + 1.0`, not just
// `x * Value(2.0)`.
Value operator+(double a, const Value& b);
Value operator+(const Value& a, double b);
Value operator*(double a, const Value& b);
Value operator*(const Value& a, double b);
Value operator-(double a, const Value& b);
Value operator-(const Value& a, double b);

}  // namespace tinygrad
