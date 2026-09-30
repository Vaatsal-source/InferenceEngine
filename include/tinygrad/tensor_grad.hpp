#pragma once
//
// Phase 3: Tensor autograd — this is Phase 1's graph machinery (Node,
// backward_fn, topological backward()) fused onto Phase 2's Tensor class.
//
// The shape of the idea is IDENTICAL to Value/Node from Phase 1:
//   - TensorNode holds `data` (a Tensor, from Phase 2), `grad` (a Tensor of
//     the SAME shape, accumulating dL/d(this)), `prev` (parent nodes), and
//     `backward_fn` (a closure that pushes this node's grad onto its
//     parents' grad).
//   - GradTensor is the thin shared_ptr<TensorNode> handle, same role as
//     Value was for Node.
// The only new wrinkle versus Phase 1 is BROADCASTING: when + or * combines
// two GradTensors of different shapes (Phase 2's broadcasting rules), the
// output shape can be bigger than either input's shape. So the backward
// pass has to do the reverse of broadcasting — called "unbroadcasting" or
// "reduction" — summing the incoming gradient back down to each input's
// original shape before accumulating it. See tensor_grad.cpp's comments for
// exactly how.
//
// Every op here is built by CALLING Phase 2's Tensor operators/matmul for
// the forward math (out.data = a.data() + b.data(), etc.) — you are not
// reimplementing tensor arithmetic, just wrapping it with graph-tracking,
// exactly like Phase 1's operator+ wrapped double addition.

#include "tinygrad/tensor.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tinygrad {

struct TensorNode {
    Tensor data;
    Tensor grad;  // always the same shape as `data`; starts at all zeros.

    std::vector<std::shared_ptr<TensorNode>> prev;

    // Same contract as Phase 1's Node::backward_fn: given that `grad` has
    // already been set to dL/d(this node), push the right contribution into
    // each parent's `.grad`. Must ACCUMULATE (+=), never overwrite — same
    // reused-value trap as Phase 1.
    std::function<void()> backward_fn = [] {};

    std::string op;

    explicit TensorNode(Tensor data_in);
};

class GradTensor {
public:
    explicit GradTensor(Tensor data);
    explicit GradTensor(std::shared_ptr<TensorNode> node);

    const Tensor& data() const;
    const Tensor& grad() const;
    const std::shared_ptr<TensorNode>& node() const { return node_; }

    // Same algorithm as Value::backward() from Phase 1: topo-sort the graph
    // (post-order DFS over prev), seed the root's grad, walk in reverse
    // calling backward_fn().
    //
    // The one new rule: the root's grad is seeded to a Tensor of the SAME
    // SHAPE as this->data(), filled with 1.0 — which only makes unambiguous
    // sense when that shape has exactly one element (a scalar loss). Call
    // sum() first if you have a multi-element tensor you want to reduce to
    // a loss. Throw std::runtime_error if data().numel() != 1.
    void backward();

    // Elementwise ops, broadcasting-aware (see header comment above).
    friend GradTensor operator+(const GradTensor& a, const GradTensor& b);
    friend GradTensor operator*(const GradTensor& a, const GradTensor& b);

    // 2D matmul only, no broadcasting (matches Phase 2's Tensor::matmul
    // restriction) — this must be {m, k}, other must be {k, n}.
    GradTensor matmul(const GradTensor& other) const;

    // Reduces every element to a single scalar Tensor of shape {1} (the sum
    // of all elements). This is how you turn a multi-element result into
    // something backward() can be called on directly.
    GradTensor sum() const;

private:
    std::shared_ptr<TensorNode> node_;
};

}  // namespace tinygrad
