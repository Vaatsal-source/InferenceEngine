#include "tinygrad/tensor_grad.hpp"

#include <stdexcept>
#include <unordered_set>

namespace tinygrad {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::GradTensor::") + fn +
                            " is not implemented yet — see include/tinygrad/tensor_grad.hpp for the spec");
}

// Sums `grad` (shaped like a broadcast op's output) back down to
// `target_shape` (one of that op's original input shapes). Mirror image of
// Phase 2's map_to_source_index: instead of reading the same source value
// for every stretched position, we accumulate every stretched position's
// gradient into that one shared source position.
Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& target_shape) {
    if (grad.shape() == target_shape) return grad;

    Tensor result(target_shape);  // zero-filled
    size_t total = grad.numel();
    const std::vector<size_t>& grad_shape = grad.shape();
    std::vector<size_t> idx(grad_shape.size(), 0);
    size_t skip = grad_shape.size() - target_shape.size();

    for (size_t linear = 0; linear < total; ++linear) {
        size_t rem = linear;
        for (size_t d = grad_shape.size(); d-- > 0;) {
            idx[d] = rem % grad_shape[d];
            rem /= grad_shape[d];
        }

        std::vector<size_t> target_idx(target_shape.size());
        for (size_t d = 0; d < target_shape.size(); ++d) {
            target_idx[d] = (target_shape[d] == 1) ? 0 : idx[skip + d];
        }
        result.at(target_idx) += grad.at(idx);
    }
    return result;
}
}  // namespace

// --- trivial plumbing (already done for you) --------------------------

TensorNode::TensorNode(Tensor data_in) : data(std::move(data_in)), grad(Tensor(data.shape())) {
    // Tensor(shape) is Phase 2's zero-init constructor — reused here to
    // build a same-shaped, all-zero grad. This is the direct analogue of
    // Node's `grad = 0.0` default from Phase 1.
}

GradTensor::GradTensor(Tensor data) : node_(std::make_shared<TensorNode>(std::move(data))) {}

GradTensor::GradTensor(std::shared_ptr<TensorNode> node) : node_(std::move(node)) {}

const Tensor& GradTensor::data() const { return node_->data; }
const Tensor& GradTensor::grad() const { return node_->grad; }

// --- everything below is yours to implement ----------------------------
//
// Suggested order (matches the test file):
//   1. backward()                                    (TensorGradBackward.*,
//                                                       reuses Phase 1's algorithm)
//   2. operator+ (start with same-shape case)         (TensorGradBackward.*)
//   3. unbroadcasting inside operator+'s backward_fn  (TensorGradBroadcast.*)
//   4. operator*                                      (TensorGradBackward.*,
//                                                       TensorGradBroadcast.*)
//   5. sum()                                          (needed to get a
//                                                       scalar loss at all)
//   6. matmul()                                       (TensorGradMatmul.*)
//
// --- backward() ---------------------------------------------------------
// Identical shape to Value::backward() from Phase 1:
//   - Post-order DFS over node_->prev, using an
//     unordered_set<TensorNode*> to avoid revisiting shared nodes.
//   - If data().numel() != 1, throw std::runtime_error — you can't
//     unambiguously seed a gradient of 1.0 into a multi-element tensor.
//   - Seed node_->grad = Tensor(shape, {1.0}) — a one-element tensor
//     holding 1.0 (dL/dL = 1), matching this node's shape (which is {1}
//     if you got here via sum()).
//   - Walk the topo order in reverse, calling backward_fn() on each node.
//
// --- unbroadcasting (the one genuinely new idea this phase) ------------
// When operator+ or operator* combines two GradTensors of DIFFERENT
// shapes, Phase 2's broadcasting rules "stretch" the smaller one — e.g. a
// {3} tensor added to a {2,3} tensor conceptually gets copied across both
// rows. The forward pass doesn't need to know this happened; but the
// BACKWARD pass does, because the incoming gradient (shaped like the
// output, {2,3}) needs to be correctly summed back down to {3} before it
// can be added into that input's own {3}-shaped grad. Otherwise shapes
// won't match and you'll be silently or loudly wrong.
//
// Concretely, write a helper (free function in an anonymous namespace,
// same style as Phase 2's helpers):
//
//   Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& target_shape) {
//       if (grad.shape() == target_shape) return grad;
//       Tensor result(target_shape);  // zero-filled
//       // Walk every index of `grad` (decode a linear counter into a
//       // multi-index, exactly like Tensor::to_vector() does internally).
//       // For each one, map it down to an index into `result`: align
//       // shapes from the right; wherever target_shape has a dim of size
//       // 1 (or is missing that dimension entirely, i.e. it was a
//       // leading dim that got broadcast away), collapse to index 0 for
//       // that dimension in `result`. Then result.at(mapped_idx) +=
//       // grad.at(idx) — note += (accumulate), not = : several positions
//       // in `grad` fold onto the same position in `result`.
//       return result;
//   }
//
// This is the exact mirror image of the map_to_source_index() helper you
// wrote in tensor.cpp's broadcasting ops — same index math, but instead of
// READING from a smaller tensor into a bigger loop, you're ACCUMULATING
// from a bigger tensor into a smaller one.
//
// --- operator+ ------------------------------------------------------------
// Worked example (the graph-building shape is identical to Phase 1's
// operator+; broadcasting is the only new piece):
//
//   GradTensor operator+(const GradTensor& a, const GradTensor& b) {
//       auto out = std::make_shared<TensorNode>(a.data() + b.data());
//       out->prev = {a.node(), b.node()};
//       out->op = "+";
//       std::weak_ptr<TensorNode> out_weak = out;
//       auto a_node = a.node();
//       auto b_node = b.node();
//       out->backward_fn = [out_weak, a_node, b_node] {
//           auto out_locked = out_weak.lock();
//           // d(a+b)/da = 1, d(a+b)/db = 1 — same as Phase 1's scalar +,
//           // just unbroadcast back to each input's own shape first.
//           a_node->grad = a_node->grad + unbroadcast(out_locked->grad, a_node->data.shape());
//           b_node->grad = b_node->grad + unbroadcast(out_locked->grad, b_node->data.shape());
//       };
//       return GradTensor(out);
//   }
//
// --- operator* ------------------------------------------------------------
// Product rule, tensor version: d(a*b)/da = b, d(a*b)/db = a — elementwise,
// broadcasting-aware, computed with Phase 2's own operator* (which already
// knows how to broadcast b.data() against out_locked->grad's shape), THEN
// unbroadcast the result down to a's original shape before accumulating:
//   a_node->grad = a_node->grad + unbroadcast(b_node->data * out_locked->grad, a_node->data.shape());
//   b_node->grad = b_node->grad + unbroadcast(a_node->data * out_locked->grad, b_node->data.shape());
//
// --- sum() ------------------------------------------------------------
// Forward: total = the sum of every element of data() (e.g. via
// data().to_vector() and a manual accumulate), wrapped as a {1}-shaped
// Tensor. Backward: the incoming grad is a single number g (out->grad's
// only element) — every element of the input contributed to that sum with
// local derivative 1, so the input's grad becomes a same-shaped Tensor
// filled entirely with g, ADDED into the input's existing grad (build it
// via Tensor(shape, vector<double>(numel, g))).
//
// --- matmul() ------------------------------------------------------------
// Forward: out.data = this->data().matmul(other.data()) — reuse Phase 2's
// Tensor::matmul directly. Backward, given dL/dC (grad of the {m,n}
// output):
//   dL/dA = dL/dC.matmul(B^T)   (shape {m,k}, matches A)
//   dL/dB = A^T.matmul(dL/dC)   (shape {k,n}, matches B)
// where ^T is Tensor::transpose(0, 1) from Phase 2. No broadcasting or
// unbroadcasting needed here — matmul doesn't broadcast in this engine.

void GradTensor::backward() { 
    std::vector<std::shared_ptr<TensorNode>> topo_order;
    std::unordered_set<TensorNode*> visited;
    //post order dfs , node only added to topo_order after all of its parents 
    std::function<void(const std::shared_ptr<TensorNode>&)> build_topo = [&](const std::shared_ptr<TensorNode>& n) {
        if (visited.count(n.get())) return;
        visited.insert(n.get());
        for (const auto& parent : n->prev) {
            build_topo(parent);
        }
        topo_order.push_back(n);
    };
    build_topo(node_);
    if (node_->data.numel() != 1) {
        throw std::runtime_error("GradTensor::backward: can only be called on a scalar (single-element) tensor — call sum() first");
    }
    node_->grad = Tensor(node_->data.shape(), std::vector<double>(node_->data.numel(), 1.0));
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        (*it)->backward_fn();
    }
}

GradTensor operator+(const GradTensor& a, const GradTensor& b) {
    auto out = std::make_shared<TensorNode>(a.data() + b.data());
    out->prev = {a.node(), b.node()};
    out->op = "+";
    std::weak_ptr<TensorNode> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad = a_node->grad + unbroadcast(out_locked->grad, a_node->data.shape());
        b_node->grad = b_node->grad + unbroadcast(out_locked->grad, b_node->data.shape());
    };
    return GradTensor(out);
}

GradTensor operator*(const GradTensor& a, const GradTensor& b) {
    auto out = std::make_shared<TensorNode>(a.data() * b.data());
    out->prev = {a.node(), b.node()};
    out->op = "*";
    std::weak_ptr<TensorNode> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad = a_node->grad + unbroadcast(b_node->data * out_locked->grad, a_node->data.shape());
        b_node->grad = b_node->grad + unbroadcast(a_node->data * out_locked->grad, b_node->data.shape());
    };
    return GradTensor(out);
}

GradTensor GradTensor::sum() const { 
    Tensor total({1});
    double sum_value = 0.0;
    for (const auto& val : data().to_vector()) {
        sum_value += val;
    }
    total.at({0}) = sum_value;

    auto out = std::make_shared<TensorNode>(total);
    out->prev = {node_};
    out->op = "sum";
    std::weak_ptr<TensorNode> out_weak = out;
    auto self_node = node_;
    out->backward_fn = [out_weak, self_node] {
        auto out_locked = out_weak.lock();
        double g = out_locked->grad.at({0});
        self_node->grad = self_node->grad + Tensor(self_node->data.shape(), std::vector<double>(self_node->data.numel(), g));
    };
    return GradTensor(out);
}

GradTensor GradTensor::matmul(const GradTensor& other) const {
    Tensor out_data = data().matmul(other.data());
    auto out = std::make_shared<TensorNode>(out_data);
    out->prev = {node_, other.node()};
    out->op = "matmul";
    std::weak_ptr<TensorNode> out_weak = out;
    auto a_node = node_;
    auto b_node = other.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad = a_node->grad + out_locked->grad.matmul(b_node->data.transpose(0, 1));
        b_node->grad = b_node->grad + a_node->data.transpose(0, 1).matmul(out_locked->grad);
    };
    return GradTensor(out);
}

}  // namespace tinygrad
