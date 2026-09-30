#include "tinygrad/tensor_grad.hpp"

#include <stdexcept>
#include <unordered_set>

namespace tinygrad {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::GradTensor::") + fn +
                            " is not implemented yet — see include/tinygrad/tensor_grad.hpp for the spec");
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

void GradTensor::backward() { not_implemented("backward"); }

GradTensor operator+(const GradTensor& a, const GradTensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator+");
}

GradTensor operator*(const GradTensor& a, const GradTensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator*");
}

GradTensor GradTensor::sum() const { not_implemented("sum"); }

GradTensor GradTensor::matmul(const GradTensor& other) const {
    (void)other;
    not_implemented("matmul");
}

}  // namespace tinygrad
