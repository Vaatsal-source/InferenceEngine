#pragma once
//
// Phase 2: n-d Tensor class — contiguous buffer + shape + strides.
//
// No autograd here yet (that's Phase 3) — this is purely the data structure
// and forward-only ops: construction, element access, reshape/transpose
// (views, not copies), elementwise arithmetic with NumPy-style broadcasting,
// and 2D matmul.
//
// Storage model:
//   - `storage_` is a shared_ptr<vector<double>> — the actual flat buffer.
//     shared_ptr so that views (reshape/transpose) can point at the SAME
//     buffer instead of copying data.
//   - `shape_`   is the size of each dimension, e.g. {2, 3} for a 2x3 matrix.
//   - `strides_` is, for each dimension, how many elements to skip in
//     `storage_` to move one step along that dimension. For a contiguous
//     row-major 2x3 tensor, strides are {3, 1}: moving one step in dim 0
//     (row) skips 3 elements, moving one step in dim 1 (col) skips 1.
//   - `offset_`  is where this view starts in `storage_` (0 unless this
//     Tensor is a view produced by another op).
//
// Why strides instead of always assuming contiguous row-major layout?
// Because transpose() needs to swap two dimensions WITHOUT copying data —
// it just swaps the corresponding entries in shape_ and strides_. That's
// the whole trick views are built on.
//
// Element access: flat_index = offset_ + sum(indices[d] * strides_[d]).
//
// Broadcasting (NumPy rules, used by +, -, *, /): to combine two tensors of
// possibly different shapes, align their shapes from the RIGHT (trailing
// dimension). Two dimensions are compatible if they're equal, or if either
// one is 1 (a size-1 dimension "stretches" to match the other). Missing
// leading dimensions are treated as size 1. E.g. {3, 1} and {1, 4} broadcast
// to {3, 4}; {5} and {2, 5} broadcast to {2, 5}; {2, 3} and {4} do NOT
// broadcast (3 != 4, neither is 1).

#include <cstddef>
#include <memory>
#include <vector>

namespace tinygrad {

class Tensor {
public:
    // Zero-initialized tensor of the given shape.
    explicit Tensor(std::vector<size_t> shape);

    // Tensor of the given shape, filled with `data` in row-major order.
    // `data.size()` must equal the product of `shape` (throw otherwise).
    Tensor(std::vector<size_t> shape, std::vector<double> data);

    size_t ndim() const;
    size_t numel() const;
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    // True if strides_ match the "standard" contiguous row-major layout for
    // shape_ (i.e. this is not a transposed/reshaped view). reshape() needs
    // this to decide whether it can return a cheap view.
    bool is_contiguous() const;

    // Multi-dimensional element access. `indices.size()` must equal ndim().
    // Throws std::out_of_range if any index is out of bounds for its dim.
    double& at(const std::vector<size_t>& indices);
    double at(const std::vector<size_t>& indices) const;

    // Flattens this tensor into a plain vector in row-major (logical) order
    // — i.e. respects strides/offset, so it works correctly on views too.
    // Mostly useful for tests and debugging.
    std::vector<double> to_vector() const;

    // Returns a NEW VIEW with the given shape, sharing storage_ with this
    // tensor (no data copy). Only valid when is_contiguous() is true and
    // new_shape's product equals numel() — throw std::runtime_error
    // otherwise (a non-contiguous reshape would need a copy, which this
    // Phase intentionally does not implement — that's a future exercise).
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    // Returns a NEW VIEW with dimensions dim0 and dim1 swapped — swap the
    // corresponding entries of shape_ and strides_, keep the same storage_,
    // offset_. No data movement. (This is what makes the result
    // non-contiguous in general.)
    Tensor transpose(size_t dim0, size_t dim1) const;

    // Elementwise ops with NumPy-style broadcasting (see header comment).
    // Each output element is computed independently; there is no graph
    // tracking here (that's Phase 3).
    friend Tensor operator+(const Tensor& a, const Tensor& b);
    friend Tensor operator-(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, const Tensor& b);
    friend Tensor operator/(const Tensor& a, const Tensor& b);

    // 2D matrix multiply only for this phase: this must be {m, k}, other
    // must be {k, n}, result is {m, n}. Throw std::invalid_argument if
    // either tensor isn't rank 2 or the inner dimensions (k) don't match.
    // Batched/N-D matmul is a Phase 3 concern once autograd needs it.
    Tensor matmul(const Tensor& other) const;

private:
    std::shared_ptr<std::vector<double>> storage_;
    size_t offset_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    // Internal ctor used by views (reshape/transpose) to share storage_.
    Tensor(std::shared_ptr<std::vector<double>> storage, size_t offset,
           std::vector<size_t> shape, std::vector<size_t> strides);
};

}  // namespace tinygrad
