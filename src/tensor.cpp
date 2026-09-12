#include "tinygrad/tensor.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tinygrad {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::Tensor::") + fn +
                            " is not implemented yet — see include/tinygrad/tensor.hpp for the spec");
}

// Computes standard contiguous row-major strides for a shape, e.g.
// {2, 3, 4} -> {12, 4, 1}. You'll want this in more than one constructor.
std::vector<size_t> contiguous_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t running = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = running;
        running *= shape[i];
    }
    return strides;
}
}  // namespace

// --- trivial plumbing (already done for you) --------------------------

Tensor::Tensor(std::shared_ptr<std::vector<double>> storage, size_t offset,
               std::vector<size_t> shape, std::vector<size_t> strides)
    : storage_(std::move(storage)), offset_(offset), shape_(std::move(shape)), strides_(std::move(strides)) {}

size_t Tensor::ndim() const { return shape_.size(); }

const std::vector<size_t>& Tensor::shape() const { return shape_; }

const std::vector<size_t>& Tensor::strides() const { return strides_; }

// --- everything below is yours to implement ----------------------------
//
// Suggested order (matches the test file):
//   1. Tensor(shape), Tensor(shape, data), numel()        (TensorBasics.*)
//   2. at() (both overloads), to_vector()                 (TensorBasics.*)
//   3. is_contiguous(), reshape()                          (TensorViews.*)
//   4. transpose()                                         (TensorViews.*)
//   5. operator+ - * / with broadcasting                   (TensorBroadcast.*)
//   6. matmul()                                            (TensorMatmul.*)
//
// Worked example — numel(), which everything else leans on:
//
//   size_t Tensor::numel() const {
//       return std::accumulate(shape_.begin(), shape_.end(),
//                               size_t{1}, std::multiplies<size_t>());
//   }
//
// For at(): flat index into storage_ is
//   offset_ + sum over d of (indices[d] * strides_[d])
// Bounds-check each indices[d] < shape_[d] first (throw std::out_of_range).
//
// For is_contiguous(): compare strides_ against contiguous_strides(shape_)
// (the helper above) — equal means this is a plain, unviewed layout.
//
// For reshape(): if !is_contiguous() or new_shape's product != numel(),
// throw std::runtime_error. Otherwise return a Tensor sharing storage_ and
// offset_, with shape_ = new_shape and strides_ = contiguous_strides(new_shape).
//
// For transpose(dim0, dim1): copy shape_ and strides_, std::swap the dim0
// and dim1 entries in each, return a Tensor sharing storage_/offset_.
//
// For broadcasting ops: align shapes from the right; a pair of dims is
// compatible if equal or one of them is 1. Build the output shape, then for
// each output index, map it back to an index into `a` and `b` separately —
// for a dimension where the source tensor's size is 1, always use index 0
// for that dimension (that's the "stretch"). Compute the result with at().
//
// For matmul(): require a.ndim() == 2 && b.ndim() == 2 && a.shape()[1] ==
// b.shape()[0], else throw std::invalid_argument. Triple nested loop
// (i, j, k) accumulating a.at({i,k}) * b.at({k,j}) into result.at({i,j}).

Tensor::Tensor(std::vector<size_t> shape) {
    size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    storage_ = std::make_shared<std::vector<double>>(total, 0.0);
    offset_ = 0;
    strides_ = contiguous_strides(shape);
    shape_ = std::move(shape);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data) {
    size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    if (data.size() != total) {
        throw std::invalid_argument("Tensor(shape, data): data.size() does not match shape's element count");
    }
    storage_ = std::make_shared<std::vector<double>>(std::move(data));
    offset_ = 0;
    strides_ = contiguous_strides(shape);
    shape_ = std::move(shape);
}

size_t Tensor::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
}

bool Tensor::is_contiguous() const { return strides_ == contiguous_strides(shape_); }

namespace {
size_t flat_index(const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset,
                   const std::vector<size_t>& indices) {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Tensor::at: wrong number of indices for this tensor's rank");
    }
    size_t flat = offset;
    for (size_t d = 0; d < indices.size(); ++d) {
        if (indices[d] >= shape[d]) {
            throw std::out_of_range("Tensor::at: index out of bounds for this dimension");
        }
        flat += indices[d] * strides[d];
    }
    return flat;
}
}  // namespace

double& Tensor::at(const std::vector<size_t>& indices) {
    return (*storage_)[flat_index(shape_, strides_, offset_, indices)];
}

double Tensor::at(const std::vector<size_t>& indices) const {
    return (*storage_)[flat_index(shape_, strides_, offset_, indices)];
}

std::vector<double> Tensor::to_vector() const {
    std::vector<double> result;
    result.reserve(numel());
    std::vector<size_t> idx(shape_.size(), 0);
    size_t total = numel();
    for (size_t linear = 0; linear < total; ++linear) {
        // Decode `linear` (0..total-1) into a multi-dimensional index, in
        // row-major order — the last dimension varies fastest.
        size_t rem = linear;
        for (size_t d = shape_.size(); d-- > 0;) {
            idx[d] = rem % shape_[d];
            rem /= shape_[d];
        }
        result.push_back(at(idx));  // at() respects strides_/offset_, so
                                     // this works correctly on views too.
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Tensor::reshape: tensor is not contiguous");
    }
    size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (new_total != numel()) {
        throw std::runtime_error("Tensor::reshape: new shape's element count does not match this tensor's");
    }
    return Tensor(storage_, offset_, new_shape, contiguous_strides(new_shape));
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    std::vector<size_t> new_shape = shape_;
    std::vector<size_t> new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(storage_, offset_, new_shape, new_strides);
}

namespace {
// NumPy-style broadcast: align shapes from the right; a size-1 dim (or a
// missing leading dim) stretches to match the other shape's size there.
std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t nd = std::max(a.size(), b.size());
    std::vector<size_t> result(nd);
    for (size_t i = 0; i < nd; ++i) {
        // i counts from the LEFT of the output; convert to "from the right"
        // so we can compare trailing dimensions of a and b.
        size_t from_right = nd - 1 - i;
        size_t a_dim = (from_right < a.size()) ? a[a.size() - 1 - from_right] : 1;
        size_t b_dim = (from_right < b.size()) ? b[b.size() - 1 - from_right] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Tensor: shapes are not broadcast-compatible");
        }
        result[i] = std::max(a_dim, b_dim);
    }
    return result;
}

// Maps a full output index (rank == out_shape's rank) back to an index
// valid for a source tensor of rank src_shape.size() <= out index's rank —
// missing leading dims are dropped, and any source dim of size 1 always
// reads index 0 (that's the "stretch").
std::vector<size_t> map_to_source_index(const std::vector<size_t>& out_idx, const std::vector<size_t>& src_shape) {
    size_t skip = out_idx.size() - src_shape.size();
    std::vector<size_t> src_idx(src_shape.size());
    for (size_t d = 0; d < src_shape.size(); ++d) {
        src_idx[d] = (src_shape[d] == 1) ? 0 : out_idx[skip + d];
    }
    return src_idx;
}

// Shared body for all four elementwise ops — only the actual arithmetic
// (`op`) differs between +, -, *, /.
template <typename BinaryOp>
Tensor elementwise(const Tensor& a, const Tensor& b, BinaryOp op) {
    std::vector<size_t> out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor result(out_shape);
    size_t total = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear) {
        size_t rem = linear;
        for (size_t d = out_shape.size(); d-- > 0;) {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        double av = a.at(map_to_source_index(idx, a.shape()));
        double bv = b.at(map_to_source_index(idx, b.shape()));
        result.at(idx) = op(av, bv);
    }
    return result;
}
}  // namespace

Tensor operator+(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x + y; });
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x - y; });
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x * y; });
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return elementwise(a, b, [](double x, double y) { return x / y; });
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2 || shape_[1] != other.shape()[0]) {
        throw std::invalid_argument("Tensor::matmul: requires two rank-2 tensors with matching inner dimension");
    }
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = other.shape()[1];
    Tensor result({m, n});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                sum += at({i, kk}) * other.at({kk, j});
            }
            result.at({i, j}) = sum;
        }
    }
    return result;
}

}  // namespace tinygrad
