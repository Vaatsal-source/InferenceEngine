#include "tinygrad/tensor.hpp"

#include <numeric>
#include <stdexcept>

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
    (void)shape;
    not_implemented("Tensor(shape)");
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data) {
    (void)shape;
    (void)data;
    not_implemented("Tensor(shape, data)");
}

size_t Tensor::numel() const { not_implemented("numel"); }

bool Tensor::is_contiguous() const { not_implemented("is_contiguous"); }

double& Tensor::at(const std::vector<size_t>& indices) {
    (void)indices;
    not_implemented("at");
}

double Tensor::at(const std::vector<size_t>& indices) const {
    (void)indices;
    not_implemented("at (const)");
}

std::vector<double> Tensor::to_vector() const { not_implemented("to_vector"); }

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    (void)new_shape;
    not_implemented("reshape");
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    (void)dim0;
    (void)dim1;
    not_implemented("transpose");
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator+");
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator-");
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator*");
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    (void)a;
    (void)b;
    not_implemented("operator/");
}

Tensor Tensor::matmul(const Tensor& other) const {
    (void)other;
    not_implemented("matmul");
}

}  // namespace tinygrad
