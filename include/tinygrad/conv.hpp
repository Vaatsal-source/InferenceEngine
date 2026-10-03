#pragma once
//
// Phase 4: conv2d + max pooling — the ops that make this engine useful for
// vision models. Both are free functions (not GradTensor methods) that
// build on Phase 3's GradTensor exactly the way operator+/matmul did: do
// the forward math with plain Tensors, wrap the result in a new TensorNode,
// and attach a backward_fn that pushes gradient back onto the inputs.
//
// Tensor layout convention used by both ops (standard NCHW, like
// PyTorch/ONNX):
//   input  shape: {N, C_in,  H,  W}   — batch, input channels, height, width
//   weight shape: {C_out, C_in, KH, KW}  — conv2d only
//   output shape: {N, C_out, H_out, W_out}
//
// conv2d is CROSS-CORRELATION (no kernel flip), matching every real ML
// framework's "conv2d" despite the name — this is deliberate, not a
// simplification.
//
// No bias parameter on conv2d: add it yourself afterward with the
// broadcasting operator+ from Phase 3 — e.g.
//   conv2d(x, w) + bias   // bias shape {C_out, 1, 1} broadcasts over N, H, W
// This is a deliberate design choice to make you reuse Phase 3's
// broadcasting-aware add instead of duplicating bias-handling inside conv2d.

#include "tinygrad/tensor_grad.hpp"

namespace tinygrad {

// Cross-correlates `input` {N, C_in, H, W} with `weight` {C_out, C_in, KH,
// KW}. `stride` moves the window that many pixels each step; `padding`
// zero-pads H and W on both sides before sliding the window.
//
// Output shape: {N, C_out, H_out, W_out} where
//   H_out = (H + 2*padding - KH) / stride + 1
//   W_out = (W + 2*padding - KW) / stride + 1
// (integer division — throw std::invalid_argument if the window doesn't
// tile evenly, i.e. if the above division has a nonzero remainder, or if
// input.shape()[1] != weight.shape()[1] (channel mismatch).
GradTensor conv2d(const GradTensor& input, const GradTensor& weight, size_t stride = 1, size_t padding = 0);

// Slides a kernel_size x kernel_size window over `input` {N, C, H, W} with
// the given stride, taking the MAX value in each window. No padding, no
// channel mixing — output shape is {N, C, H_out, W_out} where
//   H_out = (H - kernel_size) / stride + 1
//   W_out = (W - kernel_size) / stride + 1
// Throw std::invalid_argument if that division doesn't come out even.
GradTensor max_pool2d(const GradTensor& input, size_t kernel_size, size_t stride);

}  // namespace tinygrad
