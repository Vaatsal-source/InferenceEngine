#include "tinygrad/conv.hpp"

#include <limits>
#include <stdexcept>

namespace tinygrad {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::") + fn +
                            " is not implemented yet — see include/tinygrad/conv.hpp for the spec");
}

// --- trivial plumbing (already done for you) --------------------------
//
// Zero-pads the last two dimensions of a {N, C, H, W} tensor by `padding`
// on every side. E.g. padding=1 turns a {1,1,3,3} tensor into {1,1,5,5}
// with the original 3x3 sitting in the middle surrounded by zeros.
Tensor pad2d(const Tensor& input, size_t padding) {
    if (padding == 0) return input;
    const auto& s = input.shape();  // {N, C, H, W}
    Tensor result({s[0], s[1], s[2] + 2 * padding, s[3] + 2 * padding});  // zero-filled
    for (size_t n = 0; n < s[0]; ++n) {
        for (size_t c = 0; c < s[1]; ++c) {
            for (size_t h = 0; h < s[2]; ++h) {
                for (size_t w = 0; w < s[3]; ++w) {
                    result.at({n, c, h + padding, w + padding}) = input.at({n, c, h, w});
                }
            }
        }
    }
    return result;
}

// Inverse of pad2d: strips `padding` off every side of the last two
// dimensions. Used in conv2d's backward pass to undo the forward padding
// before handing the gradient back to the caller (whose input was never
// padded in the first place).
Tensor crop2d(const Tensor& padded, size_t padding) {
    if (padding == 0) return padded;
    const auto& s = padded.shape();  // {N, C, H_padded, W_padded}
    Tensor result({s[0], s[1], s[2] - 2 * padding, s[3] - 2 * padding});
    const auto& out_s = result.shape();
    for (size_t n = 0; n < out_s[0]; ++n) {
        for (size_t c = 0; c < out_s[1]; ++c) {
            for (size_t h = 0; h < out_s[2]; ++h) {
                for (size_t w = 0; w < out_s[3]; ++w) {
                    result.at({n, c, h, w}) = padded.at({n, c, h + padding, w + padding});
                }
            }
        }
    }
    return result;
}
}  // namespace

// --- everything below is yours to implement ----------------------------
//
// Suggested order (matches the test file):
//   1. conv2d forward (no padding, stride 1 first)        (Conv2dForward.*)
//   2. conv2d backward (dL/dWeight, then dL/dInput)        (Conv2dBackward.*)
//   3. conv2d with stride/padding                          (Conv2dStridePadding.*)
//   4. max_pool2d forward                                  (MaxPool2dForward.*)
//   5. max_pool2d backward (route grad to the argmax only) (MaxPool2dBackward.*)
//
// --- conv2d forward -------------------------------------------------------
// 1. Validate: input.shape()[1] == weight.shape()[1] (C_in matches), else
//    throw std::invalid_argument.
// 2. Tensor padded = pad2d(input.data(), padding);
// 3. Compute H_out, W_out from padded's H/W, weight's KH/KW, and stride
//    (see header comment for the formula) — throw std::invalid_argument if
//    either division has a remainder.
// 4. Tensor out_data({N, C_out, H_out, W_out});  // zero-filled accumulator
// 5. Six nested loops — n, co, oh, ow (output position), then ci, kh, kw
//    (reduction over input channels and kernel window):
//      out_data.at({n,co,oh,ow}) +=
//          padded.at({n, ci, oh*stride + kh, ow*stride + kw}) *
//          weight.data().at({co, ci, kh, kw});
//    This is the cross-correlation: each output pixel is a dot product
//    between the kernel and the patch of (padded) input it's centered on.
//
// --- conv2d backward -------------------------------------------------------
// Given dL/dOutput (shape {N, C_out, H_out, W_out}), by the same index
// relationship as forward:
//   dL/dWeight[co,ci,kh,kw] = sum over (n, oh, ow) of
//       dL/dOutput[n,co,oh,ow] * padded_input[n,ci,oh*stride+kh,ow*stride+kw]
//   dL/dPaddedInput[n,ci,oh*stride+kh,ow*stride+kw] +=    (note +=: many
//       (oh,ow,kh,kw) combinations can land on the same input pixel when
//       stride < kernel size — "overlapping receptive fields")
//       dL/dOutput[n,co,oh,ow] * weight[co,ci,kh,kw]
// Then dL/dInput = crop2d(dL/dPaddedInput, padding) — undo the padding you
// added forward, since the caller's original (unpadded) input is what
// needs a gradient.
// Build this with the SAME six nested loops as forward, accumulating into
// two gradient accumulators instead of one output.
//
// --- max_pool2d forward -------------------------------------------------------
// For each (n, c, oh, ow), scan the kernel_size x kernel_size window
// starting at (oh*stride, ow*stride) in that channel, take the max value,
// AND remember which (kh, kw) inside the window produced it — you'll want
// that argmax position again for backward. A natural approach: build the
// output data AND a parallel "argmax index" structure (e.g.
// std::vector<size_t> storing a flat kh*kernel_size+kw per output
// position) in the same pass, then capture that argmax vector (by value)
// in the backward_fn lambda.
//
// --- max_pool2d backward -------------------------------------------------------
// Unlike conv2d, only ONE input position per output position has a
// nonzero local derivative (derivative of "max" w.r.t. the winning element
// is 1, and 0 w.r.t. every other element in the window). So: start
// dL/dInput at all zeros (same shape as input), then for each (n, c, oh,
// ow), add dL/dOutput[n,c,oh,ow] onto dL/dInput at exactly the
// (oh*stride+kh, ow*stride+kw) position recorded by forward's argmax —
// using += since, if stride < kernel_size, the same input pixel could
// conceivably be the argmax for more than one output window.

GradTensor conv2d(const GradTensor& input, const GradTensor& weight, size_t stride, size_t padding) {
    if(input.data().shape()[1] != weight.data().shape()[1]) {
        throw std::invalid_argument("conv2d: input channels (C_in) must match weight channels (C_in)");
    }
    Tensor padded = pad2d(input.data(), padding);
    const auto& s = padded.shape();
    const auto& w_s = weight.data().shape();
    size_t H_out = (s[2] - w_s[2]) / stride + 1;
    size_t W_out = (s[3] - w_s[3]) / stride + 1;
    if ((s[2] - w_s[2]) % stride != 0 || (s[3] - w_s[3]) % stride != 0) {
        throw std::invalid_argument("conv2d: output dimensions must be integers");
    }
    Tensor out_data({s[0], w_s[0], H_out, W_out});  // zero-filled accumulator
    for (size_t n = 0; n < s[0]; ++n) {
        for (size_t co = 0; co < w_s[0]; ++co) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    for (size_t ci = 0; ci < s[1]; ++ci) {
                        for (size_t kh = 0; kh < w_s[2]; ++kh) {
                            for (size_t kw = 0; kw < w_s[3]; ++kw) {
                                out_data.at({n, co, oh, ow}) += padded.at({n, ci, oh * stride + kh, ow * stride + kw}) * weight.data().at({co, ci, kh, kw});
                            }
                        }
                    }
                }
            }
        }
    }
    auto out_node = std::make_shared<TensorNode>(out_data);
    out_node->prev = {input.node(), weight.node()};
    out_node->op = "conv2d";
    std::weak_ptr<TensorNode> out_weak = out_node;
    auto input_node = input.node();
    auto weight_node = weight.node();
    out_node->backward_fn = [out_weak, input_node, weight_node, stride, padding] {
        auto out_locked = out_weak.lock();
        Tensor padded_input = pad2d(input_node->data, padding);
        Tensor dL_dPaddedInput(padded_input.shape());  // zero-filled
        Tensor dL_dWeight(weight_node->data.shape());  // zero-filled
        const auto& s = padded_input.shape();
        const auto& w_s = weight_node->data.shape();
        size_t H_out = out_locked->data.shape()[2];
        size_t W_out = out_locked->data.shape()[3];
        for (size_t n = 0; n < s[0]; ++n) {
            for (size_t co = 0; co < w_s[0]; ++co) {
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        for (size_t ci = 0; ci < s[1]; ++ci) {
                            for (size_t kh = 0; kh < w_s[2]; ++kh) {
                                for (size_t kw = 0; kw < w_s[3]; ++kw) {
                                    dL_dWeight.at({co, ci, kh, kw}) += out_locked->grad.at({n, co, oh, ow}) * padded_input.at({n, ci, oh * stride + kh, ow * stride + kw});
                                    dL_dPaddedInput.at({n, ci, oh * stride + kh, ow * stride + kw}) += out_locked->grad.at({n, co, oh, ow}) * weight_node->data.at({co, ci, kh, kw});
                                }
                            }
                        }
                    }
                }
            }
        }
        input_node->grad = input_node->grad + crop2d(dL_dPaddedInput, padding);
        weight_node->grad = weight_node->grad + dL_dWeight;
    };
    return GradTensor(out_node);
}

GradTensor max_pool2d(const GradTensor& input, size_t kernel_size, size_t stride) {
    const auto& s = input.data().shape();  // {N, C, H, W}
    if ((s[2] - kernel_size) % stride != 0 || (s[3] - kernel_size) % stride != 0) {
        throw std::invalid_argument("max_pool2d: output dimensions must be integers");
    }
    size_t H_out = (s[2] - kernel_size) / stride + 1;
    size_t W_out = (s[3] - kernel_size) / stride + 1;

    Tensor out_data({s[0], s[1], H_out, W_out});
    // Argmax position (kh, kw) within its window, flattened as kh*kernel_size+kw,
    // one per output element, in the same row-major order to_vector() uses —
    // captured below so backward_fn doesn't need to re-scan for the max.
    std::vector<size_t> argmax(out_data.numel());
    size_t flat = 0;
    for (size_t n = 0; n < s[0]; ++n) {
        for (size_t c = 0; c < s[1]; ++c) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_kh = 0, max_kw = 0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            double val = input.data().at({n, c, oh * stride + kh, ow * stride + kw});
                            if (val > max_val) {
                                max_val = val;
                                max_kh = kh;
                                max_kw = kw;
                            }
                        }
                    }
                    out_data.at({n, c, oh, ow}) = max_val;
                    argmax[flat++] = max_kh * kernel_size + max_kw;
                }
            }
        }
    }

    auto out_node = std::make_shared<TensorNode>(out_data);
    out_node->prev = {input.node()};
    out_node->op = "max_pool2d";
    std::weak_ptr<TensorNode> out_weak = out_node;
    auto input_node = input.node();
    out_node->backward_fn = [out_weak, input_node, kernel_size, stride, argmax, H_out, W_out] {
        auto out_locked = out_weak.lock();
        Tensor dL_dInput(input_node->data.shape());  // zero-filled
        const auto& in_s = input_node->data.shape();
        size_t flat = 0;
        for (size_t n = 0; n < in_s[0]; ++n) {
            for (size_t c = 0; c < in_s[1]; ++c) {
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        size_t max_kh = argmax[flat] / kernel_size;
                        size_t max_kw = argmax[flat] % kernel_size;
                        ++flat;
                        dL_dInput.at({n, c, oh * stride + max_kh, ow * stride + max_kw}) +=
                            out_locked->grad.at({n, c, oh, ow});
                    }
                }
            }
        }
        input_node->grad = input_node->grad + dL_dInput;
    };
    return GradTensor(out_node);
}

}  // namespace tinygrad
