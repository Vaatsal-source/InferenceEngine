#include "tinygrad/nn.hpp"

#include <cmath>
#include <stdexcept>

namespace tinygrad {

namespace {
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::") + fn +
                            " is not implemented yet — see include/tinygrad/nn.hpp for the spec");
}
}  // namespace

// --- everything below is yours to implement ----------------------------
//
// Do them in this order — each one is easier once the last one works:
//   1. tanh                                    (TanhActivation.*)
//   2. Linear(weight, bias) + Linear::forward  (LinearForward.*)
//   3. Linear::parameters                      (LinearForward.*)
//   4. Linear(in_features, out_features, seed) (LinearRandomInit.*)
//   5. Sequential::forward + ::parameters      (SequentialForward.*)
//   6. SGD::step + SGD::zero_grad              (SGDOptimizer.*)
// (The actual training loop that USES all of this lives in
// examples/xor.cpp, already written for you — you don't need to touch it.)
//
// =========================================================================
// 1. tanh
// =========================================================================
// This is a copy of the exact same op you already wrote three times before
// (Value::tanh in Phase 1, and conv2d/max_pool2d in Phase 4) — same shape,
// just applied to a whole Tensor's worth of numbers instead of one.
//
// FORWARD — turn every number in x into tanh(that number):
//   1. Grab the plain numbers out: std::vector<double> vals = x.data().to_vector();
//   2. Make a new vector the same size, where new_vals[i] = std::tanh(vals[i]).
//   3. Wrap it back up as a Tensor with the SAME SHAPE as x:
//        Tensor out_data(x.data().shape(), new_vals);
//
// GRAPH WIRING — identical pattern to every op you've written before:
//   auto out = std::make_shared<TensorNode>(out_data);
//   out->prev = {x.node()};
//   std::weak_ptr<TensorNode> out_weak = out;
//   auto x_node = x.node();
//
// BACKWARD — the formula is d/dx tanh(x) = 1 - tanh(x)^2. You already have
// tanh(x) sitting in out_locked->data (that's exactly what forward just
// computed), so:
//   out->backward_fn = [out_weak, x_node] {
//       auto out_locked = out_weak.lock();
//       std::vector<double> t = out_locked->data.to_vector();       // tanh(x) values
//       std::vector<double> g = out_locked->grad.to_vector();       // incoming gradient
//       std::vector<double> local_grad(t.size());
//       for (size_t i = 0; i < t.size(); ++i) {
//           local_grad[i] = (1.0 - t[i] * t[i]) * g[i];
//       }
//       Tensor delta(out_locked->data.shape(), local_grad);
//       x_node->grad = x_node->grad + delta;   // accumulate, same as every other op
//   };
//   return GradTensor(out);
//
// =========================================================================
// 2. Linear(weight, bias) — the simple constructor
// =========================================================================
// This one just needs to remember the two Tensors it was given:
//   weight_ = GradTensor(std::move(weight));
//   bias_   = GradTensor(std::move(bias));
// That's the entire function body. (You can't write this in the
// member-initializer list the normal way here because the placeholder
// list is already filled in above — just assign inside the {} body.)
//
// =========================================================================
// 3. Linear::forward
// =========================================================================
// One line: return x.matmul(weight_) + bias_;
// Walk through WHY this works: x is shaped {N, in_features} (N = how many
// examples you're running at once). weight_ is {in_features, out_features}.
// matmul gives you {N, out_features}. bias_ is just {out_features} — a
// single row — and Phase 3's broadcasting + automatically copies that one
// row across all N rows of the matmul result. You already built all three
// pieces (matmul, broadcasting +) in earlier phases — this function is
// just plugging them together.
//
// =========================================================================
// 4. Linear::parameters
// =========================================================================
// One line: return {weight_, bias_};
// (This is just so an optimizer can find every learnable number in this
// layer later.)
//
// =========================================================================
// 5. Linear(in_features, out_features, seed) — the random-init constructor
// =========================================================================
// Goal: fill weight_ with small random numbers instead of letting the
// caller pick them by hand.
//   1. std::mt19937 rng(seed);   // a random number generator, seeded so
//                                 // tests get the same "random" numbers
//                                 // every time
//   2. double scale = 1.0 / std::sqrt(static_cast<double>(in_features));
//   3. std::uniform_real_distribution<double> dist(-scale, scale);
//   4. Make an empty std::vector<double> of size in_features * out_features,
//      and fill EVERY slot by calling dist(rng) (each call gives you one
//      new random number in [-scale, scale]).
//   5. weight_ = GradTensor(Tensor({in_features, out_features}, that_vector));
//   6. bias_ = GradTensor(Tensor({out_features}));  // Phase 2's zero-init
//      constructor — bias just starts at all zeros, no randomness needed.
//
// =========================================================================
// 6. Sequential::forward
// =========================================================================
// This is just "run each layer, in order, with tanh squeezed in between —
// except after the very last layer." Plain loop:
//   GradTensor h = x;
//   for (size_t i = 0; i < layers_.size(); ++i) {
//       h = layers_[i].forward(h);
//       if (i + 1 != layers_.size()) {       // not the last layer
//           h = tanh(h);
//       }
//   }
//   return h;
//
// =========================================================================
// 7. Sequential::parameters
// =========================================================================
// Collect every layer's parameters into one list:
//   std::vector<GradTensor> result;
//   for (const auto& layer : layers_) {
//       auto layer_params = layer.parameters();              // {weight, bias}
//       result.insert(result.end(), layer_params.begin(), layer_params.end());
//   }
//   return result;
//
// =========================================================================
// 8. SGD::step
// =========================================================================
// For EVERY parameter p in parameters_, you want: p's numbers get a little
// smaller or bigger, nudged in the direction that reduces the loss. The
// formula is: new_value = old_value - learning_rate * gradient.
// You don't need any new machinery for this — TensorNode's `data` and
// `grad` fields are public, and p.node() gives you the TensorNode directly:
//   for (auto& p : parameters_) {
//       Tensor step = p.node()->grad * Tensor({1}, {-learning_rate_});
//       p.node()->data = p.node()->data + step;
//   }
// (Multiplying by a {1}-shaped Tensor broadcasts against ANY shape — same
// broadcasting rule from Phase 2 — so this works whether p is shaped
// {2,3} or {5} or anything else.)
//
// =========================================================================
// 9. SGD::zero_grad
// =========================================================================
// After stepping, you want every parameter's grad reset back to zero
// before the next backward() call (otherwise gradients from this step and
// the next would add together, which is wrong). For every parameter p:
//   p.node()->grad = Tensor(p.node()->data.shape());  // Phase 2's
//                                                       // zero-init ctor

GradTensor tanh(const GradTensor& x) {
    std::vector<double> vals = x.data().to_vector();
    std::vector<double> new_vals(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        new_vals[i] = std::tanh(vals[i]);
    }
    return GradTensor(Tensor(x.data().shape(), new_vals));
}

// GradTensor has no default constructor (same as Phase 1's Value), so these
// stubs need SOME placeholder value in the member-initializer list before
// the body can throw — a {1}-shaped zero tensor is just filler, replaced
// the moment you implement these for real.
Linear::Linear(size_t in_features, size_t out_features, unsigned seed) : weight_(Tensor({1})), bias_(Tensor({1})) {
    (void)in_features;
    (void)out_features;
    (void)seed;
    not_implemented("Linear(in_features, out_features, seed)");
}

Linear::Linear(Tensor weight, Tensor bias) : weight_(Tensor({1})), bias_(Tensor({1})) {
    (void)weight;
    (void)bias;
    not_implemented("Linear(weight, bias)");
}

GradTensor Linear::forward(const GradTensor& x) const {
    (void)x;
    not_implemented("Linear::forward");
}

std::vector<GradTensor> Linear::parameters() const { not_implemented("Linear::parameters"); }

Sequential::Sequential(std::vector<Linear> layers) : layers_(std::move(layers)) {}

GradTensor Sequential::forward(const GradTensor& x) const {
    (void)x;
    not_implemented("Sequential::forward");
}

std::vector<GradTensor> Sequential::parameters() const { not_implemented("Sequential::parameters"); }

SGD::SGD(std::vector<GradTensor> parameters, double learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::step() { not_implemented("SGD::step"); }

void SGD::zero_grad() { not_implemented("SGD::zero_grad"); }

}  // namespace tinygrad
