// Phase 5 demo: train a tiny MLP on XOR, entirely with the engine you built
// across Phases 1-5. XOR is the classic "needs a nonlinearity" toy problem
// — no single linear layer can separate it, which is exactly why tanh
// between Linear layers matters.
//
// Run with: ./build/examples/xor_demo

#include "tinygrad/nn.hpp"

#include <cstdio>

using tinygrad::GradTensor;
using tinygrad::Linear;
using tinygrad::SGD;
using tinygrad::Sequential;
using tinygrad::Tensor;

int main() {
    // 4 examples, 2 input features each.
    GradTensor x(Tensor({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1}));
    // XOR targets, pre-negated so the training loop can compute
    // (pred - target) as (pred + neg_target) — see tensor_grad.hpp's
    // comment on why GradTensor has no operator- of its own.
    GradTensor neg_y(Tensor({4, 1}, {0, -1, -1, 0}));

    Sequential model({Linear(2, 8, /*seed=*/42), Linear(8, 1, /*seed=*/7)});
    SGD optimizer(model.parameters(), /*learning_rate=*/0.5);

    const int epochs = 3000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        GradTensor pred = model.forward(x);
        GradTensor diff = pred + neg_y;
        GradTensor loss = (diff * diff).sum();
        loss.backward();
        optimizer.step();

        if (epoch % 500 == 0 || epoch == epochs - 1) {
            std::printf("epoch %4d  loss %.6f\n", epoch, loss.data().at({0}));
        }
    }

    std::printf("\nfinal predictions:\n");
    GradTensor final_pred = model.forward(x);
    const char* labels[4] = {"0 xor 0", "0 xor 1", "1 xor 0", "1 xor 1"};
    for (size_t i = 0; i < 4; ++i) {
        std::printf("  %s = %.4f\n", labels[i], final_pred.data().at({i, 0}));
    }
    return 0;
}
