# tinygrad-cpp

A minimal tensor + autograd engine in C++, built from scratch as the
foundation for a future inference engine. Modeled on the shape of
[micrograd](https://github.com/karpathy/micrograd), but in C++ so it
real inference engine (ONNX Runtime, ncnn, llama.cpp) needs.

## Roadmap

- **Phase 1 — Scalar autograd engine** (`include/tinygrad/value.hpp`,
  `src/value.cpp`) 
  A `Value` wraps one `double` and builds a computation graph as you combine
  them with `+`, `*`, `tanh`, `relu`, etc. `backward()` walks the graph in
  reverse topological order and fills in `.grad` via the chain rule. This is
  deliberately scalar-only — get the *graph mechanics* right here, where
  it's easy to hand-verify every gradient, before generalizing to n-d
  tensors.
- **Phase 2 — n-d Tensor class.** Contiguous buffer + shape + strides,
  broadcasting rules, and the same computation-graph idea but operating on
  whole tensors instead of scalars. This is where matmul and reshape/view
  semantics live.
- **Phase 3 — Tensor autograd.** Extend the Phase 1 graph machinery to
  Phase 2 tensors: backward passes for matmul, broadcasting-aware add,
  softmax, etc.
- **Phase 4 — conv2d + pooling.** The ops that actually make this useful for
  vision models.
- **Phase 5 — A tiny NN layer on top** (`Linear`, `Sequential`,
  an optimizer) and a demo: train a small MLP on XOR or MNIST end to end,
  entirely in C++ code.


## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

GoogleTest is picked up from a system install if you have one
(`sudo apt install libgtest-dev libgmock-dev` on Debian/Ubuntu/Raspberry Pi
OS), otherwise CMake fetches it from GitHub automatically — no manual setup
either way.

## Running the tests 

```bash
cd build
ctest --output-on-failure          # run everything
./tests/tinygrad_tests             # or run the binary directly for gtest's nicer output
./tests/tinygrad_tests --gtest_filter=ValueBasics.*   # just one group
```



## Project layout

```
include/tinygrad/value.hpp   Phase 1 public API (Value, Node) — spec lives here
src/value.cpp                Phase 1 implementation (mostly TODO stubs)
tests/test_value.cpp         Phase 1 test suite — your TDD target
examples/                    (empty for now — Phase 5 demo goes here)
```
