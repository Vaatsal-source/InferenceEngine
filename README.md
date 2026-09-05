# tinygrad-cpp

A minimal tensor + autograd engine in C++, built from scratch as the
foundation for a future inference engine. Modeled on the shape of
[micrograd](https://github.com/karpathy/micrograd), but in C++ so it
actually teaches you the memory-layout and graph-construction mechanics a
real inference engine (ONNX Runtime, ncnn, llama.cpp) needs.

## Roadmap

- **Phase 1 — Scalar autograd engine** (`include/tinygrad/value.hpp`,
  `src/value.cpp`) ← **you are here**
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
  entirely in your own C++ code.

Only Phase 1 is scaffolded so far — come back to this file to scope Phase 2
once Phase 1's tests are green.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

GoogleTest is picked up from a system install if you have one
(`sudo apt install libgtest-dev libgmock-dev` on Debian/Ubuntu/Raspberry Pi
OS), otherwise CMake fetches it from GitHub automatically — no manual setup
either way.

## Running the tests (this is your TDD loop for Phase 1)

```bash
cd build
ctest --output-on-failure          # run everything
./tests/tinygrad_tests             # or run the binary directly for gtest's nicer output
./tests/tinygrad_tests --gtest_filter=ValueBasics.*   # just one group
```

Right now, **1 of 18 tests pass** (`ValueBasics.StoresData` — the only
already-implemented bit). Everything else throws
`std::logic_error("... is not implemented yet")` from
`src/value.cpp`. That's intentional: the test file **is** the spec.

## How to work through Phase 1

1. Open `include/tinygrad/value.hpp` — read the comments on `Node` and
   `Value`. This is the full interface; nothing here should need to change.
2. Open `src/value.cpp` — every function after the "everything below is
   yours to implement" comment is a stub. There's a fully worked example
   of `operator+`'s shape in a comment right above it — follow that pattern
   for the rest.
3. Implement in this order (each unblocks the next test group):
   - `operator+`, `operator*` → makes `ValueBasics.*` pass.
   - `backward()` (topological sort, see the docstring in the header for
     the exact algorithm) → makes `ValueBackward.SimpleAdd` and
     `SimpleMul` pass.
   - `ValueBackward.ReusedValueAccumulatesGradient` is the one that catches
     people — make sure every `backward_fn` does `+=`, never `=`.
   - `pow`, unary `operator-`, `operator-`, `operator/` (these can mostly
     reuse `+`/`*` — e.g. `a - b` is `a + (-1.0 * b)`, `a / b` is
     `a * b.pow(-1.0)`).
   - `tanh`, `relu`, `exp` → makes `ValueActivations.*` and
     `NeuronSanity.*` pass.
4. `ValueGradCheck.MatchesNumericalGradientForCompoundExpression` is a
   finite-difference check — keep this pattern in your back pocket, you'll
   want it again for conv2d in Phase 4 where hand-deriving gradients gets
   error-prone.
5. Once all 18 tests are green, Phase 1 is done. Ping me and we'll scope
   the Tensor class (Phase 2) — strides, broadcasting, and how the same
   graph idea generalizes from a single `double` to an n-d buffer.

## Project layout

```
include/tinygrad/value.hpp   Phase 1 public API (Value, Node) — spec lives here
src/value.cpp                Phase 1 implementation (mostly TODO stubs)
tests/test_value.cpp         Phase 1 test suite — your TDD target
examples/                    (empty for now — Phase 5 demo goes here)
```
