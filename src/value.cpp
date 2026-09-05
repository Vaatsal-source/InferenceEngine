#include "tinygrad/value.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace tinygrad {

namespace {
// Small helper so every unimplemented function fails loudly and identically
// instead of silently returning garbage. Delete these calls as you fill in
// each function.
[[noreturn]] void not_implemented(const char* fn) {
    throw std::logic_error(std::string("tinygrad::Value::") + fn +
                            " is not implemented yet — see include/tinygrad/value.hpp for the spec");
}
}  // namespace


Value::Value(double data) : node_(std::make_shared<Node>()) {
    node_->data = data;
    // grad defaults to 0.0, prev defaults to empty, backward_fn defaults to
    // a no-op — all correct for a leaf node, nothing else to do here.
}

Value::Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}

double Value::data() const { return node_->data; }
double Value::grad() const { return node_->grad; }


void Value::backward() {
    std::vector<std::shared_ptr<Node>> topo_order;
    std::unordered_set<Node*> visited;

    // Post-order DFS: a node is appended to topo_order only after all of its
    // parents (prev) have already been visited/appended.
    std::function<void(const std::shared_ptr<Node>&)> build_topo =
        [&](const std::shared_ptr<Node>& n) {
            if (visited.count(n.get())) return;
            visited.insert(n.get());
            for (const auto& parent : n->prev) {
                build_topo(parent);
            }
            topo_order.push_back(n);
        };
    build_topo(node_);

    node_->grad = 1.0;  // dL/dL = 1

    // Walk in reverse: outputs before inputs, so every node's grad is fully
    // accumulated before it pushes gradient onto its own parents.
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        (*it)->backward_fn();
    }
}

Value operator+(const Value& a, const Value& b) {

    auto out = std::make_shared<Node>();
    out->data = a.data() + b.data();
    out->prev = {a.node(), b.node()};
    out->op = "+";
    std::weak_ptr<Node> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad += out_locked->grad;   
        b_node->grad += out_locked->grad;   
    };
    return Value(out);
}

Value operator-(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>();
    out->data = a.data() - b.data();
    out->prev = {a.node(), b.node()};
    out->op = "-";
    std::weak_ptr<Node> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad += out_locked->grad;   // d(a-b)/da = 1
        b_node->grad -= out_locked->grad;   // d(a-b)/db = -1
    };
    return Value(out);
}

Value operator*(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>();
    out->data = a.data() * b.data();
    out->prev = {a.node(), b.node()};
    out->op = "*";
    std::weak_ptr<Node> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad += b_node->data * out_locked->grad;   // d(a*b)/da = b
        b_node->grad += a_node->data * out_locked->grad;   // d(a*b)/db = a
    };
    return Value(out);
}

Value operator/(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>();
    out->data = a.data() / b.data();
    out->prev = {a.node(), b.node()};
    out->op = "/";
    std::weak_ptr<Node> out_weak = out;
    auto a_node = a.node();
    auto b_node = b.node();
    out->backward_fn = [out_weak, a_node, b_node] {
        auto out_locked = out_weak.lock();
        a_node->grad += (1.0 / b_node->data) * out_locked->grad;   // d(a/b)/da = 1/b
        b_node->grad -= (a_node->data / (b_node->data * b_node->data)) * out_locked->grad;   // d(a/b)/db = -a/(b^2)
    };
    return Value(out);
}

Value Value::operator-() const { return *this * Value(-1.0); }

Value Value::pow(double exponent) const {
    auto out = std::make_shared<Node>();
    out->data = std::pow(node_->data, exponent);
    out->prev = {node_};
    out->op = "pow";
    std::weak_ptr<Node> out_weak = out;
    auto self_node = node_;
    out->backward_fn = [out_weak, self_node, exponent] {
        auto out_locked = out_weak.lock();
        // d(x^n)/dx = n * x^(n-1)
        self_node->grad += exponent * std::pow(self_node->data, exponent - 1.0) * out_locked->grad;
    };
    return Value(out);
}

Value Value::exp() const {
    auto out = std::make_shared<Node>();
    out->data = std::exp(node_->data);
    out->prev = {node_};
    out->op = "exp";
    std::weak_ptr<Node> out_weak = out;
    auto self_node = node_;
    out->backward_fn = [out_weak, self_node] {
        auto out_locked = out_weak.lock();
        // d(e^x)/dx = e^x, which is just out->data.
        self_node->grad += out_locked->data * out_locked->grad;
    };
    return Value(out);
}

Value Value::tanh() const {
    auto out = std::make_shared<Node>();
    out->data = std::tanh(node_->data);
    out->prev = {node_};
    out->op = "tanh";
    std::weak_ptr<Node> out_weak = out;
    auto self_node = node_;
    out->backward_fn = [out_weak, self_node] {
        auto out_locked = out_weak.lock();
        // d(tanh(x))/dx = 1 - tanh(x)^2, and tanh(x) is just out->data.
        self_node->grad += (1.0 - out_locked->data * out_locked->data) * out_locked->grad;
    };
    return Value(out);
}

Value Value::relu() const {
    auto out = std::make_shared<Node>();
    out->data = node_->data > 0.0 ? node_->data : 0.0;
    out->prev = {node_};
    out->op = "relu";
    std::weak_ptr<Node> out_weak = out;
    auto self_node = node_;
    out->backward_fn = [out_weak, self_node] {
        auto out_locked = out_weak.lock();
        // d(relu(x))/dx = 1 if x > 0 else 0.
        self_node->grad += (self_node->data > 0.0 ? 1.0 : 0.0) * out_locked->grad;
    };
    return Value(out);
}

Value operator+(double a, const Value& b) { return Value(a) + b; }
Value operator+(const Value& a, double b) { return a + Value(b); }
Value operator*(double a, const Value& b) { return Value(a) * b; }
Value operator*(const Value& a, double b) { return a * Value(b); }
Value operator-(double a, const Value& b) { return Value(a) - b; }
Value operator-(const Value& a, double b) { return a - Value(b); }

}  // namespace tinygrad
