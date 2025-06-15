#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <utility>
#include <variant>
using std::string;
namespace py = pybind11;

struct pair_hash {
    size_t operator() (const std::pair<string, string >& p) const {
        return std::hash<string>()(p.first) ^ (std::hash<string>()(p.second) << 1);
    }
};
class LCTNode {
    using ValueType = std::variant<string, std::pair<string, string>>;
    friend class LinkCutTree;
private:
    ValueType value;
    LCTNode* left;
    LCTNode* right;
    LCTNode* parent;
    bool reverse;
    bool is_edge;
    double weight;
    std::pair<string, string> max_edge;
    double max_weight;
public:
    explicit LCTNode(ValueType value): value(std::move(value)), left(nullptr),
                               right(nullptr), parent(nullptr), reverse(false), is_edge(false),
                               weight(0), max_edge({}), max_weight(0) {

    }
    LCTNode(ValueType value, double weight): value(std::move(value)), left(nullptr),
                                              right(nullptr), parent(nullptr), reverse(false), is_edge(true),
                                              weight(weight), max_weight(weight) {
        max_edge = std::get<std::pair<string, string>>(this->value);
    }
    ~LCTNode() = default;
    bool is_left() {
        return parent && parent->left == this;
    }
    void update() {
        max_weight = weight;
        if (is_edge) {
            max_edge = std::get<std::pair<string, string>>(value);  // 安全访问pair类型
        } else {
            max_edge = {};  // 非边节点置空
        }
        for(auto* child:{left,right}){
            if(child && child->max_weight>max_weight){
                max_weight=child->max_weight;
                max_edge=child->max_edge;
            }
        }
    }
    void push_down() {
        if (reverse) {
            if (left) left->reverse ^= true;
            if (right) right->reverse ^= true;
            reverse ^= true;
            std::swap(left, right);
        }
    }
    bool is_root() {
        return parent == nullptr || (parent->left != this && parent->right != this);
    }
    void rotate() {
        auto y = parent;
        auto z = y->parent;
        bool b = is_left();
        if (!y->is_root()) {
            if (z->right == y) z->right = this;
            else z->left = this;
        }
        parent = z;
        y->parent = this;
        if (b) {
            if (right) right->parent = y;
            y->left = right;
            right = y;
        } else {
            if (left) left->parent = y;
            y->right = left;
            left = y;
        }
        y->update();
        update();
    }
    void splay() {
        std::vector<LCTNode*> queue = {this};
        auto i = this;
        while (!i->is_root()) {
            queue.push_back(i->parent);
            i = i->parent;
        }
        while (!queue.empty()) {
            auto q = queue.at(queue.size() - 1);
            queue.pop_back();
            q->push_down();
        }
        while (!is_root()) {
            auto y = parent;
            auto z = y->parent;
            if (!y->is_root()) {
                if ((y->left == this) ^ (z->left == y)) rotate();
                else y->rotate();
            }
            rotate();
        }
    }
};


class LinkCutTree {
private:
    std::unordered_map<string , LCTNode*> nodes;
    std::unordered_map<std::pair<string, string>, LCTNode*,pair_hash> edges;

    LCTNode* get_node(const string& val) {
        if (!nodes.count(val)) {
            nodes[val] = new LCTNode(val);
        }
        return nodes[val];
    }
   static void access(LCTNode* x) {
        LCTNode* t = nullptr;
        while (x) {
            x->splay();
            x->right = t;
            x->update();
            t = x;
            x = x->parent;
        }
    }
    static void make_root(LCTNode* x) {
        access(x);
        x->splay();
        x->reverse ^= true;
    }
    static LCTNode* find_root(LCTNode* x) {
        access(x);
        x->splay();
        while (x->left) x = x->left;
        return x;
    }
    static void split(LCTNode* x, LCTNode* y) {
        make_root(x);
        access(y);
        y->splay();
    }
    static void _cut(LCTNode* x, LCTNode* y) {
        if (find_root(x) != find_root(y)) return;
        split(x, y);
        auto b = x->is_left() ? y->left : y->right;
        if (x->right != nullptr || x->parent != y || !b) return;
        y->left = nullptr;
        x->parent = nullptr;
    }
    static void _link(LCTNode* x, LCTNode* y) {
        if (find_root(x) == find_root(y)) return;
        make_root(x);
        x->parent = y;
    }
public:
    ~LinkCutTree() {
        for (auto& [k,v] : nodes){
            delete v;
            v=nullptr;
        }
        for (auto& [k,v] : edges){
            delete v;
            v=nullptr;
        }
    }
//    bool has_edge(const string& a,const string& b){
//        return edges.count(std::minmax(a,b));
//    }
    void link(const string& u, const string& v, double weight) {
        LCTNode* x = get_node(u);
        LCTNode* y = get_node(v);
        if (find_root(x) == find_root(y)) return;
        auto edge = std::minmax(u, v);
        auto* z = new LCTNode(edge, weight);
        _link(x, z);
        _link(z, y);
        edges[edge] = z;
    }
    void cut(const string& u, const string& v) {
        auto edge = std::minmax(u, v);
        if (!edges.count(edge)) {
            return;
        }
        LCTNode* z = edges[edge];
        LCTNode* x = get_node(u);
        LCTNode* y = get_node(v);
        _cut(x, z);
        _cut(z, y);
        if (edges.erase(edge)) { // 确保只删除一次
            delete z;
        }
    }
    bool is_connection(const string& u, const string& v) {
        return find_root(get_node(u)) == find_root(get_node(v));
    }
    std::pair<std::pair<std::optional<string>, std::optional<string>>, std::optional<double>> get_max_edge(const string& u, const string& v) {
        LCTNode* x = get_node(u);
        LCTNode* y = get_node(v);
        if (find_root(x) != find_root(y)) return {{std::nullopt,std::nullopt}, std::nullopt};
        split(x, y);
        return {y->max_edge, y->max_weight};
    }
};


PYBIND11_MODULE(lct, m) {
    py::class_<LinkCutTree>(m, "LinkCutTree_Mode")
            .def(py::init<>())
            .def("link", &LinkCutTree::link)
            .def("cut", &LinkCutTree::cut)
            .def("is_connection", &LinkCutTree::is_connection)
            .def("get_max_edge", &LinkCutTree::get_max_edge);
    // 注册类型转换
    py::register_exception<std::invalid_argument>(m, "InvalidArgument");
}
