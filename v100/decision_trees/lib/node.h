//
// Created by tzhao713 on 10/21/19.
//

#ifndef CUDA_BASE_NODE
#define CUDA_BASE_NODE

#include <utility>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include "cuda_main.h"
#include "cart_inference.h"

#define SPLIT_STR "SplitSerialized"
#define LEAF_STR "LeafSerialized"
#define LEAF_VAL_STR "NumberSerialized"
#define SPLIT_VAL_STR "FeatureIdxSerialized"
#define DEFAULT_VAL 0.0
#define INVALID_FLOAT_VAL -1.0
#define INVALID_INT_VAL -1

#define N_NUM_CHILDREN 1
#define N_ROOT_CHILDREN 2
#define N_SPLIT_CHILDREN 5
#define DEBUG_PRINT false

namespace qi = boost::spirit::qi;

using namespace std;
using namespace boost;

typedef make_recursive_variant<
        vector<recursive_variant_>,
        string>::type ast_t;

static const qi::rule<string::iterator, ast_t()> node =
        '(' >> *node >> ')' | +~qi::char_("()");

class Node {
public:
    explicit Node(const string &value) {
        this->is_primitive = true;
        if (iequals(value, SPLIT_STR) || iequals(value, LEAF_STR)) {
            this->value = DEFAULT_VAL;
            this->name = value;
            this->is_number = false;
        } else {
            try {
                this->name = LEAF_VAL_STR;
                this->value = lexical_cast<float>(value);
                this->is_number = true;
            } catch (const boost::bad_lexical_cast &exc) {
                // The parsed value for a split will look like this: "float,int"
                vector<string> split_vals;
                this->is_number = false;
                this->is_split_val = true;
                this->name = SPLIT_VAL_STR;
                split(split_vals, value, is_any_of(","));
                this->value = lexical_cast<float>(split_vals.front());
                this->ft_idx = lexical_cast<int>(split_vals.back());
            }
        }
    }

    Node(float value, string name, bool is_number) : value(value), name(std::move(name)), is_number(is_number) {
        is_primitive = false;
    }

    Node(float value, std::string name, std::vector<Node> children, bool is_number, int ft_idx) : value(value),
                                                                                                  name(std::move(name)),
                                                                                                  children(std::move(
                                                                                                          children)),
                                                                                                  is_number(is_number),
                                                                                                  ft_idx(ft_idx) {
        is_primitive = false;
    }

    STLTreeNode toSTLTreeNode() {
        vector<STLTreeNode> stl_children = vector<STLTreeNode>();
        for (Node child : this->children) {
            stl_children.push_back(child.toSTLTreeNode());
        }
        STLTreeNode n = STLTreeNode(this->value, stl_children);
        return n;
    }

    float get_value() const {
        return value;
    }

    vector<Node> get_children() {
        return children;
    }

    int get_ft_idx() const {
        return ft_idx;
    }

    bool is_num() {
        return is_number;
    }

    bool is_primitive_node() const {
        return is_primitive;
    }

    cuda_tree_node_t* toCUDATreeFormat(size_t *len) {
        // bfs pre-order visit nodes
        vector<struct cuda_tree_node_t> v = vector<struct cuda_tree_node_t>();
        update_cuda_vector(&v, *this);
        auto v_len = static_cast<size_t>(v.size());
        auto cpy_len = v_len * sizeof(cuda_tree_node_t);
        *len = v_len;
        auto v_c_array = static_cast<cuda_tree_node_t *>(malloc(v_len * sizeof(cuda_tree_node_t)));
        memcpy(v_c_array, &v.front(), cpy_len);
        if (DEBUG_PRINT)
            cout << v.size() << endl;
        return v_c_array;
    }

    virtual ~Node() = default;

private:
    float value = INVALID_FLOAT_VAL;
    int ft_idx = INVALID_INT_VAL;
    string name = "";
    vector<Node> children = vector<Node>();
    bool is_number = false;
    bool is_split_val = false;
    bool is_primitive = false;

    static int update_cuda_vector(vector<struct cuda_tree_node_t> *v, Node &n) {
        struct cuda_tree_node_t ctn{
                .left_idx = -1, .right_idx = -1, .value = n.get_value(), .ft_idx = 0
        };
        v->push_back(ctn);
        int idx = (int) v->size() - 1;
        vector<Node> children = n.get_children();
        int ne = children.size();
        if (ne == 2) {
            Node left = children.front();
            Node right = children.back();
            int left_idx = update_cuda_vector(v, left);
            int right_idx = update_cuda_vector(v, right);
            cuda_tree_node_t *tgt = &v->at(idx);

            tgt->left_idx = left_idx;
            tgt->right_idx = right_idx;
            tgt->ft_idx = n.get_ft_idx();
        }

        return idx;
    }
};


struct node_visitor : static_visitor<Node> {
    explicit node_visitor(int indent = 0) : _indent(indent) {}

    Node operator()(const std::string &s) const {
        // The case for a leaf node.
        string tmp_str = trim_left_copy(trim_right_copy(s));
        trim_left_if(tmp_str, is_any_of(","));
        print(tmp_str);
        return Node(tmp_str);
    }

    template<class V>
    Node operator()(const V &vec) const {
        // I'm assuming that CART is always using a two-way split at each level.
        print("(");
        vector<Node> children = vector<Node>();

        for (typename V::const_iterator it = vec.begin(); it != vec.end(); it++) {
            Node n = apply_visitor(node_visitor(_indent + 1), *it);
            children.push_back(n);
        }

        print(children.size());
        print(")");

        int csize = children.size();
        Node result = Node(LEAF_STR);
        if (csize == N_NUM_CHILDREN) {
            return children.front();
        } else if (csize == N_ROOT_CHILDREN) {
            // We use an outer parenthesis to make the ast work.
            // The only valid element in the outer-most scope is the last element.
            result = children.back();
        } else if (csize == N_SPLIT_CHILDREN) {
            vector<Node> filtered_children;
            Node split_value = children.back();
            float value = split_value.get_value();
            int ft_idx = split_value.get_ft_idx();
            // Children nodes are from the 2nd element till the 2nd last element.
            copy_if(children.begin() + 1, children.end() - 1,
                    back_inserter(filtered_children),
                    [](Node n) { return n.is_num() || !n.is_primitive_node(); });
            result = Node(value, SPLIT_STR,
                          filtered_children, false, ft_idx);
        } else {
            static_assert("ERROR: Unknown number of children of Split.", "");
        }

        return result;
    }

private:
    template<typename T>
    void print(const T &v) const {
        if (DEBUG_PRINT)
            cout << string(_indent * 4, ' ') << v << endl;
    }

    int _indent;
};

Node gen_node_tree_from_ast(const ast_t &tree) {
    return apply_visitor(node_visitor(), tree);
}

#endif //CUDA_BASE_NODE