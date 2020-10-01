//
// Created by tzhao713 on 10/22/19.
//

#ifndef CUDA_BASE_CART_INFERENCE_H
#define CUDA_BASE_CART_INFERENCE_H
#define INVALID_PTR -1
#define IS_LEAF(a, b) ((a == INVALID_PTR) && (b == INVALID_PTR))

#include <vector>

using namespace std;

struct cuda_tree_node_t {
    int left_idx;
    int right_idx;
    float value;
    int ft_idx;
};

class STLTreeNode {
public:
    STLTreeNode(float value, vector<STLTreeNode> children);

    explicit STLTreeNode(float value);

    virtual ~STLTreeNode() = default;

    float getValue() const {
        return value;
    }

    const vector<STLTreeNode> &getChildren() const {
        return children;
    }

private:
    float value = 0.0;
    std::vector<STLTreeNode> children = std::vector<STLTreeNode>();
};

void cart_inference(
        cuda_tree_node_t **&cuda_trees,
        size_t *&cuda_tree_sizes,
        float **&cuda_samples,
        int n_features,
        int n_samples,
        int n_trees);



#endif //CUDA_BASE_CART_INFERENCE_H
