#include "cuda_main.h"

int
cuda_main(
        cuda_tree_node_t **&cuda_trees,
        size_t *&cuda_tree_sizes,
        float **&cuda_samples,
        int n_features,
        int n_samples,
        int n_trees) {
    cout << "In CUDA's scope...." << endl;
    cart_inference(cuda_trees, cuda_tree_sizes, cuda_samples, n_features, n_samples, n_trees);
    return 0;
}