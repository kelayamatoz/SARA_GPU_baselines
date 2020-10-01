
#ifndef CUDA_TEST_CLION_CUDAMAIN_H
#define CUDA_TEST_CLION_CUDAMAIN_H

#include <iostream>
#include <cstdio>
#include <vector>
#include "cart_inference.h"

int
cuda_main(cuda_tree_node_t **&cuda_trees, size_t *&cuda_tree_sizes, float **&cuda_samples, int n_features, int n_samples,
          int n_trees);

#endif //CUDA_TEST_CLION_CUDAMAIN_H
