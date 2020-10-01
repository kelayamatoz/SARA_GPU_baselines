#include <iostream>
#include <cstdlib>
#include "lib/cuda_main.h"
#include "lib/node.h"
#include "lib/utils.h"

#define N_FEATURES 512
#define N_SAMPLES 102400

int main(int argc, char **argv) {
    srand(static_cast<unsigned>(1));
    vector<string> raw_trees = vector<string>();
    string raw_tree_path = "../../data/tree_serialized.cuda";

    ifstream st(raw_tree_path);
    string line;
    while (getline(st, line)) {
        string tree_str = string(line);
        raw_trees.push_back(tree_str);
    }

    auto cuda_samples = static_cast<float **>(malloc(N_SAMPLES * sizeof(float *)));
    gen_random_cuda_samples(cuda_samples, N_SAMPLES, N_FEATURES, 16.0);

    auto n_trees = static_cast<int>(raw_trees.size());
    auto cuda_trees =
            static_cast<cuda_tree_node_t **>(malloc(n_trees * sizeof(cuda_tree_node_t *)));
    auto cuda_tree_sizes = static_cast<size_t *>(malloc(n_trees * sizeof(size_t)));
    for (auto i = 0; i < n_trees; i++) {
        auto rtree = raw_trees[i];
        string::iterator f(rtree.begin()), l(rtree.end());
        ast_t tree;
        if (qi::parse(f, l, node, tree)) {
            Node n = gen_node_tree_from_ast(tree);
            size_t n_nodes = 0;
            cuda_tree_node_t *serialized = n.toCUDATreeFormat(&n_nodes);
            cuda_trees[i] = serialized;
            cuda_tree_sizes[i] = n_nodes;
        } else
            cerr << "Unparsed: " << string(f, l) << endl;
    }

    // Do some testing to make sure that things are correct
    // print the 100th line in the samples
    if (DEBUG_PRINT) {
        int test_tree_idx = 2;
        print_sample_by_idx(cuda_samples, 100, N_FEATURES);
        print_tree_by_idx(cuda_trees, 2, static_cast<int>(cuda_tree_sizes[test_tree_idx]));
    }

    clock_t time_req = clock();
    auto m = cart_inference_cpu(cuda_trees, cuda_tree_sizes, cuda_samples, N_FEATURES, N_SAMPLES, n_trees);
    cout << "CPU Passed " << static_cast<float>(clock() - time_req) / CLOCKS_PER_SEC << " seconds." << endl;
    cout << "CPU average score = " << m.avg_score << endl;
    cout << "CPU score variance = " << m.var_score << endl;
    cout << "CPU average iterations per sample = " << m.n_iter_per_sample << endl;

    cuda_main(cuda_trees, cuda_tree_sizes, cuda_samples, N_FEATURES, N_SAMPLES, n_trees);

    for (int i = 0; i < N_SAMPLES; i++) {
        free(cuda_samples[i]);
    }

    for (int i = 0; i < n_trees; i++) {
        free(cuda_trees[i]);
    }

    free(cuda_trees);
    free(cuda_tree_sizes);
    free(cuda_samples);

    return 0;
}

