//
// Created by tzhao713 on 11/6/19.
//

#ifndef CUDA_BASE_UTILS_H
#define CUDA_BASE_UTILS_H

#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>

using namespace std;
using namespace boost;

struct inference_meta_info {
    float score;
    int n_iters;
    void increment() { n_iters += 1; };
};

struct inference_results {
    double avg_score;
    double var_score;
    int n_iter_per_sample;
};

void gen_cuda_samples_from_file(float **cuda_samples, const string &file_path, int n_samples, int n_features) {
    std::ifstream s(file_path);
    for (int i = 0; i < n_samples; i++) {
        string line;
        vector<string> split_strs(n_features);
        float *sample_array;
        sample_array = static_cast<float *>(malloc(sizeof(float) * n_features));
        cuda_samples[i] = sample_array;
        getline(s, line);
        split(split_strs, line, is_any_of(","));
        for (int j = 0; j < n_features; j++) {
            cuda_samples[i][j] = lexical_cast<float>(split_strs[j]);
        }
    }
}

void gen_random_cuda_samples(float **cuda_samples, int n_samples, int n_features, const float upper_bound) {

    auto array_gen = [=] {
        auto v = static_cast<float *>(malloc(sizeof(float) * n_features));
        for (int j = 0; j < n_features; j++)
            v[j] = static_cast<float>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * upper_bound);
        return v;
    };
    for (int i = 0; i < n_samples; i++)
        cuda_samples[i] = array_gen();
}

void print_sample_by_idx(float **samples, int sample_idx, int n_features) {
    for (auto i = 0; i < n_features; i++) {
        cout << samples[sample_idx][i];
        if (i == n_features - 1)
            cout << endl;
        else
            cout << ", ";
    }
}

void print_tree_by_idx(cuda_tree_node_t **tree_list, int tree_idx, int tree_size) {
    for (int i = 0; i < tree_size; i++) {
        cuda_tree_node_t tmp_node = tree_list[tree_idx][i];
        cout << "(";
        cout << "left_idx = " << tmp_node.left_idx << ", ";
        cout << "right_idx = " << tmp_node.right_idx << ", ";
        cout << "value = " << tmp_node.value << ", ";
        cout << "ft_idx = " << tmp_node.ft_idx << ")";
        if (i == tree_size - 1)
            cout << endl;
        else
            cout << "," << endl;
    }
}

inference_results cart_inference_cpu(
        cuda_tree_node_t **&cuda_trees,
        size_t *&cuda_tree_sizes,
        float **&cuda_samples,
        int n_features,
        int n_samples,
        int n_trees) {
    auto results = vector<float>(n_samples);
    auto n_iter_list = vector<int>(n_samples);
    auto inference_fn = [&](const float *features) {
        inference_meta_info m{.score=0, .n_iters=0};
        for (int i_tree = 0; i_tree < n_trees; i_tree++) {
            size_t tree_size = cuda_tree_sizes[i_tree];
            cuda_tree_node_t *tr = cuda_trees[i_tree];
            bool is_leaf = false;
            int i_node = 0;
            float v = 0.0;
            do {
                cuda_tree_node_t n = cuda_trees[i_tree][i_node];
                v = n.value;
                int ft_idx = n.ft_idx;
                float q = features[ft_idx];
                int lp = n.left_idx;
                int rp = n.right_idx;
                i_node = q < v ? lp : rp;
                is_leaf = IS_LEAF(lp, rp);
                m.increment();
            } while (!is_leaf);

            m.score += v;
        }

        return m;
    };

    for (int i_sample = 0; i_sample < n_samples; i_sample++) {
        auto inference_results = inference_fn(cuda_samples[i_sample]);
        results[i_sample] = inference_results.score;
        n_iter_list[i_sample] = inference_results.n_iters;
    }

    double mean = accumulate(results.begin(), results.end(), 0.0) / n_samples;
    double var = sqrt([&mean, &n_samples](vector<float> v) {
             for_each(v.begin(), v.end(), [&mean](float &vv) { vv = pow(vv - mean, 2); });
             return accumulate(v.begin(), v.end(), 0.0) / n_samples;
         }(results));
    int avg_iter_per_sample = accumulate(n_iter_list.begin(), n_iter_list.end(), 0) / n_samples;
    for (int i = 0; i < n_samples; i ++) {
        cout << results[i] << endl;
    }
    return inference_results{.avg_score=mean, .var_score=var, .n_iter_per_sample=avg_iter_per_sample};
}

#endif //CUDA_BASE_UTILS_H
