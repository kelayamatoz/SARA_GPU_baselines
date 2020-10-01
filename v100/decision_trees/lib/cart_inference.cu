//
// Created by tzhao713 on 10/22/19.
//

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

#include "cart_inference.h"
#include "cuda_headers.h"
#include "helper_cuda.h"

__device__ float
get_score_single_query(float q, cuda_tree_node_t *tr) {
    bool is_leaf = false;
    float score = 0.0;
    int n_idx = 0;
    do {
        cuda_tree_node_t n = tr[n_idx];
        float v = n.value;
        int lp = n.left_idx;
        int rp = n.right_idx;
        n_idx = q < v ? lp : rp;
        is_leaf = IS_LEAF(lp, rp);
    } while (!is_leaf);
    return score;
}

__global__ void
evaluate_cart_single_query(cuda_tree_node_t *d_tree, float *d_queries, float *d_results, int n_queries) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_queries) {
        float q = d_queries[i];
        float score = get_score_single_query(q, d_tree);
        d_results[i] = score;
    }
}

__global__ void
evaluate_cart(cuda_tree_node_t **d_trees, float **d_samples, float *d_scores, int n_samples, int n_trees) {
    // Embed the trees in the block memory
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (t_idx < n_samples) {
        float accum_score = 0.0;
        float *features = d_samples[t_idx];

        for (int i = 0; i < n_trees; i++) {
            cuda_tree_node_t *tr = d_trees[i];
            bool is_leaf = false;
            float v = 0.0;
            int n_idx = 0;
            do {
                cuda_tree_node_t n = tr[n_idx];
                v = n.value;
                int ft_idx = n.ft_idx;
                int lp = n.left_idx;
                int rp = n.right_idx;
                float q = features[ft_idx];
                n_idx = q < v ? lp : rp;
                is_leaf = IS_LEAF(lp, rp);
            } while (!is_leaf);

            accum_score += v;
        }

        d_scores[t_idx] = accum_score;
    }
}

__global__ void
print_device_malloc(
        cuda_tree_node_t **d_trees,
        const size_t *d_tree_sizes,
        float **d_samples,
        int n_features,
        int n_samples,
        int n_trees) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    cuda_tree_node_t *dtr_head = d_trees[0];
    cuda_tree_node_t dtr_head_head_node = dtr_head[0];

    int last_tree_idx = n_trees - 1;
    size_t last_tree_size = d_tree_sizes[last_tree_idx];
    cuda_tree_node_t *dtr_back = d_trees[last_tree_idx];
    cuda_tree_node_t dtr_back_last_node = dtr_back[last_tree_size - 1];

    printf(">>> Accessing thread idx = %d, "
           "tree 0 node 0 = (li = %d, lr = %d, v = %f, fid = %d), "
           "tree -1 node -1 = (li = %d, lr = %d, v = %f, fid = %d), "
           "data_sample (0, 0) = %f, "
           "data_sample (-1, -1) = %f <<< \n",
           idx,
           dtr_head_head_node.left_idx, dtr_head_head_node.right_idx, dtr_head_head_node.value,
           dtr_head_head_node.ft_idx,
           dtr_back_last_node.left_idx, dtr_back_last_node.right_idx, dtr_back_last_node.value,
           dtr_back_last_node.ft_idx,
           d_samples[0][0],
           d_samples[n_samples - 1][n_features - 1]
    );
}

void cart_inference(
        cuda_tree_node_t **&cuda_trees,
        size_t *&cuda_tree_sizes,
        float **&cuda_samples,
        int n_features,
        int n_samples,
        int n_trees) {
    cudaEvent_t begin, begin_kernel, stop_kernel, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&begin_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop);

    cuda_tree_node_t **d_trees = nullptr;
    size_t *d_tree_sizes = nullptr;
    float **d_samples = nullptr;
    float *d_scores = nullptr;

    checkCudaErrors(cudaMalloc(
            reinterpret_cast<void **>(&d_trees), n_trees * sizeof(cuda_tree_node_t *)
    ));

    checkCudaErrors(cudaMalloc(
            reinterpret_cast<void **>(&d_tree_sizes), n_trees * sizeof(size_t)
    ));

    // CUDA doesn't allow me to manipulate device pointer from the host side;
    // hence I'm storing the device pointers and then do the memcpy...
    cuda_tree_node_t *d_trees_book[n_trees];
    size_t d_tree_sizes_book[n_trees];
    for (auto i = 0; i < n_trees; i++) {
        // Update device tree
        size_t n_nodes = cuda_tree_sizes[i];
        cuda_tree_node_t *d_tr;
        checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&d_tr), n_nodes * sizeof(cuda_tree_node_t)
        ));
        printf("n_nodes = %zu\n", n_nodes);
        printf("d_tr = %p\n", d_tr);
        printf("cuda_trees[i] = %p\n", cuda_trees[i]);
        printf("(cuda_trees)[i] = %p\n", (cuda_trees)[i]);

//        cout << ">>> Accessing the " << i << "th tree ..." << endl;
//        for (int j = 0; j < cuda_tree_sizes[i]; j++) {
//            cuda_tree_node_t tmp_node = cuda_trees[i][j];
//            cout << "(";
//            cout << "left_idx = " << tmp_node.left_idx << ", ";
//            cout << "right_idx = " << tmp_node.right_idx << ", ";
//            cout << "value = " << tmp_node.value << ", ";
//            cout << "ft_idx = " << tmp_node.ft_idx << "), ";
//        }
//        cout << endl;
        cout << ">>> Memcpy length = " << n_nodes * sizeof(cuda_tree_node_t) << endl;

        cout << ">>> Testing cuda memcpy" << endl;
        checkCudaErrors(cudaMemcpy(
                d_tr, cuda_trees[i], n_nodes * sizeof(cuda_tree_node_t), cudaMemcpyHostToDevice
        ));

        cout << ">>> Passed cuda memcpy" << endl;

        d_trees_book[i] = d_tr;
        d_tree_sizes_book[i] = n_nodes;
    }

    // Copy the meta pointers to device
    cout << ">>> Copying the meta pointers for d_trees" << endl;
    checkCudaErrors(cudaMemcpy(
            d_trees, d_trees_book, sizeof(cuda_tree_node_t *) * n_trees, cudaMemcpyHostToDevice
    ));
    cout << ">>> Copying the meta pointers for d_tree_sizes" << endl;
    checkCudaErrors(cudaMemcpy(
            d_tree_sizes, d_tree_sizes_book, sizeof(size_t) * n_trees, cudaMemcpyHostToDevice
    ));

    cout << ">>> Malloc-ed the tree structures..." << endl;

    // Update device samples
    cout << ">>> Pinning data samples to the device memory" << ", n_samples = " << n_samples << endl;
    checkCudaErrors(cudaMalloc(
            reinterpret_cast<void **>(&d_samples), n_samples * sizeof(float *)
    ));

    float *d_samples_book[n_samples];
    for (auto i = 0; i < n_samples; i++) {
        float *i_sample = nullptr;
        checkCudaErrors(cudaMalloc(
                reinterpret_cast<void **>(&i_sample), n_features * sizeof(float)
        ));
        checkCudaErrors(cudaMemcpy(
                i_sample, cuda_samples[i], n_features * sizeof(float), cudaMemcpyHostToDevice
        ));

        d_samples_book[i] = i_sample;
    }

    checkCudaErrors(cudaMemcpy(
            d_samples, d_samples_book, n_samples * sizeof(float *), cudaMemcpyHostToDevice
    ));

    cout << ">>> Updated device samples" << endl;

//    print_device_malloc << < 1, 1024 >> > (d_trees, d_tree_sizes, d_samples, n_features, n_samples, n_trees);
    checkCudaErrors(cudaMalloc(
            reinterpret_cast<void **>(&d_scores), n_samples * sizeof(float)
    ));

    cout << ">>> Starting kernel" << endl;
    checkCudaErrors(cudaEventRecord(begin_kernel));
    evaluate_cart << < 256, 1024 >> > (d_trees, d_samples, d_scores, n_samples, n_trees);
    checkCudaErrors(cudaEventRecord(stop_kernel));
    gpuErrchk(cudaPeekAtLastError());
    cout << ">>> End kernel execution" << endl;

    auto h_scores = static_cast<float *>(malloc(n_samples * sizeof(float)));
    checkCudaErrors(cudaMemcpy(
            h_scores, d_scores, n_samples * sizeof(float), cudaMemcpyDeviceToHost
    ));

    // Get mean and avg of h_scores
    float sum = 0., mean = 0., var_sum = 0, var = 0.;
    for (int i = 0; i < n_samples; i ++)
        sum += h_scores[i];
    mean = sum / static_cast<float>(n_samples);
    var_sum = 0;
    for (int i = 0; i < n_samples; i ++) {
        var_sum = pow(h_scores[i] - mean, 2);
    }
    var = sqrt(var_sum / static_cast<float>(n_samples));

    cout << "GPU average score = " << mean << endl;
    cout << "GPU score variance = " << var << endl;

    cout << "All data points: " << endl;
    for (int i = 0; i < n_samples; i ++) {
        cout << h_scores[i] << endl;
    }


    // Cleanup the resources...
    cout << ">>> Freeing d_trees" << endl;
    for (auto i = 0; i < n_trees; i++) {
        checkCudaErrors(cudaFree(d_trees_book[i]));
    }

    checkCudaErrors(cudaFree(d_trees));

    cout << ">>> Freeing d_tree sizes" << endl;
    checkCudaErrors(cudaFree(d_tree_sizes));

    cout << ">>> Freeing d_samples" << endl;
    for (auto i = 0; i < n_samples; i++) {
        checkCudaErrors(cudaFree(d_samples_book[i]));
    }

    cout << ">>> Freeing d_samples metadata" << endl;
    checkCudaErrors(cudaFree(d_samples));

    cout << ">>> Freeing d_scores" << endl;
    checkCudaErrors(cudaFree(d_scores));

    cout << ">>> REPORT" << endl;
    float accum = 0;
    for (int i = 0; i < n_samples; i++)
        accum += h_scores[i];

    cout << ">>> Average sample scores: " << accum / static_cast<float>(n_samples) << endl;
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, begin_kernel, stop_kernel);
    cout << ">>> Kernel execution time: " << kernel_time << "ms" << endl;
    free(h_scores);

}

void cart_inference_single_query(vector<vector<struct cuda_tree_node_t>> *root_list,
                                 vector<float *> *cuda_samples, int n_samples, int n_features) {
    // Strategy: stores the tree in scratch pad; each thread fetches a chunk of data.
    vector<cuda_tree_node_t> *root = &root_list->front();
    cuda_tree_node_t *h_tree;
    cuda_tree_node_t *d_tree;
    float *h_queries;
    float *d_queries;
    float *d_results;
    float *h_results;

    // We assume that each block can fit the model locally.
    cudaEvent_t begin, begin_kernel, stop_kernel, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&begin_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop);

    int n_nodes = root->size();
    h_tree = root->data();

    std::vector<float> queries(n_samples);
    std::generate(queries.begin(), queries.end(), rand);
    h_queries = queries.data();

    h_results = (float *) malloc(sizeof(float) * n_samples);

    cudaMalloc(reinterpret_cast<void **>(&d_tree), n_nodes * sizeof(cuda_tree_node_t));
    cudaMalloc(reinterpret_cast<void **>(&d_queries), n_samples * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_results), n_samples * sizeof(float));

    // Malloc the data samples
    float **data_src;
    data_src = (float **) malloc(sizeof(float *) * n_samples);
    for (int i = 0; i < n_samples; i++) {
        cudaMalloc(reinterpret_cast<void **>(&data_src[i]), n_features * sizeof(float));
        cudaMemcpy(data_src[i], cuda_samples->at(i), n_features * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaEventRecord(begin);
    // TODO: setup.
    // Pin the decision h_tree onto the device. Pin the queries onto device memory as well.
    cudaMemcpy(d_tree, h_tree, n_nodes * sizeof(cuda_tree_node_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, n_samples * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(begin_kernel);
    // TODO: run kernel.

    evaluate_cart_single_query << < 1, 1024 >> > (d_tree, d_queries, d_results, n_samples);

    cudaEventRecord(stop_kernel);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(h_queries, d_results, n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop);

    float kernelTime, totalTime;
    cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
    cudaEventElapsedTime(&totalTime, begin, stop);
    printf("Time for kernel execution is: %fms\n", kernelTime);
    printf("Total time is: %fms\n", totalTime);

    free(h_results);
    cudaFree(d_queries);
    cudaFree(d_results);
}

STLTreeNode::STLTreeNode(float value, vector<STLTreeNode> children) : value(value), children(std::move(children)) {}

STLTreeNode::STLTreeNode(float value) : value(value) {}
