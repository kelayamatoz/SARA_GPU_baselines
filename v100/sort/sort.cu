#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>

int main(int argc, char **argv) {
    assert(argc == 2);

    size_t count = std::atoi(argv[1]);

    thrust::host_vector<int> h_vec(count);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    thrust::device_vector<int> d_vec = h_vec;

    std::clock_t begin = std::clock();
    thrust::sort(d_vec.begin(), d_vec.end());
    std::clock_t end = std::clock();

    auto runtime = float(end - begin) / CLOCKS_PER_SEC;
    auto throughput = count / runtime;
    std::cout << "elapsed time: " << runtime << " (s)" << std::endl;
    std::cout << "throughput: " << throughput << " (elements/s)" << std::endl;
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    return 0;
}
