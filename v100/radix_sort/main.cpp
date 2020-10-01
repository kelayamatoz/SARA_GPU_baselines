#include "lib/radixSortThrust.h"

int main(int argc, char **argv) {
    // All float
//    static const char * cuda_args[] = {"cuda_radix_sort","-n=4194304","-float"};
//    const int cuda_argc = 3;

    // All int
    static const char * cuda_args[] = {"cuda_radix_sort","-n=4194304", "-keysonly", "-float"};
    const int cuda_argc = 4;
    return cudaMain(cuda_argc, const_cast<char **>(cuda_args));
}