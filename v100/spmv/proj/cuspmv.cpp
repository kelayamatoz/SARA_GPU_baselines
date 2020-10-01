#include <assert.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cusolverRf.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
#include "helper_string.h"

using namespace std;

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

int main(int argc, char *argv[]) {
  cusparseHandle_t cusparseH = nullptr; // residual evaluation
  cudaStream_t stream = nullptr;
  cusparseMatDescr_t descrA = nullptr; // A is a base-0 general matrix
  cudaEvent_t begin, begin_kernel, stop_kernel, stop;
  checkCudaErrors(cudaEventCreate(&begin));
  checkCudaErrors(cudaEventCreate(&begin_kernel));
  checkCudaErrors(cudaEventCreate(&stop_kernel));
  checkCudaErrors(cudaEventCreate(&stop));

  double lower_bound = 0;
  double upper_bound = 1;
  uniform_real_distribution<double> unif(lower_bound, upper_bound);
  default_random_engine re;

  const double one = 1;
  double *h_x = nullptr;
  double *h_r = nullptr;
  int rowsA = 0; // number of rows of A
  int colsA = 0; // number of columns of A
  int nnzA = 0;  // number of nonzeros of A
  int baseA = 0; // base index in CSR format
                 // cusolverRf only works for base-0

  // CSR(A) from I/O
  int *h_csrRowPtrA = nullptr; // <int> n+1
  int *h_csrColIndA = nullptr; // <int> nnzA
  double *h_csrValA = nullptr; // <double> nnzA

  // Device pointers
  double *d_x = nullptr;
  int *d_csrRowPtrA = nullptr; // <int> n+1
  int *d_csrColIndA = nullptr; // <int> nnzA
  double *d_csrValA = nullptr; // <double> nnzA
  double *d_r = nullptr;       // <double> n, r = b - A*x

  const char *fileName = "../data/lap2D_5pt_n100.mtx";

  if (loadMMSparseMatrix<double>((char *)fileName, 'd', true, &rowsA, &colsA,
                                 &nnzA, &h_csrValA, &h_csrRowPtrA,
                                 &h_csrColIndA, true)) {
    return 1;
  }
  baseA = h_csrRowPtrA[0]; // baseA = {0,1}

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  if (baseA) {
    for (int i = 0; i <= rowsA; i++) {
      h_csrRowPtrA[i]--;
    }
    for (int i = 0; i < nnzA; i++) {
      h_csrColIndA[i]--;
    }
    baseA = 0;
  }

  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);
  checkCudaErrors(cusparseCreate(&cusparseH));
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cusparseSetStream(cusparseH, stream));
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));

  if (baseA)
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
  else
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  // Set up the vector
  h_r = (double *)malloc(sizeof(double) * rowsA);
  h_x = (double *)malloc(sizeof(double) * colsA);
  for (int i = 0; i < rowsA; i++)
    h_x[i] = unif(re);

  // cuda setups
  cudaEventRecord(begin);

  checkCudaErrors(
      cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
  checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA));
  checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

  checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA,
                             sizeof(int) * (rowsA + 1),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, h_x, sizeof(double) * colsA, cudaMemcpyHostToDevice));

  cudaEventRecord(begin_kernel);

  checkCudaErrors(cusparseDcsrmv(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 rowsA, colsA, nnzA, &one, descrA, d_csrValA,
                                 d_csrRowPtrA, d_csrColIndA, d_x, &one, d_r));

  checkCudaErrors(
      cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

  cudaEventRecord(stop_kernel);

  auto norm_r = vec_norminf(rowsA, h_r);
  cout << "norm_r = " << norm_r << endl;

  free(h_x);
  free(h_r);
  free(h_csrRowPtrA);
  free(h_csrColIndA);
  free(h_csrValA);

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_csrRowPtrA));
  checkCudaErrors(cudaFree(d_csrColIndA));
  checkCudaErrors(cudaFree(d_csrValA));
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cusparseDestroy(cusparseH));

  cudaEventRecord(stop);

  float kernelTime, totalTime, memSetupTime, memDestroyTime;
  cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
  cudaEventElapsedTime(&totalTime, begin, stop);
  cudaEventElapsedTime(&memSetupTime, begin, begin_kernel);
  cudaEventElapsedTime(&memDestroyTime, stop_kernel, stop);
  cout << "Time for total SPMV execution is " << totalTime << " ms" << endl;
  cout << "Time for kernel SPMV execution is " << kernelTime << " ms" << endl;
  cout << "Time for setup SPMV execution is " << memSetupTime << " ms" << endl;
  cout << "Time for destroy SPMV execution is " << memDestroyTime << " ms" << endl;

  return 0;
}