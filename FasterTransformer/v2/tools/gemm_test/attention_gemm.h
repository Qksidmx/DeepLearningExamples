/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include "common.h"

using namespace std;

template<typename T>
void generate_attention_gemm_config(int batch_size,
                                    int seq_len,
                                    int head_num,
                                    int size_per_head,
                                    int poly_m,
                                    int res_cnt)
{
  FILE* fd = fopen("attention_gemm_config.in", "w");
  if(fd == NULL)
  {
    printf("Cannot write to file attention_gemm_config.in\n");
    return ;
  }

  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  const int gemm_num = 2;
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  int batchCount[gemm_num] = {1,1};
  char mess[gemm_num][256];
  
  //gemm1
  M[0] = poly_m;
  N[0] = seq_len;
  K[0] = size_per_head * head_num;
  batchCount[0] = batch_size;
  strcpy(mess[0], "attention batched Gemm1");

  //gemm2
  M[1] = res_cnt;
  N[1] = size_per_head * head_num;
  K[1] = poly_m;
  batchCount[1] = batch_size;
  strcpy(mess[1], "attention batched Gemm2");

  cublasHandle_t cublas_handle;
  check_cuda_error(cublasCreate(&cublas_handle));

  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  cudaDataType_t computeType;
  int startAlgo, endAlgo;
  const int ites = 200;
  struct timeval start, end;
  
  if(sizeof(T) == sizeof(float)){
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
    computeType = CUDA_R_32F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT;
    endAlgo = (int)CUBLAS_GEMM_ALGO23;
  }
  else{
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
    computeType = CUDA_R_16F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  }

  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  printf("***FP32 Gemm Testing***\n");
  for(int i = 0; i < gemm_num; ++i)
  {
    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A;
    T* d_B;
    T* d_C;
    check_cuda_error(cudaMalloc((void**)&d_A, sizeof(T) * m * k * batchCount[i]));
    check_cuda_error(cudaMalloc((void**)&d_B, sizeof(T) * k * n * batchCount[i]));
    check_cuda_error(cudaMalloc((void**)&d_C, sizeof(T) * m * n * batchCount[i]));

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for(int algo = startAlgo; algo <= endAlgo; algo++)
    {
      cublasStatus_t status;
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
      for(int ite = 0; ite < ites; ++ite)
      {
        if(i == 0)
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_B, BType, k, n * k,
                d_A, AType, k, m * k,
                &beta,
                d_C, CType, n, m * n,
                batchCount[i],
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_B, BType, n, n*k,
                d_A, AType, k, m*k,
                &beta,
                d_C, CType, n, n * m,
                batchCount[i],
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
      }
      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      if(status == CUBLAS_STATUS_SUCCESS)
      {
        printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
        if(diffTime(start, end) / ites < exec_time)
        {
          exec_time = diffTime(start, end) / ites;
          fast_algo = algo;
        }
      }
    }
    printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
    fprintf(fd, "%d\n", fast_algo);
  }
  return;
}

