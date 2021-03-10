#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
void stried_slice_kernel_kernelLauncher(T *source, T *target, int batch_size, int seq_len, int hidden_dim, cudaStream_t stream);

template <typename T>
void  dot_product_sum_kernel_kernelLauncher(T* result,T* target,T *source, int batch_size, int res_cnt, int hidden_dim, cudaStream_t stream);

