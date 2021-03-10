#include "fastertransformer/common.h"
#include "common.h"

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
__global__
void stried_slice(T *source, T *target, int batch_size, int seq_len, int hidden_dim){
    if (blockIdx.y == 0 && threadIdx.x < hidden_dim){
        int source_offset = blockIdx.x * hidden_dim * seq_len;
        int target_offset = blockIdx.x * hidden_dim;
        target[target_offset + threadIdx.x] = (float)source[threadIdx.x + source_offset];
    }
}

template <typename T>
__global__
void  dot_product_sum_kernel(T* result,T* target,T *source, int batch_size, int seq_len, int hidden_dim)
{

    int qk_offset = blockIdx.x * hidden_dim;
    float qk = threadIdx.x < hidden_dim ? (float)target[threadIdx.x + qk_offset] * source[threadIdx.x + qk_offset]: 0.0f;
    float sum_val = blockReduceSum<float>(qk);
    if(threadIdx.x == 0)
    {
      result[blockIdx.x] = sum_val;

    }
    __syncthreads();
}

template <typename T>
void stried_slice_kernel_kernelLauncher(T *source, T *target, int batch_size, int seq_len, int hidden_dim, cudaStream_t stream)
{
    dim3 grid(batch_size, seq_len);
    dim3 block;
    block.x = 1024;
    stried_slice<T><<<grid, block, 0, stream>>>(source,
                                                target,
                                                batch_size,
                                                seq_len,
                                                hidden_dim);
}

template <typename T>
void  dot_product_sum_kernel_kernelLauncher(T* result,T* target,T *source, int batch_size, int res_cnt, int hidden_dim,cudaStream_t stream)
{
    dim3 grid;
    dim3 block;
    block.x = 1024;
    grid.x = batch_size * res_cnt;
    dot_product_sum_kernel<T><<<grid, block, 0, stream>>>(result, target, source,
                                                            batch_size, res_cnt, hidden_dim);
}


template void
stried_slice_kernel_kernelLauncher<float>(float *source, float *target, int batch_size, int seq_len, int hidden_dim, cudaStream_t stream);

template void
dot_product_sum_kernel_kernelLauncher<float>(float* result,float* target,float *source, int batch_size, int res_cnt, int hidden_dim,cudaStream_t stream);

