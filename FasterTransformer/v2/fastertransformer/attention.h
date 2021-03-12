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

/**
 * BERT Encoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/common_structure.h"

namespace fastertransformer
{

template <typename T>
class AttentionInitParam
{
public:
  const T *q_tensor;
  const T *k_tensor;
  const T *v_tensor;

  T *attention_out;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <class Traits_>
class Attention
{
  const IAllocator &allocator_;
  typedef typename Traits_::DataType DataType_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[2];

  DataType_ *buf_;
  DataType_ *qk_buf_;
  //DataType_ *attr_out_buf_;

  int batch_size_, q_seq_len_, hidden_size_, k_seq_len_, attention_type_;

public:
  Attention(const IAllocator &allocator, int batch_size, int q_seq_len,
                         int hidden_size, int k_seq_len, int attention_type) : allocator_(allocator), batch_size_(batch_size), q_seq_len_(q_seq_len),
                                                                            hidden_size_(hidden_size),k_seq_len_(k_seq_len), attention_type_(attention_type)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_("Attention");
#endif

//    int m = batch_size_ * q_seq_len_;
//    int k = hidden_size_;
//    int n = batch_size_ * k_seq_len_

    int buf_size = batch_size * q_seq_len_ * k_seq_len_;
//    int out_size = batch_size_ * q_seq_len_ * hidden_size_;

    try
    {
      buf_ = reinterpret_cast<DataType_ *>(allocator_.malloc(sizeof(DataType_) * buf_size ));
      if (buf_ == nullptr)
        throw std::runtime_error(std::string("Tensorflow Allocator failed to allocate internal buffer."));

      qk_buf_ = buf_;

      FILE *fd = fopen("attention_gemm_config.in", "r");
      int err = 0;
      if (fd == NULL)
        printf("attention_gemm_config.in is not found\n");
      else
      {
        err = fscanf(fd, "%d%d",&cublasAlgo_[0], &cublasAlgo_[1]);
        fclose(fd);
      }
      if (err != 2)
      {
        printf("loading GEMM algorithms error, using default GEMM algorithms!\n");
        if (Traits_::OpType == OperationType::FP32)
        {
          cublasAlgo_[0] = -1;
          cublasAlgo_[1] = -1;
        }
        else
        {
          cublasAlgo_[0] = 99;
          cublasAlgo_[1] = 99;
        }
      }
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }

  /**
   * do forward 
   **/
  void forward(AttentionInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_("Attention");
#endif
    try
    {
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;
//      int m = batch_size_ * q_seq_len_;
//      int k = hidden_size_;
//      int n = batch_size_ * k_seq_len_
//
//      int buf_size = m * n;
//      int out_size = m * k

      check_cuda_error(cublasGemmStridedBatchedEx(param.cublas_handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          k_seq_len_, q_seq_len_, hidden_size_,
          &alpha,
          param.k_tensor, AType_, hidden_size_, k_seq_len_ * hidden_size_,
          param.q_tensor, BType_, hidden_size_, q_seq_len_ * hidden_size_,
          &beta,
          qk_buf_, CType_, k_seq_len_, q_seq_len_ * k_seq_len_,
          batch_size_,
          computeType_,
          static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
		  
#ifndef NDEBUG
	  cudaDeviceSynchronize();
	  check_cuda_error(cudaGetLastError());
#endif

      single_softmax_kernel_kernelLauncher(qk_buf_, batch_size_, q_seq_len_, k_seq_len_, param.stream);
	  
#ifndef NDEBUG
	  cudaDeviceSynchronize();
	  check_cuda_error(cudaGetLastError());
#endif

      check_cuda_error(cublasGemmStridedBatchedEx(param.cublas_handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          hidden_size_, q_seq_len_, k_seq_len_,
          &alpha,
          param.v_tensor, AType_, hidden_size_, k_seq_len_ * hidden_size_,
          qk_buf_, BType_, k_seq_len_, q_seq_len_ * k_seq_len_,
          &beta,
          param.attention_out, CType_, hidden_size_, q_seq_len_ * hidden_size_,
          batch_size_,
          computeType_,
          static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }

  ~Attention()
  {
    allocator_.free(buf_);
  }
};
} // namespace fastertransformer
