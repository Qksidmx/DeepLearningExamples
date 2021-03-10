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
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "fastertransformer/tf_op/attention_op.h"
#include "fastertransformer/common.h"
#include "fastertransformer/faster_transformer.h"
#include "tensorflow/core/framework/op.h"
#include <cuda_runtime.h>
#include <string>
namespace tensorflow
{
using GPUDevice = Eigen::GpuDevice;
using namespace fastertransformer;

namespace functor
{
template <typename T>
struct AttentionOpFunctor<GPUDevice, T>
{
  typedef typename TFTraits<T>::DataType DataType_;
  static Status Compute(OpKernelContext *context,
      AttentionInitParam<DataType_ > param,
      Attention<TransformerTraits< TFTraits<T>::OpType > > *attention)
  {
    const cudaStream_t &stream = context->eigen_device<GPUDevice>().stream();
    param.stream = stream;
    try
    {
      check_cuda_error(cublasSetStream(param.cublas_handle, stream));
      //attention->initialize();
      attention->forward(param);
      return Status::OK();
    }
    catch(std::runtime_error& error)
    {
      return errors::Internal(error.what());
    }
    catch(...)
    {
      return errors::Internal("Runtime error");
    }
  }
};
} //namespace functor

template struct functor::AttentionOpFunctor<GPUDevice, float>;
//template struct functor::BertTransformerOpFunctor<GPUDevice, Eigen::half>;
} //namespace tensorflow
#endif
