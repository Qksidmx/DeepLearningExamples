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
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/attention_op.h"
#include "fastertransformer/tf_op/common_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/logging.h"
#include <cuda_fp16.h>
namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Attention")
    .Input("q_tensor: T")
    .Input("k_tensor: T")
    .Input("v_tensor: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("attention_type: int")
    .Attr("batch_size: int >= 1")
    .Attr("q_seq_len: int >= 1")
    .Attr("hidden_size: int >= 1")
    .Attr("k_seq_len: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size, q_seq_len, hidden_size, k_seq_len;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("q_seq_len", &q_seq_len);
      c->GetAttr("hidden_size", &hidden_size);
      c->GetAttr("k_seq_len", &k_seq_len);
      int rank = c->Rank(c->input(0));
      if (rank != 2 && rank != 3)
      {
        return errors::InvalidArgument("[@BertTransformer::ShapeInference] "
                                       "invalid rank (from_tensor@input[0]): ",
                                       rank,
                                       ", should be 2 or 3");
      }
      // calculate output size
      shape_inference::DimensionHandle output_dim1;
      c->set_output(0, c->MakeShape({batch_size, q_seq_len, hidden_size}));


      return Status::OK();
    });
template <typename Device, typename T>
class AttentionOp : public CommonOp<T>
{
public:
  explicit AttentionOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(context, context->GetAttr("q_seq_len", &q_seq_len_));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_size", &hidden_size_));
    OP_REQUIRES_OK(context, context->GetAttr("k_seq_len", &k_seq_len_));
    OP_REQUIRES_OK(context, context->GetAttr("attention_type", &attention_type_));
  }

  void Compute(OpKernelContext *context) override
  {

    typedef TransformerTraits<traits_::OpType> TransformerTraits_;
    Attention<TransformerTraits_> *attention_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      attention_ = new Attention<TransformerTraits_>(allocator_,
                                                    batch_size_,
                                                    q_seq_len_,
                                                    hidden_size_,
                                                    k_seq_len_,
                                                    attention_type_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }

    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));

    AttentionInitParam<DataType_> param; //init param here
    param.cublas_handle = this->get_cublas_handler();
    this->get_tensor(context, 0, &param.q_tensor);
    this->get_tensor(context, 1, &param.k_tensor);
    this->get_tensor(context, 2, &param.v_tensor);


    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {batch_size_ , q_seq_len_, hidden_size_}, &output));

    param.attention_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    OP_REQUIRES_OK(
        context,
        functor::AttentionOpFunctor<Device, T>::Compute(
            context,
            param,
            attention_));
    delete attention_;
  }

private:
  int batch_size_, q_seq_len_, hidden_size_, k_seq_len_, attention_type_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Attention").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AttentionOp<GPUDevice, T>)
REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow
