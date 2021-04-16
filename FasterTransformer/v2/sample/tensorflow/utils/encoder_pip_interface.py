# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import tensorflow as tf
import numpy as np
import math
import six
import os
from common import create_initializer
import tensorflow_faster_transformer as ft

def op_encoder(inputs,
               encoder_args,
               encoder_vars,
               attention_mask):
    # transformer_op_module = tf.load_op_library(
    #     os.path.join('./lib/libtf_fastertransformer.so'))
    print("load tf model variables ... ")
    for layer_idx in range(encoder_args.num_layer):
        val_off = layer_idx * 16
        outputs = ft.bert_transformer(
            inputs,
            inputs,
            encoder_vars[val_off + 0], encoder_vars[val_off +
                                                    2], encoder_vars[val_off + 4],
            encoder_vars[val_off + 1], encoder_vars[val_off +
                                                    3], encoder_vars[val_off + 5],
            attention_mask,
            encoder_vars[val_off + 6], encoder_vars[val_off +
                                                    7], encoder_vars[val_off + 8],
            encoder_vars[val_off + 9], encoder_vars[val_off +
                                                    10], encoder_vars[val_off + 11],
            encoder_vars[val_off + 12], encoder_vars[val_off +
                                                     13], encoder_vars[val_off + 14],
            encoder_vars[val_off + 15],
            from_seq_len=encoder_args.max_seq_len, to_seq_len=encoder_args.max_seq_len,
            head_num=encoder_args.head_num, size_per_head=encoder_args.size_per_head)
        inputs = outputs
    return outputs


def op_opennmt_encoder(inputs,
                encoder_args,
                encoder_vars,
                attention_mask):
    # transformer_op_module = tf.load_op_library(
    #     os.path.join('./lib/libtf_fastertransformer.so'))
    print("load tf model variables ... ")
    for layer_idx in range(encoder_args.num_layer):
        val_off = layer_idx * 16
        outputs = ft.opennmt_transformer(
            inputs,
            inputs,
            encoder_vars[val_off + 0], # layernorm_beta
            encoder_vars[val_off + 1], # layernorm_gamma
            encoder_vars[val_off + 2], # W_Q
            encoder_vars[val_off + 4], # W_K
            encoder_vars[val_off + 6], # W_V
            encoder_vars[val_off + 3], # bias Q
            encoder_vars[val_off + 5], # bias K
            encoder_vars[val_off + 7], # bias V
            attention_mask,
            encoder_vars[val_off + 8], # attention output kernel
            encoder_vars[val_off + 9], # attention output bias
            encoder_vars[val_off + 10],# layernorm betta
            encoder_vars[val_off + 11],# layernorm gamma
            encoder_vars[val_off + 12],# intermediate kernel
            encoder_vars[val_off + 13],# intermediate bias
            encoder_vars[val_off + 14],# output kernel
            encoder_vars[val_off + 15],# output bias
            from_seq_len=encoder_args.max_seq_len, to_seq_len=encoder_args.max_seq_len,
            head_num=encoder_args.head_num, size_per_head=encoder_args.size_per_head)
        inputs = outputs
    return norm(outputs)