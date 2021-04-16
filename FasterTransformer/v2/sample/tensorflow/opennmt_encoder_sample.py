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

from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import sys
from utils.common import TransformerArgument, cross_check, time_test
from utils.common import DecodingArgument
from utils.decoding import tf_decoding, op_decoding
import utils.encoder
from opennmt.utils import misc
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.inputters import WordEmbedder
from opennmt.inputters import ExampleInputter

def restore_model_by_pkl(sess, variables):
    import pickle as pkl
    with open("model_opennmt.pkl", 'rb') as model_file:
        model_dict = pkl.load(model_file)

        assign_op_list = []
        for var in variables:
            print(var.name, end=' ')
            if var.name in model_dict:
                print("restore", end=' ')
                assign_op_list.append(tf.assign(var, np.reshape(model_dict[var.name], var.shape)))
                print("mean: {} , var: {} . ".format(np.mean(model_dict[var.name]), np.std(model_dict[var.name])), end=' ')
            print()
            if var.name not in model_dict:
                print(var.name, 'not saved!!!')
                sys.exit(-1)

        #assert(len(assign_op_list) == len(variables))
        sess.run(assign_op_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument('-encoder_head', '--encoder_head_number', type=int, default=8, metavar='NUMBER',
                        help='encoder head number (default: 12)')
    parser.add_argument('-encoder_size', '--encoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='encoder size per head (default: 64)')
    parser.add_argument('-decoder_head', '--decoder_head_number', type=int, default=8, metavar='NUMBER',
                        help='decoder head number (default: 8)')
    parser.add_argument('-decoder_size', '--decoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='decoder size per head (default: 64)')
    parser.add_argument('-encoder_layer', '--encoder_num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-decoder_layer', '--decoder_num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)')
    parser.add_argument('--use_ft', default=False, action='store_true', help='use FT opennmt encoder')
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.')
    parser.add_argument('--ft_package', default=False, action='store_true', help='if True, ft package will be imported')
    
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

    # if use ft_package is True then will not use *.so file
    if args.ft_package:
        import utils.encoder_pip_interface
        utils.encoder.op_encoder = utils.encoder_pip_interface.op_encoder
        utils.encoder.op_opennmt_encoder = utils.encoder_pip_interface.op_opennmt_encoder

    start_of_sentence_id = 1
    end_of_sentence_id = 2

    np.random.seed(1)
    tf.set_random_seed(1)
    kernel_initializer_range = 0.02
    bias_initializer_range = 0.02

    batch_size = args.batch_size
    beam_width = args.beam_width
    max_seq_len = args.max_seq_len
    encoder_head_num = args.encoder_head_number
    encoder_size_per_head = args.encoder_size_per_head
    decoder_head_num = args.decoder_head_number
    decoder_size_per_head = args.decoder_size_per_head
    encoder_num_layer = args.encoder_num_layer
    decoder_num_layer = args.decoder_num_layer
    encoder_hidden_dim = encoder_head_num * encoder_size_per_head
    decoder_hidden_dim = decoder_head_num * decoder_size_per_head
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 2e-5
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 2e-2

    initializer_range = 0.02
    # generate random data
    memory_sequence_length = max_seq_len
    source_embedding = np.random.randn(batch_size, memory_sequence_length, encoder_hidden_dim)
    source_embedding = tf.convert_to_tensor(source_embedding, dtype=tf_datatype)

    source_inputter = WordEmbedder("source_vocabulary", embedding_size=encoder_hidden_dim)
    target_inputter = WordEmbedder("target_vocabulary", embedding_size=encoder_hidden_dim)
    inputter = ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": "./utils/translation/wmtende.vocab",
        "target_vocabulary": "./utils/translation/wmtende.vocab"
        })
    vocab_size = target_inputter.vocabulary_size
    source_file = "./utils/translation/test.en"

    decoding_args = DecodingArgument(batch_size=batch_size,
                                     beam_width=beam_width,
                                     head_num=decoder_head_num,
                                     size_per_head=decoder_size_per_head,
                                     num_layer=decoder_num_layer,
                                     max_seq_len=max_seq_len,
                                     vocab_size=vocab_size,
                                     start_id=start_of_sentence_id,
                                     end_id=end_of_sentence_id,
                                     encoder_hidden_dim=encoder_head_num * encoder_size_per_head,
                                     dtype=tf_datatype)

    mode = tf.estimator.ModeKeys.PREDICT
    with tf.variable_scope("transformer/encoder"):
        # dataset = inputter.make_inference_dataset(source_file, batch_size)
        # iterator = dataset.make_initializable_iterator()
        # source = iterator.get_next()
        # source_embedding = source_inputter.make_inputs(source)
        # memory_sequence_length = source["length"]

        encoder_args = TransformerArgument(batch_size=batch_size, beam_width=beam_width,
                                           head_num=encoder_head_num,
                                           size_per_head=encoder_size_per_head,
                                           num_layer=encoder_num_layer,
                                           max_seq_len=max_seq_len,
                                           dtype=tf_datatype)

        if args.use_ft:
            encoder = utils.encoder.SelfAttentionEncoder(
                num_layers=encoder_num_layer,
                num_units=encoder_hidden_dim,
                num_heads=8,
                ffn_inner_dim=2048,
                dropout=0.1,
                attention_dropout=0.1,
                relu_dropout=0.1)
            attention_mask = tf.sequence_mask([memory_sequence_length], maxlen=max_seq_len, dtype=tf_datatype)
            attention_mask = tf.expand_dims(attention_mask, axis=1)
            attention_mask = tf.expand_dims(attention_mask, axis=1)
            attention_mask = tf.tile(attention_mask, [1, encoder_head_num, max_seq_len, 1])
            source_embedding = encoder.preprocess(source_embedding, mode=mode)
            memory, _, _ = encoder.encode(source_embedding, sequence_length=[memory_sequence_length], mode=mode)
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            op_encoder_result = utils.encoder.op_opennmt_encoder(inputs=source_embedding, encoder_args=encoder_args,
                                                      encoder_vars=all_vars,
                                                      attention_mask=attention_mask)
            op_encoder_result = tf.reshape(op_encoder_result, (batch_size, -1, encoder_hidden_dim))
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            encoder = SelfAttentionEncoder(
                num_layers=encoder_num_layer,
                num_units=encoder_hidden_dim,
                num_heads=8,
                ffn_inner_dim=2048,
                dropout=0.1,
                attention_dropout=0.1,
                relu_dropout=0.1)
            memory, _, _ = encoder.encode(source_embedding, [memory_sequence_length], mode=mode)
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        tf_encoder_result = memory

    tf_encoder_result = tf.reshape(
        tf_encoder_result, [batch_size, -1, encoder_hidden_dim])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not args.use_ft:
            saver = tf.train.Saver(all_vars)
            saver.restore(sess, "translation/ckpt/model.ckpt-500000")
        # sess.run(tf.tables_initializer())
        # sess.run(iterator.initializer)
        if args.use_ft:
            restore_model_by_pkl(sess, all_vars)

        iteration = 0
        while iteration < 3:
            try:
                if args.use_ft:
                    result = sess.run([tf_encoder_result, op_encoder_result])
                else:
                    result = sess.run([tf_encoder_result])
                if args.use_ft:
                    print("[INFO] tf : ", result[0])
                    print("[INFO] op : ", result[1])
                    cross_check("Encoder", result[0], result[1], atol_threshold)
                else:
                    print("[INFO] tf : ", result)
                iteration += 1
            except tf.errors.OutOfRangeError:
                break
        if args.test_time:
            prtint("RUN TIME TEST ...")
            tf_time = time_test(sess, tf_encoder_result, 100)
            op_time = time_test(sess, op_encoder_result, 100)

            print("tf_opennmt time cost: ", tf_time)
            print("op_opennmt time cost: ", op_time)
