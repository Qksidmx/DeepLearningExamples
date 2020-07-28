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
from utils.common import DecodingArgument
from utils.decoding import tf_decoding, op_decoding
from opennmt.utils import misc
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.inputters import WordEmbedder
from opennmt.inputters import ExampleInputter

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
    parser.add_argument('--saved_model_path', type=str, help='SavedModel path(for serving)')
    parser.add_argument('--ckpt_path', type=str, help='OpenNMT checkpoint path')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

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

    source_inputter = WordEmbedder("source_vocabulary", embedding_size=512)
    target_inputter = WordEmbedder("target_vocabulary", embedding_size=512)
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
        dataset = inputter.make_inference_dataset(source_file, batch_size)
        iterator = dataset.make_initializable_iterator()
        source = iterator.get_next()
        if args.saved_model_path:
            source_string_ph = tf.placeholder_with_default(source['tokens'], shape=[None, None])
            source_length_ph = tf.placeholder_with_default(source['length'], shape=[None])
            source_ids_ph = tf.placeholder_with_default(source['ids'], shape=[None, None])
            source_embedding = source_inputter.make_inputs(
                {'tokens': source_string_ph, 'length': source_length_ph, 'ids': source_ids_ph})
            memory_sequence_length = source_length_ph
        else:
            source_embedding = source_inputter.make_inputs(source)
            memory_sequence_length = source["length"]

        encoder = SelfAttentionEncoder(
            num_layers=encoder_num_layer,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1)
        memory, _, _ = encoder.encode(source_embedding, memory_sequence_length, mode=mode)
        tf_encoder_result = memory

    tf_encoder_result = tf.reshape(
        tf_encoder_result, [batch_size, -1, encoder_hidden_dim])

    with tf.variable_scope("transformer/decoder", reuse=tf.AUTO_REUSE):
        target_inputter.build()

    with tf.variable_scope("transformer/decoder", reuse=tf.AUTO_REUSE):
        decoder = SelfAttentionDecoder(
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0)

        start_tokens = tf.fill([batch_size], start_of_sentence_id)
        end_token = end_of_sentence_id

        target_ids, _, target_length, _ = decoder.dynamic_decode_and_search(
            target_inputter.embedding,
            start_tokens,
            end_token,
            vocab_size=vocab_size,
            beam_width=beam_width,
            memory=memory,
            memory_sequence_length=memory_sequence_length)
        target_vocab_rev = target_inputter.vocabulary_lookup_reverse()
        target_tokens = target_vocab_rev.lookup(tf.cast(target_ids, tf.int64))
        opennmt_target_length = target_length
        opennmt_target_tokens = target_tokens
        opennmt_target_ids = target_ids

    ## TF Decoding ###
    ## end of tf decoding ##

    ## op decoding ##
    ## end of op decoding

    opennmt_target_ids = tf.cast(opennmt_target_ids, tf.int32)
    opennmt_target_length = tf.minimum(opennmt_target_length + 1, tf.shape(opennmt_target_ids)[2])
    if args.saved_model_path:
        signature_def = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"source_string": source_string_ph, "source_length": source_length_ph, "source_ids": source_ids_ph},
            outputs={"output_tokens": opennmt_target_tokens, "output_length": opennmt_target_length}
        )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.ckpt_path)
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        if args.saved_model_path:
            builder = tf.saved_model.builder.SavedModelBuilder(args.saved_model_path)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.SERVING],
                signature_def_map={
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature_def
                },
                assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
                legacy_init_op=tf.tables_initializer())
            builder.save()
        iteration = 0
        while iteration < 3:
            try:
                opennmt_batch_tokens, opennmt_batch_length, \
                source_result = sess.run([opennmt_target_tokens, opennmt_target_length,
                                                                            source])
                print(source_result)
                print("[INFO] opennmt: ", end='')
                for tokens, length in zip(opennmt_batch_tokens, opennmt_batch_length):
                    misc.print_bytes(b" ".join(tokens[0][:length[0] - 1]))

                iteration += 1
            except tf.errors.OutOfRangeError:
                break

