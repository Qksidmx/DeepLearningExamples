import tensorflow as tf
import numpy as np
import argparse
import os
from utils.common import TransformerArgument, time_test, cross_check
from utils.encoder import tf_encoder, op_encoder, dot_product_attention
# from opennmt.inputters import WordEmbedder

def get_input_data(batch_size, seq_len, hidden_dim, poly_m):
    from_data = np.random.randn(batch_size, seq_len, hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)

    # mask = np.random.randint(2, size=(batch_size, seq_len, seq_len))
    mask = np.ones((batch_size, seq_len, seq_len))
    attention_mask = tf.convert_to_tensor(mask, dtype=tf_datatype)

    poly_code = np.random.randn(poly_m, hidden_dim)
    poly_code = np.array([poly_code,] * batch_size)
    tf_poly_code = tf.convert_to_tensor(poly_code, dtype=tf_datatype)

    return from_tensor, attention_mask, tf_poly_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-l', '--num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-s', '--seq_len', type=int, default=32, metavar='NUMBER',
                        help='sequence length (default: 32)')
    parser.add_argument('-n', '--head_number', type=int, default=12, metavar='NUMBER',
                        help='head number (default: 12)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)')
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.')
    parser.add_argument('-pm', "--poly_m", default=16, type=int, help="Number of m of polyencoder")
    parser.add_argument("--res_cnt", default=1, type=int, help="Number of responses")

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

    np.random.seed(1)
    tf.set_random_seed(1)
    kernel_initializer_range = 0.02
    bias_initializer_range = 0.02

    batch_size = args.batch_size
    num_layer = args.num_layer
    seq_len = args.seq_len
    head_num = args.head_number
    size_per_head = args.size_per_head
    poly_m = args.poly_m
    res_cnt = args.res_cnt
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 2e-5
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 2e-2

    hidden_dim = head_num * size_per_head
    initializer_range = 0.02

    input_a, mask_a, poly_codes =  get_input_data(batch_size, seq_len, hidden_dim, poly_m)
    input_b, mask_b, _ = get_input_data(batch_size*res_cnt, seq_len, hidden_dim, poly_m)
    # pm_embedding_table = np.random.randn(poly_m, hidden_dim).astype(np_datatype)  # a [poly_m, hidden_dim] table
    # pm_embedding_table = tf.convert_to_tensor(pm_embedding_table)



    encoder_args = TransformerArgument(batch_size=batch_size,
                                       beam_width=1,
                                       head_num=head_num,
                                       size_per_head=size_per_head,
                                       num_layer=num_layer,
                                       max_seq_len=seq_len,
                                       dtype=tf_datatype)

    tf_ctx_out = tf_encoder(input_tensor=input_a,
                                   encoder_args=encoder_args,
                                   attention_mask=mask_a)
    tf_ctx_out = tf.reshape(tf_ctx_out, (batch_size, seq_len, hidden_dim))
    '''
    with tf.variable_scope("poly_ctx", reuse=tf.AUTO_REUSE):
        pm_embedding_table = tf.get_variable(
            name="poly_embedding",
            shape=[poly_m, hidden_dim],
            dtype=tf_datatype,
            initializer=tf.truncated_normal_initializer(stddev=initializer_range, dtype=tf_datatype))
        poly_code_ids = tf.range(poly_m, dtype=tf.int32)
        poly_code_ids = tf.tile(tf.expand_dims(poly_code_ids,0),(batch_size,1))
        poly_codes = tf.gather(pm_embedding_table, poly_code_ids)
    '''

    tf_embs, _ = dot_product_attention(poly_codes, tf_ctx_out, tf_ctx_out, tf.estimator.ModeKeys.PREDICT)

    tf_cand_emb = tf_encoder(input_tensor=input_b,
                                     encoder_args=encoder_args,
                                     attention_mask=mask_b)
    # with tf.variable_scope("poly"):
    tf_cand_emb = tf.reshape(tf_cand_emb, (batch_size*res_cnt, seq_len, hidden_dim))
    tf_cand_emb = tf.squeeze(tf_cand_emb[:, 0:1, :], axis=1)
    tf_cand_emb = tf.reshape(tf_cand_emb, (batch_size, res_cnt, hidden_dim))

    tf_ctx_emb, _ = dot_product_attention(tf_cand_emb, tf_embs, tf_embs, tf.estimator.ModeKeys.PREDICT)
    tf_dot_product = tf.reduce_sum(tf_ctx_emb * tf_cand_emb ,axis=-1)





    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    '''
    for variables in encoder_variables:
        print(variables)
    '''
    op_ctx_out = op_encoder(inputs=input_a,
                                   encoder_args=encoder_args,
                                   encoder_vars=encoder_variables,
                                   attention_mask=mask_a)
    # with tf.variable_scope("poly_ctx", reuse=tf.AUTO_REUSE):
    #     pm_embedding_table = tf.get_variable(
    #         name="poly_embedding",
    #         shape=[poly_m, hidden_dim],
    #         dtype=tf_datatype,
    #         initializer=tf.truncated_normal_initializer(stddev=initializer_range, dtype=tf_datatype))
    #     poly_code_ids = tf.range(poly_m, dtype=tf.int32)
    #     poly_code_ids = tf.tile(tf.expand_dims(poly_code_ids,0),(batch_size,1))
    #     poly_codes = tf.gather(pm_embedding_table, poly_code_ids)
    transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_fastertransformer.so'))
    op_embs = transformer_op_module.attention(poly_codes, op_ctx_out, op_ctx_out,
                batch_size=batch_size,q_seq_len=poly_m,hidden_size=hidden_dim,k_seq_len=seq_len,attention_type=1
              )
    # op_embs, _ = dot_product_attention(poly_codes, op_ctx_out, op_ctx_out, tf.estimator.ModeKeys.PREDICT)

    op_cand_emb = op_encoder(inputs=input_b,
                                   encoder_args=encoder_args,
                                   encoder_vars=encoder_variables,
                                   attention_mask=mask_b)
    # with tf.variable_scope("poly"):
    op_cand_emb = tf.reshape(op_cand_emb, (batch_size * res_cnt, seq_len, hidden_dim))
    op_cand_emb = tf.squeeze(op_cand_emb[:, 0:1, :], axis=1)
    op_cand_emb = tf.reshape(op_cand_emb, (batch_size, res_cnt, hidden_dim))

    op_ctx_emb = transformer_op_module.attention(op_cand_emb, op_embs, op_embs,
                                              batch_size=batch_size, q_seq_len=res_cnt, hidden_size=hidden_dim,
                                              k_seq_len=poly_m,attention_type=1
                                             )
    op_ctx_emb = tf.reshape(op_ctx_emb,(batch_size, res_cnt, hidden_dim))
    # op_ctx_emb, _ = dot_product_attention(op_cand_emb, op_embs, op_embs, tf.estimator.ModeKeys.PREDICT)
    op_dot_product = tf.reduce_sum(op_ctx_emb * op_cand_emb, axis=-1)




    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        tf_encoder_result_val = sess.run(tf_dot_product)
        print(tf_encoder_result_val)
        op_encoder_result_val = sess.run(op_dot_product)
        print(op_encoder_result_val)

        cross_check("Encoder", tf_encoder_result_val,
                    op_encoder_result_val, atol_threshold)

        tf_time = time_test(sess, tf_dot_product, 1000)
        op_time = time_test(sess, op_dot_product, 1000)

        print("tf poly-encoder time:",tf_time)
        print("op poly-encoder time:", op_time)
