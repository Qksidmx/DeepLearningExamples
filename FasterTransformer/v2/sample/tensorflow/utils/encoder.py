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


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    tf_datatype=tf.float32):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    # [B*F, N*H]
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")
    # [B*F, N*H]
    from_tensor_2d = reshape_to_matrix(from_tensor)
    # [B*T, N*H]
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # attention_scores = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    # apply attention_mask(with broadcast)
    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        adder = (1.0 - tf.cast(attention_mask, tf_datatype)) * -10000.0

        attention_scores += adder

    # `attention_probs = [B, N, F, T]`
    attention_probs = tf.nn.softmax(attention_scores)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def tf_encoder(input_tensor,
               encoder_args,
               attention_mask=None,
               intermediate_act_fn=gelu,
               initializer_range=0.02):
    intermediate_size = encoder_args.hidden_dim * 4
    if encoder_args.hidden_dim % encoder_args.head_num != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (encoder_args.hidden_dim, encoder_args.head_num))

    attention_head_size = int(encoder_args.hidden_dim / encoder_args.head_num)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    prev_output = reshape_to_matrix(input_tensor)

    for layer_idx in range(encoder_args.num_layer):
        with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=encoder_args.head_num,
                        size_per_head=encoder_args.size_per_head,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=encoder_args.batch_size,
                        from_seq_length=encoder_args.max_seq_len,
                        to_seq_length=encoder_args.max_seq_len,
                        tf_datatype=encoder_args.dtype)
                    # `attention_output` = [B*F, N*H]
                    attention_output = attention_head

                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        encoder_args.hidden_dim,
                        use_bias=True,
                        bias_initializer=create_initializer(
                            initializer_range, encoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    use_bias=True,
                    bias_initializer=create_initializer(
                        initializer_range, encoder_args.dtype),
                    kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    encoder_args.hidden_dim,
                    use_bias=True,
                    bias_initializer=create_initializer(
                        initializer_range, encoder_args.dtype),
                    kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output

    return prev_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """
    Get tensor shape, prefer static shape.
    If expected_rank is not none, check rank also.

    :param tensor:
    :param expected_rank: integer or list of integers
    :param name:
    :return: Shape list
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def op_encoder(inputs,
               encoder_args,
               encoder_vars,
               attention_mask):
    transformer_op_module = tf.load_op_library(
        os.path.join('./lib/libtf_fastertransformer.so'))
    for layer_idx in range(encoder_args.num_layer):
        val_off = layer_idx * 16
        outputs = transformer_op_module.bert_transformer(
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
    transformer_op_module = tf.load_op_library(
        os.path.join('./lib/libtf_fastertransformer.so'))
    for layer_idx in range(encoder_args.num_layer):
        val_off = layer_idx * 16
        outputs = transformer_op_module.open_nmt_transformer(
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


@six.add_metaclass(abc.ABCMeta)
class Encoder(tf.keras.layers.Layer):
    """Base class for encoders."""

    def build_mask(self, inputs, sequence_length=None, dtype=tf.bool):
        """Builds a boolean mask for :obj:`inputs`."""
        if sequence_length is None:
            return None
        return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1], dtype=dtype)

    def call(self, inputs, sequence_length=None, training=None):  # pylint: disable=arguments-differ
        """Encodes an input sequence.

        Args:
          inputs: The inputs to encode of shape :math:`[B, T, ...]`.
          sequence_length: The length of each input with shape :math:`[B]`.
          training: Run in training mode.

        Returns:
          A tuple ``(outputs, state, sequence_length)``.
        """
        return self.encode(
            inputs,
            sequence_length=sequence_length,
            mode=tf.estimator.ModeKeys.TRAIN if training else None)

    def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
        """Encodes an input sequence.

        Args:
          inputs: The inputs to encode of shape :math:`[B, T, ...]`.
          sequence_length: The length of each input with shape :math:`[B]`.
          mode: A ``tf.estimator.ModeKeys`` mode.

        Returns:
          A tuple ``(outputs, state, sequence_length)``.
        """
        return self.call(
            inputs,
            sequence_length=sequence_length,
            training=mode == tf.estimator.ModeKeys.TRAIN)


@six.add_metaclass(abc.ABCMeta)
class PositionEncoder(tf.keras.layers.Layer):
    """Base class for position encoders."""

    def __init__(self, reducer=tf.add_n):
        super(PositionEncoder, self).__init__()
        self.reducer = reducer

    def __call__(self, inputs, sequence_length=None, position=None):  # pylint: disable=arguments-differ
        """Apply position encoding to inputs.

        Args:
          inputs: The inputs of shape :math:`[B, T, D]`.
          sequence_length: The length of each sequence of shape :math:`[B]`.
            If ``None``, sequences are assumed to have the same length.
          position: If known, the position to encode (1-indexed).

        Returns:
          A ``tf.Tensor`` of shape :math:`[B, T, D]` where :math:`D` depends on the
          :attr:`reducer`.
        """
        self._dtype = inputs.dtype
        # Build by default for backward compatibility.
        if not tf.get_variable_scope().reuse:
            self.build(inputs.shape)
        return self.call(
            inputs, sequence_length=sequence_length, position=position)

    def call(self, inputs, sequence_length=None, position=None):  # pylint: disable=arguments-differ
        _ = sequence_length

        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        input_dim = inputs.get_shape().as_list()[-1]

        if position is None:
            positions = tf.range(timesteps) + 1
        else:
            positions = [position]
        position_encoding = self.encode([positions], input_dim, dtype=inputs.dtype)
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
        return self.reducer([inputs, position_encoding])

    def apply(self, inputs, sequence_length=None):  # pylint: disable=arguments-differ
        """Shortcut for ``__call__``."""
        return self(inputs, sequence_length=sequence_length)

    def apply_one(self, inputs, position):
        """Shortcut for ``__call__``."""
        return self(inputs, position=position)

    @abc.abstractmethod
    def encode(self, positions, depth, dtype=tf.float32):
        """Creates position encodings.

        Args:
          position: The positions to encode of shape :math:`[B, ...]`.
          depth: The encoding depth :math:`D`.
          dtype: The encoding type.

        Returns:
          A ``tf.Tensor`` of shape :math:`[B, ..., D]`.
        """
        raise NotImplementedError()


class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes positions with sine waves as described in
    https://arxiv.org/abs/1706.03762.
    """

    def encode(self, positions, depth, dtype=tf.float32):
        if depth % 2 != 0:
            raise ValueError("SinusoidalPositionEncoder expects the depth to be divisble "
                             "by 2 but got %d" % depth)

        batch_size = tf.shape(positions)[0]
        positions = tf.cast(positions, tf.float32)

        log_timescale_increment = math.log(10000) / (depth / 2 - 1)
        inv_timescales = tf.exp(tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment)
        inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
        scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
        encoding = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        return tf.cast(encoding, dtype)


class SelfAttentionEncoder(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder()):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(SelfAttentionEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder

    def preprocess(self, inputs, mode=tf.estimator.ModeKeys.TRAIN):
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)
        inputs = tf.layers.dropout(
            inputs,
            rate=self.dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        return inputs

    def encode(self, inputs, mask=None, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
        if mask is None:
            mask = build_sequence_mask(
                sequence_length,
                num_heads=self.num_heads,
                maximum_length=tf.shape(inputs)[1])

        state = ()

        for l in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = multi_head_attention(
                        self.num_heads,
                        norm(inputs),
                        None,
                        mode,
                        num_units=self.num_units,
                        mask=mask,
                        dropout=self.attention_dropout)
                    context = drop_and_add(
                        inputs,
                        context,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope("ffn"):
                    transformed = feed_forward(
                        norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = drop_and_add(
                        context,
                        transformed,
                        mode,
                        dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1),)

        outputs = norm(inputs)
        return (outputs, state, sequence_length)


def build_sequence_mask(sequence_length,
                        num_heads=None,
                        maximum_length=None,
                        dtype=tf.float32):
    """Builds the dot product mask.

    Args:
      sequence_length: The sequence length.
      num_heads: The number of heads.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The type of the mask tensor.

    Returns:
      A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[batch_size, 1, 1, max_length]``.
    """
    mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
    mask = tf.expand_dims(mask, axis=1)
    if num_heads is not None:
        mask = tf.expand_dims(mask, axis=1)
    return mask


def norm(inputs):
    """Layer normalizes :obj:`inputs`."""
    return tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)


def feed_forward(x, inner_dim, mode, dropout=0.0):
    """Implements the Transformer's "Feed Forward" layer.

    .. math::

        ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2

    Args:
      x: The input.
      inner_dim: The number of units of the inner linear transformation.
      mode: A ``tf.estimator.ModeKeys`` mode.
      dropout: The probability to drop units from the inner transformation.

    Returns:
      The transformed input.
    """
    input_dim = x.get_shape().as_list()[-1]

    # modified
    # inner = tf.layers.conv1d(x, inner_dim, 1, activation=tf.nn.relu)
    inner = tf.layers.dense(x, inner_dim, activation=tf.nn.relu)
    inner = tf.layers.dropout(
        inner,
        rate=dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    # modified
    # outer = tf.layers.conv1d(inner, input_dim, 1)
    outer = tf.layers.dense(inner, input_dim)

    return outer


def drop_and_add(inputs,
                 outputs,
                 mode,
                 dropout=0.1):
    """Drops units in the outputs and adds the previous values.

    Args:
      inputs: The input of the previous layer.
      outputs: The output of the previous layer.
      mode: A ``tf.estimator.ModeKeys`` mode.
      dropout: The probability to drop units in :obj:`outputs`.

    Returns:
      The residual and normalized output.
    """
    outputs = tf.layers.dropout(
        outputs,
        rate=dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    input_dim = inputs.get_shape().as_list()[-1]
    output_dim = outputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        outputs += inputs
    return outputs


def multi_head_attention(num_heads,
                         queries,
                         memory,
                         mode,
                         num_units=None,
                         mask=None,
                         cache=None,
                         dropout=0.0,
                         return_attention=False):
    """Computes the multi-head attention as described in
    https://arxiv.org/abs/1706.03762.

    Args:
      num_heads: The number of attention heads.
      queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      mode: A ``tf.estimator.ModeKeys`` mode.
      num_units: The number of hidden units. If not set, it is set to the input
        dimension.
      mask: A ``tf.Tensor`` applied to the dot product.
      cache: A dictionary containing pre-projected keys and values.
      dropout: The probability to drop units from the inputs.
      return_attention: Return the attention head probabilities in addition to the
        context.

    Returns:
      The concatenated attention context of each head and the attention
      probabilities (if :obj:`return_attention` is set).
    """
    num_units = num_units or queries.get_shape().as_list()[-1]

    if num_units % num_heads != 0:
        raise ValueError("Multi head attention requires that num_units is a"
                         " multiple of {}".format(num_heads))

    if memory is None:
        queries, keys, values = nofused_projection(queries, num_units, num_outputs=3)

        keys = split_heads(keys, num_heads)
        values = split_heads(values, num_heads)

        if cache is not None:
            keys = tf.concat([cache["self_keys"], keys], axis=2)
            values = tf.concat([cache["self_values"], values], axis=2)
            cache["self_keys"] = keys
            cache["self_values"] = values
    else:
        queries = tf.layers.conv1d(queries, num_units, 1)

        if cache is not None:
            def _project_and_split():
                k, v = nofused_projection(memory, num_units, num_outputs=2)
                return split_heads(k, num_heads), split_heads(v, num_heads)

            keys, values = tf.cond(
                tf.equal(tf.shape(cache["memory_keys"])[2], 0),
                true_fn=_project_and_split,
                false_fn=lambda: (cache["memory_keys"], cache["memory_values"]))
            cache["memory_keys"] = keys
            cache["memory_values"] = values
        else:
            keys, values = nofused_projection(memory, num_units, num_outputs=2)
            keys = split_heads(keys, num_heads)
            values = split_heads(values, num_heads)

    queries = split_heads(queries, num_heads)
    queries *= (num_units // num_heads) ** -0.5

    heads, attn = dot_product_attention(
        queries,
        keys,
        values,
        mode,
        mask=mask,
        dropout=dropout)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    # outputs = tf.layers.conv1d(combined, num_units, 1)
    outputs = tf.layers.dense(combined, num_units)

    if not return_attention:
        return outputs
    return outputs, attn


def nofused_projection(inputs, num_units, num_outputs=1):
    """Projects the same input into multiple output spaces.

    Args:
      inputs: The inputs to project.
      num_units: The number of output units of each space.
      num_outputs: The number of output spaces.

    Returns:
      :obj:`num_outputs` ``tf.Tensor`` of depth :obj:`num_units`.
    """

    # modified
    # return tf.split(
    # tf.layers.conv1d(inputs, num_units * num_outputs, 1), num_outputs, axis=2)
    # `query_layer` = [B, F, N*H]
    query_layer = tf.layers.dense(
        inputs,
        num_units,
        name="query",
        # kernel_initializer=,
    )

    # `key_layer` = [B, T, N*H]
    key_layer = tf.layers.dense(
        inputs,
        num_units,
        name="key",
        use_bias=True,
        # kernel_initializer,
    )

    # `value_layer` = [B, T, N*H]
    value_layer = tf.layers.dense(
        inputs,
        num_units,
        name="value",
        # kernel_initializer=
    )
    return query_layer, key_layer, value_layer


def combine_heads(inputs):
    """Concatenates heads.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
    """
    static_shape = inputs.get_shape().as_list()
    depth = static_shape[-1]
    num_heads = static_shape[1]
    outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
    outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], depth * num_heads])
    return outputs


def split_heads(inputs, num_heads):
    """Splits a tensor in depth.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
      num_heads: The number of heads :math:`H`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
    """
    static_shape = inputs.get_shape().as_list()
    depth = static_shape[-1]
    outputs = tf.reshape(
        inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_heads, depth // num_heads])
    outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
    return outputs


def dot_product_attention(queries,
                          keys,
                          values,
                          mode,
                          mask=None,
                          dropout=0.0):
    """Computes the dot product attention.

    Args:
      queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      keys: The sequence use to calculate attention scores. A tensor of shape
        :math:`[B, T_2, ...]`.
      values: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
      mode: A ``tf.estimator.ModeKeys`` mode.
      mask: A ``tf.Tensor`` applied to the dot product.
      dropout: The probability to drop units from the inputs.

    Returns:
      A tuple ``(context vector, attention vector)``.
    """
    # Dot product between queries and keys.
    dot = tf.matmul(queries, keys, transpose_b=True)

    if mask is not None:
        dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)

    # Compute attention weights.
    attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
    drop_attn = tf.layers.dropout(
        attn,
        rate=dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute attention context.
    context = tf.matmul(drop_attn, values)

    return context, attn
