#!/usr/bin/env python
# coding:utf-8

import tensorflow as tf
import numpy as np
import argparse
import csv
import utils.tokenization as tokenization
from utils.common import TransformerArgument, time_test, cross_check
from utils.encoder import tf_encoder, op_encoder

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example(text_a, text_b, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""


  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return input_ids, input_mask, segment_ids

def get_input_tensor():
    embedding_table = np.random.normal()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_length', type=int, default=32, metavar='NUMBER',
                        help='sequence length (default: 32)')
    parser.add_argument('--input_file', type=str, default="", metavar='STRING',
                        help='input file path')
    parser.add_argument('--vocab_file', type=str, default="", metavar='STRING',
                        help='vocab file path')
    args = parser.parse_args()

    input_file = args.input_file
    max_seq_length = args.max_seq_length
    tokenizer = tokenization.FullTokenizer(
      vocab_file=args.vocab_file, do_lower_case=True)
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])

            a_input_ids, a_input_mask, a_segment_ids = convert_single_example(text_a,None,max_seq_length=max_seq_length,tokenizer=tokenizer)
            b_input_ids, b_input_mask, b_segment_ids = convert_single_example(text_b, None,
                                                                              max_seq_length=max_seq_length,
                                                                              tokenizer=tokenizer)
