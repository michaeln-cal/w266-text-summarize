# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import collections
import numpy as np

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================


import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops



class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


UNK_ID = 0


def create_vocab_tables(vocab_file, max_size):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  vocab =[]
  count =0
  with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
          pieces = line.split()
          vocab.append(pieces[0])
          count+=1
          if(max_size>0 and count>=max_size):
              # print("max size of",max_size," exceeded stop")
              break

  vocab = np.unique(np.array(vocab))
  vocab_table = lookup_ops.index_table_from_tensor(
      tf.convert_to_tensor(vocab), default_value=UNK_ID)
  return vocab_table

def create_id_tables(vocab_file, max_size):
      """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
      vocab = []
      count = 0
      with open(vocab_file, 'r') as vocab_f:
          for line in vocab_f:
              pieces = line.split()
              vocab.append(pieces[0])
              count += 1
              if (max_size > 0 and count >= max_size):
                  print("max size of", max_size, " exceeded stop")
                  break

      vocab = np.unique(np.array(vocab))
      vocab_table = lookup_ops.index_to_string_table_from_tensor(
          tf.convert_to_tensor(vocab))
      return vocab_table


# #vocab_file ="/Users/giang/Downloads/finished_files/vocab_copy"
# vocab_file = FLAGS.vocab_path
#
# vocab_table = create_vocab_tables(vocab_file, 50000)
# vocab_index= create_id_tables(vocab_file,50000)
#
# start_decoding = tf.cast(vocab_table.lookup(tf.constant(START_DECODING)), tf.int32)
# stop_decoding = tf.cast(vocab_table.lookup(tf.constant(STOP_DECODING)), tf.int32)
# unk_token = tf.cast(vocab_table.lookup(tf.constant(UNKNOWN_TOKEN)), tf.int32)


def get_iterator(dataset,
                 vocab_table,
                 hps,
                 random_seed=111,
                 num_threads=4,
                 output_buffer_size=None,
                 num_shards=1,
                 shard_index=0):
    if not output_buffer_size:
        output_buffer_size = hps.batch_size * 1000

    start_decoding = tf.cast(vocab_table.lookup(tf.constant(START_DECODING)), tf.int32)
    stop_decoding = tf.cast(vocab_table.lookup(tf.constant(STOP_DECODING)), tf.int32)


    dataset = dataset.shard(num_shards, shard_index)



    dataset = dataset.shuffle(output_buffer_size, random_seed)
    dataset = dataset.map(
        lambda article, abstract: (tf.string_split([article]).values,  tf.string_split([abstract]).values),
        num_parallel_calls=num_threads)

    dataset = dataset.map(
            lambda src, tgt: (src[:hps.max_enc_steps], tgt),
        num_parallel_calls=num_threads)
    dataset = dataset.map(
            lambda src, tgt: (src, tgt[:hps.max_dec_steps]),
        num_parallel_calls=num_threads)

    dataset = dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    #

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    dataset = dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([start_decoding], tgt), 0),
                          tf.concat((tgt, [stop_decoding]), 0)),
        num_parallel_calls=num_threads)
    # Add in sequence lengths.
    dataset = dataset.map(
        lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_threads)

    def batching_func(x):
        return x.padded_batch(
            hps.batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_input
                tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                stop_decoding,  # src
                stop_decoding,  # tgt_input
                stop_decoding,  # tgt_output
                0,  # src_len -- unused
                0))  # tgt_len -- unused


    batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
     tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)

def abstract2sents(abstract):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences

      return  ' '.join(sents)  # string

