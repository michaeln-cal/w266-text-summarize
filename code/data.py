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

import collections
import numpy as np
import random

UNK_ID=0
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


import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops



class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


# def create_vocab_tables(vocab_file, max_size, unk_id):
#   """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
#   vocab =[]
#   count =0
#   with open(vocab_file, 'r') as vocab_f:
#       for line in vocab_f:
#           pieces = line.split()
#           vocab.append(pieces[0])
#           count+=1
#           if(max_size>0 and count>=max_size):
#               # print("max size of",max_size," exceeded stop")
#               break
#
#   vocab = np.unique(np.array(vocab))
#   vocab_table = lookup_ops.index_table_from_tensor(
#       tf.convert_to_tensor(vocab), default_value=unk_id)
#   return vocab_table

def create_vocab_tables(vocab_file,max_size, unk_id):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value=unk_id)
  return vocab_table

# def create_id_tables(vocab_file, max_size):
#       """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
#       vocab = []
#       count = 0
#       with open(vocab_file, 'r') as vocab_f:
#           for line in vocab_f:
#               pieces = line.split()
#               vocab.append(pieces[0])
#               count += 1
#               if (max_size > 0 and count >= max_size):
#                   break
#
#       vocab = np.unique(np.array(vocab))
#       vocab_table = lookup_ops.index_to_string_table_from_tensor(
#           tf.convert_to_tensor(vocab))
#       return vocab_table

def create_id_tables(vocab_file, max_size):
    return lookup_ops.index_to_string_table_from_file(vocab_file, default_value = "<unk>")


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       source_reverse,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  if source_reverse:
    src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)

def get_iterator(dataset,
                 vocab_table,
                 hps,
                 random_seed=111,
                 num_threads=4,
                 output_buffer_size=None,
                 num_shards=1,
                 shard_index=0):
    if not output_buffer_size:
        output_buffer_size = hps.batch_size * 10

    src_eos_id = tf.cast(vocab_table.lookup(tf.constant(hps.eos)), tf.int32)
    tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(hps.sos)), tf.int32)
    tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(hps.eos)), tf.int32)


    dataset = dataset.shard(num_shards, shard_index)



    dataset = dataset.shuffle(output_buffer_size, random_seed)
    dataset = dataset.map(
        lambda article, abstract: (tf.string_split([article]).values,  tf.string_split([abstract]).values),
        num_parallel_calls=num_threads)

    dataset = dataset.map(
            lambda src, tgt: (src[:hps.src_max_len], tgt),
        num_parallel_calls=num_threads)
    dataset = dataset.map(
            lambda src, tgt: (src, tgt[:hps.tgt_max_len]),
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
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
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
                tf.TensorShape([400]),  # src
                tf.TensorShape([150]),  # tgt_input
                tf.TensorShape([150]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                tgt_eos_id,  # tgt_input
                tgt_eos_id,  # tgt_output
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





def get_dataset(data_dir, prefix, random_decode=None):
  article_filenames,abstract_filenames = get_files(data_dir,prefix)

  article_data_set = tf.data.TextLineDataset(article_filenames)
  abstract_data_set = tf.data.TextLineDataset(abstract_filenames)
  dataset = tf.data.Dataset.zip((article_data_set,abstract_data_set))
  if random_decode:
      """if this is a random sampling process during training"""
      decode_id = random.randint(0, len(abstract_filenames) - 1)
      dataset =dataset[decode_id]

  return dataset

def get_files(data_dir, prefix):
  article_filenames = []
  abstract_filenames = []
  art_dir = data_dir + '/article'
  abs_dir =data_dir+'/abstract'
  for file in os.listdir(art_dir):
    if file.startswith(prefix):
        article_filenames.append(art_dir + "/" + file)
  for file in os.listdir(abs_dir):
    if file.startswith(prefix):
        abstract_filenames.append(abs_dir + "/" + file)



  return article_filenames,abstract_filenames
