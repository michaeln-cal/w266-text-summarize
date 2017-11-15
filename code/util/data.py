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
                           ("initializer", "article", "dec_input",
                            "dec_target","enc_mask","dec_mask"))):
  pass

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


UNK_ID = 0

# def check_vocab(vocab_file, out_dir, check_special_token=True, pad=None,
#                  unk=None, start_de=None, stop_de=None):
#   """Check if vocab_file doesn't exist, create from corpus_file."""
#   if tf.gfile.Exists(vocab_file):
#     utils.print_out("# Vocab file %s exists" % vocab_file)
#     vocab = []
#     with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
#       vocab_size = 0
#       for word in f:
#         vocab_size += 1
#         vocab.append(word.strip())
#     if check_special_token:
#       # Verify if the vocab starts with unk, sos, eos
#       # If not, prepend those tokens & generate a new vocab file
#       if not unk: unk = UNKNOWN_TOKEN
#       if not pad: pad = PAD_TOKEN
#       if not start_de: start_de=START_DECODING
#       if not stop_de: stop_de=STOP_DECODING
#       assert len(vocab) >= 4
#       if vocab[0] != unk or vocab[1] != pad or vocab[2] != start_de or vocab[3!=stop_de]:
#         utils.print_out("The first 4 vocab words [%s, %s, %s,%s]"
#                         " are not [%s, %s, %s, %s]" %
#                         (vocab[0], vocab[1], vocab[2], vocab[3], unk, pad, start_de, stop_de))
#         vocab = [unk, pad, start_de, stop_de] + vocab
#         vocab_size += 4
#         new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
#         with codecs.getwriter("utf-8")(
#             tf.gfile.GFile(new_vocab_file, "wb")) as f:
#           for word in vocab:
#             f.write("%s\n" % word)
#         vocab_file = new_vocab_file
#   else:
#     raise ValueError("vocab_file '%s' does not exist." % vocab_file)
#
#   vocab_size = len(vocab)
#   return vocab_size, vocab_file
#

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
              print("max size of",max_size," exceeded stop")
              break

  vocab = np.unique(np.array(vocab))
  vocab_table = lookup_ops.index_table_from_tensor(
      tf.convert_to_tensor(vocab), default_value=UNK_ID)

  return vocab_table


  #
  # def write_metadata(self, fpath):
  #   """Writes metadata file for Tensorboard word embedding visualizer as described here:
  #     https://www.tensorflow.org/get_started/embedding_viz
  #
  #   Args:
  #     fpath: place to write the metadata file
  #   """
  #   print "Writing word embedding metadata file to %s..." % (fpath)
  #   with open(fpath, "w") as f:
  #     fieldnames = ['word']
  #     writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
  #     for i in xrange(self.size()):
  #       writer.writerow({"word": self._id_to_word[i]})

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
    # pad_token = tf.cast(vocab_table.lookup(tf.constant(PAD_TOKEN)), tf.int32)
    pad_token =0
    # def process_art_abs(article, abstract_sentences):
    #     # Process the article
    #     article_words = article.split()
    #     if len(article_words) > hps.max_enc_steps:
    #       article_words = article_words[:hps.max_enc_steps]
    #     enc_len = len(article_words) # store the length after truncation but before padding
    #
    #     # Process the abstract
    #     abstract = ' '.join(abstract_sentences) # string
    #     abstract_words = abstract.split() # list of strings
    #     return article_words, enc_len, abstract_words
    #

    dataset = dataset.shard(num_shards, shard_index)



    dataset = dataset.shuffle(output_buffer_size, random_seed)
    dataset = dataset.map(
        lambda article, abstract: (tf.string_split([article]).values,  tf.string_split([abstract]).values),
        num_parallel_calls=num_threads)

    dataset = dataset.map(
        lambda article, abstract: tf.cond(tf.size(abstract) >=tf.constant(hps.max_dec_steps),lambda :
        (article[:hps.max_enc_steps], tf.concat(([tf.constant(START_DECODING)], abstract), 0)[:hps.max_dec_steps],abstract[:hps.max_dec_steps])
                          ,lambda: (article[:hps.max_enc_steps], tf.concat(([tf.constant(START_DECODING)], abstract), 0),tf.concat((abstract,[tf.constant(START_DECODING)]),0))),num_parallel_calls=num_threads)

    # dataset = dataset.map(
    #     lambda article, abstract: (article,  abstract, abstract),
    #     num_parallel_calls=num_threads)


    # dataset = dataset.map(
    #     lambda article, abstract: (article[:hps.max_enc_steps], tf.cond(tf.size(abstract) >=tf.constant(hps.max_dec_steps),(
    #                       tf.concat(([tf.constant(START_DECODING)], abstract), 0)[:hps.max_dec_steps],abstract[:hps.max_dec_steps])
    #                       ,(tf.concat(([tf.constant(START_DECODING)], abstract), 0),tf.concat((abstract,[tf.constant(START_DECODING)]),0)))),num_parallel_calls=num_threads)



    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda article, dec_input, dec_target: (tf.cast(vocab_table.lookup(article), tf.int32),tf.cast(vocab_table.lookup(dec_input), tf.int32),
                          tf.cast(vocab_table.lookup(dec_target), tf.int32)),
        num_parallel_calls=num_threads)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    #
    # # Add in sequence lengths.
    dataset = dataset.map(
        lambda article, dec_input, dec_target: (article,tf.size(article), dec_input, dec_target,tf.ones([tf.size(article)],dtype=tf.float32),
                                                tf.ones([tf.size(dec_target)], dtype=tf.float32),tf.size(dec_input)),
        num_parallel_calls=num_threads)
    #
    # # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            hps.batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([hps.max_enc_steps]),  # article
                tf.TensorShape([]),  # article_len

                tf.TensorShape([hps.max_dec_steps]),  # dec_input
                tf.TensorShape([hps.max_dec_steps]),  # dec_target
                tf.TensorShape([hps.max_enc_steps]),  # mask for article
                tf.TensorShape([hps.max_dec_steps]),  # mask for encoder
                tf.TensorShape([])),  # dec_input_len  # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                pad_token,  # src
                0,
                pad_token,  # tgt_input
                pad_token,  # tgt_output
                0.0,

                0.0,
                0))  # tgt_len -- unused

    # if num_buckets > 1:
    #
    #     def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
    #         # Calculate bucket_width by maximum source sequence length.
    #         # Pairs with length [0, bucket_width) go to bucket 0, length
    #         # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
    #         # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
    #         if max_enc_steps:
    #             bucket_width = (src_max_len + num_buckets - 1) // num_buckets
    #         else:
    #             bucket_width = 10
    #
    #         # Bucket sentence pairs by the length of their source sentence and target
    #         # sentence.
    #         bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
    #         return tf.to_int64(tf.minimum(num_buckets, bucket_id))
    #
    #     def reduce_func(unused_key, windowed_data):
    #         return batching_func(windowed_data)
    #
    #     batched_dataset = dataset.apply(
    #         tf.contrib.data.group_by_window(
    #             key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    #
    # else:
    batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (article, art_size,dec_input, dec_target, enc_mask, dec_mask, dec_input_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        article=(article, art_size),
        dec_input=(dec_input,dec_input_len),
        dec_target=dec_target,
        enc_mask = enc_mask,
        dec_mask=dec_mask
        )

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

