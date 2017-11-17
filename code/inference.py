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

"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
import util.data as data
import util.util as util
from tensorflow.python.ops import lookup_ops

import os
import time

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint



class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, vocab,vocab_table):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()

    self._vocab = vocab
    self._vocab_table =vocab_table
    train_dir = os.path.join(FLAGS.log_root, "train")
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding

    ckpt_state = tf.train.get_checkpoint_state(train_dir)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    sess = tf.Session(config=util.get_config())
    self._sess = sess
    self._saver.restore(sess, ckpt_state.model_checkpoint_path)
    self._sess.run([self._model.init_iter])
    sess.run(tf.tables_initializer())

  FLAGS = tf.app.flags.FLAGS

  def decode(self):
      """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
      t0 = time.time()
      counter = 0
      while True:

            # Run beam search to get best Hypothesis
            best_hyp = run_beam_search(self._sess, self._model, self._vocab)


          # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
          # decoded_words = data.outputids2words(output_ids, self._vocab,
          #                                      (batch.art_oovs[0] if FLAGS.pointer_gen else None))

            decoded_words = tf.contrib.lookup.index_to_string(output_ids, self._vocab_table,default_value='UNK')
            print("the output is", decoded_words)


          # # Remove the [STOP] token from decoded_words, if necessary
          # try:
          #     fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
          #     decoded_words = decoded_words[:fst_stop_idx]
          # except ValueError:
          #     decoded_words = decoded_words
          # decoded_output = ' '.join(decoded_words)  # single string


          # # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
          # t1 = time.time()
          # if t1 - t0 > SECS_UNTIL_NEW_CKPT:
          #     tf.logging.info(
          #         'We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint',
          #         t1 - t0)
          #     _ = util.load_ckpt(self._saver, self._sess)
          #     t0 = time.time()


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
      """Hypothesis constructor.

      Args:
        tokens: List of integers. The ids of the tokens that form the summary so far.
        log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
        state: Current state of the decoder, a LSTMStateTuple.
        attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
        p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
        coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
      """
      self.tokens = tokens
      self.log_probs = log_probs
      self.state = state
      self.attn_dists = attn_dists
      self.p_gens = p_gens
      self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
      """Return a NEW hypothesis, extended with the information from the latest step of beam search.

      Args:
        token: Integer. Latest token produced by beam search.
        log_prob: Float. Log prob of the latest token.
        state: Current decoder state, a LSTMStateTuple.
        attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
        p_gen: Generation probability on latest step. Float.
        coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
      Returns:
        New Hypothesis for next step.
      """
      return Hypothesis(tokens=self.tokens + [token],
                        log_probs=self.log_probs + [log_prob],
                        state=state,
                        attn_dists=self.attn_dists + [attn_dist],
                        p_gens=self.p_gens + [p_gen],
                        coverage=coverage)
    @property
    def latest_token(self):
      return self.tokens[-1]

    @property
    def log_prob(self):
      # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
      return sum(self.log_probs)

    @property
    def avg_log_prob(self):
      # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
      return self.log_prob / len(self.tokens)

def run_beam_search(sess, model, vocab):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  enc_states, dec_in_state = model.run_encoder(sess)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[model.start_decoding],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([model.iterator.article[0].shape[1]])  # zero vector of length attention_length
                     ) for _ in range(FLAGS.beam_size)]
  results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)
  print("hello 1")
  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
      print("hello 2")

      latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
      latest_tokens = [t if t in range(vocab[1]) else data.unk_token for t in
                       latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
      states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
      prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)



      # Run one step of the decoder to get the new info
      (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                      latest_tokens=latest_tokens,
                                                                                                      enc_states=enc_states,
                                                                                                      dec_init_states=states,
                                                                                                      prev_coverage=prev_coverage)


      # Extend each hypothesis and collect them all in all_hyps
      all_hyps = []
      num_orig_hyps = 1 if steps == 0 else len(
          hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
      for i in range(num_orig_hyps):
          h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                           new_coverage[
                                                               i]  # take the ith hypothesis and new decoder state info
          for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
              # Extend the ith hypothesis with the jth option
              new_hyp = h.extend(token=topk_ids[i, j],
                                 log_prob=topk_log_probs[i, j],
                                 state=new_state,
                                 attn_dist=attn_dist,
                                 p_gen=p_gen,
                                 coverage=new_coverage_i)
              all_hyps.append(new_hyp)

      # Filter and collect any hypotheses that have produced the end token.
      hyps = []  # will contain hypotheses for the next step
      for h in sort_hyps(all_hyps):  # in order of most likely h
          if h.latest_token == data.stop_decoding:  # if stop token is reached...
              # If this hypothesis is sufficiently long, put in results. Otherwise discard.
              if steps >= FLAGS.min_dec_steps:
                  results.append(h)
          else:  # hasn't reached stop token, so continue to extend this hypothesis
              hyps.append(h)
          if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
              # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
              break

      steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  print("result is", results)
  if len(results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
   results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

     # Return the hypothesis with highest average log prob
  return hyps_sorted[0]


def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)



  #
  #
  # def beam_search(self):
  #   # enc_states, dec_in_state = self._model.run_encoder(self._sess, self._iterator)
  #   to_return = {
  #     "ids": self._model._topk_ids,
  #     "probs": self._model._topk_log_probs,
  #     "states": self._model._dec_out_state,
  #     "attn_dists": self._model.attn_dists
  #   }
  #   results =self._sess.run(to_return)
  #   print("hello", results["ids"])
  #   return results
  #
  #
  # def get_decode_dir_name(ckpt_name):
  #   """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""
  #
  #   if "train" in FLAGS.data_path: dataset = "train"
  #   elif "val" in FLAGS.data_path: dataset = "val"
  #   elif "test" in FLAGS.data_path: dataset = "test"
  #   else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  #   dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  #   if ckpt_name is not None:
  #       dirname += "_%s" % ckpt_name
  #   return dirname
  #
  #
  #

