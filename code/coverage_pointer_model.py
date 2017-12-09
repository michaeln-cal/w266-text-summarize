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
"""Attention-based sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import core as layers_core

from util import misc_utils as utils
import numpy as np
import time

import model
import model_helper



__all__ = ["CoveragePointerModel"]


class CoveragePointerModel(model.Model):
  """Sequence-to-sequence dynamic model with attention.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
  This class also allows to use GRU cells in addition to LSTM cells with
  support for dropout.
  """

  def __init__(self,
               hps,
               mode,
               iterator,
               vocab_table,
               reverse_target_vocab_table=None,
               scope=None):
    # Set attention_mechanism_fn

    self.rand_unif_init = tf.random_uniform_initializer(-0.02, 0.02, seed=123)
    self.trunc_norm_init = tf.truncated_normal_initializer(stddev=1e-4)
    self.vocab_size = hps.vocab_size
    # self.global_step = tf.Variable(0, trainable=False)





    super(CoveragePointerModel, self).__init__(
        hps=hps,
        mode=mode,
        iterator=iterator,
        vocab_table=vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope)

  def attention_decoder(self,hps, decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                        initial_state_attention=False, pointer_gen=True, use_coverage=True, prev_coverage=None, scope=None):
      """
      Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
        enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
        cell: rnn_cell.RNNCell defining the cell function and size.
        initial_state_attention:
          Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
        pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
        use_coverage: boolean. If True, use coverage mechanism.
        prev_coverage:
          If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

      Returns:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x cell.output_size]. The output vectors.
        state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
        attn_dists: A list containing tensors of shape (batch_size,attn_length).
          The attention distributions for each decoder step.
        p_gens: List of length input_size, containing tensors of shape [batch_size, 1]. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
        coverage: Coverage vector on the last step computed. None if use_coverage=False.
      """

      with tf.variable_scope(scope or "Attention_Decoder"):
          batch_size = hps.batch_size  # if this line fails, it's because the batch size isn't defined
          attn_size = encoder_states.get_shape()[
              2].value  # if this line fails, it's because the attention length isn't defined

          # Reshape encoder_states (need to insert a dim)
          encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

          # To calculate attention, we calculate
          #   v^T tanh(W_h h_i + W_s s_t + b_attn)
          # where h_i is an encoder state, and s_t a decoder state.
          # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
          # We set it to be equal to the size of the encoder states.
          attention_vec_size = attn_size

          # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
          W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
          encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                           "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

          # Get the weight vectors v and w_c (w_c is for coverage)
          v = tf.get_variable("v", [attention_vec_size])
          if use_coverage:
              with tf.variable_scope("coverage"):
                  w_c = tf.get_variable("w_c", [1, 1, 1, attention_vec_size])

          if prev_coverage is not None:  # for beam search mode with coverage
              # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
              prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

          def attention(decoder_state, coverage=None):
              """Calculate the context vector and attention distribution from the decoder state.

              Args:
                decoder_state: state of the decoder
                coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

              Returns:
                context_vector: weighted sum of encoder_states
                attn_dist: attention distribution
                coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
              """
              with tf.variable_scope("Attention"):
                  # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                  decoder_features = self.linear(decoder_state, attention_vec_size,
                                            True)  # shape (batch_size, attention_vec_size)
                  decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                                    1)  # reshape to (batch_size, 1, 1, attention_vec_size)

                  def masked_attention(e):
                      """Take softmax of e then apply enc_padding_mask and re-normalize"""
                      attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                      attn_dist *= enc_padding_mask  # apply mask
                      masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                      return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                  if use_coverage and coverage is not None:  # non-first step of coverage
                      # Multiply coverage vector by w_c to get coverage_features.
                      coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1],
                                                        "SAME")  # c has shape (batch_size, attn_length, 1, attention_vec_size)

                      # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                      e = math_ops.reduce_sum(
                          v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                          [2, 3])  # shape (batch_size,attn_length)

                      # Calculate attention distribution
                      attn_dist = masked_attention(e)

                      # Update coverage vector
                      coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                  else:
                      # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                      e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                              [2, 3])  # calculate e

                      # Calculate attention distribution
                      attn_dist = masked_attention(e)

                      if use_coverage:  # first step of training
                          coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

                  # Calculate the context vector from attn_dist and encoder_states
                  context_vector = math_ops.reduce_sum(
                      array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                      [1, 2])  # shape (batch_size, attn_size).
                  context_vector = array_ops.reshape(context_vector, [-1, attn_size])

              return context_vector, attn_dist, coverage

          outputs = []
          attn_dists = []
          p_gens = []
          state = initial_state
          coverage = prev_coverage  # initialize coverage to None or whatever was passed in
          context_vector = array_ops.zeros([batch_size, attn_size])
          context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
          if initial_state_attention:  # true in decode mode
              # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
              context_vector, _, coverage = attention(initial_state,
                                                      coverage)  # in decode mode, this is what updates the coverage vector
          for i, inp in enumerate(decoder_inputs):
              tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
              if i > 0:
                  tf.get_variable_scope().reuse_variables()

              # Merge input and previous attentions into one vector x of the same size as inp
              input_size = inp.get_shape().with_rank(2)[1]
              if input_size.value is None:
                  raise ValueError("Could not infer input size from input: %s" % inp.name)
              x = self.linear([inp] + [context_vector], input_size, True)

              # Run the decoder RNN cell. cell_output = decoder state
              # print("before here", i, x, state)
              cell_output, state = cell(x, state)

              # Run the attention mechanism.
              if i == 0 and initial_state_attention:  # always true in decode mode
                  with tf.variable_scope(tf.get_variable_scope(),
                                                     reuse=True):  # you need this because you've already run the initial attention(...) call
                      context_vector, attn_dist, _ = attention(state, coverage)  # don't allow coverage to update
              else:
                  context_vector, attn_dist, coverage = attention(state, coverage)
              attn_dists.append(attn_dist)

              # Calculate p_gen
              if pointer_gen:
                  with tf.variable_scope('calculate_pgen'):
                      p_gen = self.linear([context_vector, state.c, state.h, x], 1, True)  # Tensor shape (batch_size, 1)
                      p_gen = tf.sigmoid(p_gen)
                      p_gens.append(p_gen)

              # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
              # This is V[s_t, h*_t] + b in the paper
              with tf.variable_scope("AttnOutputProjection"):
                  output = self.linear([cell_output] + [context_vector], cell.output_size, True)
              outputs.append(output)

          # If using coverage, reshape it
          if coverage is not None:
              coverage = array_ops.reshape(coverage, [batch_size, -1])

          return outputs, state, attn_dists, p_gens, coverage

  def linear(self,args, output_size, bias, bias_start=0.0, scope=None):
      """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

      Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.

      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      """
      if args is None or (isinstance(args, (list, tuple)) and not args):
          raise ValueError("`args` must be specified")
      if not isinstance(args, (list, tuple)):
          args = [args]

      # Calculate the total size of arguments on dimension 1.
      total_arg_size = 0
      shapes = [a.get_shape().as_list() for a in args]
      for shape in shapes:
          if len(shape) != 2:
              raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
          if not shape[1]:
              raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
          else:
              total_arg_size += shape[1]

      # Now the computation.
      with tf.variable_scope(scope or "Linear"):
          matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
          if len(args) == 1:
              res = tf.matmul(args[0], matrix)
          else:
              res = tf.matmul(tf.concat(axis=1, values=args), matrix)
          if not bias:
              return res
          bias_term = tf.get_variable(
              "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
      return res + bias_term

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self.hps.num_units
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state



  def _build_decoder(self, encoder_outputs, encoder_state, hps):
      """Build and run a RNN decoder with a final projection layer.

        Args:
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          hps: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """

      tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.sos)),
                             tf.int32)
      tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.eos)),
                             tf.int32)

      iterator = self.iterator

      with tf.variable_scope('decoder') as decoder_scope:
        sample_id, final_context_state,coverage_loss= tf.no_op(), tf.no_op(),tf.no_op()

        dec_in_state = self._reduce_states(encoder_state[0], encoder_state[1])

        cell, decoder_initial_state = self._build_decoder_cell(hps, None, dec_in_state, None)
        # target_output = iterator.target_output
        # target_input = iterator.target_input
        source_sequence_length = iterator.source_sequence_length
        # target_sequence_length = iterator.target_sequence_length

        source = iterator.source


        if self.time_major:
            print("it's time major")

            source = tf.transpose(source)
            source_sequence_length = tf.transpose(source_sequence_length)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            target_input = iterator.target_input
            target_output = iterator.target_output
            target_sequence_length = iterator.target_sequence_length

            if self.time_major:
                utils.print_out("time major")
                target_input = tf.transpose(target_input)
                target_output = tf.transpose(target_output)
                target_sequence_length = tf.transpose(target_sequence_length)

            target_max_time = self.get_max_time(target_output)
            dec_target_mask = tf.sequence_mask(target_sequence_length, target_max_time, dtype=tf.float32)
            self._target_batch = target_output


            emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding_decoder, x) for x in tf.unstack(target_input,
                                                                                       axis=1, num =150)]  # list length max_dec_steps containing shape (batch_size, emb_size)



            source_max_time = self.get_max_time(source)

            en_src_mask = tf.sequence_mask(source_sequence_length, source_max_time, dtype=tf.float32)
            if self.time_major:
             en_src_mask = tf.transpose(en_src_mask)

            decoder_outputs, dec_out_state, attn_dists, p_gens, coverage = self.attention_decoder(hps,emb_dec_inputs, dec_in_state, encoder_outputs, en_src_mask, cell,
                                                                                 initial_state_attention=False,
                                                                                 pointer_gen=hps.pointer_gen,
                                                                                 use_coverage=True,
                                                                                 prev_coverage=None, scope =decoder_scope)


            logits = self.output_layer(tf.stack(decoder_outputs))

            coverage_loss = _coverage_loss(attn_dists, dec_target_mask)



        else:
            # maximum_iteration: The maximum decoding steps.
            maximum_iterations = self._get_infer_maximum_iterations(
                hps, source_sequence_length)
            beam_width = hps.beam_width
            length_penalty_weight = hps.length_penalty_weight
            start_tokens = tf.fill([self.batch_size], tgt_sos_id)
            end_token = tgt_eos_id

            if beam_width > 0:
                utils.print_out("beam search activated")
                my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=beam_width,
                    output_layer=self.output_layer,
                    length_penalty_weight=length_penalty_weight)
            else:
                # Helper
                utils.print_out("greedy decoding activated")

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding_decoder, start_tokens, end_token)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    dec_in_state,
                    output_layer=self.output_layer  # applied per timestep
                )

            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                maximum_iterations=maximum_iterations,
                output_time_major=self.time_major,
                swap_memory=True,
                scope=decoder_scope)

            if beam_width > 0:
                logits = tf.no_op()
                sample_id = outputs.predicted_ids
                # if self.time_major:
                #     sample_id = tf.transpose(sample_id, perm=[1, 2, 0])
                #     utils.print_out("transpose activated for sampleid   ")



            else:
                logits = outputs.rnn_output
                sample_id = outputs.sample_id

        return coverage_loss,logits, sample_id, final_context_state


  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
      """Build an RNN cell that can be used by decoder."""
      # We only make use of encoder_outputs in attention-based models
      # if hparams.attention:
      #     raise ValueError("BasicModel doesn't support attention.")

      num_layers = 1
      num_residual_layers = hparams.num_residual_layers

      cell = model_helper.create_rnn_cell(
          unit_type=hparams.unit_type,
          num_units=hparams.num_units,
          num_layers=num_layers,
          num_residual_layers=num_residual_layers,
          forget_bias=hparams.forget_bias,
          dropout=hparams.dropout,
          num_gpus=hparams.num_gpus,
          mode=self.mode,
          single_cell_fn=self.single_cell_fn)

      # For beam search, we need to replicate encoder infos beam_width times
      if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
          decoder_initial_state = tf.contrib.seq2seq.tile_batch(
              encoder_state, multiplier=hparams.beam_width)
      else:
          decoder_initial_state = encoder_state

      return cell, decoder_initial_state


  def _get_infer_maximum_iterations(self, hps, source_sequence_length):
      """Maximum decoding steps at inference time."""
      if hps.tgt_max_len_infer:
          maximum_iterations = hps.tgt_max_len_infer
          utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
      else:
          # TODO(thangluong): add decoding_length_factor flag
          decoding_length_factor = 0.3
          max_encoder_length = tf.reduce_max(source_sequence_length)
          maximum_iterations = tf.to_int32(tf.round(
              tf.to_float(max_encoder_length) * decoding_length_factor))
      return maximum_iterations

  def _coverage_loss(attn_dists, padding_mask):
      """Calculates the coverage loss from the attention distributions.

      Args:
        attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
        padding_mask: shape (batch_size, max_dec_steps).

      Returns:
        coverage_loss: scalar
      """
      coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
      covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
      for a in attn_dists:
          covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
          covlosses.append(covloss)
          coverage += a  # update the coverage vector
      coverage_loss = _mask_and_avg(covlosses, padding_mask)
      return coverage_loss


  def get_max_time(self, tensor):
      time_axis = 0 if self.time_major else 1
      return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]



  def build_graph(self, hparams, scope=None):
      """Subclass must implement this method.

      Creates a sequence-to-sequence model with dynamic RNN decoder API.
      Args:
        hparams: Hyperparameter configurations.
        scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

      Returns:
        A tuple of the form (logits, loss, final_context_state),
        where:
          logits: float32 Tensor [batch_size x num_decoder_symbols].
          loss: the total loss / batch_size.
          final_context_state: The final state of decoder RNN.

      Raises:
        ValueError: if encoder_type differs from mono and bi, or
          attention_option is not (luong | scaled_luong |
          bahdanau | normed_bahdanau).
      """
      utils.print_out("# creating %s graph ..." % self.mode)
      dtype = tf.float32
      num_layers = hparams.num_layers
      num_gpus = hparams.num_gpus

      with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
          # Encoder
          encoder_outputs, encoder_state = self._build_encoder(hparams)

          ## Decoder
          coverage_loss,logits, sample_id, final_context_state = self._build_decoder(
              encoder_outputs, encoder_state, hparams)

          ## Loss
          if self.mode != tf.contrib.learn.ModeKeys.INFER:
              with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
                  loss = self._compute_loss(logits)+coverage_loss
          else:
              loss = None

          return logits, loss, final_context_state, sample_id,coverage_loss




def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss

def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
  values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
  values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex)  # overall average

