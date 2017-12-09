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

import model
import model_helper
import collections
import functools
import math

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapper

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapperState
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism

# from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention




class CoverageAttentionWrapper(AttentionWrapper):

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        history = previous_alignment_history[0]
        history.clear_after_read =False

        history = tf.cond(history.size()>0,lambda : history.read(0),
                          lambda :tf.zeros([1,30,400], dtypes.float32))
        all_alignments = []
        all_attentions = []
        all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments = _compute_attention(
                attention_mechanism, cell_output, previous_alignments[i],history,
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = array_ops.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            print("is multi")
            return t
        else:
            print("not multi")
            print(t[0])

            return t[0]

def _compute_attention(attention_mechanism, cell_output, previous_alignments,previous_alignment_history,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments = attention_mechanism(
      cell_output, previous_alignments=previous_alignments, alignment_history =previous_alignment_history)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments

def _bahdanau_coverage_score(processed_query,alignment_history, keys, normalize):
  """Implements Bahdanau-style (additive) scoring function.
  This attention has two forms.  The first is Bhandanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, set `normalize=True`.
  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    normalize: Whether to normalize the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  # history_sum now has the shape of [batch,max_time] that represent the sum of past history of alignment in this
  # particular timestep

  # print("history shape is", tf.shape(history_sum))
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  max_time = keys.shape[1].value or array_ops.shape(keys)[2]

  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  c = variable_scope.get_variable(
      "coverage_c", [400,num_units], dtype=dtype)
  history = alignment_history
  # history_sum= math_ops.reduce_sum(history,[0])


  #expand history into [batch,1, max_time]
  # history_sum_expanded =array_ops.expand_dims(history_sum, 1)
  #now become [batch,max_time,num_units]
  coverage_features =  math_ops.matmul(history,c)
  coverage_features = array_ops.expand_dims(coverage_features, 1)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=math.sqrt((1. / num_units)))
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query +coverage_features+ b), [2])
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query+coverage_features), [2])

class CoverageBahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.
  This attention has two forms.  The first is Bahdanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               name="CoverageBahdanauAttention"):
    """Construct the Attention mechanism.
    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(CoverageBahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, previous_alignments,alignment_history):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_coverage_score(processed_query,alignment_history, self._keys, self._normalize)
    alignments = self._probability_fn(score, previous_alignments)
    return alignments