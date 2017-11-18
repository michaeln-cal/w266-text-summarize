
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import model_helper


class SummarizationModel(object):

    def __init__(self, iterator, hps, mode, vocab_table,reverse_target_vocab_table=None, scope=None):
        self.init_iter = iterator.initializer
        self.hps = hps
        self.vocab_table = vocab_table
        self.reverse_target_vocab_table = reverse_target_vocab_table
        self.iterator = iterator

        self.mode = mode
        self.single_cell_fn = None
        self.time_major = False
        self.batch_size = hps.batch_size

        # self._output_layer = layers_core.Dense(
        #     self._vocab[1], use_bias=False, name="output_projection")
        self.start_decoding = tf.cast(vocab_table.lookup(tf.constant(hps.START_DECODING)), tf.int32)
        self.stop_decoding = tf.cast(vocab_table.lookup(tf.constant(hps.STOP_DECODING)), tf.int32)

        #init
        self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)

        self.attention_mechanism_fn = create_attention_mechanism
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
        self.init_embeddings(hps)

     # Projection
        with tf.variable_scope(scope or "build_network"):
         with tf.variable_scope("decoder/output_projection"):
          self.output_layer = layers_core.Dense(hps.vocab_size, use_bias=False, name="output_projection")

         ## Train graph
         res = self.build_graph(hps, scope=scope)

         if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
             self.train_loss = res[1]
             self.word_count = tf.reduce_sum(
                 self.iterator.source_sequence_length) + tf.reduce_sum(
                 self.iterator.target_sequence_length)
         elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
             self.eval_loss = res[1]
         elif self.mode == tf.contrib.learn.ModeKeys.INFER:
             self.infer_logits, _, self.final_context_state, self.sample_id = res
             self.sample_words = reverse_target_vocab_table.lookup(
                 tf.to_int64(self.sample_id))

         if self.mode != tf.contrib.learn.ModeKeys.INFER:
             ## Count the number of predicted words for compute ppl.
             self.predict_count = tf.reduce_sum(
                 self.iterator.target_sequence_length)

         self.global_step = tf.Variable(0, trainable=False)
         params = tf.trainable_variables()

         # Gradients and SGD update operation for training the model.
         # Arrage for the embedding vars to appear at the beginning.
         if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
             self.learning_rate = tf.constant(hps.learning_rate)
             # warm-up
             self.learning_rate = self._get_learning_rate_warmup(hps)
             # decay
             self.learning_rate = self._get_learning_rate_decay(hps)

             # Optimizer
             if hps.optimizer == "sgd":
                 opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                 tf.summary.scalar("lr", self.learning_rate)
             elif hps.optimizer == "adam":
                 opt = tf.train.AdamOptimizer(self.learning_rate)

             # Gradients
             gradients = tf.gradients(
                 self.train_loss,
                 params,
                 colocate_gradients_with_ops=hps.colocate_gradients_with_ops)

             clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
                 gradients, max_gradient_norm=hps.max_grad_norm)

             self.update = opt.apply_gradients(
                 zip(clipped_gradients, params), global_step=self.global_step)

             # Summary
             self.train_summary = tf.summary.merge([
                                                       tf.summary.scalar("lr", self.learning_rate),
                                                       tf.summary.scalar("train_loss", self.train_loss),
                                                   ] + gradient_norm_summary)

         if self.mode == tf.contrib.learn.ModeKeys.INFER:
             self.infer_summary = self._get_infer_summary(hps)

         # Saver
         self.saver = tf.train.Saver(tf.global_variables())

         # Print trainable variables
         print("# Trainable variables")
         for param in params:
             print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                               param.op.device))

    def init_embeddings(self, hps):
        """Init embeddings."""

        embedding = tf.get_variable('embedding', [self.hps.vocab_size, self.hps.emb_dim], dtype=tf.float32,
                                        initializer=self.trunc_norm_init)

        self.embedding_encoder, self.embedding_decoder =embedding,embedding

    def _build_encoder_cell(self, hps, num_layers, num_residual_layers=0,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return model_helper.create_rnn_cell(
            unit_type=hps.unit_type,
            num_units=hps.hidden_dim,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hps.forget_bias,
            dropout=hps.dropout,
            num_gpus=hps.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].


        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple withb hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self.hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

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
        tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.START_DECODING)),
                             tf.int32)
        tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.STOP_DECODING)),
                             tf.int32)

        num_layers = hps.num_layers
        num_gpus = hps.num_gpus

        iterator = self.iterator

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(
            hps, iterator.source_sequence_length)

        ## Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hps, encoder_outputs, encoder_state,
                iterator.source_sequence_length)

            ## Train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = iterator.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, target_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, iterator.target_sequence_length,
                    time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state, )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id

                # Note: there's a subtle difference here between train and inference.
                # We could have set output_layer when create my_decoder
                #   and shared more code between train and inference.
                # We chose to apply the output_layer to all timesteps for speed:
                #   10% improvements for small models & 20% for larger ones.
                # If memory is a concern, we should apply output_layer per timestep.
                device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
                with tf.device(model_helper.get_device_str(device_id, num_gpus)):
                    logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:
                beam_size = hps.beam_size
                length_penalty_weight = hps.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_size > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_size,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    # Helper
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder, start_tokens, end_token)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                if beam_size > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def _build_decoder_cell(self, hps, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        attention_option = hps.attention
        attention_architecture = hps.attention_architecture

        if attention_architecture != "standard":
            raise ValueError(
                "Unknown attention architecture %s" % attention_architecture)

        num_units = hps.num_units
        num_layers = hps.num_layers
        num_residual_layers = hps.num_residual_layers
        num_gpus = hps.num_gpus
        beam_size = hps.beam_size

        dtype = tf.float32
        # Ensure memory is batch-major
        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_size > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=beam_size)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=beam_size)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=beam_size)
            batch_size = self.batch_size * beam_size
        else:
            batch_size = self.batch_size

        attention_mechanism = self.attention_mechanism_fn(
            attention_option, num_units, memory, source_sequence_length, self.mode)

        cell = model_helper.create_rnn_cell(
            unit_type=hps.unit_type,
            num_units=num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hps.forget_bias,
            dropout=hps.dropout,
            num_gpus=num_gpus,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                             beam_size == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            name="attention")

        # TODO(thangluong): do we need num_layers, num_gpus?
        cell = tf.contrib.rnn.DeviceWrapper(cell,
                                            model_helper.get_device_str(
                                                num_layers - 1, num_gpus))

        if hps.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state

    def _get_infer_summary(self, hps):
        return tf.no_op()

    def _get_infer_maximum_iterations(self, hps, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if hps.tgt_max_len_infer:
            maximum_iterations = hps.tgt_max_len_infer
            print("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            # TODO(thangluong): add decoding_length_factor flag
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def build_graph(self, hps, scope=None):
        """Subclass must implement this method.

        Creates a sequence-to-sequence model with dynamic RNN decoder API.
        Args:
          hps: Hyperparameter configurations.
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
        print("# creating %s graph ..." % self.mode)
        dtype = tf.float32
        num_layers = hps.num_layers
        num_gpus = hps.num_gpus


        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Encoder
            with tf.variable_scope('embedding'):

                emb_enc_inputs = tf.nn.embedding_lookup(self.embedding_encoder,
                                                        self.iterator.source)  # tensor with shape (batch_size, max_enc_steps, emb_size)

            with tf.variable_scope('encoding'):
            # Add the encoder.
                encoder_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self.iterator.source_sequence_length)
                # enc_states = enc_outputs
                # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state

                dec_in_state = self._reduce_states(fw_st, bw_st)



            ## Decoder
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, dec_in_state, self.hps)

            ## Loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _get_learning_rate_warmup(self, hps):
        """Get learning rate warmup."""
        warmup_steps = hps.warmup_steps
        warmup_scheme = hps.warmup_scheme
        print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                        (hps.learning_rate, warmup_steps, warmup_scheme))

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if warmup_scheme == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor ** (
                tf.to_float(warmup_steps - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(
            self.global_step < hps.warmup_steps,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self, hps):
        """Get learning rate decay."""
        if (hps.learning_rate_decay_scheme and
                    hps.learning_rate_decay_scheme == "luong"):
            start_decay_step = int(hps.num_train_steps / 2)
            decay_steps = int(hps.num_train_steps / 10)  # decay 5 times
            decay_factor = 0.5
        else:
            start_decay_step = hps.start_decay_step
            decay_steps = hps.decay_steps
            decay_factor = hps.decay_factor
        print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
              "decay_factor %g" %
              (hps.learning_rate_decay_scheme,
               hps.start_decay_step, hps.decay_steps, hps.decay_factor))

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
        ])

    def decode(self, sess):
        """Decode a batch.

        Args:
          sess: tensorflow session to use.

        Returns:
          A tuple consiting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        _, infer_summary, _, sample_words = self.infer(sess)

        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words, infer_summary

def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length, mode):
    """Create attention mechanism based on the attention_option."""
    del mode  # unused

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


