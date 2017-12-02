
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import model_helper
from util import misc_utils as utils


class Model(object):

    def __init__(self, iterator, hps, mode, vocab_table,reverse_target_vocab_table=None, scope=None):
        self.init_iter = iterator.initializer
        self.hps = hps
        self.vocab_table = vocab_table
        self.reverse_target_vocab_table = reverse_target_vocab_table
        self.iterator = iterator
        self.use_test_set = False
        self.mode = mode
        self.single_cell_fn = None
        self.time_major = hps.time_major
        self.batch_size = hps.batch_size

        # self._output_layer = layers_core.Dense(
        #     self._vocab[1], use_bias=False, name="output_projection")
        # self.start_decoding = tf.cast(vocab_table.lookup(tf.constant(hps.START_DECODING)), tf.int32)
        # self.stop_decoding = tf.cast(vocab_table.lookup(tf.constant(hps.STOP_DECODING)), tf.int32)

        #init
        # self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        #
        # self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

        self.init_embeddings(hps, scope)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

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



             clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
                 gradients, max_gradient_norm=hps.max_gradient_norm)
             self.grad_norm = grad_norm

             self.update = opt.apply_gradients(
                 zip(clipped_grads, params), global_step=self.global_step)

             # Summary
             # Summary
             self.train_summary = tf.summary.merge([
                                                       tf.summary.scalar("lr", self.learning_rate),
                                                       tf.summary.scalar("train_loss", self.train_loss),
                                                   ] + grad_norm_summary)
         if self.mode == tf.contrib.learn.ModeKeys.INFER:
             self.infer_summary = self._get_infer_summary(hps)

         # Saver
         self.saver = tf.train.Saver(tf.global_variables())

         # Print trainable variables
         utils.print_out("# Trainable variables")
         for param in params:
             utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                               param.op.device))

    # def init_embeddings(self, hps):
    #     """Init embeddings."""
    #
    #     embedding = tf.get_variable('embedding', [self.hps.vocab_size, self.hps.emb_dim], dtype=tf.float32,
    #                                     initializer=self.trunc_norm_init)
    #
    #     self.embedding_encoder, self.embedding_decoder =embedding,embedding

    def init_embeddings(self, hparams, scope):
        """Init embeddings."""
        self.embedding_encoder, self.embedding_decoder = (
            model_helper.create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=hparams.vocab_size,
                tgt_vocab_size=hparams.vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                num_partitions=hparams.num_embeddings_partitions,
                scope=scope, ))

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predict_count,
                         self.batch_size])

    def _build_encoder(self, hparams):
        """Build an encoder."""
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        iterator = self.iterator

        source = iterator.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, source)

            # Encoder_outpus: [max_time, batch_size, num_units]
            if hparams.encoder_type == "uni":
                utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                                (num_layers, num_residual_layers))
                cell = self._build_encoder_cell(
                    hparams, num_layers, num_residual_layers)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=iterator.source_sequence_length,
                    time_major=self.time_major,
                    swap_memory=True)
            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                                (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = (
                    self._build_bidirectional_rnn(
                        inputs=encoder_emb_inp,
                        sequence_length=iterator.source_sequence_length,
                        dtype=dtype,
                        hparams=hparams,
                        num_bi_layers=num_bi_layers,
                        num_bi_residual_layers=num_bi_residual_layers))

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
        return encoder_outputs, encoder_state

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)



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
        tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.sos)),
                             tf.int32)
        tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hps.eos)),
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
                    utils.print_out("time major")
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

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                    # if self.time_major:
                    #     sample_id = tf.transpose(sample_id, perm=[1, 2, 0])
                    #     utils.print_out("transpose activated for sampleid   ")



                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build an RNN cell that can be used by decoder."""
        # We only make use of encoder_outputs in attention-based models
        if hparams.attention:
            raise ValueError("BasicModel doesn't support attention.")

        num_layers = hparams.num_layers
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

    def _get_infer_summary(self, hps):
        return tf.no_op()

    def _get_infer_maximum_iterations(self, hps, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if hps.tgt_max_len_infer:
            maximum_iterations = hps.tgt_max_len_infer
            utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            # TODO(thangluong): add decoding_length_factor flag
            decoding_length_factor = 0.2
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

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
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams)

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
        utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
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
            utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
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
                         self.word_count,
                         self.batch_size,
                         self.grad_norm,
                         self.learning_rate])

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

    def _build_bidirectional_rnn(self, inputs, sequence_length,
                                 dtype, hparams,
                                 num_bi_layers,
                                 num_bi_residual_layers,
                                 base_gpu=0):
        """Create and call biddirectional RNN cells.

        Args:
          num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
            layers in each RNN cell will be wrapped with `ResidualWrapper`.
          base_gpu: The gpu device id to use for the first forward RNN layer. The
            i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
            device id. The `base_gpu` for backward RNN cell is `(base_gpu +
            num_bi_layers)`.

        Returns:
          The concatenated bidirectional output and the bidirectional RNN cell"s
          state.
        """
        # Construct forward and backward cells
        fw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=base_gpu)
        bw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=(base_gpu + num_bi_layers))

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state

