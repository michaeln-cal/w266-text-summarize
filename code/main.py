
import time
import datetime
import numpy as np
import os
import struct
import tensorflow as tf
from collections import namedtuple
from tensorflow.core.example import example_pb2
from tensorflow.python import debug as tf_debug

import data
import misc_utils as utils
from model import SummarizationModel
import util
import util.util as util
FLAGS = tf.app.flags.FLAGS

# Where to find data

tf.app.flags.DEFINE_string('data_path', '../data/cnn-dailymail/finished_files/chunked', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '../data/cnn-dailymail/finished_files/vocab', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_boolean('pass_hidden_state', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('warmup_scheme', 't2t', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('learning_rate_decay_scheme', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'SGD.')
tf.app.flags.DEFINE_string('unit_type', 'lstm', 'start of sentence')


tstamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
tf.app.flags.DEFINE_string('exp_name', 'exp_'+tstamp, 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('num_units', 256, 'dimension of RNN hidden states')

tf.app.flags.DEFINE_integer('num_gpus', 1, 'number of GPU')
tf.app.flags.DEFINE_integer('num_residual_layers', 0, 'number residual')

tf.app.flags.DEFINE_integer('num_layers', 1, 'number of layers')

tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 6, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_integer('warmup_steps', 1, 'warm up step ')
tf.app.flags.DEFINE_integer('start_decay_step', 0, 'decay step ')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'decay step ')
tf.app.flags.DEFINE_integer('decay_factor', 1, 'decay step ')
tf.app.flags.DEFINE_integer('num_train_steps', 12000, 'num train step ')
tf.app.flags.DEFINE_integer('tgt_max_len_infer', None, 'num tag max len infer ')






tf.app.flags.DEFINE_float('learning_rate', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('forget_bias', 1.0, 'forget bias ')
tf.app.flags.DEFINE_float('dropout', 0.2, 'drop out')
tf.app.flags.DEFINE_float('length_penalty_weight', 0.0, 'length penalty ')



tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')



tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', False, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

tf.app.flags.DEFINE_boolean('colocate_gradients_with_ops', True, "colocate gradient with norm")

# Constant
tf.app.flags.DEFINE_string('SENTENCE_START', '<s>', 'start of sentence')

tf.app.flags.DEFINE_string('SENTENCE_END', '</s>', 'end of sentence')
tf.app.flags.DEFINE_string('PAD_TOKEN', '[PAD]', 'Padding token')
tf.app.flags.DEFINE_string('START_DECODING', '[START]', 'start decoding token')

tf.app.flags.DEFINE_string('UNKNOWN_TOKEN', '[UNK]', 'unknown  token')

tf.app.flags.DEFINE_string('STOP_DECODING', '[STOP]', 'start decoding token')
tf.app.flags.DEFINE_string('attention', 'luong', 'Mechansim')

tf.app.flags.DEFINE_string('attention_architecture', 'standard', 'standard')






train_filenames = []
test_filenames =[]
val_filenames =[]

#dir = "/Users/giang/Downloads/finished_files/chunked"
dir = FLAGS.data_path

for file in os.listdir(dir):
    if file.startswith("train_"):
        train_filenames.append(dir+"/"+file)

for file in os.listdir(dir):
    if file.startswith("test_"):
        test_filenames.append(dir+"/"+file)


for file in os.listdir(dir):
    if file.startswith("val_"):
        val_filenames.append(dir+"/"+file)


#vocab_file ="/Users/giang/Downloads/finished_files/vocab_copy"
vocab_file = FLAGS.vocab_path

# filename =['/Users/giang/train_127.bin']
# filename2 ='/Users/giang/a.story'
def read_file(filename):
    reader = open(filename, 'rb')
    len_bytes = reader.read(8)
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    example = example_pb2.Example.FromString(example_str)
    article_text = example.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
    abstract_text = data.abstract2sents(str(example.features.feature['abstract'].bytes_list.value[0])) # the abstract text was saved under the key 'abstract' in the data files
    # print(abstract_text)
    return article_text,abstract_text

# filenames =['/Users/giang/train_127.bin','/Users/giang/train_090.bin']



def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def restore_best_model():
  """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
  tf.logging.info("Restoring bestmodel for training...")

  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print("Initializing all variables...")
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
  print("Restoring all non-adagrad variables from best model in eval dir...")
  curr_ckpt = util.load_ckpt(saver, sess, "eval")
  print("Restored %s." % curr_ckpt)

  # Save this model to train dir and quit
  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
  print("Saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
  new_saver.save(sess, new_fname)
  print("Saved.")
  exit()


def convert_to_coverage_model():
  """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
  tf.logging.info("converting non-coverage model to coverage model..")

  # initialize an entire coverage model from scratch
  sess = tf.Session(config=util.get_config())
  print("initializing everything...")
  sess.run(tf.global_variables_initializer())

  # load all non-coverage weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
  print("restoring non-coverage variables...")
  curr_ckpt = util.load_ckpt(saver, sess)
  print("restored.")

  # save this model and quit
  new_fname = curr_ckpt + '_cov_init'
  print("saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print("saved.")
  exit()


def setup_training(hps,model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph(hps) # build the graph
  if FLAGS.convert_to_coverage_model:
    assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
    convert_to_coverage_model()
  if FLAGS.restore_best_model:
    restore_best_model()
  saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=60, # checkpoint every 60 secs
                     global_step=model.global_step)
  # summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, sess_context_manager,train_dir) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, sess_context_manager,train_dir):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")

  with sess_context_manager as sess:
    sess.run([model.init_iter])
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    t0=time.time()
    epoch_step = 1
    while True: # repeats until interrupted

      tf.logging.debug('running training step...')
      t1=time.time()
      try:
        results = model.train(sess)
        # return sess.run([self.update,
        #                  self.train_loss,
        #                  self.predict_count,
        #                  self.train_summary,
        #                  self.global_step,
        #                  self.word_count,
        #                  self.batch_size])
        summaries = results[3]  # we will write these summaries to tensorboard using summary_writer
        train_step = results[4]  # we need this to update our running average loss
      except tf.errors.OutOfRangeError:
        # Finished going through the training dataset.  Go to next epoch.
        epoch_step += 1
        print(
          "# Finished epoch: ",epoch_step, "step %d. Perform external evaluation" %train_step)
        sess.run([model.init_iter])
        continue

      t2=time.time()
      loss = results[1]
      tf.logging.info('training: epoch:%2d; time elapsed:%8.2f (%+6.2f); loss: %f.2', epoch_step, t2-t0, t2-t1, loss)

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      # if FLAGS.coverage:
      #   coverage_loss = results['coverage_loss']
      #   tf.logging.info("coverage_loss: %f", coverage_loss) # print the coverage loss to screen

      # get the summaries and iteration number so we can write summaries to tensorboard


      # summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0: # flush the summary writer every so often
        model.saver.save(
          sess,
          os.path.join(train_dir, "summarization.ckpt"),
          global_step=model.global_step)
        # summary_writer.flush()





def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = (FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  if FLAGS.mode == 'decode':
    FLAGS.batch_size = FLAGS.beam_size

  # If single_pass=True, check we're in decode mode


  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    # if key in hparam_list: # if it's in the list
    hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  # batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

  train_dataset = tf.data.TextLineDataset.from_tensor_slices(train_filenames)

  train_dataset = train_dataset.map(lambda filename: tf.py_func(read_file, [filename], [tf.string, tf.string]))

  test_dataset = tf.data.TextLineDataset.from_tensor_slices(test_filenames)
  test_dataset = test_dataset.map(lambda filename: tf.py_func(read_file, [filename], [tf.string, tf.string]))

  val_dataset = tf.data.TextLineDataset.from_tensor_slices(test_filenames)
  val_dataset = val_dataset.map(lambda filename: tf.py_func(read_file, [filename], [tf.string, tf.string]))



  # it = dataset.make_one_shot_iterator()
  # x_it = it.get_next()
  vocab_table = data.create_vocab_tables(vocab_file, hps.vocab_size)
  reverse_vocab_table = data.create_id_tables(vocab_file, hps.vocab_size)



  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'train':
    print("creating model...")
    iterator = data.get_iterator(train_dataset, vocab_table, hps)
    # init = iterator.initializer



    model = SummarizationModel(iterator,hps,hps.mode,vocab_table,reverse_vocab_table)
    setup_training(hps,model)

  elif hps.mode == 'decode':
    train_dir = os.path.join(FLAGS.log_root, "train")

    iterator = data.get_iterator(val_dataset, vocab_table, hps)
    # init = iterator.initializer
    ckpt_state = tf.train.get_checkpoint_state(train_dir)


    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

    model = SummarizationModel(iterator,hps,tf.contrib.learn.ModeKeys.INFER,vocab_table,reverse_vocab_table)
    sess = tf.Session(config=util.get_config())
    saver =tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    sess.run(tf.tables_initializer())

    sess.run([model.init_iter])
    start_time = time.time()
    num_sentences = 0
    num_translations_per_input=1
    while True:
      try:
        nmt_outputs, _ = model.decode(sess)

        batch_size = nmt_outputs.shape[1]
        num_sentences += batch_size

        for sent_id in range(batch_size):
          for beam_id in range(num_translations_per_input):
            translation = get_translation(
              nmt_outputs[beam_id],
              sent_id,
              tgt_eos=hps.STOP_DECODING,
              subword_option="bpe")
            print((translation + b"\n").decode("utf-8"))
      except tf.errors.OutOfRangeError:
        utils.print_time(
          "  done, num sentences %d, num translations per input %d" %
          (num_sentences, num_translations_per_input), start_time)
        break

  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")
def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]

  if subword_option == "bpe":  # BPE
    translation = utils.format_bpe_text(output)
  elif subword_option == "spm":  # SPM
    translation = utils.format_spm_text(output)
  else:
    translation = utils.format_text(output)

  return translation
if __name__ == '__main__':
  tf.app.run()
