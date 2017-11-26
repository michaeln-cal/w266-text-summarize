import inference
import os
import tensorflow as tf
import train
from collections import namedtuple

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_boolean('pass_hidden_state', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

tf.app.flags.DEFINE_boolean('log_device_placement', False, 'debug of GPU device')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('warmup_scheme', 't2t', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('learning_rate_decay_scheme', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'SGD.')
tf.app.flags.DEFINE_string('unit_type', 'lstm', 'start of sentence')


tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('num_units', 256, 'dimension of RNN hidden states')

tf.app.flags.DEFINE_integer('num_gpus', 0, 'number of GPU')
tf.app.flags.DEFINE_integer('num_residual_layers', 0, 'number residual')

tf.app.flags.DEFINE_integer('num_layers', 1, 'number of layers')

tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('infer_batch_size', 32, 'infer size')

tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 150, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_width', 6, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_integer('warmup_steps', 1, 'warm up step ')
tf.app.flags.DEFINE_integer('start_decay_step', 0, 'decay step ')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'decay step ')
tf.app.flags.DEFINE_integer('decay_factor', 1, 'decay step ')
tf.app.flags.DEFINE_integer('num_train_steps', 12000, 'num train step ')
tf.app.flags.DEFINE_integer('steps_per_stats', 100, 'stat frequency')
tf.app.flags.DEFINE_integer('epoch_step', 0, 'stat frequency')


tf.app.flags.DEFINE_integer('steps_per_external_eval', None, 'external eval')



tf.app.flags.DEFINE_integer('tgt_max_len_infer', None, 'num tag max len infer ')
tf.app.flags.DEFINE_integer('unk_id', 0, 'unk_id ')






tf.app.flags.DEFINE_float('learning_rate', 1.0, 'learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'learning rate')

tf.app.flags.DEFINE_float('forget_bias', 1.0, 'forget bias ')
tf.app.flags.DEFINE_float('dropout', 0.1, 'drop out')
tf.app.flags.DEFINE_float('length_penalty_weight', 0.0, 'length penalty ')



tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')



tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_float('best_rouge', 0, 'rouge')
tf.app.flags.DEFINE_float('best_bleu', 0, 'rouge')


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
tf.app.flags.DEFINE_string('sos', '<s>', 'start of sentence')

tf.app.flags.DEFINE_string('eos', '</s>', 'end of sentence')
tf.app.flags.DEFINE_string('PAD_TOKEN', '[PAD]', 'Padding token')
tf.app.flags.DEFINE_string('START_DECODING', '[START]', 'start decoding token')

tf.app.flags.DEFINE_string('unk_token', '[UNK]', 'unknown  token')

tf.app.flags.DEFINE_string('STOP_DECODING', '[STOP]', 'start decoding token')
tf.app.flags.DEFINE_string('attention', 'luong', 'Mechansim')

tf.app.flags.DEFINE_string('attention_architecture', 'standard', 'standard')

tf.app.flags.DEFINE_string('data_dir', '/Users/giang/PycharmProjects/w266-text-summarize/data', 'directory that stores input data')
tf.app.flags.DEFINE_string('vocab_file', "/Users/giang/PycharmProjects/w266-text-summarize/data/vocab/vocab", 'vocab file')

tf.app.flags.DEFINE_string('test_src_file', "/Users/giang/PycharmProjects/w266-text-summarize/data/article/test_000.bin", 'file for external valuation')
tf.app.flags.DEFINE_string('test_tgt_file', "/Users/giang/PycharmProjects/w266-text-summarize/data/abstract/test_000.bin", 'file for external valuation')

tf.app.flags.DEFINE_string('dev_src_file', "/Users/giang/PycharmProjects/w266-text-summarize/data/article/val_000.bin", 'file for external valuation')
tf.app.flags.DEFINE_string('dev_tgt_file', "/Users/giang/PycharmProjects/w266-text-summarize/data/abstract/val_000.bin", 'file for external valuation')

tf.app.flags.DEFINE_string('out_dir', None, 'dir to store model')

tf.app.flags.DEFINE_string('best_rouge_dir', '/Users/giang/PycharmProjects/w266-text-summarize/out_dir/best_rouge', 'rouge')
tf.app.flags.DEFINE_string('best_bleu_dir', '/Users/giang/PycharmProjects/w266-text-summarize/out_dir/best_bleu', 'rouge')


tf.app.flags.DEFINE_string('train_prefix', 'train_', 'directory that stores input data')

tf.app.flags.DEFINE_string('dev_prefix', 'val_', 'dev directory that stores input data')
tf.app.flags.DEFINE_string('test_prefix', 'test_', 'test directory that stores input data')

tf.app.flags.DEFINE_string('subword_option', '', 'Set to bpe or spm to activate subword desegmentation')

tf.app.flags.DEFINE_string('metrics', ['rouge','bleu'], 'metrics to measure: bleu,rouge,accuracy')







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
  ## Train / Decode


  train_fn = train.train
  inference_fn = inference.inference


  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    # if key in hparam_list: # if it's in the list
    hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  out_dir = hps.out_dir
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)




  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'train':
    train_fn(hps, target_session="")

  #
  # elif hps.mode == 'decode':

if __name__ == '__main__':
  tf.app.run()



