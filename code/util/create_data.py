from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import struct
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


def abstract2sents(abstract, start, end):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(start, cur)
      end_p = abstract.index(end, start_p + 1)
      cur = end_p + len(end)
      sents.append(abstract[start_p+len(start):end_p])
    except ValueError as e: # no more sentences

      return  ' '.join(sents)  # string


def read_article_file(filename):
    article_text=[]
    reader = open(filename, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        example = example_pb2.Example.FromString(example_str)
        article_text.append(example.features.feature['article'].bytes_list.value[0]) # the article text was saved under the key 'article' in the data files

    return article_text

def read_abstract_file(filename):
    abstract_text=[]
    reader = open(filename, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        example = example_pb2.Example.FromString(example_str)
        abstract_text.append(abstract2sents(str(example.features.feature['abstract'].bytes_list.value[0]),SENTENCE_START, SENTENCE_END)) # the abstract text was saved under the key 'abstract' in the data files

    return abstract_text

def write_file(write_dir, filename,data):
    # if not tf.gfile.Exists(write_dir): tf.gfile.MakeDirs(write_dir)

    with open(write_dir+"/"+filename, "w") as output:
        for s in data:
            output.write(str(s) + '\n')


if __name__ == '__main__':
    data_dir ='/Users/giang/Downloads/finished_files/chunked'
    target_dir= '/Users/giang/PycharmProjects/w266-text-summarize/data'
    vocab_file='/Users/giang/Downloads/finished_files/vocab_copy'


    # for file in os.listdir(data_dir):
    #     if file.startswith('val_') or file.startswith('train_')or file.startswith('test_') :
    #
    #         write_file(target_dir+'/article',file,read_article_file(data_dir+'/'+file))
    #         write_file(target_dir+'/abstract',file,read_abstract_file(data_dir+'/'+file))



    vocab = []
    with open(vocab_file, 'r') as vocab_f:
        for line in vocab_f:
            pieces = line.split()
            vocab.append(pieces[0])


    vocab = np.unique(np.array(vocab))
    np.random.shuffle(vocab)
    write_file(target_dir+'/vocab','vocab',vocab)
