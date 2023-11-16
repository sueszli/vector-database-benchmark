"""PTB data loader and helpers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf
EOS_INDEX = 0

def _read_words(filename):
    if False:
        return 10
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode('utf-8').replace('\n', '<eos>').split()

def build_vocab(filename):
    if False:
        return 10
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    (words, _) = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    print('<eos>:', word_to_id['<eos>'])
    global EOS_INDEX
    EOS_INDEX = word_to_id['<eos>']
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    if False:
        return 10
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None):
    if False:
        return 10
    'Load PTB raw data from data directory "data_path".\n  Reads PTB text files, converts strings to integer ids,\n  and performs mini-batching of the inputs.\n  The PTB dataset comes from Tomas Mikolov\'s webpage:\n  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz\n  Args:\n    data_path: string path to the directory where simple-examples.tgz has\n      been extracted.\n  Returns:\n    tuple (train_data, valid_data, test_data, vocabulary)\n    where each of the data objects can be passed to PTBIterator.\n  '
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')
    word_to_id = build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return (train_data, valid_data, test_data, vocabulary)

def ptb_iterator(raw_data, batch_size, num_steps, epoch_size_override=None):
    if False:
        while True:
            i = 10
    'Iterate on the raw PTB data.\n\n  This generates batch_size pointers into the raw PTB data, and allows\n  minibatch iteration along these pointers.\n\n  Args:\n    raw_data: one of the raw data outputs from ptb_raw_data.\n    batch_size: int, the batch size.\n    num_steps: int, the number of unrolls.\n\n  Yields:\n    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].\n    The second element of the tuple is the same data time-shifted to the\n    right by one.\n\n  Raises:\n    ValueError: if batch_size or num_steps are too high.\n  '
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.full([batch_size, batch_len], EOS_INDEX, dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    if epoch_size_override:
        epoch_size = epoch_size_override
    else:
        epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError('epoch_size == 0, decrease batch_size or num_steps')
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        w = np.ones_like(x)
        yield (x, y, w)