"""Reads data from a large unlabeled corpus."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from base import embeddings
from corpus_processing import example
from corpus_processing import minibatching

class UnlabeledDataReader(object):

    def __init__(self, config, starting_file=0, starting_line=0, one_pass=False):
        if False:
            print('Hello World!')
        self.config = config
        self.current_file = starting_file
        self.current_line = starting_line
        self._one_pass = one_pass

    def endless_minibatches(self):
        if False:
            print('Hello World!')
        for examples in self.get_unlabeled_examples():
            d = minibatching.Dataset(self.config, examples, 'unlabeled')
            for mb in d.get_minibatches(self.config.train_batch_size):
                yield mb

    def _make_examples(self, sentences):
        if False:
            i = 10
            return i + 15
        word_vocab = embeddings.get_word_vocab(self.config)
        char_vocab = embeddings.get_char_vocab()
        return [example.Example(sentence, word_vocab, char_vocab) for sentence in sentences]

    def get_unlabeled_examples(self):
        if False:
            while True:
                i = 10
        lines = []
        for words in self.get_unlabeled_sentences():
            lines.append(words)
            if len(lines) >= 10000:
                yield self._make_examples(lines)
                lines = []

    def get_unlabeled_sentences(self):
        if False:
            i = 10
            return i + 15
        while True:
            file_ids_and_names = sorted([(int(fname.split('-')[1].replace('.txt', '')), fname) for fname in tf.gfile.ListDirectory(self.config.unsupervised_data)])
            for (fid, fname) in file_ids_and_names:
                if fid < self.current_file:
                    continue
                self.current_file = fid
                self.current_line = 0
                with tf.gfile.FastGFile(os.path.join(self.config.unsupervised_data, fname), 'r') as f:
                    for (i, line) in enumerate(f):
                        if i < self.current_line:
                            continue
                        self.current_line = i
                        words = line.strip().split()
                        if len(words) < self.config.max_sentence_length:
                            yield words
            self.current_file = 0
            self.current_line = 0
            if self._one_pass:
                break