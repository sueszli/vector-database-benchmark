from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import utils
tf.app.flags.DEFINE_string('data_dir', 'reproduce', 'Path to directory containing data and model checkpoints.')
FLAGS = tf.app.flags.FLAGS

class EnsembleLM(object):
    """Ensemble of language models."""

    def __init__(self, test_data_name='wsc273'):
        if False:
            i = 10
            return i + 15
        vocab_file = os.path.join(FLAGS.data_dir, 'vocab.txt')
        self.vocab = utils.CharsVocabulary(vocab_file, 50)
        assert test_data_name in ['pdp60', 'wsc273'], 'Test data must be pdp60 or wsc273, got {}'.format(test_data_name)
        self.test_data_name = test_data_name
        test_data = utils.parse_commonsense_reasoning_test(test_data_name)
        (self.question_ids, self.sentences, self.labels) = test_data
        self.all_probs = []

    def add_single_model(self, model_name='lm1'):
        if False:
            i = 10
            return i + 15
        'Add a single model into the current ensemble.'
        single_lm = SingleRecurrentLanguageModel(self.vocab, model_name)
        probs = single_lm.assign_probs(self.sentences, self.test_data_name)
        self.all_probs.append(probs)
        print('Done adding {}'.format(model_name))

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate the current ensemble.'
        ensembled_probs = sum(self.all_probs) / len(self.all_probs)
        scorings = []
        for (i, sentence) in enumerate(self.sentences):
            correctness = self.labels[i]
            word_probs = ensembled_probs[i, :len(sentence)]
            joint_prob = np.prod(word_probs, dtype=np.float64)
            scorings.append(dict(correctness=correctness, sentence=sentence, joint_prob=joint_prob, word_probs=word_probs))
        scoring_mode = 'full' if self.test_data_name == 'pdp60' else 'partial'
        return utils.compare_substitutions(self.question_ids, scorings, scoring_mode)

class SingleRecurrentLanguageModel(object):
    """Single Recurrent Language Model."""

    def __init__(self, vocab, model_name='lm01'):
        if False:
            while True:
                i = 10
        self.vocab = vocab
        self.log_dir = os.path.join(FLAGS.data_dir, model_name)

    def reset(self):
        if False:
            return 10
        self.sess.run(self.tensors['states_init'])

    def _score(self, word_patch):
        if False:
            while True:
                i = 10
        'Score a matrix of shape (batch_size, num_timesteps+1) str tokens.'
        word_ids = np.array([[self.vocab.word_to_id(word) for word in row] for row in word_patch])
        char_ids = np.array([[self.vocab.word_to_char_ids(word) for word in row] for row in word_patch])
        print('Probs for \n{}\n='.format(np.array(word_patch)[:, 1:]))
        (input_ids, target_ids) = (word_ids[:, :-1], word_ids[:, 1:])
        input_char_ids = char_ids[:, :-1, :]
        softmax = self.sess.run(self.tensors['softmax_out'], feed_dict={self.tensors['inputs_in']: input_ids, self.tensors['char_inputs_in']: input_char_ids})
        (batch_size, num_timesteps) = self.shape
        softmax = softmax.reshape((num_timesteps, batch_size, -1))
        softmax = np.transpose(softmax, [1, 0, 2])
        probs = np.array([[softmax[row, col, target_ids[row, col]] for col in range(num_timesteps)] for row in range(batch_size)])
        print(probs)
        return probs

    def _score_patches(self, word_patches):
        if False:
            i = 10
            return i + 15
        'Score a 2D matrix of word_patches and stitch results together.'
        (batch_size, num_timesteps) = self.shape
        (nrow, ncol) = (len(word_patches), len(word_patches[0]))
        max_len = num_timesteps * ncol
        probs = np.zeros([0, max_len])
        for (i, row) in enumerate(word_patches):
            print('Reset RNN states.')
            self.reset()
            row_probs = np.zeros([batch_size, 0])
            for (j, word_patch) in enumerate(row):
                print('Processing patch ({}, {}) / ({}, {})'.format(i + 1, j + 1, nrow, ncol))
                patch_probs = self._score(word_patch) if word_patch else np.zeros([batch_size, num_timesteps])
                row_probs = np.concatenate([row_probs, patch_probs], 1)
            probs = np.concatenate([probs, row_probs], 0)
        return probs

    def assign_probs(self, sentences, test_data_name='wsc273'):
        if False:
            while True:
                i = 10
        'Return prediction accuracy using this LM for a test.'
        probs_cache = os.path.join(self.log_dir, '{}.probs'.format(test_data_name))
        if os.path.exists(probs_cache):
            print('Reading cached result from {}'.format(probs_cache))
            with tf.gfile.Open(probs_cache, 'r') as f:
                probs = pkl.load(f)
        else:
            tf.reset_default_graph()
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph(os.path.join(self.log_dir, 'ckpt-best.meta'))
            saver.restore(self.sess, os.path.join(self.log_dir, 'ckpt-best'))
            print('Restored from {}'.format(self.log_dir))
            graph = tf.get_default_graph()
            self.tensors = dict(inputs_in=graph.get_tensor_by_name('test_inputs_in:0'), char_inputs_in=graph.get_tensor_by_name('test_char_inputs_in:0'), softmax_out=graph.get_tensor_by_name('SotaRNN_1/softmax_out:0'), states_init=graph.get_operation_by_name('SotaRNN_1/states_init'))
            self.shape = self.tensors['inputs_in'].shape.as_list()
            (batch_size, num_timesteps) = self.shape
            word_patches = utils.cut_to_patches(sentences, batch_size, num_timesteps)
            probs = self._score_patches(word_patches)
            with tf.gfile.Open(probs_cache, 'w') as f:
                pkl.dump(probs, f)
        return probs

def evaluate_ensemble(test_data_name, number_of_lms):
    if False:
        for i in range(10):
            print('nop')
    ensemble = EnsembleLM(test_data_name)
    model_list = ['lm{:02d}'.format(i + 1) for i in range(number_of_lms)]
    for model_name in model_list:
        ensemble.add_single_model(model_name)
    accuracy = ensemble.evaluate()
    print('Accuracy of {} LM(s) on {} = {}'.format(number_of_lms, test_data_name, accuracy))

def main(_):
    if False:
        return 10
    evaluate_ensemble('pdp60', 1)
    evaluate_ensemble('pdp60', 5)
    evaluate_ensemble('wsc273', 10)
    evaluate_ensemble('wsc273', 14)
if __name__ == '__main__':
    tf.app.run(main)