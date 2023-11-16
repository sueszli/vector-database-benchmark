"""Tests for fivo.ghmm_runners."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from fivo import ghmm_runners

class GHMMRunnersTest(tf.test.TestCase):

    def default_config(self):
        if False:
            while True:
                i = 10

        class Config(object):
            pass
        config = Config()
        config.model = 'ghmm'
        config.bound = 'fivo'
        config.proposal_type = 'prior'
        config.batch_size = 4
        config.num_samples = 4
        config.num_timesteps = 10
        config.variance = 0.1
        config.resampling_type = 'multinomial'
        config.random_seed = 1234
        config.parallel_iterations = 1
        config.learning_rate = 0.0001
        config.summarize_every = 1
        config.max_steps = 1
        return config

    def test_eval_ghmm_notraining_fivo_prior(self):
        if False:
            print('Hello World!')
        self.eval_ghmm_notraining('fivo', 'prior', -3.063864)

    def test_eval_ghmm_notraining_fivo_true_filtering(self):
        if False:
            print('Hello World!')
        self.eval_ghmm_notraining('fivo', 'true-filtering', -1.1409812)

    def test_eval_ghmm_notraining_fivo_true_smoothing(self):
        if False:
            for i in range(10):
                print('nop')
        self.eval_ghmm_notraining('fivo', 'true-smoothing', -0.85592091)

    def test_eval_ghmm_notraining_iwae_prior(self):
        if False:
            for i in range(10):
                print('nop')
        self.eval_ghmm_notraining('iwae', 'prior', -5.9730167)

    def test_eval_ghmm_notraining_iwae_true_filtering(self):
        if False:
            for i in range(10):
                print('nop')
        self.eval_ghmm_notraining('iwae', 'true-filtering', -1.1485999)

    def test_eval_ghmm_notraining_iwae_true_smoothing(self):
        if False:
            print('Hello World!')
        self.eval_ghmm_notraining('iwae', 'true-smoothing', -0.85592091)

    def eval_ghmm_notraining(self, bound, proposal_type, expected_bound_avg):
        if False:
            return 10
        config = self.default_config()
        config.proposal_type = proposal_type
        config.bound = bound
        config.logdir = os.path.join(tf.test.get_temp_dir(), 'test-ghmm-%s-%s' % (proposal_type, bound))
        ghmm_runners.run_eval(config)
        data = np.load(os.path.join(config.logdir, 'out.npz')).item()
        self.assertAlmostEqual(expected_bound_avg, data['mean'], places=3)

    def test_train_ghmm_for_one_step_and_eval_fivo_filtering(self):
        if False:
            i = 10
            return i + 15
        self.train_ghmm_for_one_step_and_eval('fivo', 'filtering', -16.727108)

    def test_train_ghmm_for_one_step_and_eval_fivo_smoothing(self):
        if False:
            print('Hello World!')
        self.train_ghmm_for_one_step_and_eval('fivo', 'smoothing', -19.381277)

    def test_train_ghmm_for_one_step_and_eval_iwae_filtering(self):
        if False:
            for i in range(10):
                print('nop')
        self.train_ghmm_for_one_step_and_eval('iwae', 'filtering', -33.31966)

    def test_train_ghmm_for_one_step_and_eval_iwae_smoothing(self):
        if False:
            print('Hello World!')
        self.train_ghmm_for_one_step_and_eval('iwae', 'smoothing', -46.388447)

    def train_ghmm_for_one_step_and_eval(self, bound, proposal_type, expected_bound_avg):
        if False:
            i = 10
            return i + 15
        config = self.default_config()
        config.proposal_type = proposal_type
        config.bound = bound
        config.max_steps = 1
        config.logdir = os.path.join(tf.test.get_temp_dir(), 'test-ghmm-training-%s-%s' % (proposal_type, bound))
        ghmm_runners.run_train(config)
        ghmm_runners.run_eval(config)
        data = np.load(os.path.join(config.logdir, 'out.npz')).item()
        self.assertAlmostEqual(expected_bound_avg, data['mean'], places=2)
if __name__ == '__main__':
    tf.test.main()