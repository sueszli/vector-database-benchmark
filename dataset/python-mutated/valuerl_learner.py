import tensorflow as tf
import numpy as np
import os
from learner import Learner
from valuerl import ValueRL
from worldmodel import DeterministicWorldModel

class ValueRLLearner(Learner):
    """
  ValueRL-specific training loop details.
  """

    def learner_name(self):
        if False:
            print('Hello World!')
        return 'valuerl'

    def make_loader_placeholders(self):
        if False:
            while True:
                i = 10
        self.obs_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], np.prod(self.env_config['obs_dims'])])
        self.next_obs_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], np.prod(self.env_config['obs_dims'])])
        self.action_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], self.env_config['action_dim']])
        self.reward_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size']])
        self.done_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size']])
        self.datasize_loader = tf.placeholder(tf.float64, [])
        return [self.obs_loader, self.next_obs_loader, self.action_loader, self.reward_loader, self.done_loader, self.datasize_loader]

    def make_core_model(self):
        if False:
            while True:
                i = 10
        if self.config['model_config'] is not False:
            self.worldmodel = DeterministicWorldModel(self.config['name'], self.env_config, self.config['model_config'])
        else:
            self.worldmodel = None
        valuerl = ValueRL(self.config['name'], self.env_config, self.learner_config)
        ((policy_loss, Q_loss), inspect_losses) = valuerl.build_training_graph(*self.current_batch, worldmodel=self.worldmodel)
        policy_optimizer = tf.train.AdamOptimizer(0.0003)
        policy_gvs = policy_optimizer.compute_gradients(policy_loss, var_list=valuerl.policy_params)
        capped_policy_gvs = policy_gvs
        policy_train_op = policy_optimizer.apply_gradients(capped_policy_gvs)
        Q_optimizer = tf.train.AdamOptimizer(0.0003)
        Q_gvs = Q_optimizer.compute_gradients(Q_loss, var_list=valuerl.Q_params)
        capped_Q_gvs = Q_gvs
        Q_train_op = Q_optimizer.apply_gradients(capped_Q_gvs)
        return (valuerl, (policy_loss, Q_loss), (policy_train_op, Q_train_op), inspect_losses)

    def initialize(self):
        if False:
            print('Hello World!')
        if self.config['model_config'] is not False:
            while not self.load_worldmodel():
                pass

    def resume_from_checkpoint(self, epoch):
        if False:
            while True:
                i = 10
        if self.config['model_config'] is not False:
            with self.bonus_kwargs['model_lock']:
                self.worldmodel.load(self.sess, self.save_path, epoch)

    def checkpoint(self):
        if False:
            i = 10
            return i + 15
        self.core.copy_to_old(self.sess)
        if self.config['model_config'] is not False:
            self.load_worldmodel()

    def backup(self):
        if False:
            print('Hello World!')
        pass

    def load_worldmodel(self):
        if False:
            while True:
                i = 10
        if not os.path.exists('%s/%s.params.index' % (self.save_path, self.worldmodel.saveid)):
            return False
        with self.bonus_kwargs['model_lock']:
            self.worldmodel.load(self.sess, self.save_path)
        return True