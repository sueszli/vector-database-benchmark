import tensorflow as tf
import numpy as np
from learner import Learner
from worldmodel import DeterministicWorldModel

class WorldmodelLearner(Learner):
    """
    Worldmodel-specific training loop details.
    """

    def learner_name(self):
        if False:
            print('Hello World!')
        return 'worldmodel'

    def make_loader_placeholders(self):
        if False:
            print('Hello World!')
        self.obs_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], np.prod(self.env_config['obs_dims'])])
        self.next_obs_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], np.prod(self.env_config['obs_dims'])])
        self.action_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size'], self.env_config['action_dim']])
        self.reward_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size']])
        self.done_loader = tf.placeholder(tf.float32, [self.learner_config['batch_size']])
        self.datasize_loader = tf.placeholder(tf.float64, [])
        return [self.obs_loader, self.next_obs_loader, self.action_loader, self.reward_loader, self.done_loader, self.datasize_loader]

    def make_core_model(self):
        if False:
            print('Hello World!')
        worldmodel = DeterministicWorldModel(self.config['name'], self.env_config, self.learner_config)
        (worldmodel_loss, inspect_losses) = worldmodel.build_training_graph(*self.current_batch)
        model_optimizer = tf.train.AdamOptimizer(0.0003)
        model_gvs = model_optimizer.compute_gradients(worldmodel_loss, var_list=worldmodel.model_params)
        capped_model_gvs = model_gvs
        worldmodel_train_op = model_optimizer.apply_gradients(capped_model_gvs)
        return (worldmodel, (worldmodel_loss,), (worldmodel_train_op,), inspect_losses)

    def initialize(self):
        if False:
            while True:
                i = 10
        pass

    def resume_from_checkpoint(self, epoch):
        if False:
            print('Hello World!')
        pass

    def checkpoint(self):
        if False:
            return 10
        pass

    def backup(self):
        if False:
            for i in range(10):
                print('nop')
        pass