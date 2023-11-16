"""Used to run estimators for distributed tests.

In distributed tests, we spawn processes to run estimator tasks like chief,
workers, parameter servers. The role of each task is determined by the TF_CONFIG
environment variable.

For more information on how tf.estimator.RunConfig uses TF_CONFIG, see
https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
from absl import app
from absl import flags
from absl import logging
from adanet import tf_compat
from adanet.autoensemble.estimator import AutoEnsembleEstimator
from adanet.core.estimator import Estimator
from adanet.core.evaluator import Evaluator
from adanet.distributed.placement import RoundRobinStrategy
from adanet.subnetwork import Builder
from adanet.subnetwork import SimpleGenerator
from adanet.subnetwork import Subnetwork
import tensorflow.compat.v1 as tf
try:
    from tensorflow.contrib.boosted_trees.python.utils import losses as bt_losses
except ImportError:
    bt_losses = None
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.training import session_manager as session_manager_lib
from tensorflow_estimator.python.estimator import training as training_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
flags.DEFINE_enum('estimator_type', 'estimator', ['estimator', 'autoensemble', 'autoensemble_trees_multiclass', 'estimator_with_experimental_multiworker_strategy'], 'The estimator type to train.')
flags.DEFINE_enum('placement_strategy', 'replication', ['replication', 'round_robin'], 'The distributed placement strategy.')
flags.DEFINE_string('model_dir', '', 'The model directory.')
FLAGS = flags.FLAGS

class SessionManager(session_manager_lib.SessionManager):
    """A session manager with a shorter recovery time."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['recovery_wait_secs'] = 0.5
        super(SessionManager, self).__init__(*args, **kwargs)

@contextlib.contextmanager
def _monkey_patch_distributed_training_times():
    if False:
        i = 10
        return i + 15
    'Monkey-patches global attributes with subnetwork-specifics ones.'
    old_delay_secs_per_worker = training_lib._DELAY_SECS_PER_WORKER
    old_session_manager = session_manager_lib.SessionManager
    old_min_max_variable_partitioner = partitioned_variables.min_max_variable_partitioner
    session_manager_lib.SessionManager = SessionManager
    training_lib._DELAY_SECS_PER_WORKER = 0.2

    def patched_min_max_variable_partitioner(max_partitions=1, axis=0, min_slice_size=64, bytes_per_string_element=16):
        if False:
            for i in range(10):
                print('nop')
        del min_slice_size
        return old_min_max_variable_partitioner(max_partitions=max_partitions, axis=axis, min_slice_size=64, bytes_per_string_element=bytes_per_string_element)
    partitioned_variables.min_max_variable_partitioner = patched_min_max_variable_partitioner
    try:
        yield
    finally:
        session_manager_lib.SessionManager = old_session_manager
        training_lib._DELAY_SECS_PER_WORKER = old_delay_secs_per_worker
        partitioned_variables.min_max_variable_partitioner = old_min_max_variable_partitioner

class _DNNBuilder(Builder):
    """A simple DNN subnetwork builder."""

    def __init__(self, name, config, layer_size=3, seed=13):
        if False:
            i = 10
            return i + 15
        self._name = name
        self._layer_size = layer_size
        self._config = config
        self._seed = seed

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def build_subnetwork(self, features, logits_dimension, training, iteration_step, summary, previous_ensemble=None):
        if False:
            i = 10
            return i + 15
        seed = self._seed
        if previous_ensemble:
            seed += 1
        num_ps_replicas = self._config.num_ps_replicas if self._config else 0
        partitioner = tf_compat.v1.min_max_variable_partitioner(max_partitions=num_ps_replicas)
        with tf_compat.v1.variable_scope('dnn', partitioner=partitioner):
            shared = {}
            with tf_compat.v1.variable_scope('hidden_layer'):
                w = tf_compat.v1.get_variable(shape=[2, self._layer_size], initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed), name='weight')
                hidden_layer = tf.matmul(features['x'], w)
            if previous_ensemble:
                other_hidden_layer = previous_ensemble.weighted_subnetworks[-1].subnetwork.shared['hidden_layer']
                hidden_layer = tf.concat([hidden_layer, other_hidden_layer], axis=1)
            hidden_layer = tf.nn.leaky_relu(hidden_layer, alpha=0.2)
            shared['hidden_layer'] = hidden_layer
            with tf_compat.v1.variable_scope('logits'):
                logits = tf_compat.v1.layers.dense(hidden_layer, logits_dimension, kernel_initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed))
            summary.scalar('scalar', 3)
            return Subnetwork(last_layer=logits, logits=logits, complexity=3, shared=shared)

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels, iteration_step, summary, previous_ensemble):
        if False:
            for i in range(10):
                print('nop')
        optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(loss, var_list=var_list)

def train_and_evaluate_estimator():
    if False:
        i = 10
        return i + 15
    'Runs Estimator distributed training.'
    config = tf.estimator.RunConfig(tf_random_seed=42, save_checkpoints_steps=10, save_checkpoints_secs=None, keep_checkpoint_max=None, model_dir=FLAGS.model_dir, session_config=tf_compat.v1.ConfigProto(log_device_placement=False, device_filters=['/job:ps']))

    def input_fn():
        if False:
            while True:
                i = 10
        input_features = {'x': tf.constant(features, name='x')}
        input_labels = tf.constant(labels, name='y')
        return tf.data.Dataset.from_tensors((input_features, input_labels)).repeat()
    kwargs = {'max_iteration_steps': 100, 'force_grow': True, 'delay_secs_per_worker': 0.2, 'max_worker_delay_secs': 1, 'worker_wait_secs': 1, 'worker_wait_timeout_secs': 180, 'evaluator': Evaluator(input_fn, steps=10), 'config': config}
    head = head_lib._regression_head(loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    features = [[1.0, 0.0], [0.0, 0], [0.0, 1.0], [1.0, 1.0]]
    labels = [[1.0], [0.0], [1.0], [0.0]]
    estimator_type = FLAGS.estimator_type
    if FLAGS.placement_strategy == 'round_robin':
        kwargs['experimental_placement_strategy'] = RoundRobinStrategy()
    if estimator_type == 'autoensemble':
        feature_columns = [tf.feature_column.numeric_column('x', shape=[2])]
        candidate_pool = {'linear': tf.estimator.LinearEstimator(head=head, feature_columns=feature_columns, optimizer=lambda : tf_compat.v1.train.AdamOptimizer(learning_rate=0.001)), 'dnn': tf.estimator.DNNEstimator(head=head, feature_columns=feature_columns, optimizer=lambda : tf_compat.v1.train.AdamOptimizer(learning_rate=0.001), hidden_units=[3]), 'dnn2': tf.estimator.DNNEstimator(head=head, feature_columns=feature_columns, optimizer=lambda : tf_compat.v1.train.AdamOptimizer(learning_rate=0.001), hidden_units=[10, 10])}
        estimator = AutoEnsembleEstimator(head=head, candidate_pool=candidate_pool, **kwargs)
    elif estimator_type == 'estimator':
        subnetwork_generator = SimpleGenerator([_DNNBuilder('dnn1', config, layer_size=3), _DNNBuilder('dnn2', config, layer_size=4), _DNNBuilder('dnn3', config, layer_size=5)])
        estimator = Estimator(head=head, subnetwork_generator=subnetwork_generator, **kwargs)
    elif FLAGS.estimator_type == 'autoensemble_trees_multiclass':
        if not bt_losses:
            logging.warning('Skipped autoensemble_trees_multiclass test since contrib is missing.')
            return
        n_classes = 3
        head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(n_classes=n_classes, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

        def tree_loss_fn(labels, logits):
            if False:
                i = 10
                return i + 15
            result = bt_losses.per_example_maxent_loss(labels=labels, logits=logits, num_classes=n_classes, weights=None)
            return result[0]
        tree_head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(loss_fn=tree_loss_fn, n_classes=n_classes, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        labels = [[1], [0], [1], [2]]
        feature_columns = [tf.feature_column.numeric_column('x', shape=[2])]
        candidate_pool = lambda config: {'linear': tf.estimator.LinearEstimator(head=head, feature_columns=feature_columns, optimizer=tf_compat.v1.train.AdamOptimizer(learning_rate=0.001), config=config), 'gbdt': tf.estimator.BoostedTreesEstimator(head=tree_head, feature_columns=feature_columns, n_trees=10, n_batches_per_layer=1, center_bias=False, config=config)}
        estimator = AutoEnsembleEstimator(head=head, candidate_pool=candidate_pool, **kwargs)
    elif estimator_type == 'estimator_with_experimental_multiworker_strategy':

        def _model_fn(features, labels, mode):
            if False:
                while True:
                    i = 10
            'Test model_fn.'
            layer = tf.keras.layers.Dense(1)
            logits = layer(features['x'])
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'logits': logits}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            loss = tf.losses.mean_squared_error(labels=labels, predictions=logits, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(0.2)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        if json.loads(os.environ['TF_CONFIG'])['task']['type'] == 'evaluator':
            distribution = None
        else:
            distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        multiworker_config = tf.estimator.RunConfig(tf_random_seed=42, model_dir=FLAGS.model_dir, train_distribute=distribution, session_config=tf_compat.v1.ConfigProto(log_device_placement=False))
        estimator = tf.estimator.Estimator(model_fn=_model_fn, config=multiworker_config)
    train_hooks = [tf.estimator.ProfilerHook(save_steps=50, output_dir=FLAGS.model_dir)]
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=300, hooks=train_hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=1, start_delay_secs=0.5, throttle_secs=0.05)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    del argv
    with _monkey_patch_distributed_training_times():
        train_and_evaluate_estimator()
if __name__ == '__main__':
    app.run(main)