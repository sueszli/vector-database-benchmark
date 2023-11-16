"""Ensembler definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six

class TrainOpSpec(collections.namedtuple('TrainOpSpec', ['train_op', 'chief_hooks', 'hooks'])):
    """A data structure for specifying ensembler training operations.

  Args:
    train_op: Op for the training step.
    chief_hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on
      the chief worker during training.
    hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on all
      workers during training.

  Returns:
    An :class:`adanet.ensemble.TrainOpSpec` object.
  """

    def __new__(cls, train_op, chief_hooks=None, hooks=None):
        if False:
            print('Hello World!')
        chief_hooks = tuple(chief_hooks) if chief_hooks else ()
        hooks = tuple(hooks) if hooks else ()
        return super(TrainOpSpec, cls).__new__(cls, train_op, chief_hooks, hooks)

@six.add_metaclass(abc.ABCMeta)
class Ensemble(object):
    """An abstract ensemble of subnetworks."""

    @abc.abstractproperty
    def logits(self):
        if False:
            i = 10
            return i + 15
        'Ensemble logits :class:`tf.Tensor`.'

    @abc.abstractproperty
    def subnetworks(self):
        if False:
            return 10
        "Returns an ordered :class:`Iterable` of the ensemble's subnetworks."

    @property
    def predictions(self):
        if False:
            return 10
        'Optional dict of Ensemble predictions to be merged in EstimatorSpec.\n\n    These will be additional (over the default included by the head) predictions\n    which will be included in the EstimatorSpec in `predictions` and\n    `export_outputs` (wrapped as PredictOutput).\n    '
        return None

@six.add_metaclass(abc.ABCMeta)
class Ensembler(object):
    """An abstract ensembler."""

    @abc.abstractproperty
    def name(self):
        if False:
            while True:
                i = 10
        "This ensembler's unique string name."

    @abc.abstractmethod
    def build_ensemble(self, subnetworks, previous_ensemble_subnetworks, features, labels, logits_dimension, training, iteration_step, summary, previous_ensemble, previous_iteration_checkpoint):
        if False:
            while True:
                i = 10
        'Builds an ensemble of subnetworks.\n\n    Accessing the global step via :meth:`tf.train.get_or_create_global_step()`\n    or :meth:`tf.train.get_global_step()` within this scope will return an\n    incrementable iteration step since the beginning of the iteration.\n\n    Args:\n      subnetworks: Ordered iterable of :class:`adanet.subnetwork.Subnetwork`\n        instances to ensemble. Must have at least one element.\n      previous_ensemble_subnetworks: Ordered iterable of\n        :class:`adanet.subnetwork.Subnetwork` instances present in previous\n        ensemble to be used. The subnetworks from previous_ensemble not\n        included in this list should be pruned. Can be set to None or empty.\n      features: Input :code:`dict` of :class:`tf.Tensor` objects.\n      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to\n        :class:`tf.Tensor` (for multi-head). Can be :code:`None`.\n      logits_dimension: Size of the last dimension of the logits\n        :class:`tf.Tensor`. Typically, logits have for shape `[batch_size,\n        logits_dimension]`.\n      training: A python boolean indicating whether the graph is in training\n        mode or prediction mode.\n      iteration_step: Integer :class:`tf.Tensor` representing the step since the\n        beginning of the current iteration, as opposed to the global step.\n      summary: An :class:`adanet.Summary` for scoping summaries to individual\n        ensembles in Tensorboard. Using :meth:`tf.summary` within this scope\n        will use this :class:`adanet.Summary` under the hood.\n      previous_ensemble: The best :class:`adanet.Ensemble` from iteration *t-1*.\n        The created subnetwork will extend the previous ensemble to form the\n        :class:`adanet.Ensemble` at iteration *t*.\n      previous_iteration_checkpoint: The `tf.train.Checkpoint` object associated\n        with the previous iteration.\n\n    Returns:\n      An :class:`adanet.ensemble.Ensemble` subclass instance.\n    '

    @abc.abstractmethod
    def build_train_op(self, ensemble, loss, var_list, labels, iteration_step, summary, previous_ensemble):
        if False:
            while True:
                i = 10
        "Returns an op for training an ensemble.\n\n    Accessing the global step via :meth:`tf.train.get_or_create_global_step`\n    or :meth:`tf.train.get_global_step` within this scope will return an\n    incrementable iteration step since the beginning of the iteration.\n\n    Args:\n      ensemble: The :class:`adanet.ensemble.Ensemble` subclass instance returned\n        by this instance's :meth:`build_ensemble`.\n      loss: A :class:`tf.Tensor` containing the ensemble's loss to minimize.\n      var_list: List of ensemble :class:`tf.Variable` parameters to update as\n        part of the training operation.\n      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to\n        :class:`tf.Tensor` (for multi-head).\n      iteration_step: Integer :class:`tf.Tensor` representing the step since the\n        beginning of the current iteration, as opposed to the global step.\n      summary: An :class:`adanet.Summary` for scoping summaries to individual\n        ensembles in Tensorboard. Using :code:`tf.summary` within this scope\n        will use this :class:`adanet.Summary` under the hood.\n      previous_ensemble: The best :class:`adanet.ensemble.Ensemble` from the\n        previous iteration.\n    Returns:\n      Either a train op or an :class:`adanet.ensemble.TrainOpSpec`.\n    "