"""Base minibatch sampler module.

The job of the minibatch_sampler is to subsample a minibatch based on some
criterion.

The main function call is:
    subsample(indicator, batch_size, **params).
Indicator is a 1d boolean tensor where True denotes which examples can be
sampled. It returns a boolean indicator where True denotes an example has been
sampled..

Subclasses should implement the Subsample function and can make use of the
@staticmethod SubsampleIndicator.

This is originally implemented in TensorFlow Object Detection API.
"""
from abc import ABCMeta
from abc import abstractmethod
import tensorflow.compat.v2 as tf
from official.vision.detection.utils.object_detection import ops

class MinibatchSampler(object):
    """Abstract base class for subsampling minibatches."""
    __metaclass__ = ABCMeta

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Constructs a minibatch sampler.'
        pass

    @abstractmethod
    def subsample(self, indicator, batch_size, **params):
        if False:
            i = 10
            return i + 15
        'Returns subsample of entries in indicator.\n\n    Args:\n      indicator: boolean tensor of shape [N] whose True entries can be sampled.\n      batch_size: desired batch size.\n      **params: additional keyword arguments for specific implementations of\n          the MinibatchSampler.\n\n    Returns:\n      sample_indicator: boolean tensor of shape [N] whose True entries have been\n      sampled. If sum(indicator) >= batch_size, sum(is_sampled) = batch_size\n    '
        pass

    @staticmethod
    def subsample_indicator(indicator, num_samples):
        if False:
            print('Hello World!')
        'Subsample indicator vector.\n\n    Given a boolean indicator vector with M elements set to `True`, the function\n    assigns all but `num_samples` of these previously `True` elements to\n    `False`. If `num_samples` is greater than M, the original indicator vector\n    is returned.\n\n    Args:\n      indicator: a 1-dimensional boolean tensor indicating which elements\n        are allowed to be sampled and which are not.\n      num_samples: int32 scalar tensor\n\n    Returns:\n      a boolean tensor with the same shape as input (indicator) tensor\n    '
        indices = tf.where(indicator)
        indices = tf.random.shuffle(indices)
        indices = tf.reshape(indices, [-1])
        num_samples = tf.minimum(tf.size(input=indices), num_samples)
        selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))
        selected_indicator = ops.indices_to_dense_vector(selected_indices, tf.shape(input=indicator)[0])
        return tf.equal(selected_indicator, 1)