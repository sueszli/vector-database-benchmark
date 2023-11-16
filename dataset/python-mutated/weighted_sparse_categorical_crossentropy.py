"""Sparse categorical cross-entropy losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def _adjust_labels(labels, predictions):
    if False:
        i = 10
        return i + 15
    "Adjust the 'labels' tensor by squeezing it if needed."
    labels = tf.cast(labels, tf.int32)
    if len(predictions.shape) == len(labels.shape):
        labels = tf.squeeze(labels, [-1])
    return (labels, predictions)

def _validate_rank(labels, predictions, weights):
    if False:
        return 10
    if weights is not None and len(weights.shape) != len(labels.shape):
        raise RuntimeError('Weight and label tensors were not of the same rank. weights.shape was %s, and labels.shape was %s.' % (predictions.shape, labels.shape))
    if len(predictions.shape) - 1 != len(labels.shape):
        raise RuntimeError('Weighted sparse categorical crossentropy expects `labels` to have a rank of one less than `predictions`. labels.shape was %s, and predictions.shape was %s.' % (labels.shape, predictions.shape))

def per_example_loss(labels, predictions, weights=None):
    if False:
        while True:
            i = 10
    "Calculate a per-example sparse categorical crossentropy loss.\n\n  This loss function assumes that the predictions are post-softmax.\n  Args:\n    labels: The labels to evaluate against. Should be a set of integer indices\n      ranging from 0 to (vocab_size-1).\n    predictions: The network predictions. Should have softmax already applied.\n    weights: An optional weight array of the same shape as the 'labels' array.\n      If None, all examples will be used.\n\n  Returns:\n    A tensor of shape predictions.shape[:-1] containing the per-example\n      loss.\n  "
    (labels, predictions) = _adjust_labels(labels, predictions)
    _validate_rank(labels, predictions, weights)
    labels_one_hot = tf.keras.backend.one_hot(labels, predictions.shape[-1])
    labels_one_hot = tf.keras.backend.cast(labels_one_hot, predictions.dtype)
    per_example_loss_data = -tf.keras.backend.sum(predictions * labels_one_hot, axis=[-1])
    if weights is not None:
        weights = tf.keras.backend.cast(weights, per_example_loss_data.dtype)
        per_example_loss_data = weights * per_example_loss_data
    return per_example_loss_data

def loss(labels, predictions, weights=None):
    if False:
        while True:
            i = 10
    "Calculate a per-batch sparse categorical crossentropy loss.\n\n  This loss function assumes that the predictions are post-softmax.\n  Args:\n    labels: The labels to evaluate against. Should be a set of integer indices\n      ranging from 0 to (vocab_size-1).\n    predictions: The network predictions. Should have softmax already applied.\n    weights: An optional weight array of the same shape as the 'labels' array.\n      If None, all examples will be used.\n\n  Returns:\n    A loss scalar.\n\n  Raises:\n    RuntimeError if the passed tensors do not have the same rank.\n  "
    (labels, predictions) = _adjust_labels(labels, predictions)
    _validate_rank(labels, predictions, weights)
    per_example_loss_data = per_example_loss(labels, predictions, weights)
    if weights is None:
        return tf.keras.backend.mean(per_example_loss_data)
    else:
        numerator = tf.keras.backend.sum(per_example_loss_data)
        weights = tf.keras.backend.cast(weights, predictions.dtype)
        denominator = tf.keras.backend.sum(weights) + 1e-05
        return numerator / denominator