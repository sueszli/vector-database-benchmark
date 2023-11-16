"""Hyperparameters for the object detection model in TF.learn.

This file consolidates and documents the hyperparameters used by the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def create_hparams(hparams_overrides=None):
    if False:
        while True:
            i = 10
    'Returns hyperparameters, including any flag value overrides.\n\n  Args:\n    hparams_overrides: Optional hparams overrides, represented as a\n      string containing comma-separated hparam_name=value pairs.\n\n  Returns:\n    The hyperparameters as a tf.HParams object.\n  '
    hparams = tf.contrib.training.HParams(load_pretrained=True)
    if hparams_overrides:
        hparams = hparams.parse(hparams_overrides)
    return hparams