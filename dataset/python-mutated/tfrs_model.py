from turtle import Turtle
import tensorflow as tf
import warnings
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.tasks import base
from bigdl.dllib.utils import log4Error
from typing import Dict

class TFRSModel(tf.keras.Model):

    def __init__(self, tfrs_model: tfrs.Model) -> None:
        if False:
            return 10
        super().__init__()
        log4Error.invalidInputError(isinstance(tfrs_model, tfrs.Model), 'FriesianTFRSModel only support tfrs.Model, but got ' + tfrs_model.__class__.__name__)
        log4Error.invalidInputError(not tfrs_model._is_compiled, 'TFRSModel should be initialized before compiling.')
        attr = tfrs_model.__dict__
        task_dict = dict()
        for (k, v) in attr.items():
            if isinstance(v, base.Task):
                task_dict[k] = v
        for (k, v) in task_dict.items():
            try:
                v._loss.reduction = tf.keras.losses.Reduction.NONE
            except:
                warnings.warn('Model task ' + k + ' has no attribute _loss, please use `tf.keras.losses.Reduction.SUM` or `tf.keras.losses.Reduction.NONE` for loss reduction in this task if the Estimator throw an error.')
        self.model = tfrs_model

    def call(self, features):
        if False:
            while True:
                i = 10
        return self.model.call(features)

    def train_step(self, inputs) -> Dict[str, tf.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Custom train step using the `compute_loss` method.\n\n        Args:\n        inputs: A data structure of tensors: raw inputs to the model. These will\n            usually contain labels and weights as well as features.\n\n        Returns:\n        metrics: A dict of loss tensors of metrics names.\n        '
        with tf.GradientTape() as tape:
            loss = self.model.compute_loss(inputs, training=True)
            loss_rank = loss.shape.rank
            if loss_rank is not None and loss_rank != 0:
                loss = tf.nn.compute_average_loss(loss)
            regularization_loss = tf.cast(tf.nn.scale_regularization_loss(sum(self.model.losses)), tf.float32)
            total_loss = loss + regularization_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics['loss'] = loss
        metrics['regularization_loss'] = regularization_loss
        metrics['total_loss'] = total_loss
        return metrics

    def test_step(self, inputs) -> Dict[str, tf.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Custom test step using the `compute_loss` method.\n\n        Args:\n        inputs: A data structure of tensors: raw inputs to the model. These will\n            usually contain labels and weights as well as features.\n\n        Returns:\n        metrics: A dict of loss tensors of metrics names.\n        '
        loss = self.model.compute_loss(inputs, training=False)
        regularization_loss = sum(self.model.losses)
        total_loss = loss + regularization_loss
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics['loss'] = loss
        metrics['regularization_loss'] = regularization_loss
        metrics['total_loss'] = total_loss
        return metrics