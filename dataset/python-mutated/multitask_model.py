"""A multi-task and semi-supervised NLP model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model import encoder
from model import shared_inputs

class Inference(object):

    def __init__(self, config, inputs, pretrained_embeddings, tasks):
        if False:
            for i in range(10):
                print('nop')
        with tf.variable_scope('encoder'):
            self.encoder = encoder.Encoder(config, inputs, pretrained_embeddings)
        self.modules = {}
        for task in tasks:
            with tf.variable_scope(task.name):
                self.modules[task.name] = task.get_module(inputs, self.encoder)

class Model(object):

    def __init__(self, config, pretrained_embeddings, tasks):
        if False:
            while True:
                i = 10
        self._config = config
        self._tasks = tasks
        (self._global_step, self._optimizer) = self._get_optimizer()
        self._inputs = shared_inputs.Inputs(config)
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            inference = Inference(config, self._inputs, pretrained_embeddings, tasks)
            self._trainer = inference
            self._tester = inference
            self._teacher = inference
            if config.ema_test or config.ema_teacher:
                ema = tf.train.ExponentialMovingAverage(config.ema_decay)
                model_vars = tf.get_collection('trainable_variables', 'model')
                ema_op = ema.apply(model_vars)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

                def ema_getter(getter, name, *args, **kwargs):
                    if False:
                        return 10
                    var = getter(name, *args, **kwargs)
                    return ema.average(var)
                scope.set_custom_getter(ema_getter)
                inference_ema = Inference(config, self._inputs, pretrained_embeddings, tasks)
                if config.ema_teacher:
                    self._teacher = inference_ema
                if config.ema_test:
                    self._tester = inference_ema
        self._unlabeled_loss = self._get_consistency_loss(tasks)
        self._unlabeled_train_op = self._get_train_op(self._unlabeled_loss)
        self._labeled_train_ops = {}
        for task in self._tasks:
            task_loss = self._trainer.modules[task.name].supervised_loss
            self._labeled_train_ops[task.name] = self._get_train_op(task_loss)

    def _get_consistency_loss(self, tasks):
        if False:
            for i in range(10):
                print('nop')
        return sum([self._trainer.modules[task.name].unsupervised_loss for task in tasks])

    def _get_optimizer(self):
        if False:
            print('Hello World!')
        global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        warm_up_multiplier = tf.minimum(tf.to_float(global_step), self._config.warm_up_steps) / self._config.warm_up_steps
        decay_multiplier = 1.0 / (1 + self._config.lr_decay * tf.sqrt(tf.to_float(global_step)))
        lr = self._config.lr * warm_up_multiplier * decay_multiplier
        optimizer = tf.train.MomentumOptimizer(lr, self._config.momentum)
        return (global_step, optimizer)

    def _get_train_op(self, loss):
        if False:
            while True:
                i = 10
        (grads, vs) = zip(*self._optimizer.compute_gradients(loss))
        (grads, _) = tf.clip_by_global_norm(grads, self._config.grad_clip)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self._optimizer.apply_gradients(zip(grads, vs), global_step=self._global_step)

    def _create_feed_dict(self, mb, model, is_training=True):
        if False:
            while True:
                i = 10
        feed = self._inputs.create_feed_dict(mb, is_training)
        if mb.task_name in model.modules:
            model.modules[mb.task_name].update_feed_dict(feed, mb)
        else:
            for module in model.modules.values():
                module.update_feed_dict(feed, mb)
        return feed

    def train_unlabeled(self, sess, mb):
        if False:
            i = 10
            return i + 15
        return sess.run([self._unlabeled_train_op, self._unlabeled_loss], feed_dict=self._create_feed_dict(mb, self._trainer))[1]

    def train_labeled(self, sess, mb):
        if False:
            while True:
                i = 10
        return sess.run([self._labeled_train_ops[mb.task_name], self._trainer.modules[mb.task_name].supervised_loss], feed_dict=self._create_feed_dict(mb, self._trainer))[1]

    def run_teacher(self, sess, mb):
        if False:
            for i in range(10):
                print('nop')
        result = sess.run({task.name: self._teacher.modules[task.name].probs for task in self._tasks}, feed_dict=self._create_feed_dict(mb, self._teacher, False))
        for (task_name, probs) in result.iteritems():
            mb.teacher_predictions[task_name] = probs.astype('float16')

    def test(self, sess, mb):
        if False:
            print('Hello World!')
        return sess.run([self._tester.modules[mb.task_name].supervised_loss, self._tester.modules[mb.task_name].preds], feed_dict=self._create_feed_dict(mb, self._tester, False))

    def get_global_step(self, sess):
        if False:
            return 10
        return sess.run(self._global_step)