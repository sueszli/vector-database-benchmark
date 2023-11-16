"""A light weight utilities to train NLP models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from absl import logging
import tensorflow as tf
from official.utils.misc import distribution_utils
from official.utils.misc import tpu_lib
_SUMMARY_TXT = 'training_summary.txt'
_MIN_SUMMARY_STEPS = 10

def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    if False:
        while True:
            i = 10
    'Saves model to with provided checkpoint prefix.'
    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    logging.info('Saving model as TF checkpoint: %s', saved_path)
    return

def _get_input_iterator(input_fn, strategy):
    if False:
        print('Hello World!')
    'Returns distributed dataset iterator.'
    if not callable(input_fn):
        raise ValueError('`input_fn` should be a closure that returns a dataset.')
    iterator = iter(strategy.experimental_distribute_datasets_from_function(input_fn))
    return iterator

def _float_metric_value(metric):
    if False:
        return 10
    'Gets the value of a float-value keras metric.'
    return metric.result().numpy().astype(float)

def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    if False:
        return 10
    'Calculates steps to run on device.'
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
        return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop

def write_txt_summary(training_summary, summary_dir):
    if False:
        while True:
            i = 10
    'Writes a summary text file to record stats.'
    summary_path = os.path.join(summary_dir, _SUMMARY_TXT)
    with tf.io.gfile.GFile(summary_path, 'wb') as f:
        logging.info('Training Summary: \n%s', str(training_summary))
        f.write(json.dumps(training_summary, indent=4))

def run_customized_training_loop(_sentinel=None, strategy=None, model_fn=None, loss_fn=None, model_dir=None, train_input_fn=None, steps_per_epoch=None, steps_per_loop=1, epochs=1, eval_input_fn=None, eval_steps=None, metric_fn=None, init_checkpoint=None, custom_callbacks=None, run_eagerly=False, sub_model_export_name=None):
    if False:
        print('Hello World!')
    "Run BERT pretrain model training using low-level API.\n\n  Arguments:\n      _sentinel: Used to prevent positional parameters. Internal, do not use.\n      strategy: Distribution strategy on which to run low level training loop.\n      model_fn: Function that returns a tuple (model, sub_model). Caller of this\n        function should add optimizer to the `model` via calling\n        `model.compile()` API or manually setting `model.optimizer` attribute.\n        Second element of the returned tuple(sub_model) is an optional sub model\n        to be used for initial checkpoint -- if provided.\n      loss_fn: Function with signature func(labels, logits) and returns a loss\n        tensor.\n      model_dir: Model directory used during training for restoring/saving model\n        weights.\n      train_input_fn: Function that returns a tf.data.Dataset used for training.\n      steps_per_epoch: Number of steps to run per epoch. At the end of each\n        epoch, model checkpoint will be saved and evaluation will be conducted\n        if evaluation dataset is provided.\n      steps_per_loop: Number of steps per graph-mode loop. In order to reduce\n        communication in eager context, training logs are printed every\n        steps_per_loop.\n      epochs: Number of epochs to train.\n      eval_input_fn: Function that returns evaluation dataset. If none,\n        evaluation is skipped.\n      eval_steps: Number of steps to run evaluation. Required if `eval_input_fn`\n        is not none.\n      metric_fn: A metrics function that returns a Keras Metric object to record\n        evaluation result using evaluation dataset or with training dataset\n        after every epoch.\n      init_checkpoint: Optional checkpoint to load to `sub_model` returned by\n        `model_fn`.\n      custom_callbacks: A list of Keras Callbacks objects to run during\n        training. More specifically, `on_batch_begin()`, `on_batch_end()`,\n        methods are invoked during training.\n      run_eagerly: Whether to run model training in pure eager execution. This\n        should be disable for TPUStrategy.\n      sub_model_export_name: If not None, will export `sub_model` returned by\n        `model_fn` into checkpoint files. The name of intermediate checkpoint\n        file is {sub_model_export_name}_step_{step}.ckpt and the last\n        checkpint's name is {sub_model_export_name}.ckpt;\n        if None, `sub_model` will not be exported as checkpoint.\n\n  Returns:\n      Trained model.\n\n  Raises:\n      ValueError: (1) When model returned by `model_fn` does not have optimizer\n        attribute or when required parameters are set to none. (2) eval args are\n        not specified correctly. (3) metric_fn must be a callable if specified.\n        (4) sub_model_checkpoint_name is specified, but `sub_model` returned\n        by `model_fn` is None.\n  "
    if _sentinel is not None:
        raise ValueError('only call `run_customized_training_loop()` with named arguments.')
    required_arguments = [strategy, model_fn, loss_fn, model_dir, steps_per_epoch, train_input_fn]
    if [arg for arg in required_arguments if arg is None]:
        raise ValueError('`strategy`, `model_fn`, `loss_fn`, `model_dir`, `steps_per_loop` and `steps_per_epoch` are required parameters.')
    if steps_per_loop > steps_per_epoch:
        logging.error('steps_per_loop: %d is specified to be greater than  steps_per_epoch: %d, we will use steps_per_epoch as steps_per_loop.', steps_per_loop, steps_per_epoch)
        steps_per_loop = steps_per_epoch
    assert tf.executing_eagerly()
    if run_eagerly:
        if steps_per_loop > 1:
            raise ValueError('steps_per_loop is used for performance optimization. When you want to run eagerly, you cannot leverage graph mode loop.')
        if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
            raise ValueError('TPUStrategy should not run eagerly as it heavily replies on graph optimization for the distributed system.')
    if eval_input_fn and (eval_steps is None or metric_fn is None):
        raise ValueError('`eval_step` and `metric_fn` are required when `eval_input_fn ` is not none.')
    if metric_fn and (not callable(metric_fn)):
        raise ValueError('if `metric_fn` is specified, metric_fn must be a callable.')
    total_training_steps = steps_per_epoch * epochs
    train_iterator = _get_input_iterator(train_input_fn, strategy)
    with distribution_utils.get_strategy_scope(strategy):
        (model, sub_model) = model_fn()
        if not hasattr(model, 'optimizer'):
            raise ValueError('User should set optimizer attribute to model inside `model_fn`.')
        if sub_model_export_name and sub_model is None:
            raise ValueError('sub_model_export_name is specified as %s, but sub_model is None.' % sub_model_export_name)
        optimizer = model.optimizer
        use_float16 = isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)
        if init_checkpoint:
            logging.info('Checkpoint file %s found and restoring from initial checkpoint for core model.', init_checkpoint)
            checkpoint = tf.train.Checkpoint(model=sub_model)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
            logging.info('Loading from checkpoint file completed')
        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        eval_metrics = [metric_fn()] if metric_fn else []
        train_metrics = [metric.__class__.from_config(metric.get_config()) for metric in eval_metrics]
        summary_dir = os.path.join(model_dir, 'summaries')
        eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
        if steps_per_loop >= _MIN_SUMMARY_STEPS:
            train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))
        else:
            train_summary_writer = None
        training_vars = model.trainable_variables

        def _replicated_step(inputs):
            if False:
                while True:
                    i = 10
            'Replicated training step.'
            (inputs, labels) = inputs
            with tf.GradientTape() as tape:
                model_outputs = model(inputs, training=True)
                loss = loss_fn(labels, model_outputs)
                if use_float16:
                    scaled_loss = optimizer.get_scaled_loss(loss)
            if use_float16:
                scaled_grads = tape.gradient(scaled_loss, training_vars)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, training_vars)
            optimizer.apply_gradients(zip(grads, training_vars))
            train_loss_metric.update_state(loss)
            for metric in train_metrics:
                metric.update_state(labels, model_outputs)

        @tf.function
        def train_steps(iterator, steps):
            if False:
                print('Hello World!')
            'Performs distributed training steps in a loop.\n\n      Args:\n        iterator: the distributed iterator of training datasets.\n        steps: an tf.int32 integer tensor to specify number of steps to run\n          inside host training loop.\n\n      Raises:\n        ValueError: Any of the arguments or tensor shapes are invalid.\n      '
            if not isinstance(steps, tf.Tensor):
                raise ValueError('steps should be an Tensor. Python object may cause retracing.')
            for _ in tf.range(steps):
                strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

        def train_single_step(iterator):
            if False:
                i = 10
                return i + 15
            'Performs a distributed training step.\n\n      Args:\n        iterator: the distributed iterator of training datasets.\n\n      Raises:\n        ValueError: Any of the arguments or tensor shapes are invalid.\n      '
            strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

        def test_step(iterator):
            if False:
                while True:
                    i = 10
            'Calculates evaluation metrics on distributed devices.'

            def _test_step_fn(inputs):
                if False:
                    print('Hello World!')
                'Replicated accuracy calculation.'
                (inputs, labels) = inputs
                model_outputs = model(inputs, training=False)
                for metric in eval_metrics:
                    metric.update_state(labels, model_outputs)
            strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))
        if not run_eagerly:
            train_single_step = tf.function(train_single_step)
            test_step = tf.function(test_step)

        def _run_evaluation(current_training_step, test_iterator):
            if False:
                return 10
            'Runs validation steps and aggregate metrics.'
            for _ in range(eval_steps):
                test_step(test_iterator)
            with eval_summary_writer.as_default():
                for metric in eval_metrics + model.metrics:
                    metric_value = _float_metric_value(metric)
                    logging.info('Step: [%d] Validation %s = %f', current_training_step, metric.name, metric_value)
                    tf.summary.scalar(metric.name, metric_value, step=current_training_step)
                eval_summary_writer.flush()

        def _run_callbacks_on_batch_begin(batch):
            if False:
                i = 10
                return i + 15
            'Runs custom callbacks at the start of every step.'
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_begin(batch)

        def _run_callbacks_on_batch_end(batch):
            if False:
                i = 10
                return i + 15
            'Runs custom callbacks at the end of every step.'
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_end(batch)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        sub_model_checkpoint = tf.train.Checkpoint(model=sub_model) if sub_model_export_name else None
        latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint_file:
            logging.info('Checkpoint file %s found and restoring from checkpoint', latest_checkpoint_file)
            checkpoint.restore(latest_checkpoint_file)
            logging.info('Loading from checkpoint file completed')
        current_step = optimizer.iterations.numpy()
        checkpoint_name = 'ctl_step_{step}.ckpt'
        while current_step < total_training_steps:
            train_loss_metric.reset_states()
            for metric in train_metrics + model.metrics:
                metric.reset_states()
            _run_callbacks_on_batch_begin(current_step)
            steps = steps_to_run(current_step, steps_per_epoch, steps_per_loop)
            if steps == 1:
                train_single_step(train_iterator)
            else:
                train_steps(train_iterator, tf.convert_to_tensor(steps, dtype=tf.int32))
            _run_callbacks_on_batch_end(current_step)
            current_step += steps
            train_loss = _float_metric_value(train_loss_metric)
            training_status = 'Train Step: %d/%d  / loss = %s' % (current_step, total_training_steps, train_loss)
            if train_summary_writer:
                with train_summary_writer.as_default():
                    tf.summary.scalar(train_loss_metric.name, train_loss, step=current_step)
                    for metric in train_metrics + model.metrics:
                        metric_value = _float_metric_value(metric)
                        training_status += '  %s = %f' % (metric.name, metric_value)
                        tf.summary.scalar(metric.name, metric_value, step=current_step)
                    train_summary_writer.flush()
            logging.info(training_status)
            if current_step % steps_per_epoch == 0:
                if current_step < total_training_steps:
                    _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))
                    if sub_model_export_name:
                        _save_checkpoint(sub_model_checkpoint, model_dir, '%s_step_%d.ckpt' % (sub_model_export_name, current_step))
                if eval_input_fn:
                    logging.info('Running evaluation after step: %s.', current_step)
                    _run_evaluation(current_step, _get_input_iterator(eval_input_fn, strategy))
                    for metric in eval_metrics + model.metrics:
                        metric.reset_states()
        _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))
        if sub_model_export_name:
            _save_checkpoint(sub_model_checkpoint, model_dir, '%s.ckpt' % sub_model_export_name)
        if eval_input_fn:
            logging.info('Running final evaluation after training is complete.')
            _run_evaluation(current_step, _get_input_iterator(eval_input_fn, strategy))
        training_summary = {'total_training_steps': total_training_steps, 'train_loss': _float_metric_value(train_loss_metric)}
        if eval_metrics:
            training_summary['last_train_metrics'] = _float_metric_value(train_metrics[0])
            training_summary['eval_metrics'] = _float_metric_value(eval_metrics[0])
        write_txt_summary(training_summary, summary_dir)
        return model