"""Hooks helper to return a list of TensorFlow hooks for training by name.

More hooks can be added to this set. To add a new hook, 1) add the new hook to
the registry in HOOKS, 2) add a corresponding function that parses out necessary
parameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.utils.logs import hooks
from official.utils.logs import logger
from official.utils.logs import metric_hook
_TENSORS_TO_LOG = dict(((x, x) for x in ['learning_rate', 'cross_entropy', 'train_accuracy']))

def get_train_hooks(name_list, use_tpu=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Factory for getting a list of TensorFlow hooks for training by name.\n\n  Args:\n    name_list: a list of strings to name desired hook classes. Allowed:\n      LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook, which are defined\n      as keys in HOOKS\n    use_tpu: Boolean of whether computation occurs on a TPU. This will disable\n      hooks altogether.\n    **kwargs: a dictionary of arguments to the hooks.\n\n  Returns:\n    list of instantiated hooks, ready to be used in a classifier.train call.\n\n  Raises:\n    ValueError: if an unrecognized name is passed.\n  '
    if not name_list:
        return []
    if use_tpu:
        tf.compat.v1.logging.warning('hooks_helper received name_list `{}`, but a TPU is specified. No hooks will be used.'.format(name_list))
        return []
    train_hooks = []
    for name in name_list:
        hook_name = HOOKS.get(name.strip().lower())
        if hook_name is None:
            raise ValueError('Unrecognized training hook requested: {}'.format(name))
        else:
            train_hooks.append(hook_name(**kwargs))
    return train_hooks

def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Function to get LoggingTensorHook.\n\n  Args:\n    every_n_iter: `int`, print the values of `tensors` once every N local\n      steps taken on the current worker.\n    tensors_to_log: List of tensor names or dictionary mapping labels to tensor\n      names. If not set, log _TENSORS_TO_LOG by default.\n    **kwargs: a dictionary of arguments to LoggingTensorHook.\n\n  Returns:\n    Returns a LoggingTensorHook with a standard set of tensors that will be\n    printed to stdout.\n  '
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG
    return tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=every_n_iter)

def get_profiler_hook(model_dir, save_steps=1000, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Function to get ProfilerHook.\n\n  Args:\n    model_dir: The directory to save the profile traces to.\n    save_steps: `int`, print profile traces every N steps.\n    **kwargs: a dictionary of arguments to ProfilerHook.\n\n  Returns:\n    Returns a ProfilerHook that writes out timelines that can be loaded into\n    profiling tools like chrome://tracing.\n  '
    return tf.estimator.ProfilerHook(save_steps=save_steps, output_dir=model_dir)

def get_examples_per_second_hook(every_n_steps=100, batch_size=128, warm_steps=5, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Function to get ExamplesPerSecondHook.\n\n  Args:\n    every_n_steps: `int`, print current and average examples per second every\n      N steps.\n    batch_size: `int`, total batch size used to calculate examples/second from\n      global time.\n    warm_steps: skip this number of steps before logging and running average.\n    **kwargs: a dictionary of arguments to ExamplesPerSecondHook.\n\n  Returns:\n    Returns a ProfilerHook that writes out timelines that can be loaded into\n    profiling tools like chrome://tracing.\n  '
    return hooks.ExamplesPerSecondHook(batch_size=batch_size, every_n_steps=every_n_steps, warm_steps=warm_steps, metric_logger=logger.get_benchmark_logger())

def get_logging_metric_hook(tensors_to_log=None, every_n_secs=600, **kwargs):
    if False:
        while True:
            i = 10
    'Function to get LoggingMetricHook.\n\n  Args:\n    tensors_to_log: List of tensor names or dictionary mapping labels to tensor\n      names. If not set, log _TENSORS_TO_LOG by default.\n    every_n_secs: `int`, the frequency for logging the metric. Default to every\n      10 mins.\n    **kwargs: a dictionary of arguments.\n\n  Returns:\n    Returns a LoggingMetricHook that saves tensor values in a JSON format.\n  '
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG
    return metric_hook.LoggingMetricHook(tensors=tensors_to_log, metric_logger=logger.get_benchmark_logger(), every_n_secs=every_n_secs)

def get_step_counter_hook(**kwargs):
    if False:
        print('Hello World!')
    'Function to get StepCounterHook.'
    del kwargs
    return tf.estimator.StepCounterHook()
HOOKS = {'loggingtensorhook': get_logging_tensor_hook, 'profilerhook': get_profiler_hook, 'examplespersecondhook': get_examples_per_second_hook, 'loggingmetrichook': get_logging_metric_hook, 'stepcounterhook': get_step_counter_hook}