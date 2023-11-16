"""Some common SessionRunHook classes.

Note that the symbols that are exported to v1 tf.train namespace are also
exported to v2 in tf.estimator namespace. See
https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
"""
import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
_HOOKS = 'hooks'
_STEPS_PER_RUN_VAR = 'steps_per_run'

class _HookTimer:
    """Base timer for determining when Hooks should trigger.

  Should not be instantiated directly.
  """

    def __init__(self):
        if False:
            return 10
        pass

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Resets the timer.'
        pass

    def should_trigger_for_step(self, step):
        if False:
            i = 10
            return i + 15
        'Return true if the timer should trigger for the specified step.'
        raise NotImplementedError

    def update_last_triggered_step(self, step):
        if False:
            i = 10
            return i + 15
        'Update the last triggered time and step number.\n\n    Args:\n      step: The current step.\n\n    Returns:\n      A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number\n      of seconds between the current trigger and the last one (a float), and\n      `elapsed_steps` is the number of steps between the current trigger and\n      the last one. Both values will be set to `None` on the first trigger.\n    '
        raise NotImplementedError

    def last_triggered_step(self):
        if False:
            while True:
                i = 10
        'Returns the last triggered time step or None if never triggered.'
        raise NotImplementedError

@tf_export(v1=['train.SecondOrStepTimer'])
class SecondOrStepTimer(_HookTimer):
    """Timer that triggers at most once every N seconds or once every N steps.

  This symbol is also exported to v2 in tf.estimator namespace. See
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
  """

    def __init__(self, every_secs=None, every_steps=None):
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        self._every_secs = every_secs
        self._every_steps = every_steps
        if self._every_secs is None and self._every_steps is None:
            raise ValueError('Either every_secs or every_steps should be provided.')
        if self._every_secs is not None and self._every_steps is not None:
            raise ValueError('Can not provide both every_secs and every_steps.')
        super(SecondOrStepTimer, self).__init__()

    def reset(self):
        if False:
            return 10
        self._last_triggered_step = None
        self._last_triggered_time = None

    def should_trigger_for_step(self, step):
        if False:
            print('Hello World!')
        'Return true if the timer should trigger for the specified step.\n\n    Args:\n      step: Training step to trigger on.\n\n    Returns:\n      True if the difference between the current time and the time of the last\n      trigger exceeds `every_secs`, or if the difference between the current\n      step and the last triggered step exceeds `every_steps`. False otherwise.\n    '
        if self._last_triggered_step is None:
            return True
        if self._last_triggered_step == step:
            return False
        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True
        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True
        return False

    def update_last_triggered_step(self, step):
        if False:
            for i in range(10):
                print('nop')
        current_time = time.time()
        if self._last_triggered_time is None:
            elapsed_secs = None
            elapsed_steps = None
        else:
            elapsed_secs = current_time - self._last_triggered_time
            elapsed_steps = step - self._last_triggered_step
        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return (elapsed_secs, elapsed_steps)

    def last_triggered_step(self):
        if False:
            i = 10
            return i + 15
        return self._last_triggered_step

class NeverTriggerTimer(_HookTimer):
    """Timer that never triggers."""

    def should_trigger_for_step(self, step):
        if False:
            i = 10
            return i + 15
        _ = step
        return False

    def update_last_triggered_step(self, step):
        if False:
            while True:
                i = 10
        _ = step
        return (None, None)

    def last_triggered_step(self):
        if False:
            return 10
        return None

@tf_export(v1=['train.LoggingTensorHook'])
class LoggingTensorHook(session_run_hook.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.

  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:

  ```python
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  ```

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility

  """

    def __init__(self, tensors, every_n_iter=None, every_n_secs=None, at_end=False, formatter=None):
        if False:
            print('Hello World!')
        'Initializes a `LoggingTensorHook`.\n\n    Args:\n      tensors: `dict` that maps string-valued tags to tensors/tensor names, or\n        `iterable` of tensors/tensor names.\n      every_n_iter: `int`, print the values of `tensors` once every N local\n        steps taken on the current worker.\n      every_n_secs: `int` or `float`, print the values of `tensors` once every N\n        seconds. Exactly one of `every_n_iter` and `every_n_secs` should be\n        provided.\n      at_end: `bool` specifying whether to print the values of `tensors` at the\n        end of the run.\n      formatter: function, takes dict of `tag`->`Tensor` and returns a string.\n        If `None` uses default printing all tensors.\n\n    Raises:\n      ValueError: if `every_n_iter` is non-positive.\n    '
        only_log_at_end = at_end and every_n_iter is None and (every_n_secs is None)
        if not only_log_at_end and (every_n_iter is None) == (every_n_secs is None):
            raise ValueError('either at_end and/or exactly one of every_n_iter and every_n_secs must be provided.')
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError('invalid every_n_iter=%s.' % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = sorted(tensors.keys())
        self._tensors = tensors
        self._formatter = formatter
        self._timer = NeverTriggerTimer() if only_log_at_end else SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)
        self._log_at_end = at_end

    def begin(self):
        if False:
            return 10
        self._timer.reset()
        self._iter_count = 0
        self._current_tensors = {tag: _as_graph_element(tensor) for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):
        if False:
            print('Hello World!')
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        if False:
            while True:
                i = 10
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        (elapsed_secs, _) = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append('%s = %s' % (tag, tensor_values[tag]))
            if elapsed_secs is not None:
                logging.info('%s (%.3f sec)', ', '.join(stats), elapsed_secs)
            else:
                logging.info('%s', ', '.join(stats))
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        if False:
            return 10
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)
        self._iter_count += 1

    def end(self, session):
        if False:
            return 10
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)

def get_or_create_steps_per_run_variable():
    if False:
        print('Hello World!')
    'Gets or creates the steps_per_run variable.\n\n  In Estimator, the user provided computation, the model_fn, is wrapped\n  inside a tf.while_loop for peak performance. The iterations of the loop are\n  specified by this variable, which adjusts its value on the CPU after each\n  device program execution and before the next execution.\n\n  The purpose of using a variable, rather than a constant, is to allow\n  Estimator adapt the device training iterations according to the final steps\n  specified by users. For example, if the user sets the steps_per_run as\n  4 and steps as 10 in Estimator.train(), the steps_per_run\n  variable will have the following value before each training run.\n\n      - 1-st execution: steps_per_run = 4\n      - 2-nd execution: steps_per_run = 4\n      - 3-rd execution: steps_per_run = 2\n\n  As model_fn increases the global step once per train_op invocation, the global\n  step is 10 after all executions, matching the steps=10 inputs passed in by\n  users.\n\n  Returns:\n    A TF non-trainable resource variable.\n\n  Raises:\n    RuntimeError: If multi steps_per_run variables were found.\n  '
    graph = ops.get_default_graph()
    collection_name = '{}_{}'.format(_HOOKS, _STEPS_PER_RUN_VAR)
    steps_per_run_vars = graph.get_collection(collection_name)
    if len(steps_per_run_vars) == 1:
        return steps_per_run_vars[0]
    elif len(steps_per_run_vars) > 1:
        raise RuntimeError('Multiple steps_per_run_var in collection.')
    with variable_scope.variable_scope(_HOOKS, reuse=variable_scope.AUTO_REUSE):
        return variable_scope.get_variable(_STEPS_PER_RUN_VAR, initializer=init_ops.ones_initializer(), shape=[], dtype=dtypes.int32, trainable=False, collections=[collection_name, ops.GraphKeys.LOCAL_VARIABLES], use_resource=True)

class _MultiStepStopAtStepHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, num_steps=None, last_step=None, steps_per_run=1):
        if False:
            while True:
                i = 10
        'Initializes a `MultiStepStopAtStepHook`.\n\n    This hook requests stop after either a number of steps have been\n    executed or a last step has been reached. Only one of the two options can be\n    specified.\n\n    if `num_steps` is specified, it indicates the number of steps to execute\n    after `begin()` is called. If instead `last_step` is specified, it\n    indicates the last step we want to execute, as passed to the `after_run()`\n    call.\n\n    In Estimator, the user provided computation, the model_fn, is wrapped\n    inside a tf.while_loop for peak performance. The steps_per_run variable\n    determines the number of iterations of the loop before returning to the CPU.\n\n    Args:\n      num_steps: Number of steps to execute.\n      last_step: Step after which to stop.\n      steps_per_run: Number of steps executed per run call.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        if num_steps is None and last_step is None:
            raise ValueError('One of num_steps or last_step must be specified.')
        if num_steps is not None and last_step is not None:
            raise ValueError('Only one of num_steps or last_step can be specified.')
        if steps_per_run is None or steps_per_run < 1:
            raise ValueError('steps_per_run should be greater than 0')
        self._num_steps = num_steps
        self._last_step = last_step
        self._steps_per_run_initial_value = steps_per_run

    def begin(self):
        if False:
            while True:
                i = 10
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StopAtStepHook.')
        self._steps_per_run_variable = get_or_create_steps_per_run_variable()

    def _update_steps_per_run_variable(self, global_step, session):
        if False:
            i = 10
            return i + 15
        steps = min(self._last_step - global_step, self._steps_per_run_initial_value)
        self._steps_per_run_variable.load(steps, session=session)

    def after_create_session(self, session, coord):
        if False:
            for i in range(10):
                print('nop')
        global_step = session.run(self._global_step_tensor)
        if self._last_step is None:
            self._last_step = global_step + self._num_steps
        self._update_steps_per_run_variable(global_step, session)

    def after_run(self, run_context, run_values):
        if False:
            return 10
        global_step = run_context.session.run(self._global_step_tensor)
        if global_step >= self._last_step:
            run_context.request_stop()
        else:
            self._update_steps_per_run_variable(global_step, run_context.session)

@tf_export(v1=['train.StopAtStepHook'])
class StopAtStepHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility
  """

    def __init__(self, num_steps=None, last_step=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a `StopAtStepHook`.\n\n    This hook requests stop after either a number of steps have been\n    executed or a last step has been reached. Only one of the two options can be\n    specified.\n\n    if `num_steps` is specified, it indicates the number of steps to execute\n    after `begin()` is called. If instead `last_step` is specified, it\n    indicates the last step we want to execute, as passed to the `after_run()`\n    call.\n\n    Args:\n      num_steps: Number of steps to execute.\n      last_step: Step after which to stop.\n\n    Raises:\n      ValueError: If one of the arguments is invalid.\n    '
        if num_steps is None and last_step is None:
            raise ValueError('One of num_steps or last_step must be specified.')
        if num_steps is not None and last_step is not None:
            raise ValueError('Only one of num_steps or last_step can be specified.')
        self._num_steps = num_steps
        self._last_step = last_step

    def begin(self):
        if False:
            while True:
                i = 10
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StopAtStepHook.')

    def after_create_session(self, session, coord):
        if False:
            i = 10
            return i + 15
        if self._last_step is None:
            global_step = session.run(self._global_step_tensor)
            self._last_step = global_step + self._num_steps

    def before_run(self, run_context):
        if False:
            for i in range(10):
                print('nop')
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        if False:
            for i in range(10):
                print('nop')
        global_step = run_values.results + 1
        if global_step >= self._last_step:
            step = run_context.session.run(self._global_step_tensor)
            if step >= self._last_step:
                run_context.request_stop()

@tf_export(v1=['train.CheckpointSaverListener'])
class CheckpointSaverListener:
    """Interface for listeners that take action before or after checkpoint save.

  `CheckpointSaverListener` triggers only in steps when `CheckpointSaverHook` is
  triggered, and provides callbacks at the following points:
   - before using the session
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - at the end of session

  To use a listener, implement a class and pass the listener to a
  `CheckpointSaverHook`, as in this example:

  ```python
  class ExampleCheckpointSaverListener(CheckpointSaverListener):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def before_save(self, session, global_step_value):
      print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
      print('Done writing checkpoint.')
      if decided_to_stop_training():
        return True

    def end(self, session, global_step_value):
      print('Done with the session.')

  ...
  listener = ExampleCheckpointSaverListener()
  saver_hook = tf.estimator.CheckpointSaverHook(
      checkpoint_dir, listeners=[listener])
  with
  tf.compat.v1.train.MonitoredTrainingSession(chief_only_hooks=[saver_hook]):
    ...
  ```

  A `CheckpointSaverListener` may simply take some action after every
  checkpoint save. It is also possible for the listener to use its own schedule
  to act less frequently, e.g. based on global_step_value. In this case,
  implementors should implement the `end()` method to handle actions related to
  the last checkpoint save. But the listener should not act twice if
  `after_save()` already handled this last checkpoint save.

  A `CheckpointSaverListener` can request training to be stopped, by returning
  True in `after_save`. Please note that, in replicated distributed training
  setting, only `chief` should use this behavior. Otherwise each worker will do
  their own evaluation, which may be wasteful of resources.
  """

    def begin(self):
        if False:
            while True:
                i = 10
        pass

    def before_save(self, session, global_step_value):
        if False:
            i = 10
            return i + 15
        pass

    def after_save(self, session, global_step_value):
        if False:
            for i in range(10):
                print('nop')
        pass

    def end(self, session, global_step_value):
        if False:
            i = 10
            return i + 15
        pass

@tf_export(v1=['train.CheckpointSaverHook'])
class CheckpointSaverHook(session_run_hook.SessionRunHook):
    """Saves checkpoints every N steps or seconds."""

    def __init__(self, checkpoint_dir, save_secs=None, save_steps=None, saver=None, checkpoint_basename='model.ckpt', scaffold=None, listeners=None, save_graph_def=True):
        if False:
            return 10
        'Initializes a `CheckpointSaverHook`.\n\n    Args:\n      checkpoint_dir: `str`, base directory for the checkpoint files.\n      save_secs: `int`, save every N secs.\n      save_steps: `int`, save every N steps.\n      saver: `Saver` object, used for saving.\n      checkpoint_basename: `str`, base name for the checkpoint files.\n      scaffold: `Scaffold`, use to get saver object.\n      listeners: List of `CheckpointSaverListener` subclass instances. Used for\n        callbacks that run immediately before or after this hook saves the\n        checkpoint.\n      save_graph_def: Whether to save the GraphDef and MetaGraphDef to\n        `checkpoint_dir`. The GraphDef is saved after the session is created as\n        `graph.pbtxt`. MetaGraphDefs are saved out for every checkpoint as\n        `model.ckpt-*.meta`.\n\n    Raises:\n      ValueError: One of `save_steps` or `save_secs` should be set.\n      ValueError: At most one of `saver` or `scaffold` should be set.\n    '
        logging.info('Create CheckpointSaverHook.')
        if saver is not None and scaffold is not None:
            raise ValueError('You cannot provide both saver and scaffold.')
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._scaffold = scaffold
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)
        self._listeners = listeners or []
        self._steps_per_run = 1000000
        self._save_graph_def = save_graph_def

    def _set_steps_per_run(self, steps_per_run):
        if False:
            print('Hello World!')
        self._steps_per_run = steps_per_run

    def begin(self):
        if False:
            while True:
                i = 10
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use CheckpointSaverHook.')
        for l in self._listeners:
            l.begin()

    def after_create_session(self, session, coord):
        if False:
            print('Hello World!')
        global_step = session.run(self._global_step_tensor)
        if self._save_graph_def:
            training_util.write_graph(ops.get_default_graph().as_graph_def(add_shapes=True), self._checkpoint_dir, 'graph.pbtxt')
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        graph = ops.get_default_graph()
        meta_graph_def = meta_graph.create_meta_graph_def(graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
        self._summary_writer.add_graph(graph)
        self._summary_writer.add_meta_graph(meta_graph_def)
        self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):
        if False:
            return 10
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        if False:
            print('Hello World!')
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        if False:
            i = 10
            return i + 15
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._save(session, last_step)
        for l in self._listeners:
            l.end(session, last_step)

    def _save(self, session, step):
        if False:
            while True:
                i = 10
        'Saves the latest checkpoint, returns should_stop.'
        logging.info('Calling checkpoint listeners before saving checkpoint %d...', step)
        for l in self._listeners:
            l.before_save(session, step)
        logging.info('Saving checkpoints for %d into %s.', step, self._save_path)
        self._get_saver().save(session, self._save_path, global_step=step, write_meta_graph=self._save_graph_def)
        self._summary_writer.add_session_log(SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path), step)
        logging.info('Calling checkpoint listeners after saving checkpoint %d...', step)
        should_stop = False
        for l in self._listeners:
            if l.after_save(session, step):
                logging.info('A CheckpointSaverListener requested that training be stopped. listener: {}'.format(l))
                should_stop = True
        return should_stop

    def _get_saver(self):
        if False:
            return 10
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError('No items in collection {}. Please add a saver to the collection or provide a saver or scaffold.'.format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError('More than one item in collection {}. Please indicate which one to use by passing it to the constructor.'.format(collection_key))
        self._saver = savers[0]
        return savers[0]

@tf_export(v1=['train.StepCounterHook'])
class StepCounterHook(session_run_hook.SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self, every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None):
        if False:
            return 10
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps and every_n_secs should be provided.')
        self._timer = SecondOrStepTimer(every_steps=every_n_steps, every_secs=every_n_secs)
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._steps_per_run = 1

    def _set_steps_per_run(self, steps_per_run):
        if False:
            return 10
        self._steps_per_run = steps_per_run

    def begin(self):
        if False:
            print('Hello World!')
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StepCounterHook.')
        self._summary_tag = training_util.get_global_step().op.name + '/sec'

    def before_run(self, run_context):
        if False:
            while True:
                i = 10
        return SessionRunArgs(self._global_step_tensor)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        if False:
            print('Hello World!')
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
            self._summary_writer.add_summary(summary, global_step)
        logging.info('%s: %g', self._summary_tag, steps_per_sec)

    def after_run(self, run_context, run_values):
        if False:
            print('Hello World!')
        _ = run_context
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                (elapsed_time, elapsed_steps) = self._timer.update_last_triggered_step(global_step)
                if elapsed_time is not None:
                    self._log_and_record(elapsed_steps, elapsed_time, global_step)
        if stale_global_step == self._last_global_step:
            logging.log_first_n(logging.WARN, 'It seems that global step (tf.train.get_global_step) has not been increased. Current value (could be stable): %s vs previous value: %s. You could increase the global step by passing tf.train.get_global_step() to Optimizer.apply_gradients or Optimizer.minimize.', 5, stale_global_step, self._last_global_step)
        self._last_global_step = stale_global_step

@tf_export(v1=['train.NanLossDuringTrainingError'])
class NanLossDuringTrainingError(RuntimeError):

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'NaN loss during training.'

@tf_export(v1=['train.NanTensorHook'])
class NanTensorHook(session_run_hook.SessionRunHook):
    """Monitors the loss tensor and stops training if loss is NaN.

  Can either fail with exception or just stop training.
  """

    def __init__(self, loss_tensor, fail_on_nan_loss=True):
        if False:
            i = 10
            return i + 15
        'Initializes a `NanTensorHook`.\n\n    Args:\n      loss_tensor: `Tensor`, the loss tensor.\n      fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.\n    '
        self._loss_tensor = loss_tensor
        self._fail_on_nan_loss = fail_on_nan_loss

    def before_run(self, run_context):
        if False:
            for i in range(10):
                print('nop')
        return SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        if np.isnan(run_values.results):
            failure_message = 'Model diverged with loss = NaN.'
            if self._fail_on_nan_loss:
                logging.error(failure_message)
                raise NanLossDuringTrainingError
            else:
                logging.warning(failure_message)
                run_context.request_stop()

@tf_export(v1=['train.SummarySaverHook'])
class SummarySaverHook(session_run_hook.SessionRunHook):
    """Saves summaries every N steps."""

    def __init__(self, save_steps=None, save_secs=None, output_dir=None, summary_writer=None, scaffold=None, summary_op=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a `SummarySaverHook`.\n\n    Args:\n      save_steps: `int`, save summaries every N steps. Exactly one of\n        `save_secs` and `save_steps` should be set.\n      save_secs: `int`, save summaries every N seconds.\n      output_dir: `string`, the directory to save the summaries to. Only used if\n        no `summary_writer` is supplied.\n      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,\n        one will be created accordingly.\n      scaffold: `Scaffold` to get summary_op if it's not provided.\n      summary_op: `Tensor` of type `string` containing the serialized `Summary`\n        protocol buffer or a list of `Tensor`. They are most likely an output by\n        TF summary methods like `tf.compat.v1.summary.scalar` or\n        `tf.compat.v1.summary.merge_all`. It can be passed in as one tensor; if\n        more than one, they must be passed in as a list.\n\n    Raises:\n      ValueError: Exactly one of scaffold or summary_op should be set.\n    "
        if scaffold is None and summary_op is None or (scaffold is not None and summary_op is not None):
            raise ValueError('Exactly one of scaffold or summary_op must be provided.')
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._scaffold = scaffold
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        if False:
            while True:
                i = 10
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._next_step = None
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use SummarySaverHook.')

    def before_run(self, run_context):
        if False:
            return 10
        self._request_summary = self._next_step is None or self._timer.should_trigger_for_step(self._next_step)
        requests = {'global_step': self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                requests['summary'] = self._get_summary_op()
        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        _ = run_context
        if not self._summary_writer:
            return
        stale_global_step = run_values.results['global_step']
        global_step = stale_global_step + 1
        if self._next_step is None or self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
        if self._next_step is None:
            self._summary_writer.add_session_log(SessionLog(status=SessionLog.START), global_step)
        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            if 'summary' in run_values.results:
                for summary in run_values.results['summary']:
                    self._summary_writer.add_summary(summary, global_step)
        self._next_step = global_step + 1

    def end(self, session=None):
        if False:
            while True:
                i = 10
        if self._summary_writer:
            self._summary_writer.flush()

    def _get_summary_op(self):
        if False:
            while True:
                i = 10
        'Fetches the summary op either from self._summary_op or self._scaffold.\n\n    Returns:\n      Returns a list of summary `Tensor`.\n    '
        summary_op = None
        if self._summary_op is not None:
            summary_op = self._summary_op
        elif self._scaffold.summary_op is not None:
            summary_op = self._scaffold.summary_op
        if summary_op is None:
            return None
        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op

@tf_export(v1=['train.GlobalStepWaiterHook'])
class GlobalStepWaiterHook(session_run_hook.SessionRunHook):
    """Delays execution until global step reaches `wait_until_step`.

  This hook delays execution until global step reaches to `wait_until_step`. It
  is used to gradually start workers in distributed settings. One example usage
  would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
  task_id=0 is the chief.
  """

    def __init__(self, wait_until_step):
        if False:
            return 10
        'Initializes a `GlobalStepWaiterHook`.\n\n    Args:\n      wait_until_step: an `int` shows until which global step should we wait.\n    '
        self._wait_until_step = wait_until_step

    def begin(self):
        if False:
            i = 10
            return i + 15
        self._worker_is_started = False
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use _GlobalStepWaiterHook.')

    def before_run(self, run_context):
        if False:
            return 10
        if self._worker_is_started:
            return None
        if self._wait_until_step <= 0:
            self._worker_is_started = True
            return None
        logging.info('Waiting for global step %d before starting training.', self._wait_until_step)
        last_logged_step = 0
        while True:
            current_step = run_context.session.run(self._global_step_tensor)
            if current_step >= self._wait_until_step:
                self._worker_is_started = True
                return None
            if current_step - last_logged_step > 1000:
                logging.info('Waiting for global step %d before starting training. Current step is %d.', self._wait_until_step, current_step)
                last_logged_step = current_step
            time.sleep(0.5)

@tf_export(v1=['train.FinalOpsHook'])
class FinalOpsHook(session_run_hook.SessionRunHook):
    """A hook which evaluates `Tensors` at the end of a session."""

    def __init__(self, final_ops, final_ops_feed_dict=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes `FinalOpHook` with ops to run at the end of the session.\n\n    Args:\n      final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names\n        to `Tensors`.\n      final_ops_feed_dict: A feed dictionary to use when running\n        `final_ops_dict`.\n    '
        self._final_ops = final_ops
        self._final_ops_feed_dict = final_ops_feed_dict
        self._final_ops_values = None

    @property
    def final_ops_values(self):
        if False:
            print('Hello World!')
        return self._final_ops_values

    def end(self, session):
        if False:
            for i in range(10):
                print('nop')
        if self._final_ops is not None:
            try:
                self._final_ops_values = session.run(self._final_ops, feed_dict=self._final_ops_feed_dict)
            except (errors.OutOfRangeError, StopIteration) as e:
                logging.warning('An OutOfRangeError or StopIteration exception is raised by the code in FinalOpsHook. This typically means the Ops running by the FinalOpsHook have a dependency back to some input source, which should not happen. For example, for metrics in tf.estimator.Estimator, all metrics functions return two Ops: `value_op` and  `update_op`. Estimator.evaluate calls the `update_op` for each batch of the data in input source and, once it is exhausted, it call the `value_op` to get the metric values. The `value_op` here should have dependency back to variables reading only, rather than reading another batch from input. Otherwise, the `value_op`, executed by `FinalOpsHook`, triggers another data reading, which ends OutOfRangeError/StopIteration. Please fix that.')
                raise e

@tf_export(v1=['train.FeedFnHook'])
class FeedFnHook(session_run_hook.SessionRunHook):
    """Runs `feed_fn` and sets the `feed_dict` accordingly."""

    def __init__(self, feed_fn):
        if False:
            return 10
        'Initializes a `FeedFnHook`.\n\n    Args:\n      feed_fn: function that takes no arguments and returns `dict` of `Tensor`\n        to feed.\n    '
        self.feed_fn = feed_fn

    def before_run(self, run_context):
        if False:
            i = 10
            return i + 15
        return session_run_hook.SessionRunArgs(fetches=None, feed_dict=self.feed_fn())

@tf_export(v1=['train.ProfilerHook'])
class ProfilerHook(session_run_hook.SessionRunHook):
    """Captures CPU/GPU profiling information every N steps or seconds.

  This produces files called "timeline-<step>.json", which are in Chrome
  Trace format.

  For more information see:
  https://github.com/catapult-project/catapult/blob/master/tracing/README.md
  """

    def __init__(self, save_steps=None, save_secs=None, output_dir='', show_dataflow=True, show_memory=False):
        if False:
            while True:
                i = 10
        'Initializes a hook that takes periodic profiling snapshots.\n\n    `options.run_metadata` argument of `tf.Session.Run` is used to collect\n    metadata about execution. This hook sets the metadata and dumps it in Chrome\n    Trace format.\n\n\n    Args:\n      save_steps: `int`, save profile traces every N steps. Exactly one of\n        `save_secs` and `save_steps` should be set.\n      save_secs: `int` or `float`, save profile traces every N seconds.\n      output_dir: `string`, the directory to save the profile traces to.\n        Defaults to the current directory.\n      show_dataflow: `bool`, if True, add flow events to the trace connecting\n        producers and consumers of tensors.\n      show_memory: `bool`, if True, add object snapshot events to the trace\n        showing the sizes and lifetimes of tensors.\n    '
        self._output_file = os.path.join(output_dir, 'timeline-{}.json')
        self._file_writer = SummaryWriterCache.get(output_dir)
        self._show_dataflow = show_dataflow
        self._show_memory = show_memory
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        if False:
            print('Hello World!')
        self._next_step = None
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use ProfilerHook.')

    def before_run(self, run_context):
        if False:
            while True:
                i = 10
        self._request_summary = self._next_step is not None and self._timer.should_trigger_for_step(self._next_step)
        requests = {'global_step': self._global_step_tensor}
        opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE) if self._request_summary else None
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        stale_global_step = run_values.results['global_step']
        if self._next_step is None:
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, self._output_file.format(global_step), run_values.run_metadata.step_stats)
            self._file_writer.add_run_metadata(run_values.run_metadata, 'step_%d' % global_step)
        self._next_step = global_step + 1

    def _save(self, step, save_path, step_stats):
        if False:
            i = 10
            return i + 15
        logging.info("Saving timeline for %d into '%s'.", step, save_path)
        with gfile.Open(save_path, 'w') as f:
            trace = timeline.Timeline(step_stats)
            f.write(trace.generate_chrome_trace_format(show_dataflow=self._show_dataflow, show_memory=self._show_memory))

def _as_graph_element(obj):
    if False:
        for i in range(10):
            print('nop')
    'Retrieves Graph element.'
    graph = ops.get_default_graph()
    if not isinstance(obj, str):
        if not hasattr(obj, 'graph') or obj.graph != graph:
            raise ValueError('Passed %s should have graph attribute that is equal to current graph %s.' % (obj, graph))
        return obj
    if ':' in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ':0')
        try:
            graph.as_graph_element(obj + ':1')
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError('Name %s is ambiguous, as this `Operation` has multiple outputs (at least 2).' % obj)
    return element