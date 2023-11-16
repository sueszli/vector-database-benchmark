"""Iterator ops."""
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

def _convert_external_state_policy_to_enum(external_state_policy):
    if False:
        print('Hello World!')
    if isinstance(external_state_policy, options_lib.ExternalStatePolicy):
        return external_state_policy
    if external_state_policy == 'warn':
        return options_lib.ExternalStatePolicy.WARN
    if external_state_policy == 'ignore':
        return options_lib.ExternalStatePolicy.IGNORE
    if external_state_policy == 'fail':
        return options_lib.ExternalStatePolicy.FAIL
    raise ValueError(f"Invalid `ExternalStatePolicy.` Supported values include 'warn', 'ignore', and 'fail.' Received {external_state_policy}.")

@tf_export('data.experimental.make_saveable_from_iterator')
@deprecation.deprecated(None, '`make_saveable_from_iterator` is intended for use in TF1 with `tf.compat.v1.Saver`. In TF2, use `tf.train.Checkpoint` instead.')
def make_saveable_from_iterator(iterator, external_state_policy=None):
    if False:
        i = 10
        return i + 15
    "Returns a SaveableObject for saving/restoring iterator state using Saver.\n\n  Args:\n    iterator: Iterator.\n    external_state_policy: A string that identifies how to handle input\n      pipelines that depend on external state. Possible values are\n      'ignore': The external state is silently ignored.\n      'warn': The external state is ignored, logging a warning.\n      'fail': The operation fails upon encountering external state.\n      By default we set it to 'fail'.\n\n  Returns:\n    A SaveableObject for saving/restoring iterator state using Saver.\n\n  Raises:\n    ValueError: If iterator does not support checkpointing.\n    ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or\n      'fail'.\n\n  For example:\n\n  ```python\n  with tf.Graph().as_default():\n    ds = tf.data.Dataset.range(10)\n    iterator = ds.make_initializable_iterator()\n    # Build the iterator SaveableObject.\n    saveable_obj = tf.data.experimental.make_saveable_from_iterator(iterator)\n    # Add the SaveableObject to the SAVEABLE_OBJECTS collection so\n    # it can be automatically saved using Saver.\n    tf.compat.v1.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)\n    saver = tf.compat.v1.train.Saver()\n\n    while continue_training:\n      ... Perform training ...\n      if should_save_checkpoint:\n        saver.save()\n  ```\n\n  Note: When restoring the iterator, the existing iterator state is completely\n  discarded. This means that any changes you may have made to the Dataset\n  graph will be discarded as well! This includes the new Dataset graph\n  that you may have built during validation. So, while running validation,\n  make sure to run the initializer for the validation input pipeline after\n  restoring the checkpoint.\n\n  Note: Not all iterators support checkpointing yet. Attempting to save the\n  state of an unsupported iterator will throw an error.\n  "
    if external_state_policy is None:
        external_state_policy = 'fail'
    policy_enum = _convert_external_state_policy_to_enum(external_state_policy)
    return iterator_ops._IteratorSaveable(iterator._iterator_resource, iterator._iterator_resource.name, external_state_policy=policy_enum)

@tf_export('data.experimental.CheckpointInputPipelineHook')
class CheckpointInputPipelineHook(session_run_hook.SessionRunHook):
    """Checkpoints input pipeline state every N steps or seconds.

  This hook saves the state of the iterators in the `Graph` so that when
  training is resumed the input pipeline continues from where it left off.
  This could potentially avoid overfitting in certain pipelines where the
  number of training steps per eval are small compared to the dataset
  size or if the training pipeline is pre-empted.

  Differences from `CheckpointSaverHook`:
  1. Saves only the input pipelines in the "iterators" collection and not the
     global variables or other saveable objects.
  2. Does not write the `GraphDef` and `MetaGraphDef` to the summary.

  Example of checkpointing the training pipeline:

  ```python
  est = tf.estimator.Estimator(model_fn)
  while True:
    est.train(
        train_input_fn,
        hooks=[tf.data.experimental.CheckpointInputPipelineHook(est)],
        steps=train_steps_per_eval)
    # Note: We do not pass the hook here.
    metrics = est.evaluate(eval_input_fn)
    if should_stop_the_training(metrics):
      break
  ```

  This hook should be used if the input pipeline state needs to be saved
  separate from the model checkpoint. Doing so may be useful for a few reasons:
  1. The input pipeline checkpoint may be large, if there are large shuffle
     or prefetch buffers for instance, and may bloat the checkpoint size.
  2. If the input pipeline is shared between training and validation, restoring
     the checkpoint during validation may override the validation input
     pipeline.

  For saving the input pipeline checkpoint alongside the model weights use
  `tf.data.experimental.make_saveable_from_iterator` directly to create a
  `SaveableObject` and add to the `SAVEABLE_OBJECTS` collection. Note, however,
  that you will need to be careful not to restore the training iterator during
  eval. You can do that by not adding the iterator to the SAVEABLE_OBJECTS
  collector when building the eval graph.
  """

    def __init__(self, estimator, external_state_policy=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a `CheckpointInputPipelineHook`.\n\n    If the input pipeline depends on external state (e.g. seeds for\n    RandomUniform) beyond the input pipeline, this hook would be unable to\n    serialize and deserialize that state. If its acceptable to ignore that state\n    change the external_state_policy argument to 'warn' or 'ignore'. For e.g.\n\n    ```python\n    est = tf.estimator.Estimator(model_fn)\n    while True:\n      est.train(\n          train_input_fn,\n          hooks=[tf.data.experimental.CheckpointInputPipelineHook(\n              est, external_state_policy='warn')],\n          steps=train_steps_per_eval)\n      # Note: We do not pass the hook here.\n      metrics = est.evaluate(eval_input_fn)\n      if should_stop_the_training(metrics):\n        break\n    ```\n\n    Args:\n      estimator: Estimator.\n      external_state_policy: A string that identifies how to handle input\n        pipelines that depend on external state. Possible values are\n        'ignore': The external state is silently ignored.\n        'warn': The external state is ignored, logging a warning.\n        'fail': The operation fails upon encountering external state.\n        By default we set it to 'fail'.\n\n    Raises:\n      ValueError: One of `save_steps` or `save_secs` should be set.\n      ValueError: At most one of saver or scaffold should be set.\n      ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or\n        'fail'.\n    "
        if external_state_policy is None:
            external_state_policy = 'fail'
        self._external_state_policy = _convert_external_state_policy_to_enum(external_state_policy)
        checkpoint_prefix = 'input'
        if estimator._config.num_worker_replicas > 1:
            suffix = '_{}_{}'.format(estimator._config.task_type, estimator._config.task_id)
            checkpoint_prefix += suffix
        self._checkpoint_saver_hook = basic_session_run_hooks.CheckpointSaverHook(estimator.model_dir, save_secs=estimator._config.save_checkpoints_secs, save_steps=estimator._config.save_checkpoints_steps, checkpoint_basename=checkpoint_prefix + '.ckpt')
        self._latest_filename = 'checkpoint_' + checkpoint_prefix

    def begin(self):
        if False:
            print('Hello World!')
        if self._checkpoint_saver_hook._saver is None and self._checkpoint_saver_hook._scaffold is None:
            iterators = ops.get_collection(iterator_ops.GLOBAL_ITERATORS)
            saveables = [iterator_ops._IteratorSaveable(i, i.name, external_state_policy=self._external_state_policy) for i in iterators]
            self._checkpoint_saver_hook._saver = _CustomSaver(saveables, self._latest_filename, sharded=True)
        self._checkpoint_saver_hook.begin()

    def after_create_session(self, session, coord):
        if False:
            print('Hello World!')
        self._first_run = True

    def _restore_or_save_initial_ckpt(self, session):
        if False:
            i = 10
            return i + 15
        latest_checkpoint_path = checkpoint_management.latest_checkpoint(self._checkpoint_saver_hook._checkpoint_dir, latest_filename=self._latest_filename)
        if latest_checkpoint_path:
            self._checkpoint_saver_hook._get_saver().restore(session, latest_checkpoint_path)
        else:
            global_step = session.run(self._checkpoint_saver_hook._global_step_tensor)
            self._checkpoint_saver_hook._save(session, global_step)
            self._checkpoint_saver_hook._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):
        if False:
            return 10
        if self._first_run:
            self._restore_or_save_initial_ckpt(run_context.session)
            self._first_run = False
        return self._checkpoint_saver_hook.before_run(run_context)

    def after_run(self, run_context, run_values):
        if False:
            i = 10
            return i + 15
        self._checkpoint_saver_hook.after_run(run_context, run_values)

    def end(self, session):
        if False:
            print('Hello World!')
        self._checkpoint_saver_hook.end(session)

class _CustomSaver(saver_lib.Saver):
    """`Saver` with a different default `latest_filename`.

  This is used in the `CheckpointInputPipelineHook` to avoid conflicts with
  the model ckpt saved by the `CheckpointSaverHook`.
  """

    def __init__(self, var_list, latest_filename, sharded=False):
        if False:
            return 10
        super(_CustomSaver, self).__init__(var_list, sharded=sharded)
        self._latest_filename = latest_filename

    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True, strip_default_attrs=False):
        if False:
            print('Hello World!')
        return super(_CustomSaver, self).save(sess, save_path, global_step, latest_filename or self._latest_filename, meta_graph_suffix, write_meta_graph, write_state, strip_default_attrs)