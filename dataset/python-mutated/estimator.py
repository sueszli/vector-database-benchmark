"""Utilities to use Modules with Estimators."""
import os
from absl import logging
import tensorflow as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow_hub import tf_utils
_EXPORT_MODULES_COLLECTION = ('__tfhub_export_modules',)

def register_module_for_export(module, export_name):
    if False:
        i = 10
        return i + 15
    'Register a Module to be exported under `export_name`.\n\n  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format.\n\n  This function registers `module` to be exported by `LatestModuleExporter`\n  under a subdirectory named `export_name`.\n\n  Note that `export_name` must be unique for each module exported from the\n  current graph. It only controls the export subdirectory name and it has\n  no scope effects such as the `name` parameter during Module instantiation.\n\n  THIS FUNCTION IS DEPRECATED.\n\n  Args:\n    module: Module instance to be exported.\n    export_name: subdirectory name to use when performing the export.\n\n  Raises:\n    ValueError: if `export_name` is already taken in the current graph.\n  '
    for (used_name, _) in tf.compat.v1.get_collection(_EXPORT_MODULES_COLLECTION):
        if used_name == export_name:
            raise ValueError('There is already a module registered to be exported as %r' % export_name)
    tf.compat.v1.add_to_collection(_EXPORT_MODULES_COLLECTION, (export_name, module))

class LatestModuleExporter(tf_estimator.Exporter):
    """Regularly exports registered modules into timestamped directories.

  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format.

  Modules can be registered to be exported by this class by calling
  `register_module_for_export` when constructing the graph. The
  `export_name` provided determines the subdirectory name used when
  exporting.

  In addition to exporting, this class also garbage collects older exports.

  Example use with EvalSpec:

  ```python
    train_spec = tf.estimator.TrainSpec(...)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        exporters=[
            hub.LatestModuleExporter("tf_hub", serving_input_fn),
        ])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  ```

  See `LatestModuleExporter.export()` for a direct use example.

  THIS FUNCTION IS DEPRECATED.
  """

    def __init__(self, name, serving_input_fn, exports_to_keep=5):
        if False:
            while True:
                i = 10
        'Creates an `Exporter` to use with `tf.estimator.EvalSpec`.\n\n    Args:\n      name: unique name of this `Exporter`, which will be used in the export\n        path.\n      serving_input_fn: A function with no arguments that returns a\n        ServingInputReceiver. This is used with the `estimator` passed to\n        `export()` to build the graph (in PREDICT mode) that registers the\n        modules for export. The model in that graph is never run, so the actual\n        data provided by this input fn does not matter.\n      exports_to_keep: Number of exports to keep. Older exports will be garbage\n        collected. Defaults to 5. Set to None to disable garbage collection.\n\n    Raises:\n      ValueError: if any argument is invalid.\n    '
        self._name = name
        self._serving_input_fn = serving_input_fn
        self._exports_to_keep = exports_to_keep
        if exports_to_keep is not None and exports_to_keep <= 0:
            raise ValueError('`exports_to_keep`, if provided, must be a positive number')

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def export(self, estimator, export_path, checkpoint_path=None, eval_result=None, is_the_final_export=None):
        if False:
            for i in range(10):
                print('nop')
        'Actually performs the export of registered Modules.\n\n    This method creates a timestamped directory under `export_path`\n    with one sub-directory (named `export_name`) per module registered\n    via `register_module_for_export`.\n\n    Example use:\n\n    ```python\n      estimator = ... (Create estimator with modules registered for export)...\n      exporter = hub.LatestModuleExporter("tf_hub", serving_input_fn)\n      exporter.export(estimator, export_path, estimator.latest_checkpoint())\n    ```\n\n    Args:\n      estimator: the `Estimator` from which to export modules.\n      export_path: A string containing a directory where to write the export\n        timestamped directories.\n      checkpoint_path: The checkpoint path to export. If `None`,\n        `estimator.latest_checkpoint()` is used.\n      eval_result: Unused.\n      is_the_final_export: Unused.\n\n    Returns:\n      The path to the created timestamped directory containing the exported\n      modules.\n    '
        if checkpoint_path is None:
            checkpoint_path = estimator.latest_checkpoint()
        export_dir = tf_utils.get_timestamped_export_dir(export_path)
        temp_export_dir = tf_utils.get_temp_export_dir(export_dir)
        session = _make_estimator_serving_session(estimator, self._serving_input_fn, checkpoint_path)
        with session:
            export_modules = tf.compat.v1.get_collection(_EXPORT_MODULES_COLLECTION)
            if export_modules:
                for (export_name, module) in export_modules:
                    module_export_path = os.path.join(temp_export_dir, tf.compat.as_bytes(export_name))
                    module.export(module_export_path, session)
                tf.compat.v1.gfile.Rename(temp_export_dir, export_dir)
                tf_utils.garbage_collect_exports(export_path, self._exports_to_keep)
                return export_dir
            else:
                logging.warn('LatestModuleExporter found zero modules to export. Use hub.register_module_for_export() if needed.')
                return None

def _make_estimator_serving_session(estimator, serving_input_fn, checkpoint_path):
    if False:
        for i in range(10):
            print('nop')
    'Returns a session constructed using `estimator` and `serving_input_fn`.\n\n  The Estimator API does not provide an API to construct a graph and session,\n  making it necessary for this function to replicate how an estimator builds\n  a graph.\n\n  This code is based on `Estimator.export_savedmodel` (another function that\n  has to replicate how an estimator builds a graph).\n\n  Args:\n    estimator: tf.Estimator to use when constructing the session.\n    serving_input_fn: A function that takes no arguments and returns a\n      `ServingInputReceiver`. It is used to construct the session.\n    checkpoint_path: The checkpoint path to restore in the session. Must not be\n      None.\n  '
    with tf.Graph().as_default() as g:
        mode = tf_estimator.ModeKeys.PREDICT
        tf.compat.v1.train.create_global_step(g)
        tf.compat.v1.set_random_seed(estimator.config.tf_random_seed)
        serving_input_receiver = serving_input_fn()
        estimator_spec = estimator.model_fn(features=serving_input_receiver.features, labels=None, mode=mode, config=estimator.config)
        session = tf.compat.v1.Session(config=estimator._session_config)
        with session.as_default():
            saver_for_restore = estimator_spec.scaffold.saver or tf.compat.v1.train.Saver(sharded=True)
            saver_for_restore.restore(session, checkpoint_path)
        return session