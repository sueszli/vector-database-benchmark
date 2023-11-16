"""Deprecated experimental Keras SavedModel implementation."""
import os
import warnings
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving import utils_v1 as model_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import compat
from tensorflow.python.util import nest
metrics_lib = LazyLoader('metrics_lib', globals(), 'tensorflow.python.keras.metrics')
models_lib = LazyLoader('models_lib', globals(), 'tensorflow.python.keras.models')
sequential = LazyLoader('sequential', globals(), 'tensorflow.python.keras.engine.sequential')
SAVED_MODEL_FILENAME_JSON = 'saved_model.json'

def export_saved_model(model, saved_model_path, custom_objects=None, as_text=False, input_signature=None, serving_only=False):
    if False:
        print('Hello World!')
    "Exports a `tf.keras.Model` as a Tensorflow SavedModel.\n\n  Note that at this time, subclassed models can only be saved using\n  `serving_only=True`.\n\n  The exported `SavedModel` is a standalone serialization of Tensorflow objects,\n  and is supported by TF language APIs and the Tensorflow Serving system.\n  To load the model, use the function\n  `tf.keras.experimental.load_from_saved_model`.\n\n  The `SavedModel` contains:\n\n  1. a checkpoint containing the model weights.\n  2. a `SavedModel` proto containing the Tensorflow backend graph. Separate\n     graphs are saved for prediction (serving), train, and evaluation. If\n     the model has not been compiled, then only the graph computing predictions\n     will be exported.\n  3. the model's json config. If the model is subclassed, this will only be\n     included if the model's `get_config()` method is overwritten.\n\n  Example:\n\n  ```python\n  import tensorflow as tf\n\n  # Create a tf.keras model.\n  model = tf.keras.Sequential()\n  model.add(tf.keras.layers.Dense(1, input_shape=[10]))\n  model.summary()\n\n  # Save the tf.keras model in the SavedModel format.\n  path = '/tmp/simple_keras_model'\n  tf.keras.experimental.export_saved_model(model, path)\n\n  # Load the saved keras model back.\n  new_model = tf.keras.experimental.load_from_saved_model(path)\n  new_model.summary()\n  ```\n\n  Args:\n    model: A `tf.keras.Model` to be saved. If the model is subclassed, the flag\n      `serving_only` must be set to True.\n    saved_model_path: a string specifying the path to the SavedModel directory.\n    custom_objects: Optional dictionary mapping string names to custom classes\n      or functions (e.g. custom loss functions).\n    as_text: bool, `False` by default. Whether to write the `SavedModel` proto\n      in text format. Currently unavailable in serving-only mode.\n    input_signature: A possibly nested sequence of `tf.TensorSpec` objects, used\n      to specify the expected model inputs. See `tf.function` for more details.\n    serving_only: bool, `False` by default. When this is true, only the\n      prediction graph is saved.\n\n  Raises:\n    NotImplementedError: If the model is a subclassed model, and serving_only is\n      False.\n    ValueError: If the input signature cannot be inferred from the model.\n    AssertionError: If the SavedModel directory already exists and isn't empty.\n  "
    warnings.warn('`tf.keras.experimental.export_saved_model` is deprecatedand will be removed in a future version. Please use `model.save(..., save_format="tf")` or `tf.keras.models.save_model(..., save_format="tf")`.')
    if serving_only:
        save_lib.save(model, saved_model_path, signatures=saving_utils.trace_model_call(model, input_signature))
    else:
        _save_v1_format(model, saved_model_path, custom_objects, as_text, input_signature)
    try:
        _export_model_json(model, saved_model_path)
    except NotImplementedError:
        logging.warning('Skipped saving model JSON, subclassed model does not have get_config() defined.')

def _export_model_json(model, saved_model_path):
    if False:
        return 10
    'Saves model configuration as a json string under assets folder.'
    model_json = model.to_json()
    model_json_filepath = os.path.join(_get_or_create_assets_dir(saved_model_path), compat.as_text(SAVED_MODEL_FILENAME_JSON))
    with gfile.Open(model_json_filepath, 'w') as f:
        f.write(model_json)

def _export_model_variables(model, saved_model_path):
    if False:
        return 10
    'Saves model weights in checkpoint format under variables folder.'
    _get_or_create_variables_dir(saved_model_path)
    checkpoint_prefix = _get_variables_path(saved_model_path)
    model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True)
    return checkpoint_prefix

def _save_v1_format(model, path, custom_objects, as_text, input_signature):
    if False:
        print('Hello World!')
    'Exports model to v1 SavedModel format.'
    if not model._is_graph_network:
        if isinstance(model, sequential.Sequential):
            if not model.built:
                raise ValueError('Weights for sequential model have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`, or the first layer in the model has `input_shape` during construction.')
        else:
            raise NotImplementedError('Subclassed models can only be exported for serving. Please set argument serving_only=True.')
    builder = saved_model_builder._SavedModelBuilder(path)
    checkpoint_path = _export_model_variables(model, path)
    export_args = {'builder': builder, 'model': model, 'custom_objects': custom_objects, 'checkpoint_path': checkpoint_path, 'input_signature': input_signature}
    has_saved_vars = False
    if model.optimizer:
        if isinstance(model.optimizer, (optimizer_v1.TFOptimizer, optimizer_v2.OptimizerV2)):
            _export_mode(mode_keys.ModeKeys.TRAIN, has_saved_vars, **export_args)
            has_saved_vars = True
            _export_mode(mode_keys.ModeKeys.TEST, has_saved_vars, **export_args)
        else:
            logging.warning('Model was compiled with an optimizer, but the optimizer is not from `tf.train` (e.g. `tf.train.AdagradOptimizer`). Only the serving graph was exported. The train and evaluate graphs were not added to the SavedModel.')
    _export_mode(mode_keys.ModeKeys.PREDICT, has_saved_vars, **export_args)
    builder.save(as_text)

def _get_var_list(model):
    if False:
        i = 10
        return i + 15
    'Returns list of all checkpointed saveable objects in the model.'
    (var_list, _, _) = graph_view.ObjectGraphView(model).serialize_object_graph()
    return var_list

def create_placeholder(spec):
    if False:
        return 10
    return backend.placeholder(shape=spec.shape, dtype=spec.dtype, name=spec.name)

def _export_mode(mode, has_saved_vars, builder, model, custom_objects, checkpoint_path, input_signature):
    if False:
        print('Hello World!')
    'Exports a model, and optionally saves new vars from the clone model.\n\n  Args:\n    mode: A `tf.estimator.ModeKeys` string.\n    has_saved_vars: A `boolean` indicating whether the SavedModel has already\n      exported variables.\n    builder: A `SavedModelBuilder` object.\n    model: A `tf.keras.Model` object.\n    custom_objects: A dictionary mapping string names to custom classes\n      or functions.\n    checkpoint_path: String path to checkpoint.\n    input_signature: Nested TensorSpec containing the expected inputs. Can be\n      `None`, in which case the signature will be inferred from the model.\n\n  Raises:\n    ValueError: If the train/eval mode is being exported, but the model does\n      not have an optimizer.\n  '
    compile_clone = mode != mode_keys.ModeKeys.PREDICT
    if compile_clone and (not model.optimizer):
        raise ValueError('Model does not have an optimizer. Cannot export mode %s' % mode)
    model_graph = ops.get_default_graph()
    with ops.Graph().as_default() as g, backend.learning_phase_scope(mode == mode_keys.ModeKeys.TRAIN):
        if input_signature is None:
            input_tensors = None
        else:
            input_tensors = nest.map_structure(create_placeholder, input_signature)
        clone = models_lib.clone_and_build_model(model, input_tensors=input_tensors, custom_objects=custom_objects, compile_clone=compile_clone)
        if compile_clone:
            g.add_to_collection(ops.GraphKeys.GLOBAL_STEP, clone.optimizer.iterations)
        train_op = None
        if mode == mode_keys.ModeKeys.TRAIN:
            clone._make_train_function()
            train_op = clone.train_function.updates_op
        elif mode == mode_keys.ModeKeys.TEST:
            clone._make_test_function()
        else:
            clone._make_predict_function()
        g.get_collection_ref(ops.GraphKeys.UPDATE_OPS).extend(clone.state_updates)
        with session.Session().as_default():
            clone_var_list = _get_var_list(clone)
            if has_saved_vars:
                status = clone.load_weights(checkpoint_path)
                status.assert_existing_objects_matched()
            else:
                _assert_same_non_optimizer_objects(model, model_graph, clone, g)
                clone.load_weights(checkpoint_path)
                clone.save_weights(checkpoint_path, save_format='tf', overwrite=True)
                builder._has_saved_variables = True
            builder.add_meta_graph(model_utils.EXPORT_TAG_MAP[mode], signature_def_map=_create_signature_def_map(clone, mode), saver=saver_lib.Saver(clone_var_list, allow_empty=True), init_op=variables.local_variables_initializer(), train_op=train_op)
        return None

def _create_signature_def_map(model, mode):
    if False:
        return 10
    'Creates a SignatureDef map from a Keras model.'
    inputs_dict = {name: x for (name, x) in zip(model.input_names, model.inputs)}
    if model.optimizer:
        targets_dict = {x.name.split(':')[0]: x for x in model._targets if x is not None}
        inputs_dict.update(targets_dict)
    outputs_dict = {name: x for (name, x) in zip(model.output_names, model.outputs)}
    metrics = saving_utils.extract_model_metrics(model)
    local_vars = set(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
    vars_to_add = set()
    if metrics is not None:
        for (key, value) in metrics.items():
            if isinstance(value, metrics_lib.Metric):
                vars_to_add.update(value.variables)
                metrics[key] = (value.result(), value.updates[0])
    vars_to_add = vars_to_add.difference(local_vars)
    for v in vars_to_add:
        ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)
    export_outputs = model_utils.export_outputs_for_mode(mode, predictions=outputs_dict, loss=model.total_loss if model.optimizer else None, metrics=metrics)
    return model_utils.build_all_signature_defs(inputs_dict, export_outputs=export_outputs, serving_only=mode == mode_keys.ModeKeys.PREDICT)

def _assert_same_non_optimizer_objects(model, model_graph, clone, clone_graph):
    if False:
        i = 10
        return i + 15
    'Asserts model and clone contain the same trackable objects.'
    return True

def load_from_saved_model(saved_model_path, custom_objects=None):
    if False:
        return 10
    "Loads a keras Model from a SavedModel created by `export_saved_model()`.\n\n  This function reinstantiates model state by:\n  1) loading model topology from json (this will eventually come\n     from metagraph).\n  2) loading model weights from checkpoint.\n\n  Example:\n\n  ```python\n  import tensorflow as tf\n\n  # Create a tf.keras model.\n  model = tf.keras.Sequential()\n  model.add(tf.keras.layers.Dense(1, input_shape=[10]))\n  model.summary()\n\n  # Save the tf.keras model in the SavedModel format.\n  path = '/tmp/simple_keras_model'\n  tf.keras.experimental.export_saved_model(model, path)\n\n  # Load the saved keras model back.\n  new_model = tf.keras.experimental.load_from_saved_model(path)\n  new_model.summary()\n  ```\n\n  Args:\n    saved_model_path: a string specifying the path to an existing SavedModel.\n    custom_objects: Optional dictionary mapping names\n        (strings) to custom classes or functions to be\n        considered during deserialization.\n\n  Returns:\n    a keras.Model instance.\n  "
    warnings.warn('`tf.keras.experimental.load_from_saved_model` is deprecatedand will be removed in a future version. Please switch to `tf.keras.models.load_model`.')
    model_json_filepath = os.path.join(compat.as_bytes(saved_model_path), compat.as_bytes(constants.ASSETS_DIRECTORY), compat.as_bytes(SAVED_MODEL_FILENAME_JSON))
    with gfile.Open(model_json_filepath, 'r') as f:
        model_json = f.read()
    model = model_config.model_from_json(model_json, custom_objects=custom_objects)
    checkpoint_prefix = os.path.join(compat.as_text(saved_model_path), compat.as_text(constants.VARIABLES_DIRECTORY), compat.as_text(constants.VARIABLES_FILENAME))
    model.load_weights(checkpoint_prefix)
    return model

def _get_or_create_variables_dir(export_dir):
    if False:
        print('Hello World!')
    "Return variables sub-directory, or create one if it doesn't exist."
    variables_dir = _get_variables_dir(export_dir)
    file_io.recursive_create_dir(variables_dir)
    return variables_dir

def _get_variables_dir(export_dir):
    if False:
        while True:
            i = 10
    'Return variables sub-directory in the SavedModel.'
    return os.path.join(compat.as_text(export_dir), compat.as_text(constants.VARIABLES_DIRECTORY))

def _get_variables_path(export_dir):
    if False:
        print('Hello World!')
    'Return the variables path, used as the prefix for checkpoint files.'
    return os.path.join(compat.as_text(_get_variables_dir(export_dir)), compat.as_text(constants.VARIABLES_FILENAME))

def _get_or_create_assets_dir(export_dir):
    if False:
        while True:
            i = 10
    "Return assets sub-directory, or create one if it doesn't exist."
    assets_destination_dir = _get_assets_dir(export_dir)
    file_io.recursive_create_dir(assets_destination_dir)
    return assets_destination_dir

def _get_assets_dir(export_dir):
    if False:
        return 10
    'Return path to asset directory in the SavedModel.'
    return os.path.join(compat.as_text(export_dir), compat.as_text(constants.ASSETS_DIRECTORY))