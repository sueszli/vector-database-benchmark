"""Utils related to keras model saving."""
import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

def extract_model_metrics(model):
    if False:
        for i in range(10):
            print('nop')
    'Convert metrics from a Keras model `compile` API to dictionary.\n\n  This is used for converting Keras models to Estimators and SavedModels.\n\n  Args:\n    model: A `tf.keras.Model` object.\n\n  Returns:\n    Dictionary mapping metric names to metric instances. May return `None` if\n    the model does not contain any metrics.\n  '
    if getattr(model, '_compile_metrics', None):
        return {m.name: m for m in model._compile_metric_functions}
    return None

def model_input_signature(model, keep_original_batch_size=False):
    if False:
        for i in range(10):
            print('nop')
    "Inspect model to get its input signature.\n\n  The model's input signature is a list with a single (possibly-nested) object.\n  This is due to the Keras-enforced restriction that tensor inputs must be\n  passed in as the first argument.\n\n  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}\n  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]\n\n  Args:\n    model: Keras Model object.\n    keep_original_batch_size: A boolean indicating whether we want to keep using\n      the original batch size or set it to None. Default is `False`, which means\n      that the batch dim of the returned input signature will always be set to\n      `None`.\n\n  Returns:\n    A list containing either a single TensorSpec or an object with nested\n    TensorSpecs. This list does not contain the `training` argument.\n  "
    input_specs = model._get_save_spec(dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
        return None
    input_specs = _enforce_names_consistency(input_specs)
    if isinstance(input_specs, collections.abc.Sequence) and len(input_specs) == 1:
        return input_specs
    else:
        return [input_specs]

def raise_model_input_error(model):
    if False:
        print('Hello World!')
    raise ValueError('Model {} cannot be saved because the input shapes have not been set. Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`. To manually set the shapes, call `model.build(input_shape)`.'.format(model))

def trace_model_call(model, input_signature=None):
    if False:
        while True:
            i = 10
    "Trace the model call to create a tf.function for exporting a Keras model.\n\n  Args:\n    model: A Keras model.\n    input_signature: optional, a list of tf.TensorSpec objects specifying the\n      inputs to the model.\n\n  Returns:\n    A tf.function wrapping the model's call function with input signatures set.\n\n  Raises:\n    ValueError: if input signature cannot be inferred from the model.\n  "
    if input_signature is None:
        if isinstance(model.call, def_function.Function):
            input_signature = model.call.input_signature
    if input_signature is None:
        input_signature = model_input_signature(model)
    if input_signature is None:
        raise_model_input_error(model)

    @def_function.function(input_signature=input_signature)
    def _wrapped_model(*args):
        if False:
            i = 10
            return i + 15
        "A concrete tf.function that wraps the model's call function."
        inputs = args[0] if len(input_signature) == 1 else list(args)
        with base_layer_utils.call_context().enter(model, inputs=inputs, build_graph=False, training=False, saving=True):
            outputs = model(inputs, training=False)
        output_names = model.output_names
        if output_names is None:
            from tensorflow.python.keras.engine import compile_utils
            output_names = compile_utils.create_pseudo_output_names(outputs)
        outputs = nest.flatten(outputs)
        return {name: output for (name, output) in zip(output_names, outputs)}
    return _wrapped_model

def model_metadata(model, include_optimizer=True, require_config=True):
    if False:
        while True:
            i = 10
    'Returns a dictionary containing the model metadata.'
    from tensorflow.python.keras import __version__ as keras_version
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2
    model_config = {'class_name': model.__class__.__name__}
    try:
        model_config['config'] = model.get_config()
    except NotImplementedError as e:
        if require_config:
            raise e
    metadata = dict(keras_version=str(keras_version), backend=K.backend(), model_config=model_config)
    if model.optimizer and include_optimizer:
        if isinstance(model.optimizer, optimizer_v1.TFOptimizer):
            logging.warning('TensorFlow optimizers do not make it possible to access optimizer attributes or optimizer state after instantiation. As a result, we cannot save the optimizer as part of the model save file. You will have to compile your model again after loading it. Prefer using a Keras optimizer instead (see keras.io/optimizers).')
        elif model._compile_was_called:
            training_config = model._get_compile_args(user_metrics=False)
            training_config.pop('optimizer', None)
            metadata['training_config'] = _serialize_nested_config(training_config)
            if isinstance(model.optimizer, optimizer_v2.RestoredOptimizer):
                raise NotImplementedError("As of now, Optimizers loaded from SavedModel cannot be saved. If you're calling `model.save` or `tf.keras.models.save_model`, please set the `include_optimizer` option to `False`. For `tf.saved_model.save`, delete the optimizer from the model.")
            else:
                optimizer_config = {'class_name': generic_utils.get_registered_name(model.optimizer.__class__), 'config': model.optimizer.get_config()}
            metadata['training_config']['optimizer_config'] = optimizer_config
    return metadata

def should_overwrite(filepath, overwrite):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the filepath should be overwritten.'
    if not overwrite and os.path.isfile(filepath):
        return ask_to_proceed_with_overwrite(filepath)
    return True

def compile_args_from_training_config(training_config, custom_objects=None):
    if False:
        i = 10
        return i + 15
    'Return model.compile arguments from training config.'
    if custom_objects is None:
        custom_objects = {}
    with generic_utils.CustomObjectScope(custom_objects):
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config)
        loss = None
        loss_config = training_config.get('loss', None)
        if loss_config is not None:
            loss = _deserialize_nested_config(losses.deserialize, loss_config)
        metrics = None
        metrics_config = training_config.get('metrics', None)
        if metrics_config is not None:
            metrics = _deserialize_nested_config(_deserialize_metric, metrics_config)
        weighted_metrics = None
        weighted_metrics_config = training_config.get('weighted_metrics', None)
        if weighted_metrics_config is not None:
            weighted_metrics = _deserialize_nested_config(_deserialize_metric, weighted_metrics_config)
        sample_weight_mode = training_config['sample_weight_mode'] if hasattr(training_config, 'sample_weight_mode') else None
        loss_weights = training_config['loss_weights']
    return dict(optimizer=optimizer, loss=loss, metrics=metrics, weighted_metrics=weighted_metrics, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode)

def _deserialize_nested_config(deserialize_fn, config):
    if False:
        for i in range(10):
            print('nop')
    'Deserializes arbitrary Keras `config` using `deserialize_fn`.'

    def _is_single_object(obj):
        if False:
            while True:
                i = 10
        if isinstance(obj, dict) and 'class_name' in obj:
            return True
        if isinstance(obj, str):
            return True
        return False
    if config is None:
        return None
    if _is_single_object(config):
        return deserialize_fn(config)
    elif isinstance(config, dict):
        return {k: _deserialize_nested_config(deserialize_fn, v) for (k, v) in config.items()}
    elif isinstance(config, (tuple, list)):
        return [_deserialize_nested_config(deserialize_fn, obj) for obj in config]
    raise ValueError('Saved configuration not understood.')

def _serialize_nested_config(config):
    if False:
        i = 10
        return i + 15
    'Serialized a nested structure of Keras objects.'

    def _serialize_fn(obj):
        if False:
            print('Hello World!')
        if callable(obj):
            return generic_utils.serialize_keras_object(obj)
        return obj
    return nest.map_structure(_serialize_fn, config)

def _deserialize_metric(metric_config):
    if False:
        i = 10
        return i + 15
    'Deserialize metrics, leaving special strings untouched.'
    from tensorflow.python.keras import metrics as metrics_module
    if metric_config in ['accuracy', 'acc', 'crossentropy', 'ce']:
        return metric_config
    return metrics_module.deserialize(metric_config)

def _enforce_names_consistency(specs):
    if False:
        i = 10
        return i + 15
    'Enforces that either all specs have names or none do.'

    def _has_name(spec):
        if False:
            i = 10
            return i + 15
        return hasattr(spec, 'name') and spec.name is not None

    def _clear_name(spec):
        if False:
            for i in range(10):
                print('nop')
        spec = copy.deepcopy(spec)
        if hasattr(spec, 'name'):
            spec._name = None
        return spec
    flat_specs = nest.flatten(specs)
    name_inconsistency = any((_has_name(s) for s in flat_specs)) and (not all((_has_name(s) for s in flat_specs)))
    if name_inconsistency:
        specs = nest.map_structure(_clear_name, specs)
    return specs

def try_build_compiled_arguments(model):
    if False:
        i = 10
        return i + 15
    if not version_utils.is_v1_layer_or_model(model) and model.outputs is not None:
        try:
            if not model.compiled_loss.built:
                model.compiled_loss.build(model.outputs)
            if not model.compiled_metrics.built:
                model.compiled_metrics.build(model.outputs, model.outputs)
        except:
            logging.warning('Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.')

def is_hdf5_filepath(filepath):
    if False:
        i = 10
        return i + 15
    return filepath.endswith('.h5') or filepath.endswith('.keras') or filepath.endswith('.hdf5')