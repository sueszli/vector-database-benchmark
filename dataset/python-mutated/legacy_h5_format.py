import json
import os
import warnings
import numpy as np
from absl import logging
from keras import backend
from keras import optimizers
from keras.backend.common import global_state
from keras.legacy.saving import json_utils
from keras.legacy.saving import saving_options
from keras.legacy.saving import saving_utils
from keras.saving import object_registration
from keras.utils import io_utils
try:
    import h5py
except ImportError:
    h5py = None
HDF5_OBJECT_HEADER_LIMIT = 64512

def save_model_to_hdf5(model, filepath, overwrite=True, include_optimizer=True):
    if False:
        print('Hello World!')
    if h5py is None:
        raise ImportError('`save_model()` using h5 format requires h5py. Could not import h5py.')
    if not isinstance(filepath, h5py.File):
        if not overwrite and os.path.isfile(filepath):
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        dirpath = os.path.dirname(filepath)
        if dirpath and (not os.path.exists(dirpath)):
            os.makedirs(dirpath, exist_ok=True)
        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False
    try:
        with saving_options.keras_option_scope(use_legacy_config=True):
            model_metadata = saving_utils.model_metadata(model, include_optimizer)
            for (k, v) in model_metadata.items():
                if isinstance(v, (dict, list, tuple)):
                    f.attrs[k] = json.dumps(v, default=json_utils.get_json_type).encode('utf8')
                else:
                    f.attrs[k] = v
            model_weights_group = f.create_group('model_weights')
            save_weights_to_hdf5_group(model_weights_group, model)
            if include_optimizer and hasattr(model, 'optimizer'):
                save_optimizer_weights_to_hdf5_group(f, model.optimizer)
        f.flush()
    finally:
        if opened_new_file:
            f.close()

def load_model_from_hdf5(filepath, custom_objects=None, compile=True):
    if False:
        for i in range(10):
            print('nop')
    'Loads a model saved via `save_model_to_hdf5`.\n\n    Args:\n        filepath: One of the following:\n            - String, path to the saved model\n            - `h5py.File` object from which to load the model\n        custom_objects: Optional dictionary mapping names\n            (strings) to custom classes or functions to be\n            considered during deserialization.\n        compile: Boolean, whether to compile the model\n            after loading.\n\n    Returns:\n        A Keras model instance. If an optimizer was found\n        as part of the saved model, the model is already\n        compiled. Otherwise, the model is uncompiled and\n        a warning will be displayed. When `compile` is set\n        to False, the compilation is omitted without any\n        warning.\n\n    Raises:\n        ImportError: if h5py is not available.\n        ValueError: In case of an invalid savefile.\n    '
    if h5py is None:
        raise ImportError('`load_model()` using h5 format requires h5py. Could not import h5py.')
    if not custom_objects:
        custom_objects = {}
    gco = object_registration.GLOBAL_CUSTOM_OBJECTS
    tlco = global_state.get_global_attribute('custom_objects_scope_dict', {})
    custom_objects = {**custom_objects, **gco, **tlco}
    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    model = None
    try:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError(f'No model config found in the file at {filepath}.')
        if hasattr(model_config, 'decode'):
            model_config = model_config.decode('utf-8')
        model_config = json_utils.decode(model_config)
        with saving_options.keras_option_scope(use_legacy_config=True):
            model = saving_utils.model_from_config(model_config, custom_objects=custom_objects)
            load_weights_from_hdf5_group(f['model_weights'], model)
        if compile:
            training_config = f.attrs.get('training_config')
            if hasattr(training_config, 'decode'):
                training_config = training_config.decode('utf-8')
            if training_config is None:
                logging.warning('No training configuration found in the save file, so the model was *not* compiled. Compile it manually.')
                return model
            training_config = json_utils.decode(training_config)
            model.compile(**saving_utils.compile_args_from_training_config(training_config, custom_objects))
            saving_utils.try_build_compiled_arguments(model)
            if 'optimizer_weights' in f:
                try:
                    if isinstance(model.optimizer, optimizers.Optimizer):
                        model.optimizer.build(model._trainable_variables)
                    else:
                        model.optimizer._create_all_weights(model._trainable_variables)
                except (NotImplementedError, AttributeError):
                    logging.warning('Error when creating the weights of optimizer {}, making it impossible to restore the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.')
                optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
                try:
                    model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    logging.warning('Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.')
    finally:
        if opened_new_file:
            f.close()
    return model

def save_weights_to_hdf5_group(f, model):
    if False:
        return 10
    'Saves the weights of a list of layers to a HDF5 group.\n\n    Args:\n        f: HDF5 group.\n        model: Model instance.\n    '
    from keras import __version__ as keras_version
    save_attributes_to_hdf5_group(f, 'layer_names', [layer.name.encode('utf8') for layer in model.layers])
    f.attrs['backend'] = backend.backend().encode('utf8')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')
    for layer in sorted(model.layers, key=lambda x: x.name):
        g = f.create_group(layer.name)
        weights = _legacy_weights(layer)
        save_subset_weights_to_hdf5_group(g, weights)
    weights = list((v for v in model._trainable_variables + model._non_trainable_variables if v in model.weights))
    g = f.create_group('top_level_model_weights')
    save_subset_weights_to_hdf5_group(g, weights)

def save_subset_weights_to_hdf5_group(f, weights):
    if False:
        return 10
    'Save top-level weights of a model to a HDF5 group.\n\n    Args:\n        f: HDF5 group.\n        weights: List of weight variables.\n    '
    weight_values = [backend.convert_to_numpy(w) for w in weights]
    weight_names = [w.name.encode('utf8') for w in weights]
    save_attributes_to_hdf5_group(f, 'weight_names', weight_names)
    for (name, val) in zip(weight_names, weight_values):
        param_dset = f.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            param_dset[()] = val
        else:
            param_dset[:] = val

def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
    if False:
        while True:
            i = 10
    'Saves optimizer weights of a optimizer to a HDF5 group.\n\n    Args:\n        hdf5_group: HDF5 group.\n        optimizer: optimizer instance.\n    '
    if isinstance(optimizer, optimizers.Optimizer):
        symbolic_weights = optimizer.variables
    else:
        symbolic_weights = getattr(optimizer, 'weights')
    if symbolic_weights:
        weights_group = hdf5_group.create_group('optimizer_weights')
        weight_names = [str(w.name).encode('utf8') for w in symbolic_weights]
        save_attributes_to_hdf5_group(weights_group, 'weight_names', weight_names)
        weight_values = [backend.convert_to_numpy(w) for w in symbolic_weights]
        for (name, val) in zip(weight_names, weight_values):
            param_dset = weights_group.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val

def save_attributes_to_hdf5_group(group, name, data):
    if False:
        while True:
            i = 10
    'Saves attributes (data) of the specified name into the HDF5 group.\n\n    This method deals with an inherent problem of HDF5 file which is not\n    able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.\n\n    Args:\n        group: A pointer to a HDF5 group.\n        name: A name of the attributes to save.\n        data: Attributes data to store.\n\n    Raises:\n      RuntimeError: If any single attribute is too large to be saved.\n    '
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]
    if bad_attributes:
        raise RuntimeError(f'The following attributes cannot be saved to HDF5 file because they are larger than {HDF5_OBJECT_HEADER_LIMIT} bytes: {bad_attributes}')
    data_npy = np.asarray(data)
    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)
    while any((x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data)):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)
    if num_chunks > 1:
        for (chunk_id, chunk_data) in enumerate(chunked_data):
            group.attrs['%s%d' % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data

def load_weights_from_hdf5_group(f, model):
    if False:
        while True:
            i = 10
    'Implements topological (order-based) weight loading.\n\n    Args:\n        f: A pointer to a HDF5 group.\n        model: Model instance.\n\n    Raises:\n        ValueError: in case of mismatch between provided layers\n            and weights file.\n    '
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version']
        if hasattr(original_keras_version, 'decode'):
            original_keras_version = original_keras_version.decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend']
        if hasattr(original_backend, 'decode'):
            original_backend = original_backend.decode('utf8')
    else:
        original_backend = None
    filtered_layers = []
    for layer in model.layers:
        weights = _legacy_weights(layer)
        if weights:
            filtered_layers.append(layer)
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError(f'Layer count mismatch when loading weights from file. Model expected {len(filtered_layers)} layers, found {len(layer_names)} saved layers.')
    for (k, name) in enumerate(layer_names):
        g = f[name]
        layer = filtered_layers[k]
        symbolic_weights = _legacy_weights(layer)
        weight_values = load_subset_weights_from_hdf5_group(g)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(f'Weight count mismatch for layer #{k} (named {layer.name} in the current model, {name} in the save file). Layer expects {len(symbolic_weights)} weight(s). Received {len(weight_values)} saved weight(s)')
        for (ref_v, val) in zip(symbolic_weights, weight_values):
            ref_v.assign(val)
    if 'top_level_model_weights' in f:
        symbolic_weights = list((v for v in model._trainable_variables + model._non_trainable_variables if v in model.weights))
        weight_values = load_subset_weights_from_hdf5_group(f['top_level_model_weights'])
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(f'Weight count mismatch for top-level weights when loading weights from file. Model expects {len(symbolic_weights)} top-level weight(s). Received {len(weight_values)} saved top-level weight(s)')
        for (ref_v, val) in zip(symbolic_weights, weight_values):
            ref_v.assign(val)

def load_weights_from_hdf5_group_by_name(f, model, skip_mismatch=False):
    if False:
        for i in range(10):
            print('nop')
    'Implements name-based weight loading (instead of topological loading).\n\n    Layers that have no matching name are skipped.\n\n    Args:\n        f: A pointer to a HDF5 group.\n        model: Model instance.\n        skip_mismatch: Boolean, whether to skip loading of layers\n            where there is a mismatch in the number of weights,\n            or a mismatch in the shape of the weights.\n\n    Raises:\n        ValueError: in case of mismatch between provided layers\n            and weights file and skip_match=False.\n    '
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version']
        if hasattr(original_keras_version, 'decode'):
            original_keras_version = original_keras_version.decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend']
        if hasattr(original_backend, 'decode'):
            original_backend = original_backend.decode('utf8')
    else:
        original_backend = None
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)
    for (k, name) in enumerate(layer_names):
        g = f[name]
        weight_values = load_subset_weights_from_hdf5_group(g)
        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn(f'Skipping loading of weights for layer #{k} (named {layer.name}) due to mismatch in number of weights. Layer expects {len(symbolic_weights)} weight(s). Received {len(weight_values)} saved weight(s)', stacklevel=2)
                    continue
                raise ValueError(f'Weight count mismatch for layer #{k} (named {layer.name}). Layer expects {len(symbolic_weights)} weight(s). Received {len(weight_values)} saved weight(s)')
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(f'Skipping loading weights for layer #{k} (named {layer.name}) due to mismatch in shape for weight {symbolic_weights[i].name}. Weight expects shape {expected_shape}. Received saved weight with shape {received_shape}', stacklevel=2)
                        continue
                    raise ValueError(f'Shape mismatch in layer #{k} (named {layer.name}) for weight {symbolic_weights[i].name}. Weight expects shape {expected_shape}. Received saved weight with shape {received_shape}')
                else:
                    symbolic_weights[i].assign(weight_values[i])
    if 'top_level_model_weights' in f:
        symbolic_weights = model.trainable_weights + model.non_trainable_weights
        weight_values = load_subset_weights_from_hdf5_group(f['top_level_model_weights'])
        if len(weight_values) != len(symbolic_weights):
            if skip_mismatch:
                warnings.warn(f'Skipping loading top-level weights for model due to mismatch in number of weights. Model expects {len(symbolic_weights)} top-level weight(s). Received {len(weight_values)} saved top-level weight(s)', stacklevel=2)
            else:
                raise ValueError(f'Weight count mismatch for top-level weights of model. Model expects {len(symbolic_weights)} top-level weight(s). Received {len(weight_values)} saved top-level weight(s)')
        else:
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(f'Skipping loading top-level weight for model due to mismatch in shape for weight {symbolic_weights[i].name}. Weight expects shape {expected_shape}. Received saved weight with shape {received_shape}', stacklevel=2)
                    else:
                        raise ValueError(f'Shape mismatch in model for top-level weight {symbolic_weights[i].name}. Weight expects shape {expected_shape}. Received saved weight with shape {received_shape}')
                else:
                    symbolic_weights[i].assign(weight_values[i])

def load_subset_weights_from_hdf5_group(f):
    if False:
        print('Hello World!')
    'Load layer weights of a model from hdf5.\n\n    Args:\n        f: A pointer to a HDF5 group.\n\n    Returns:\n        List of NumPy arrays of the weight values.\n\n    Raises:\n        ValueError: in case of mismatch between provided model\n            and weights file.\n    '
    weight_names = load_attributes_from_hdf5_group(f, 'weight_names')
    return [np.asarray(f[weight_name]) for weight_name in weight_names]

def load_optimizer_weights_from_hdf5_group(hdf5_group):
    if False:
        i = 10
        return i + 15
    'Load optimizer weights from a HDF5 group.\n\n    Args:\n        hdf5_group: A pointer to a HDF5 group.\n\n    Returns:\n        data: List of optimizer weight names.\n    '
    weights_group = hdf5_group['optimizer_weights']
    optimizer_weight_names = load_attributes_from_hdf5_group(weights_group, 'weight_names')
    return [weights_group[weight_name] for weight_name in optimizer_weight_names]

def load_attributes_from_hdf5_group(group, name):
    if False:
        for i in range(10):
            print('nop')
    'Loads attributes of the specified name from the HDF5 group.\n\n    This method deals with an inherent problem\n    of HDF5 file which is not able to store\n    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.\n\n    Args:\n        group: A pointer to a HDF5 group.\n        name: A name of the attributes to load.\n\n    Returns:\n        data: Attributes data.\n    '
    if name in group.attrs:
        data = [n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while f'{name}{chunk_id}' in group.attrs:
            data.extend([n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs[f'{name}{chunk_id}']])
            chunk_id += 1
    return data

def _legacy_weights(layer):
    if False:
        i = 10
        return i + 15
    'Legacy weight order converter.\n\n    For legacy reason, the layer.weights was in the order of\n    [self.trainable_weights + self.non_trainable_weights], and this order was\n    used for preserving the weights in h5 format. The new order of layer.weights\n    are the same as layer.get_weights() which is more intuitive for user. To\n    keep supporting the existing saved h5 file, this method should be used to\n    save/load weights.\n\n    Args:\n        layer: a `Model` or `Layer` instance.\n\n    Returns:\n        A list of variables with the legacy weight order.\n    '
    return layer.trainable_weights + layer.non_trainable_weights