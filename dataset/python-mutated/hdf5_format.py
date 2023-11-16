"""Functions for saving and loading a Keras Model from HDF5 format."""
import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None
sequential_lib = LazyLoader('sequential_lib', globals(), 'tensorflow.python.keras.engine.sequential')

def save_model_to_hdf5(model, filepath, overwrite=True, include_optimizer=True):
    if False:
        while True:
            i = 10
    "Saves a model to a HDF5 file.\n\n  The saved model contains:\n      - the model's configuration (topology)\n      - the model's weights\n      - the model's optimizer's state (if any)\n\n  Thus the saved model can be reinstantiated in\n  the exact same state, without any of the code\n  used for model definition or training.\n\n  Args:\n      model: Keras model instance to be saved.\n      filepath: One of the following:\n          - String, path where to save the model\n          - `h5py.File` object where to save the model\n      overwrite: Whether we should overwrite any existing\n          model at the target location, or instead\n          ask the user with a manual prompt.\n      include_optimizer: If True, save optimizer's state together.\n\n  Raises:\n      ImportError: if h5py is not available.\n  "
    if h5py is None:
        raise ImportError('`save_model` requires h5py.')
    if len(model.weights) != len(model._undeduplicated_weights):
        logging.warning('Found duplicated `Variable`s in Model\'s `weights`. This is usually caused by `Variable`s being shared by Layers in the Model. These `Variable`s will be treated as separate `Variable`s when the Model is restored. To avoid this, please save with `save_format="tf"`.')
    if not isinstance(filepath, h5py.File):
        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            gfile.MakeDirs(dirpath)
        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False
    try:
        model_metadata = saving_utils.model_metadata(model, include_optimizer)
        for (k, v) in model_metadata.items():
            if isinstance(v, (dict, list, tuple)):
                f.attrs[k] = json.dumps(v, default=json_utils.get_json_type).encode('utf8')
            else:
                f.attrs[k] = v
        model_weights_group = f.create_group('model_weights')
        model_layers = model.layers
        save_weights_to_hdf5_group(model_weights_group, model_layers)
        if include_optimizer and model.optimizer and (not isinstance(model.optimizer, optimizer_v1.TFOptimizer)):
            save_optimizer_weights_to_hdf5_group(f, model.optimizer)
        f.flush()
    finally:
        if opened_new_file:
            f.close()

def load_model_from_hdf5(filepath, custom_objects=None, compile=True):
    if False:
        return 10
    'Loads a model saved via `save_model_to_hdf5`.\n\n  Args:\n      filepath: One of the following:\n          - String, path to the saved model\n          - `h5py.File` object from which to load the model\n      custom_objects: Optional dictionary mapping names\n          (strings) to custom classes or functions to be\n          considered during deserialization.\n      compile: Boolean, whether to compile the model\n          after loading.\n\n  Returns:\n      A Keras model instance. If an optimizer was found\n      as part of the saved model, the model is already\n      compiled. Otherwise, the model is uncompiled and\n      a warning will be displayed. When `compile` is set\n      to False, the compilation is omitted without any\n      warning.\n\n  Raises:\n      ImportError: if h5py is not available.\n      ValueError: In case of an invalid savefile.\n  '
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')
    if not custom_objects:
        custom_objects = {}
    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    model = None
    try:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        if hasattr(model_config, 'decode'):
            model_config = model_config.decode('utf-8')
        model_config = json_utils.decode(model_config)
        model = model_config_lib.model_from_config(model_config, custom_objects=custom_objects)
        load_weights_from_hdf5_group(f['model_weights'], model.layers)
        if compile:
            training_config = f.attrs.get('training_config')
            if hasattr(training_config, 'decode'):
                training_config = training_config.decode('utf-8')
            if training_config is None:
                logging.warning('No training configuration found in the save file, so the model was *not* compiled. Compile it manually.')
                return model
            training_config = json_utils.decode(training_config)
            model.compile(**saving_utils.compile_args_from_training_config(training_config, custom_objects), from_serialized=True)
            saving_utils.try_build_compiled_arguments(model)
            if 'optimizer_weights' in f:
                try:
                    model.optimizer._create_all_weights(model.trainable_variables)
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

def preprocess_weights_for_loading(layer, weights, original_keras_version=None, original_backend=None):
    if False:
        for i in range(10):
            print('nop')
    'Preprocess layer weights between different Keras formats.\n\n  Converts layers weights from Keras 1 format to Keras 2 and also weights of\n  CuDNN layers in Keras 2.\n\n  Args:\n      layer: Layer instance.\n      weights: List of weights values (Numpy arrays).\n      original_keras_version: Keras version for the weights, as a string.\n      original_backend: Keras backend the weights were trained with,\n          as a string.\n\n  Returns:\n      A list of weights values (Numpy arrays).\n  '

    def convert_nested_bidirectional(weights):
        if False:
            i = 10
            return i + 15
        'Converts layers nested in `Bidirectional` wrapper.\n\n    This function uses `preprocess_weights_for_loading()` for converting\n    layers.\n\n    Args:\n        weights: List of weights values (Numpy arrays).\n\n    Returns:\n        A list of weights values (Numpy arrays).\n    '
        num_weights_per_layer = len(weights) // 2
        forward_weights = preprocess_weights_for_loading(layer.forward_layer, weights[:num_weights_per_layer], original_keras_version, original_backend)
        backward_weights = preprocess_weights_for_loading(layer.backward_layer, weights[num_weights_per_layer:], original_keras_version, original_backend)
        return forward_weights + backward_weights

    def convert_nested_time_distributed(weights):
        if False:
            while True:
                i = 10
        'Converts layers nested in `TimeDistributed` wrapper.\n\n    This function uses `preprocess_weights_for_loading()` for converting nested\n    layers.\n\n    Args:\n        weights: List of weights values (Numpy arrays).\n\n    Returns:\n        A list of weights values (Numpy arrays).\n    '
        return preprocess_weights_for_loading(layer.layer, weights, original_keras_version, original_backend)

    def convert_nested_model(weights):
        if False:
            print('Hello World!')
        'Converts layers nested in `Model` or `Sequential`.\n\n    This function uses `preprocess_weights_for_loading()` for converting nested\n    layers.\n\n    Args:\n        weights: List of weights values (Numpy arrays).\n\n    Returns:\n        A list of weights values (Numpy arrays).\n    '
        trainable_weights = weights[:len(layer.trainable_weights)]
        non_trainable_weights = weights[len(layer.trainable_weights):]
        new_trainable_weights = []
        new_non_trainable_weights = []
        for sublayer in layer.layers:
            num_trainable_weights = len(sublayer.trainable_weights)
            num_non_trainable_weights = len(sublayer.non_trainable_weights)
            if sublayer.weights:
                preprocessed = preprocess_weights_for_loading(layer=sublayer, weights=trainable_weights[:num_trainable_weights] + non_trainable_weights[:num_non_trainable_weights], original_keras_version=original_keras_version, original_backend=original_backend)
                new_trainable_weights.extend(preprocessed[:num_trainable_weights])
                new_non_trainable_weights.extend(preprocessed[num_trainable_weights:])
                trainable_weights = trainable_weights[num_trainable_weights:]
                non_trainable_weights = non_trainable_weights[num_non_trainable_weights:]
        return new_trainable_weights + new_non_trainable_weights
    if layer.__class__.__name__ == 'Bidirectional':
        weights = convert_nested_bidirectional(weights)
    if layer.__class__.__name__ == 'TimeDistributed':
        weights = convert_nested_time_distributed(weights)
    elif layer.__class__.__name__ in ['Model', 'Sequential', 'Functional']:
        weights = convert_nested_model(weights)
    if original_keras_version == '1':
        if layer.__class__.__name__ == 'TimeDistributed':
            weights = preprocess_weights_for_loading(layer.layer, weights, original_keras_version, original_backend)
        if layer.__class__.__name__ == 'Conv1D':
            shape = weights[0].shape
            if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
                assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0], 1)
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
            weights[0] = weights[0][:, 0, :, :]
        if layer.__class__.__name__ == 'Conv2D':
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
        if layer.__class__.__name__ == 'Conv2DTranspose':
            if layer.data_format == 'channels_last':
                weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
        if layer.__class__.__name__ == 'Conv3D':
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))
        if layer.__class__.__name__ == 'GRU':
            if len(weights) == 9:
                kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1], weights[4], weights[7]], axis=-1)
                bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]
        if layer.__class__.__name__ == 'LSTM':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0], weights[6], weights[3], weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1], weights[7], weights[4], weights[10]], axis=-1)
                bias = np.concatenate([weights[2], weights[8], weights[5], weights[11]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]
        if layer.__class__.__name__ == 'ConvLSTM2D':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0], weights[6], weights[3], weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1], weights[7], weights[4], weights[10]], axis=-1)
                bias = np.concatenate([weights[2], weights[8], weights[5], weights[11]], axis=-1)
                if layer.data_format == 'channels_first':
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                    recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
                weights = [kernel, recurrent_kernel, bias]
    conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
    if layer.__class__.__name__ in conv_layers:
        if backend.int_shape(layer.weights[0]) != weights[0].shape:
            weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = np.transpose(weights[1], (3, 2, 0, 1))
    return _convert_rnn_weights(layer, weights)

def _convert_rnn_weights(layer, weights):
    if False:
        for i in range(10):
            print('nop')
    'Converts weights for RNN layers between native and CuDNN format.\n\n  Input kernels for each gate are transposed and converted between Fortran\n  and C layout, recurrent kernels are transposed. For LSTM biases are summed/\n  split in half, for GRU biases are reshaped.\n\n  Weights can be converted in both directions between `LSTM` and`CuDNNSLTM`\n  and between `CuDNNGRU` and `GRU(reset_after=True)`. Default `GRU` is not\n  compatible with `CuDNNGRU`.\n\n  For missing biases in `LSTM`/`GRU` (`use_bias=False`) no conversion is made.\n\n  Args:\n      layer: Target layer instance.\n      weights: List of source weights values (input kernels, recurrent\n          kernels, [biases]) (Numpy arrays).\n\n  Returns:\n      A list of converted weights values (Numpy arrays).\n\n  Raises:\n      ValueError: for incompatible GRU layer/weights or incompatible biases\n  '

    def transform_kernels(kernels, func, n_gates):
        if False:
            print('Hello World!')
        'Transforms kernel for each gate separately using given function.\n\n    Args:\n        kernels: Stacked array of kernels for individual gates.\n        func: Function applied to kernel of each gate.\n        n_gates: Number of gates (4 for LSTM, 3 for GRU).\n\n    Returns:\n        Stacked array of transformed kernels.\n    '
        return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])

    def transpose_input(from_cudnn):
        if False:
            for i in range(10):
                print('nop')
        "Makes a function that transforms input kernels from/to CuDNN format.\n\n    It keeps the shape, but changes between the layout (Fortran/C). Eg.:\n\n    ```\n    Keras                 CuDNN\n    [[0, 1, 2],  <--->  [[0, 2, 4],\n     [3, 4, 5]]          [1, 3, 5]]\n    ```\n\n    It can be passed to `transform_kernels()`.\n\n    Args:\n        from_cudnn: `True` if source weights are in CuDNN format, `False`\n            if they're in plain Keras format.\n\n    Returns:\n        Function that converts input kernel to the other format.\n    "
        order = 'F' if from_cudnn else 'C'

        def transform(kernel):
            if False:
                while True:
                    i = 10
            return kernel.T.reshape(kernel.shape, order=order)
        return transform
    target_class = layer.__class__.__name__
    if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 4
        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNLSTM'
        elif bias_shape == (units * n_gates,):
            source = 'LSTM'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))

        def convert_lstm_weights(weights, from_cudnn=True):
            if False:
                i = 10
                return i + 15
            'Converts the weights between CuDNNLSTM and LSTM.\n\n      Args:\n        weights: Original weights.\n        from_cudnn: Indicates whether original weights are from CuDNN layer.\n\n      Returns:\n        Updated weights compatible with LSTM.\n      '
            kernels = transform_kernels(weights[0], transpose_input(from_cudnn), n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            if from_cudnn:
                biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
            else:
                biases = np.tile(0.5 * weights[2], 2)
            return [kernels, recurrent_kernels, biases]
        if source != target_class:
            weights = convert_lstm_weights(weights, from_cudnn=source == 'CuDNNLSTM')
    if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:
        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 3

        def convert_gru_weights(weights, from_cudnn=True):
            if False:
                for i in range(10):
                    print('nop')
            'Converts the weights between CuDNNGRU and GRU.\n\n      Args:\n        weights: Original weights.\n        from_cudnn: Indicates whether original weights are from CuDNN layer.\n\n      Returns:\n        Updated weights compatible with GRU.\n      '
            kernels = transform_kernels(weights[0], transpose_input(from_cudnn), n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
            return [kernels, recurrent_kernels, biases]
        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNGRU'
        elif bias_shape == (2, units * n_gates):
            source = 'GRU(reset_after=True)'
        elif bias_shape == (units * n_gates,):
            source = 'GRU(reset_after=False)'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))
        if target_class == 'CuDNNGRU':
            target = 'CuDNNGRU'
        elif layer.reset_after:
            target = 'GRU(reset_after=True)'
        else:
            target = 'GRU(reset_after=False)'
        if source != target:
            types = (source, target)
            if 'GRU(reset_after=False)' in types:
                raise ValueError('%s is not compatible with %s' % types)
            if source == 'CuDNNGRU':
                weights = convert_gru_weights(weights, from_cudnn=True)
            elif source == 'GRU(reset_after=True)':
                weights = convert_gru_weights(weights, from_cudnn=False)
    return weights

def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
    if False:
        return 10
    'Saves optimizer weights of a optimizer to a HDF5 group.\n\n  Args:\n      hdf5_group: HDF5 group.\n      optimizer: optimizer instance.\n  '
    symbolic_weights = getattr(optimizer, 'weights')
    if symbolic_weights:
        weights_group = hdf5_group.create_group('optimizer_weights')
        weight_names = [str(w.name).encode('utf8') for w in symbolic_weights]
        save_attributes_to_hdf5_group(weights_group, 'weight_names', weight_names)
        weight_values = backend.batch_get_value(symbolic_weights)
        for (name, val) in zip(weight_names, weight_values):
            param_dset = weights_group.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val

def load_optimizer_weights_from_hdf5_group(hdf5_group):
    if False:
        return 10
    'Load optimizer weights from a HDF5 group.\n\n  Args:\n      hdf5_group: A pointer to a HDF5 group.\n\n  Returns:\n      data: List of optimizer weight names.\n  '
    weights_group = hdf5_group['optimizer_weights']
    optimizer_weight_names = load_attributes_from_hdf5_group(weights_group, 'weight_names')
    return [weights_group[weight_name] for weight_name in optimizer_weight_names]

def save_weights_to_hdf5_group(f, layers):
    if False:
        return 10
    'Saves the weights of a list of layers to a HDF5 group.\n\n  Args:\n      f: HDF5 group.\n      layers: List of layer instances.\n  '
    from tensorflow.python.keras import __version__ as keras_version
    save_attributes_to_hdf5_group(f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
    f.attrs['backend'] = backend.backend().encode('utf8')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')
    for layer in sorted(layers, key=lambda x: x.name):
        g = f.create_group(layer.name)
        weights = _legacy_weights(layer)
        weight_values = backend.batch_get_value(weights)
        weight_names = [w.name.encode('utf8') for w in weights]
        save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
        for (name, val) in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val

def load_weights_from_hdf5_group(f, layers):
    if False:
        while True:
            i = 10
    'Implements topological (order-based) weight loading.\n\n  Args:\n      f: A pointer to a HDF5 group.\n      layers: a list of target layers.\n\n  Raises:\n      ValueError: in case of mismatch between provided layers\n          and weights file.\n  '
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
    for layer in layers:
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
        raise ValueError('You are trying to load a weight file containing ' + str(len(layer_names)) + ' layers into a model with ' + str(len(filtered_layers)) + ' layers.')
    weight_value_tuples = []
    for (k, name) in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = _legacy_weights(layer)
        weight_values = preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '" in the current model) was found to correspond to layer ' + name + ' in the save file. However the new layer ' + layer.name + ' expects ' + str(len(symbolic_weights)) + ' weights, but the saved weights have ' + str(len(weight_values)) + ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    backend.batch_set_value(weight_value_tuples)

def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False):
    if False:
        return 10
    'Implements name-based weight loading.\n\n  (instead of topological weight loading).\n\n  Layers that have no matching name are skipped.\n\n  Args:\n      f: A pointer to a HDF5 group.\n      layers: a list of target layers.\n      skip_mismatch: Boolean, whether to skip loading of layers\n          where there is a mismatch in the number of weights,\n          or a mismatch in the shape of the weights.\n\n  Raises:\n      ValueError: in case of mismatch between provided layers\n          and weights file and skip_match=False.\n  '
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
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)
    weight_value_tuples = []
    for (k, name) in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            weight_values = preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning('Skipping loading of weights for layer {}'.format(layer.name) + ' due to mismatch in number of weights ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' + str(len(symbolic_weights)) + ' weight(s), but the saved weights' + ' have ' + str(len(weight_values)) + ' element(s).')
            for i in range(len(weight_values)):
                if backend.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                    if skip_mismatch:
                        logging.warning('Skipping loading of weights for layer {}'.format(layer.name) + ' due to mismatch in shape ({} vs {}).'.format(symbolic_weights[i].shape, weight_values[i].shape))
                        continue
                    raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '"), weight ' + str(symbolic_weights[i]) + ' has shape {}'.format(backend.int_shape(symbolic_weights[i])) + ', but the saved weight has shape ' + str(weight_values[i].shape) + '.')
                else:
                    weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
    backend.batch_set_value(weight_value_tuples)

def save_attributes_to_hdf5_group(group, name, data):
    if False:
        while True:
            i = 10
    'Saves attributes (data) of the specified name into the HDF5 group.\n\n  This method deals with an inherent problem of HDF5 file which is not\n  able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.\n\n  Args:\n      group: A pointer to a HDF5 group.\n      name: A name of the attributes to save.\n      data: Attributes data to store.\n\n  Raises:\n    RuntimeError: If any single attribute is too large to be saved.\n  '
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]
    if bad_attributes:
        raise RuntimeError('The following attributes cannot be saved to HDF5 file because they are larger than %d bytes: %s' % (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))
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

def load_attributes_from_hdf5_group(group, name):
    if False:
        while True:
            i = 10
    'Loads attributes of the specified name from the HDF5 group.\n\n  This method deals with an inherent problem\n  of HDF5 file which is not able to store\n  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.\n\n  Args:\n      group: A pointer to a HDF5 group.\n      name: A name of the attributes to load.\n\n  Returns:\n      data: Attributes data.\n  '
    if name in group.attrs:
        data = [n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while '%s%d' % (name, chunk_id) in group.attrs:
            data.extend([n.decode('utf8') if hasattr(n, 'decode') else n for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data

def _legacy_weights(layer):
    if False:
        return 10
    'DO NOT USE.\n\n  For legacy reason, the layer.weights was in the order of\n  [self.trainable_weights + self.non_trainable_weights], and this order was\n  used for preserving the weights in h5 format. The new order of layer.weights\n  are the same as layer.get_weights() which is more intuitive for user. To\n  keep supporting the existing saved h5 file, this method should be used to\n  save/load weights. In future version, we will delete this method and\n  introduce a breaking change for h5 and stay with the new order for weights.\n\n  Args:\n    layer: a `tf.keras.Model` or `tf.keras.layers.Layer` instance.\n\n  Returns:\n    A list of variables with the order of trainable_weights, followed by\n      non_trainable_weights.\n  '
    weights = layer.trainable_weights + layer.non_trainable_weights
    if any((not isinstance(w, variables_module.Variable) for w in weights)):
        raise NotImplementedError("Save or restore weights that is not an instance of `tf.Variable` is not supported in h5, use `save_format='tf'` instead. Got a model or layer {} with weights {}".format(layer.__class__.__name__, weights))
    return weights