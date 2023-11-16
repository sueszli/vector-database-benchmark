import os
import zipfile
from absl import logging
from keras.api_export import keras_export
from keras.legacy.saving import legacy_h5_format
from keras.saving import saving_lib
from keras.utils import file_utils
from keras.utils import io_utils
try:
    import h5py
except ImportError:
    h5py = None

@keras_export(['keras.saving.save_model', 'keras.models.save_model'])
def save_model(model, filepath, overwrite=True, **kwargs):
    if False:
        return 10
    'Saves a model as a `.keras` file.\n\n    Args:\n        model: Keras model instance to be saved.\n        filepath: `str` or `pathlib.Path` object. Path where to save the model.\n        overwrite: Whether we should overwrite any existing model at the target\n            location, or instead ask the user via an interactive prompt.\n\n    Example:\n\n    ```python\n    model = keras.Sequential(\n        [\n            keras.layers.Dense(5, input_shape=(3,)),\n            keras.layers.Softmax(),\n        ],\n    )\n    model.save("model.keras")\n    loaded_model = keras.saving.load_model("model.keras")\n    x = keras.random.uniform((10, 3))\n    assert np.allclose(model.predict(x), loaded_model.predict(x))\n    ```\n\n    Note that `model.save()` is an alias for `keras.saving.save_model()`.\n\n    The saved `.keras` file contains:\n\n    - The model\'s configuration (architecture)\n    - The model\'s weights\n    - The model\'s optimizer\'s state (if any)\n\n    Thus models can be reinstantiated in the exact same state.\n    '
    include_optimizer = kwargs.pop('include_optimizer', True)
    save_format = kwargs.pop('save_format', False)
    if save_format:
        if str(filepath).endswith(('.h5', '.hdf5')) or str(filepath).endswith('.keras'):
            logging.warning(f'The `save_format` argument is deprecated in Keras 3. We recommend removing this argument as it can be inferred from the file path. Received: save_format={save_format}')
        else:
            raise ValueError(f'The `save_format` argument is deprecated in Keras 3. Please remove this argument and pass a file path with either `.keras` or `.h5` extension.Received: save_format={save_format}')
    if kwargs:
        raise ValueError(f'The following argument(s) are not supported: {list(kwargs.keys())}')
    if str(filepath).endswith(('.h5', '.hdf5')):
        logging.warning("You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.")
    if str(filepath).endswith('.keras'):
        try:
            exists = os.path.exists(filepath)
        except TypeError:
            exists = False
        if exists and (not overwrite):
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        saving_lib.save_model(model, filepath)
    elif str(filepath).endswith(('.h5', '.hdf5')):
        legacy_h5_format.save_model_to_hdf5(model, filepath, overwrite, include_optimizer)
    else:
        raise ValueError(f'Invalid filepath extension for saving. Please add either a `.keras` extension for the native Keras format (recommended) or a `.h5` extension. Use `tf.saved_model.save()` if you want to export a SavedModel for use with TFLite/TFServing/etc. Received: filepath={filepath}.')

@keras_export(['keras.saving.load_model', 'keras.models.load_model'])
def load_model(filepath, custom_objects=None, compile=True, safe_mode=True):
    if False:
        for i in range(10):
            print('nop')
    'Loads a model saved via `model.save()`.\n\n    Args:\n        filepath: `str` or `pathlib.Path` object, path to the saved model file.\n        custom_objects: Optional dictionary mapping names\n            (strings) to custom classes or functions to be\n            considered during deserialization.\n        compile: Boolean, whether to compile the model after loading.\n        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.\n            When `safe_mode=False`, loading an object has the potential to\n            trigger arbitrary code execution. This argument is only\n            applicable to the Keras v3 model format. Defaults to True.\n\n    Returns:\n        A Keras model instance. If the original model was compiled,\n        and the argument `compile=True` is set, then the returned model\n        will be compiled. Otherwise, the model will be left uncompiled.\n\n    Example:\n\n    ```python\n    model = keras.Sequential([\n        keras.layers.Dense(5, input_shape=(3,)),\n        keras.layers.Softmax()])\n    model.save("model.keras")\n    loaded_model = keras.saving.load_model("model.keras")\n    x = np.random.random((10, 3))\n    assert np.allclose(model.predict(x), loaded_model.predict(x))\n    ```\n\n    Note that the model variables may have different name values\n    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.\n    It is recommended that you use layer attributes to\n    access specific variables, e.g. `model.get_layer("dense_1").kernel`.\n    '
    is_keras_zip = str(filepath).endswith('.keras') and zipfile.is_zipfile(filepath)
    if file_utils.is_remote_path(filepath) and (not file_utils.isdir(filepath)) and (not is_keras_zip):
        local_path = os.path.join(saving_lib.get_temp_dir(), os.path.basename(filepath))
        file_utils.copy(filepath, local_path)
        if zipfile.is_zipfile(local_path):
            filepath = local_path
            is_keras_zip = True
    if is_keras_zip:
        return saving_lib.load_model(filepath, custom_objects=custom_objects, compile=compile, safe_mode=safe_mode)
    if str(filepath).endswith(('.h5', '.hdf5')):
        return legacy_h5_format.load_model_from_hdf5(filepath)
    elif str(filepath).endswith('.keras'):
        raise ValueError(f'File not found: filepath={filepath}. Please ensure the file is an accessible `.keras` zip file.')
    else:
        raise ValueError(f"File format not supported: filepath={filepath}. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer({filepath}, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).")

def load_weights(model, filepath, skip_mismatch=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if str(filepath).endswith('.keras'):
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')
        saving_lib.load_weights_only(model, filepath, skip_mismatch=skip_mismatch)
    elif str(filepath).endswith('.weights.h5'):
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')
        saving_lib.load_weights_only(model, filepath, skip_mismatch=skip_mismatch)
    elif str(filepath).endswith('.h5') or str(filepath).endswith('.hdf5'):
        by_name = kwargs.pop('by_name', False)
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')
        if not h5py:
            raise ImportError('Loading a H5 file requires `h5py` to be installed.')
        with h5py.File(filepath, 'r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                legacy_h5_format.load_weights_from_hdf5_group_by_name(f, model, skip_mismatch)
            else:
                legacy_h5_format.load_weights_from_hdf5_group(f, model)
    else:
        raise ValueError(f'File format not supported: filepath={filepath}. Keras 3 only supports V3 `.keras` and `.weights.h5` files, or legacy V1/V2 `.h5` files.')