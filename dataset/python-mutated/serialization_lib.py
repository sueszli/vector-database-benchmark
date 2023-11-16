"""Object config serialization and deserialization logic."""
import importlib
import inspect
import types
import warnings
import numpy as np
from keras import api_export
from keras import backend
from keras.api_export import keras_export
from keras.backend.common import global_state
from keras.saving import object_registration
from keras.utils import python_utils
from keras.utils.module_utils import tensorflow as tf
PLAIN_TYPES = (str, int, float, bool)
BUILTIN_MODULES = ('activations', 'constraints', 'initializers', 'losses', 'metrics', 'optimizers', 'regularizers')

class SerializableDict:

    def __init__(self, **config):
        if False:
            i = 10
            return i + 15
        self.config = config

    def serialize(self):
        if False:
            i = 10
            return i + 15
        return serialize_keras_object(self.config)

class SafeModeScope:
    """Scope to propagate safe mode flag to nested deserialization calls."""

    def __init__(self, safe_mode=True):
        if False:
            for i in range(10):
                print('nop')
        self.safe_mode = safe_mode

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.original_value = in_safe_mode()
        global_state.set_global_attribute('safe_mode_saving', self.safe_mode)

    def __exit__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        global_state.set_global_attribute('safe_mode_saving', self.original_value)

@keras_export('keras.config.enable_unsafe_deserialization')
def enable_unsafe_deserialization():
    if False:
        for i in range(10):
            print('nop')
    'Disables safe mode globally, allowing deserialization of lambdas.'
    global_state.set_global_attribute('safe_mode_saving', False)

def in_safe_mode():
    if False:
        return 10
    return global_state.get_global_attribute('safe_mode_saving')

class ObjectSharingScope:
    """Scope to enable detection and reuse of previously seen objects."""

    def __enter__(self):
        if False:
            print('Hello World!')
        global_state.set_global_attribute('shared_objects/id_to_obj_map', {})
        global_state.set_global_attribute('shared_objects/id_to_config_map', {})

    def __exit__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        global_state.set_global_attribute('shared_objects/id_to_obj_map', None)
        global_state.set_global_attribute('shared_objects/id_to_config_map', None)

def get_shared_object(obj_id):
    if False:
        print('Hello World!')
    'Retrieve an object previously seen during deserialization.'
    id_to_obj_map = global_state.get_global_attribute('shared_objects/id_to_obj_map')
    if id_to_obj_map is not None:
        return id_to_obj_map.get(obj_id, None)

def record_object_after_serialization(obj, config):
    if False:
        i = 10
        return i + 15
    'Call after serializing an object, to keep track of its config.'
    if config['module'] == '__main__':
        config['module'] = None
    id_to_config_map = global_state.get_global_attribute('shared_objects/id_to_config_map')
    if id_to_config_map is None:
        return
    obj_id = int(id(obj))
    if obj_id not in id_to_config_map:
        id_to_config_map[obj_id] = config
    else:
        config['shared_object_id'] = obj_id
        prev_config = id_to_config_map[obj_id]
        prev_config['shared_object_id'] = obj_id

def record_object_after_deserialization(obj, obj_id):
    if False:
        while True:
            i = 10
    'Call after deserializing an object, to keep track of it in the future.'
    id_to_obj_map = global_state.get_global_attribute('shared_objects/id_to_obj_map')
    if id_to_obj_map is None:
        return
    id_to_obj_map[obj_id] = obj

@keras_export(['keras.saving.serialize_keras_object', 'keras.utils.serialize_keras_object'])
def serialize_keras_object(obj):
    if False:
        print('Hello World!')
    'Retrieve the config dict by serializing the Keras object.\n\n    `serialize_keras_object()` serializes a Keras object to a python dictionary\n    that represents the object, and is a reciprocal function of\n    `deserialize_keras_object()`. See `deserialize_keras_object()` for more\n    information about the config format.\n\n    Args:\n        obj: the Keras object to serialize.\n\n    Returns:\n        A python dict that represents the object. The python dict can be\n        deserialized via `deserialize_keras_object()`.\n    '
    if obj is None:
        return obj
    if isinstance(obj, PLAIN_TYPES):
        return obj
    if isinstance(obj, (list, tuple)):
        config_arr = [serialize_keras_object(x) for x in obj]
        return tuple(config_arr) if isinstance(obj, tuple) else config_arr
    if isinstance(obj, dict):
        return serialize_dict(obj)
    if isinstance(obj, bytes):
        return {'class_name': '__bytes__', 'config': {'value': obj.decode('utf-8')}}
    if isinstance(obj, slice):
        return {'class_name': '__slice__', 'config': {'start': serialize_keras_object(obj.start), 'stop': serialize_keras_object(obj.stop), 'step': serialize_keras_object(obj.step)}}
    if isinstance(obj, backend.KerasTensor):
        history = getattr(obj, '_keras_history', None)
        if history:
            history = list(history)
            history[0] = history[0].name
        return {'class_name': '__keras_tensor__', 'config': {'shape': obj.shape, 'dtype': obj.dtype, 'keras_history': history}}
    if tf.available and isinstance(obj, tf.TensorShape):
        return obj.as_list() if obj._dims is not None else None
    if backend.is_tensor(obj):
        return {'class_name': '__tensor__', 'config': {'value': backend.convert_to_numpy(obj).tolist(), 'dtype': backend.standardize_dtype(obj.dtype)}}
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray) and obj.ndim > 0:
            return {'class_name': '__numpy__', 'config': {'value': obj.tolist(), 'dtype': backend.standardize_dtype(obj.dtype)}}
        else:
            return obj.item()
    if tf.available and isinstance(obj, tf.DType):
        return obj.name
    if isinstance(obj, types.FunctionType) and obj.__name__ == '<lambda>':
        warnings.warn(f'The object being serialized includes a `lambda`. This is unsafe. In order to reload the object, you will have to pass `safe_mode=False` to the loading function. Please avoid using `lambda` in the future, and use named Python functions instead. This is the `lambda` being serialized: {inspect.getsource(obj)}', stacklevel=2)
        return {'class_name': '__lambda__', 'config': {'value': python_utils.func_dump(obj)}}
    if tf.available and isinstance(obj, tf.TypeSpec):
        ts_config = obj._serialize()
        ts_config = list(map(lambda x: x.as_list() if isinstance(x, tf.TensorShape) else x.name if isinstance(x, tf.DType) else x, ts_config))
        return {'class_name': '__typespec__', 'spec_name': obj.__class__.__name__, 'module': obj.__class__.__module__, 'config': ts_config, 'registered_name': None}
    inner_config = _get_class_or_fn_config(obj)
    config_with_public_class = serialize_with_public_class(obj.__class__, inner_config)
    if config_with_public_class is not None:
        get_build_and_compile_config(obj, config_with_public_class)
        record_object_after_serialization(obj, config_with_public_class)
        return config_with_public_class
    if isinstance(obj, types.FunctionType):
        module = obj.__module__
    else:
        module = obj.__class__.__module__
    class_name = obj.__class__.__name__
    if module == 'builtins':
        registered_name = None
    elif isinstance(obj, types.FunctionType):
        registered_name = object_registration.get_registered_name(obj)
    else:
        registered_name = object_registration.get_registered_name(obj.__class__)
    config = {'module': module, 'class_name': class_name, 'config': inner_config, 'registered_name': registered_name}
    get_build_and_compile_config(obj, config)
    record_object_after_serialization(obj, config)
    return config

def get_build_and_compile_config(obj, config):
    if False:
        print('Hello World!')
    if hasattr(obj, 'get_build_config'):
        build_config = obj.get_build_config()
        if build_config is not None:
            config['build_config'] = serialize_dict(build_config)
    if hasattr(obj, 'get_compile_config'):
        compile_config = obj.get_compile_config()
        if compile_config is not None:
            config['compile_config'] = serialize_dict(compile_config)
    return

def serialize_with_public_class(cls, inner_config=None):
    if False:
        return 10
    'Serializes classes from public Keras API or object registration.\n\n    Called to check and retrieve the config of any class that has a public\n    Keras API or has been registered as serializable via\n    `keras.saving.register_keras_serializable()`.\n    '
    keras_api_name = api_export.get_name_from_symbol(cls)
    if keras_api_name is None:
        registered_name = object_registration.get_registered_name(cls)
        if registered_name is None:
            return None
        return {'module': cls.__module__, 'class_name': cls.__name__, 'config': inner_config, 'registered_name': registered_name}
    parts = keras_api_name.split('.')
    return {'module': '.'.join(parts[:-1]), 'class_name': parts[-1], 'config': inner_config, 'registered_name': None}

def serialize_with_public_fn(fn, config, fn_module_name=None):
    if False:
        return 10
    "Serializes functions from public Keras API or object registration.\n\n    Called to check and retrieve the config of any function that has a public\n    Keras API or has been registered as serializable via\n    `keras.saving.register_keras_serializable()`. If function's module name\n    is already known, returns corresponding config.\n    "
    if fn_module_name:
        return {'module': fn_module_name, 'class_name': 'function', 'config': config, 'registered_name': config}
    keras_api_name = api_export.get_name_from_symbol(fn)
    if keras_api_name:
        parts = keras_api_name.split('.')
        return {'module': '.'.join(parts[:-1]), 'class_name': 'function', 'config': config, 'registered_name': config}
    else:
        registered_name = object_registration.get_registered_name(fn)
        if not registered_name and (not fn.__module__ == 'builtins'):
            return None
        return {'module': fn.__module__, 'class_name': 'function', 'config': config, 'registered_name': registered_name}

def _get_class_or_fn_config(obj):
    if False:
        i = 10
        return i + 15
    "Return the object's config depending on its type."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    if hasattr(obj, 'get_config'):
        config = obj.get_config()
        if not isinstance(config, dict):
            raise TypeError(f'The `get_config()` method of {obj} should return a dict. It returned: {config}')
        return serialize_dict(config)
    elif hasattr(obj, '__name__'):
        return object_registration.get_registered_name(obj)
    else:
        raise TypeError(f'Cannot serialize object {obj} of type {type(obj)}. To be serializable, a class must implement the `get_config()` method.')

def serialize_dict(obj):
    if False:
        while True:
            i = 10
    return {key: serialize_keras_object(value) for (key, value) in obj.items()}

@keras_export(['keras.saving.deserialize_keras_object', 'keras.utils.deserialize_keras_object'])
def deserialize_keras_object(config, custom_objects=None, safe_mode=True, **kwargs):
    if False:
        while True:
            i = 10
    'Retrieve the object by deserializing the config dict.\n\n    The config dict is a Python dictionary that consists of a set of key-value\n    pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,\n    `Metrics`, etc. The saving and loading library uses the following keys to\n    record information of a Keras object:\n\n    - `class_name`: String. This is the name of the class,\n      as exactly defined in the source\n      code, such as "LossesContainer".\n    - `config`: Dict. Library-defined or user-defined key-value pairs that store\n      the configuration of the object, as obtained by `object.get_config()`.\n    - `module`: String. The path of the python module. Built-in Keras classes\n      expect to have prefix `keras`.\n    - `registered_name`: String. The key the class is registered under via\n      `keras.saving.register_keras_serializable(package, name)` API. The\n      key has the format of \'{package}>{name}\', where `package` and `name` are\n      the arguments passed to `register_keras_serializable()`. If `name` is not\n      provided, it uses the class name. If `registered_name` successfully\n      resolves to a class (that was registered), the `class_name` and `config`\n      values in the dict will not be used. `registered_name` is only used for\n      non-built-in classes.\n\n    For example, the following dictionary represents the built-in Adam optimizer\n    with the relevant config:\n\n    ```python\n    dict_structure = {\n        "class_name": "Adam",\n        "config": {\n            "amsgrad": false,\n            "beta_1": 0.8999999761581421,\n            "beta_2": 0.9990000128746033,\n            "decay": 0.0,\n            "epsilon": 1e-07,\n            "learning_rate": 0.0010000000474974513,\n            "name": "Adam"\n        },\n        "module": "keras.optimizers",\n        "registered_name": None\n    }\n    # Returns an `Adam` instance identical to the original one.\n    deserialize_keras_object(dict_structure)\n    ```\n\n    If the class does not have an exported Keras namespace, the library tracks\n    it by its `module` and `class_name`. For example:\n\n    ```python\n    dict_structure = {\n      "class_name": "MetricsList",\n      "config": {\n          ...\n      },\n      "module": "keras.trainers.compile_utils",\n      "registered_name": "MetricsList"\n    }\n\n    # Returns a `MetricsList` instance identical to the original one.\n    deserialize_keras_object(dict_structure)\n    ```\n\n    And the following dictionary represents a user-customized `MeanSquaredError`\n    loss:\n\n    ```python\n    @keras.saving.register_keras_serializable(package=\'my_package\')\n    class ModifiedMeanSquaredError(keras.losses.MeanSquaredError):\n      ...\n\n    dict_structure = {\n        "class_name": "ModifiedMeanSquaredError",\n        "config": {\n            "fn": "mean_squared_error",\n            "name": "mean_squared_error",\n            "reduction": "auto"\n        },\n        "registered_name": "my_package>ModifiedMeanSquaredError"\n    }\n    # Returns the `ModifiedMeanSquaredError` object\n    deserialize_keras_object(dict_structure)\n    ```\n\n    Args:\n        config: Python dict describing the object.\n        custom_objects: Python dict containing a mapping between custom\n            object names the corresponding classes or functions.\n        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.\n            When `safe_mode=False`, loading an object has the potential to\n            trigger arbitrary code execution. This argument is only\n            applicable to the Keras v3 model format. Defaults to `True`.\n\n    Returns:\n        The object described by the `config` dictionary.\n    '
    safe_scope_arg = in_safe_mode()
    safe_mode = safe_scope_arg if safe_scope_arg is not None else safe_mode
    module_objects = kwargs.pop('module_objects', None)
    custom_objects = custom_objects or {}
    tlco = global_state.get_global_attribute('custom_objects_scope_dict', {})
    gco = object_registration.GLOBAL_CUSTOM_OBJECTS
    custom_objects = {**custom_objects, **tlco, **gco}
    if config is None:
        return None
    if isinstance(config, str) and custom_objects and (custom_objects.get(config) is not None):
        return custom_objects[config]
    if isinstance(config, (list, tuple)):
        return [deserialize_keras_object(x, custom_objects=custom_objects, safe_mode=safe_mode) for x in config]
    if module_objects is not None:
        (inner_config, fn_module_name, has_custom_object) = (None, None, False)
        if isinstance(config, dict):
            if 'config' in config:
                inner_config = config['config']
            if 'class_name' not in config:
                raise ValueError(f'Unknown `config` as a `dict`, config={config}')
            if custom_objects and (config['class_name'] in custom_objects or config.get('registered_name') in custom_objects or (isinstance(inner_config, str) and inner_config in custom_objects)):
                has_custom_object = True
            elif config['class_name'] == 'function':
                fn_module_name = config['module']
                if fn_module_name == 'builtins':
                    config = config['config']
                else:
                    config = config['registered_name']
            else:
                if config.get('module', '_') is None:
                    raise TypeError(f"Cannot deserialize object of type `{config['class_name']}`. If `{config['class_name']}` is a custom class, please register it using the `@keras.saving.register_keras_serializable()` decorator.")
                config = config['class_name']
        if not has_custom_object:
            if config not in module_objects:
                return config
            if isinstance(module_objects[config], types.FunctionType):
                return deserialize_keras_object(serialize_with_public_fn(module_objects[config], config, fn_module_name), custom_objects=custom_objects)
            return deserialize_keras_object(serialize_with_public_class(module_objects[config], inner_config=inner_config), custom_objects=custom_objects)
    if isinstance(config, PLAIN_TYPES):
        return config
    if not isinstance(config, dict):
        raise TypeError(f'Could not parse config: {config}')
    if 'class_name' not in config or 'config' not in config:
        return {key: deserialize_keras_object(value, custom_objects=custom_objects, safe_mode=safe_mode) for (key, value) in config.items()}
    class_name = config['class_name']
    inner_config = config['config'] or {}
    custom_objects = custom_objects or {}
    if class_name == '__keras_tensor__':
        obj = backend.KerasTensor(inner_config['shape'], dtype=inner_config['dtype'])
        obj._pre_serialization_keras_history = inner_config['keras_history']
        return obj
    if class_name == '__tensor__':
        return backend.convert_to_tensor(inner_config['value'], dtype=inner_config['dtype'])
    if class_name == '__numpy__':
        return np.array(inner_config['value'], dtype=inner_config['dtype'])
    if config['class_name'] == '__bytes__':
        return inner_config['value'].encode('utf-8')
    if config['class_name'] == '__slice__':
        return slice(deserialize_keras_object(inner_config['start'], custom_objects=custom_objects, safe_mode=safe_mode), deserialize_keras_object(inner_config['stop'], custom_objects=custom_objects, safe_mode=safe_mode), deserialize_keras_object(inner_config['step'], custom_objects=custom_objects, safe_mode=safe_mode))
    if config['class_name'] == '__lambda__':
        if safe_mode:
            raise ValueError('Requested the deserialization of a `lambda` object. This carries a potential risk of arbitrary code execution and thus it is disallowed by default. If you trust the source of the saved model, you can pass `safe_mode=False` to the loading function in order to allow `lambda` loading, or call `keras.config.enable_unsafe_deserialization()`.')
        return python_utils.func_load(inner_config['value'])
    if tf is not None and config['class_name'] == '__typespec__':
        obj = _retrieve_class_or_fn(config['spec_name'], config['registered_name'], config['module'], obj_type='class', full_config=config, custom_objects=custom_objects)
        inner_config = map(lambda x: tf.TensorShape(x) if isinstance(x, list) else getattr(tf, x) if hasattr(tf.dtypes, str(x)) else x, inner_config)
        return obj._deserialize(tuple(inner_config))
    module = config.get('module', None)
    registered_name = config.get('registered_name', class_name)
    if class_name == 'function':
        fn_name = inner_config
        return _retrieve_class_or_fn(fn_name, registered_name, module, obj_type='function', full_config=config, custom_objects=custom_objects)
    if 'shared_object_id' in config:
        obj = get_shared_object(config['shared_object_id'])
        if obj is not None:
            return obj
    cls = _retrieve_class_or_fn(class_name, registered_name, module, obj_type='class', full_config=config, custom_objects=custom_objects)
    if isinstance(cls, types.FunctionType):
        return cls
    if not hasattr(cls, 'from_config'):
        raise TypeError(f"Unable to reconstruct an instance of '{class_name}' because the class is missing a `from_config()` method. Full object config: {config}")
    custom_obj_scope = object_registration.CustomObjectScope(custom_objects)
    safe_mode_scope = SafeModeScope(safe_mode)
    with custom_obj_scope, safe_mode_scope:
        try:
            instance = cls.from_config(inner_config)
        except TypeError as e:
            raise TypeError(f"{cls} could not be deserialized properly. Please ensure that components that are Python object instances (layers, models, etc.) returned by `get_config()` are explicitly deserialized in the model's `from_config()` method.\n\nconfig={config}.\n\nException encountered: {e}")
        build_config = config.get('build_config', None)
        if build_config and (not instance.built):
            instance.build_from_config(build_config)
            instance.built = True
        compile_config = config.get('compile_config', None)
        if compile_config:
            instance.compile_from_config(compile_config)
            instance.compiled = True
    if 'shared_object_id' in config:
        record_object_after_deserialization(instance, config['shared_object_id'])
    return instance

def _retrieve_class_or_fn(name, registered_name, module, obj_type, full_config, custom_objects=None):
    if False:
        for i in range(10):
            print('nop')
    if obj_type == 'function':
        custom_obj = object_registration.get_registered_object(name, custom_objects=custom_objects)
    else:
        custom_obj = object_registration.get_registered_object(registered_name, custom_objects=custom_objects)
    if custom_obj is not None:
        return custom_obj
    if module:
        if module == 'keras' or module.startswith('keras.'):
            api_name = module + '.' + name
            obj = api_export.get_symbol_from_name(api_name)
            if obj is not None:
                return obj
        if obj_type == 'function' and module == 'builtins':
            for mod in BUILTIN_MODULES:
                obj = api_export.get_symbol_from_name('keras.' + mod + '.' + name)
                if obj is not None:
                    return obj
            filtered_dict = {k: v for (k, v) in custom_objects.items() if k.endswith(full_config['config'])}
            if filtered_dict:
                return next(iter(filtered_dict.values()))
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            raise TypeError(f"Could not deserialize {obj_type} '{name}' because its parent module {module} cannot be imported. Full object config: {full_config}")
        obj = vars(mod).get(name, None)
        if obj is None and registered_name is not None:
            obj = vars(mod).get(registered_name, None)
        if obj is not None:
            return obj
    raise TypeError(f"Could not locate {obj_type} '{name}'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`. Full object config: {full_config}")