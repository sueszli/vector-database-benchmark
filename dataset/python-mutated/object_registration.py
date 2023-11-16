import inspect
from keras.api_export import keras_export
from keras.backend.common import global_state
GLOBAL_CUSTOM_OBJECTS = {}
GLOBAL_CUSTOM_NAMES = {}

@keras_export(['keras.saving.CustomObjectScope', 'keras.saving.custom_object_scope', 'keras.utils.CustomObjectScope', 'keras.utils.custom_object_scope'])
class CustomObjectScope:
    """Exposes custom classes/functions to Keras deserialization internals.

    Under a scope `with custom_object_scope(objects_dict)`, Keras methods such
    as `keras.models.load_model()` or
    `keras.models.model_from_config()` will be able to deserialize any
    custom object referenced by a saved config (e.g. a custom layer or metric).

    Example:

    Consider a custom regularizer `my_regularizer`:

    ```python
    layer = Dense(3, kernel_regularizer=my_regularizer)
    # Config contains a reference to `my_regularizer`
    config = layer.get_config()
    ...
    # Later:
    with custom_object_scope({'my_regularizer': my_regularizer}):
        layer = Dense.from_config(config)
    ```

    Args:
        custom_objects: Dictionary of `{name: object}` pairs.
    """

    def __init__(self, custom_objects):
        if False:
            while True:
                i = 10
        self.custom_objects = custom_objects or {}
        self.backup = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.backup = global_state.get_global_attribute('custom_objects_scope_dict', {}).copy()
        global_state.set_global_attribute('custom_objects_scope_dict', self.custom_objects.copy())
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        global_state.set_global_attribute('custom_objects_scope_dict', self.backup.copy())
custom_object_scope = CustomObjectScope

@keras_export(['keras.saving.get_custom_objects', 'keras.utils.get_custom_objects'])
def get_custom_objects():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves a live reference to the global dictionary of custom objects.\n\n    Custom objects set using using `custom_object_scope()` are not added to the\n    global dictionary of custom objects, and will not appear in the returned\n    dictionary.\n\n    Example:\n\n    ```python\n    get_custom_objects().clear()\n    get_custom_objects()['MyObject'] = MyObject\n    ```\n\n    Returns:\n        Global dictionary mapping registered class names to classes.\n    "
    return GLOBAL_CUSTOM_OBJECTS

@keras_export(['keras.saving.register_keras_serializable', 'keras.utils.register_keras_serializable'])
def register_keras_serializable(package='Custom', name=None):
    if False:
        print('Hello World!')
    'Registers an object with the Keras serialization framework.\n\n    This decorator injects the decorated class or function into the Keras custom\n    object dictionary, so that it can be serialized and deserialized without\n    needing an entry in the user-provided custom object dict. It also injects a\n    function that Keras will call to get the object\'s serializable string key.\n\n    Note that to be serialized and deserialized, classes must implement the\n    `get_config()` method. Functions do not have this requirement.\n\n    The object will be registered under the key `\'package>name\'` where `name`,\n    defaults to the object name if not passed.\n\n    Example:\n\n    ```python\n    # Note that `\'my_package\'` is used as the `package` argument here, and since\n    # the `name` argument is not provided, `\'MyDense\'` is used as the `name`.\n    @register_keras_serializable(\'my_package\')\n    class MyDense(keras.layers.Dense):\n        pass\n\n    assert get_registered_object(\'my_package>MyDense\') == MyDense\n    assert get_registered_name(MyDense) == \'my_package>MyDense\'\n    ```\n\n    Args:\n        package: The package that this class belongs to. This is used for the\n            `key` (which is `"package>name"`) to idenfify the class. Note that\n            this is the first argument passed into the decorator.\n        name: The name to serialize this class under in this package. If not\n            provided or `None`, the class\' name will be used (note that this is\n            the case when the decorator is used with only one argument, which\n            becomes the `package`).\n\n    Returns:\n        A decorator that registers the decorated class with the passed names.\n    '

    def decorator(arg):
        if False:
            print('Hello World!')
        'Registers a class with the Keras serialization framework.'
        class_name = name if name is not None else arg.__name__
        registered_name = package + '>' + class_name
        if inspect.isclass(arg) and (not hasattr(arg, 'get_config')):
            raise ValueError('Cannot register a class that does not have a get_config() method.')
        GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        GLOBAL_CUSTOM_NAMES[arg] = registered_name
        return arg
    return decorator

@keras_export(['keras.saving.get_registered_name', 'keras.utils.get_registered_name'])
def get_registered_name(obj):
    if False:
        for i in range(10):
            print('nop')
    'Returns the name registered to an object within the Keras framework.\n\n    This function is part of the Keras serialization and deserialization\n    framework. It maps objects to the string names associated with those objects\n    for serialization/deserialization.\n\n    Args:\n        obj: The object to look up.\n\n    Returns:\n        The name associated with the object, or the default Python name if the\n            object is not registered.\n    '
    if obj in GLOBAL_CUSTOM_NAMES:
        return GLOBAL_CUSTOM_NAMES[obj]
    else:
        return obj.__name__

@keras_export(['keras.saving.get_registered_object', 'keras.utils.get_registered_object'])
def get_registered_object(name, custom_objects=None, module_objects=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns the class associated with `name` if it is registered with Keras.\n\n    This function is part of the Keras serialization and deserialization\n    framework. It maps strings to the objects associated with them for\n    serialization/deserialization.\n\n    Example:\n\n    ```python\n    def from_config(cls, config, custom_objects=None):\n        if 'my_custom_object_name' in config:\n            config['hidden_cls'] = tf.keras.saving.get_registered_object(\n                config['my_custom_object_name'], custom_objects=custom_objects)\n    ```\n\n    Args:\n        name: The name to look up.\n        custom_objects: A dictionary of custom objects to look the name up in.\n            Generally, custom_objects is provided by the user.\n        module_objects: A dictionary of custom objects to look the name up in.\n            Generally, module_objects is provided by midlevel library\n            implementers.\n\n    Returns:\n        An instantiable class associated with `name`, or `None` if no such class\n            exists.\n    "
    custom_objects_scope_dict = global_state.get_global_attribute('custom_objects_scope_dict', {})
    if name in custom_objects_scope_dict:
        return custom_objects_scope_dict[name]
    elif name in GLOBAL_CUSTOM_OBJECTS:
        return GLOBAL_CUSTOM_OBJECTS[name]
    elif custom_objects and name in custom_objects:
        return custom_objects[name]
    elif module_objects and name in module_objects:
        return module_objects[name]
    return None