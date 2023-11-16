"""Functions that save the model's config into different formats."""
from tensorflow.python.keras.saving.saved_model import json_utils

def model_from_config(config, custom_objects=None):
    if False:
        return 10
    'Instantiates a Keras model from its config.\n\n  Usage:\n  ```\n  # for a Functional API model\n  tf.keras.Model().from_config(model.get_config())\n\n  # for a Sequential model\n  tf.keras.Sequential().from_config(model.get_config())\n  ```\n\n  Args:\n      config: Configuration dictionary.\n      custom_objects: Optional dictionary mapping names\n          (strings) to custom classes or functions to be\n          considered during deserialization.\n\n  Returns:\n      A Keras model instance (uncompiled).\n\n  Raises:\n      TypeError: if `config` is not a dictionary.\n  '
    if isinstance(config, list):
        raise TypeError('`model_from_config` expects a dictionary, not a list. Maybe you meant to use `Sequential.from_config(config)`?')
    from tensorflow.python.keras.layers import deserialize
    return deserialize(config, custom_objects=custom_objects)

def model_from_yaml(yaml_string, custom_objects=None):
    if False:
        print('Hello World!')
    'Parses a yaml model configuration file and returns a model instance.\n\n  Note: Since TF 2.6, this method is no longer supported and will raise a\n  RuntimeError.\n\n  Args:\n      yaml_string: YAML string or open file encoding a model configuration.\n      custom_objects: Optional dictionary mapping names\n          (strings) to custom classes or functions to be\n          considered during deserialization.\n\n  Returns:\n      A Keras model instance (uncompiled).\n\n  Raises:\n      RuntimeError: announces that the method poses a security risk\n  '
    raise RuntimeError('Method `model_from_yaml()` has been removed due to security risk of arbitrary code execution. Please use `Model.to_json()` and `model_from_json()` instead.')

def model_from_json(json_string, custom_objects=None):
    if False:
        print('Hello World!')
    'Parses a JSON model configuration string and returns a model instance.\n\n  Usage:\n\n  >>> model = tf.keras.Sequential([\n  ...     tf.keras.layers.Dense(5, input_shape=(3,)),\n  ...     tf.keras.layers.Softmax()])\n  >>> config = model.to_json()\n  >>> loaded_model = tf.keras.models.model_from_json(config)\n\n  Args:\n      json_string: JSON string encoding a model configuration.\n      custom_objects: Optional dictionary mapping names\n          (strings) to custom classes or functions to be\n          considered during deserialization.\n\n  Returns:\n      A Keras model instance (uncompiled).\n  '
    config = json_utils.decode(json_string)
    from tensorflow.python.keras.layers import deserialize
    return deserialize(config, custom_objects=custom_objects)