"""Helper classes that list&validate all attributes to serialize to SavedModel."""
import abc
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils

class SavedModelSaver(object, metaclass=abc.ABCMeta):
    """Saver defining the methods and properties used to serialize Keras objects.
  """

    def __init__(self, obj):
        if False:
            print('Hello World!')
        self.obj = obj

    @abc.abstractproperty
    def object_identifier(self):
        if False:
            return 10
        'String stored in object identifier field in the SavedModel proto.\n\n    Returns:\n      A string with the object identifier, which is used at load time.\n    '
        raise NotImplementedError

    @property
    def tracking_metadata(self):
        if False:
            while True:
                i = 10
        'String stored in metadata field in the SavedModel proto.\n\n    Returns:\n      A serialized JSON storing information necessary for recreating this layer.\n    '
        return json_utils.Encoder().encode(self.python_properties)

    def trackable_children(self, serialization_cache):
        if False:
            for i in range(10):
                print('nop')
        'Lists all Trackable children connected to this object.'
        if not utils.should_save_traces():
            return {}
        children = self.objects_to_serialize(serialization_cache)
        children.update(self.functions_to_serialize(serialization_cache))
        return children

    @abc.abstractproperty
    def python_properties(self):
        if False:
            return 10
        'Returns dictionary of python properties to save in the metadata.\n\n    This dictionary must be serializable and deserializable to/from JSON.\n\n    When loading, the items in this dict are used to initialize the object and\n    define attributes in the revived object.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def objects_to_serialize(self, serialization_cache):
        if False:
            i = 10
            return i + 15
        "Returns dictionary of extra checkpointable objects to serialize.\n\n    See `functions_to_serialize` for an explanation of this function's\n    effects.\n\n    Args:\n      serialization_cache: Dictionary passed to all objects in the same object\n        graph during serialization.\n\n    Returns:\n        A dictionary mapping attribute names to checkpointable objects.\n    "
        raise NotImplementedError

    @abc.abstractmethod
    def functions_to_serialize(self, serialization_cache):
        if False:
            return 10
        "Returns extra functions to include when serializing a Keras object.\n\n    Normally, when calling exporting an object to SavedModel, only the\n    functions and objects defined by the user are saved. For example:\n\n    ```\n    obj = tf.Module()\n    obj.v = tf.Variable(1.)\n\n    @tf.function\n    def foo(...): ...\n\n    obj.foo = foo\n\n    w = tf.Variable(1.)\n\n    tf.saved_model.save(obj, 'path/to/saved/model')\n    loaded = tf.saved_model.load('path/to/saved/model')\n\n    loaded.v  # Variable with the same value as obj.v\n    loaded.foo  # Equivalent to obj.foo\n    loaded.w  # AttributeError\n    ```\n\n    Assigning trackable objects to attributes creates a graph, which is used for\n    both checkpointing and SavedModel serialization.\n\n    When the graph generated from attribute tracking is insufficient, extra\n    objects and functions may be added at serialization time. For example,\n    most models do not have their call function wrapped with a @tf.function\n    decorator. This results in `model.call` not being saved. Since Keras objects\n    should be revivable from the SavedModel format, the call function is added\n    as an extra function to serialize.\n\n    This function and `objects_to_serialize` is called multiple times when\n    exporting to SavedModel. Please use the cache to avoid generating new\n    functions and objects. A fresh cache is created for each SavedModel export.\n\n    Args:\n      serialization_cache: Dictionary passed to all objects in the same object\n        graph during serialization.\n\n    Returns:\n        A dictionary mapping attribute names to `Function` or\n        `ConcreteFunction`.\n    "
        raise NotImplementedError