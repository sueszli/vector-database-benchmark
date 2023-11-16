"""Types specific to tf.feature_column."""
import abc
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.feature_column.FeatureColumn', v1=[])
class FeatureColumn(object, metaclass=abc.ABCMeta):
    """Represents a feature column abstraction.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  To distinguish between the concept of a feature family and a specific binary
  feature within a family, we refer to a feature family like "country" as a
  feature column. For example, we can have a feature in a `tf.Example` format:
    {key: "country",  value: [ "US" ]}
  In this example the value of feature is "US" and "country" refers to the
  column of the feature.

  This class is an abstract class. Users should not create instances of this.
  """

    @abc.abstractproperty
    def name(self):
        if False:
            print('Hello World!')
        'Returns string. Used for naming.'
        pass

    def __lt__(self, other):
        if False:
            return 10
        'Allows feature columns to be sorted in Python 3 as they are in Python 2.\n\n    Feature columns need to occasionally be sortable, for example when used as\n    keys in a features dictionary passed to a layer.\n\n    In CPython, `__lt__` must be defined for all objects in the\n    sequence being sorted.\n\n    If any objects in the sequence being sorted do not have an `__lt__` method\n    compatible with feature column objects (such as strings), then CPython will\n    fall back to using the `__gt__` method below.\n    https://docs.python.org/3/library/stdtypes.html#list.sort\n\n    Args:\n      other: The other object to compare to.\n\n    Returns:\n      True if the string representation of this object is lexicographically less\n      than the string representation of `other`. For FeatureColumn objects,\n      this looks like "<__main__.FeatureColumn object at 0xa>".\n    '
        return str(self) < str(other)

    def __gt__(self, other):
        if False:
            print('Hello World!')
        'Allows feature columns to be sorted in Python 3 as they are in Python 2.\n\n    Feature columns need to occasionally be sortable, for example when used as\n    keys in a features dictionary passed to a layer.\n\n    `__gt__` is called when the "other" object being compared during the sort\n    does not have `__lt__` defined.\n    Example:\n    ```\n    # __lt__ only class\n    class A():\n      def __lt__(self, other): return str(self) < str(other)\n\n    a = A()\n    a < "b" # True\n    "0" < a # Error\n\n    # __lt__ and __gt__ class\n    class B():\n      def __lt__(self, other): return str(self) < str(other)\n      def __gt__(self, other): return str(self) > str(other)\n\n    b = B()\n    b < "c" # True\n    "0" < b # True\n    ```\n\n    Args:\n      other: The other object to compare to.\n\n    Returns:\n      True if the string representation of this object is lexicographically\n      greater than the string representation of `other`. For FeatureColumn\n      objects, this looks like "<__main__.FeatureColumn object at 0xa>".\n    '
        return str(self) > str(other)

    @abc.abstractmethod
    def transform_feature(self, transformation_cache, state_manager):
        if False:
            return 10
        "Returns intermediate representation (usually a `Tensor`).\n\n    Uses `transformation_cache` to create an intermediate representation\n    (usually a `Tensor`) that other feature columns can use.\n\n    Example usage of `transformation_cache`:\n    Let's say a Feature column depends on raw feature ('raw') and another\n    `FeatureColumn` (input_fc). To access corresponding `Tensor`s,\n    transformation_cache will be used as follows:\n\n    ```python\n    raw_tensor = transformation_cache.get('raw', state_manager)\n    fc_tensor = transformation_cache.get(input_fc, state_manager)\n    ```\n\n    Args:\n      transformation_cache: A `FeatureTransformationCache` object to access\n        features.\n      state_manager: A `StateManager` to create / access resources such as\n        lookup tables.\n\n    Returns:\n      Transformed feature `Tensor`.\n    "
        pass

    @abc.abstractproperty
    def parse_example_spec(self):
        if False:
            while True:
                i = 10
        "Returns a `tf.Example` parsing spec as dict.\n\n    It is used for get_parsing_spec for `tf.io.parse_example`. Returned spec is\n    a dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other\n    supported objects. Please check documentation of `tf.io.parse_example` for\n    all supported spec objects.\n\n    Let's say a Feature column depends on raw feature ('raw') and another\n    `FeatureColumn` (input_fc). One possible implementation of\n    parse_example_spec is as follows:\n\n    ```python\n    spec = {'raw': tf.io.FixedLenFeature(...)}\n    spec.update(input_fc.parse_example_spec)\n    return spec\n    ```\n    "
        pass

    def create_state(self, state_manager):
        if False:
            print('Hello World!')
        'Uses the `state_manager` to create state for the FeatureColumn.\n\n    Args:\n      state_manager: A `StateManager` to create / access resources such as\n        lookup tables and variables.\n    '
        pass

    @abc.abstractproperty
    def _is_v2_column(self):
        if False:
            while True:
                i = 10
        'Returns whether this FeatureColumn is fully conformant to the new API.\n\n    This is needed for composition type cases where an EmbeddingColumn etc.\n    might take in old categorical columns as input and then we want to use the\n    old API.\n    '
        pass

    @abc.abstractproperty
    def parents(self):
        if False:
            print('Hello World!')
        "Returns a list of immediate raw feature and FeatureColumn dependencies.\n\n    For example:\n    # For the following feature columns\n    a = numeric_column('f1')\n    c = crossed_column(a, 'f2')\n    # The expected parents are:\n    a.parents = ['f1']\n    c.parents = [a, 'f2']\n    "
        pass

    def get_config(self):
        if False:
            i = 10
            return i + 15
        "Returns the config of the feature column.\n\n    A FeatureColumn config is a Python dictionary (serializable) containing the\n    configuration of a FeatureColumn. The same FeatureColumn can be\n    reinstantiated later from this configuration.\n\n    The config of a feature column does not include information about feature\n    columns depending on it nor the FeatureColumn class name.\n\n    Example with (de)serialization practices followed in this file:\n    ```python\n    class SerializationExampleFeatureColumn(\n        FeatureColumn, collections.namedtuple(\n            'SerializationExampleFeatureColumn',\n            ('dimension', 'parent', 'dtype', 'normalizer_fn'))):\n\n      def get_config(self):\n        # Create a dict from the namedtuple.\n        # Python attribute literals can be directly copied from / to the config.\n        # For example 'dimension', assuming it is an integer literal.\n        config = dict(zip(self._fields, self))\n\n        # (De)serialization of parent FeatureColumns should use the provided\n        # (de)serialize_feature_column() methods that take care of de-duping.\n        config['parent'] = serialize_feature_column(self.parent)\n\n        # Many objects provide custom (de)serialization e.g: for tf.DType\n        # tf.DType.name, tf.as_dtype() can be used.\n        config['dtype'] = self.dtype.name\n\n        # Non-trivial dependencies should be Keras-(de)serializable.\n        config['normalizer_fn'] = generic_utils.serialize_keras_object(\n            self.normalizer_fn)\n\n        return config\n\n      @classmethod\n      def from_config(cls, config, custom_objects=None, columns_by_name=None):\n        # This should do the inverse transform from `get_config` and construct\n        # the namedtuple.\n        kwargs = config.copy()\n        kwargs['parent'] = deserialize_feature_column(\n            config['parent'], custom_objects, columns_by_name)\n        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])\n        kwargs['normalizer_fn'] = generic_utils.deserialize_keras_object(\n          config['normalizer_fn'], custom_objects=custom_objects)\n        return cls(**kwargs)\n\n    ```\n    Returns:\n      A serializable Dict that can be used to deserialize the object with\n      from_config.\n    "
        return self._get_config()

    def _get_config(self):
        if False:
            return 10
        raise NotImplementedError('Must be implemented in subclasses.')

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            print('Hello World!')
        'Creates a FeatureColumn from its config.\n\n    This method should be the reverse of `get_config`, capable of instantiating\n    the same FeatureColumn from the config dictionary. See `get_config` for an\n    example of common (de)serialization practices followed in this file.\n\n    TODO(b/118939620): This is a private method until consensus is reached on\n    supporting object deserialization deduping within Keras.\n\n    Args:\n      config: A Dict config acquired with `get_config`.\n      custom_objects: Optional dictionary mapping names (strings) to custom\n        classes or functions to be considered during deserialization.\n      columns_by_name: A Dict[String, FeatureColumn] of existing columns in\n        order to avoid duplication. Should be passed to any calls to\n        deserialize_feature_column().\n\n    Returns:\n      A FeatureColumn for the input config.\n    '
        return cls._from_config(config, custom_objects, columns_by_name)

    @classmethod
    def _from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            print('Hello World!')
        raise NotImplementedError('Must be implemented in subclasses.')