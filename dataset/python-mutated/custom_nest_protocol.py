"""Protocol class for custom tf.nest support."""
import typing
from typing import Protocol

@typing.runtime_checkable
class CustomNestProtocol(Protocol):
    """Protocol for adding custom tf.nest support in user-defined classes.

  User classes should implement the two methods defined in this protocol in
  order to be supported by nest functions.
    - `__tf_flatten__` for generating the flattened components and the metadata
      of the current object.
    - `__tf_unflatten__` for creating a new object based on the input metadata
      and the components.
  See the method doc for details.

  In terms of support level, classes implementing this protocol
    - are supported by tf.nest and tf.data functions.
    - have limited support from tf.function, which requires writing a custom
      TraceType subclass to be used as the input or output of a tf.function.
    - are NOT supported by SavedModel.

  Code Examples:

  >>> import dataclasses
  >>> @dataclasses.dataclass
  ... class MaskedTensor:
  ...   mask: bool
  ...   value: tf.Tensor
  ...
  ...   def __tf_flatten__(self):
  ...     metadata = (self.mask,)  # static config.
  ...     components = (self.value,)  # dynamic values.
  ...     return metadata, components
  ...
  ...   @classmethod
  ...   def __tf_unflatten__(cls, metadata, components):
  ...     mask = metadata[0]
  ...     value = components[0]
  ...     return MaskedTensor(mask=mask, value=value)
  ...
  >>> mt = MaskedTensor(mask=True, value=tf.constant([1]))
  >>> mt
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([1], dtype=int32)>)
  >>> tf.nest.is_nested(mt)
  True
  >>> mt2 = MaskedTensor(mask=False, value=tf.constant([2]))
  >>> tf.nest.assert_same_structure(mt, mt2)

  >>> leaves = tf.nest.flatten(mt)
  >>> leaves
  [<tf.Tensor: shape=(1,), dtype=int32, numpy=array([1], dtype=int32)>]

  >>> mt3 = tf.nest.pack_sequence_as(mt, leaves)
  >>> mt3
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([1], dtype=int32)>)
  >>> bool(mt == mt3)
  True

  >>> tf.nest.map_structure(lambda x: x * 2, mt)
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([2], dtype=int32)>)

  More examples are available in the unit tests (nest_test.py).
  """

    def __tf_flatten__(self):
        if False:
            return 10
        'Flatten current object into (metadata, components).\n\n    Returns:\n      A `tuple` of (metadata, components), where\n        - metadata is a custom Python object that stands for the static config\n          of the current object, which is supposed to be fixed and not affected\n          by data transformation.\n        - components is a `tuple` that contains the modifiable fields of the\n          current object.\n\n    Implementation Note:\n    - This method should not invoke any TensorFlow ops.\n    - This method only needs to flatten the current level. If current object has\n      an attribute that also need custom flattening, nest functions (such as\n      `nest.flatten`) will utilize this method to do recursive flattening.\n    - Components must ba a `tuple`, not a `list`\n    '

    @classmethod
    def __tf_unflatten__(cls, metadata, components):
        if False:
            i = 10
            return i + 15
        'Create a user-defined object from (metadata, components).\n\n    Args:\n      metadata: a custom Python objet that stands for the static config for\n        reconstructing a new object of the current class.\n      components: a `tuple` that contains the dynamic data fields of the current\n        class, for object reconstruction.\n\n    Returns:\n      The user-defined object, with the same class of the current object.\n\n    Implementation Note:\n    - This method should not invoke any TensorFlow ops.\n    - This method only needs to unflatten the current level. If the object has\n      an attribute that also need custom unflattening, nest functions will\n      utilize this method to do recursive unflattening.\n    '