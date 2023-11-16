"""Tensor-like objects that are composed from tf.Tensors."""
import abc
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.CompositeTensor', v1=[])
class CompositeTensor(metaclass=abc.ABCMeta):
    """Abstract base class for Tensor-like objects that are composed from Tensors.

  Each `CompositeTensor` can be decomposed into a structured collection of
  component `tf.Tensor`s, and reconstructed from those components.

  The `tensorflow.python.util.nest` module has support for treating composite
  tensors as structure, which makes it easy to flatten and reconstruct
  composite tensors (or larger structures that contain composite tensors).
  E.g.:

  ```python
  ct = ...  # Create a composite tensor.
  flat_list_of_tensors = nest.flatten(ct, expand_composites=True)
  transformed_list_of_tensors = ...  # do something with the flat tensors.
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors,
                                 expand_composites=True)
  ```
  """

    @abc.abstractproperty
    def _type_spec(self):
        if False:
            print('Hello World!')
        'A `TypeSpec` describing the type of this value.'
        raise NotImplementedError(f'{type(self).__name__}._type_spec()')

    def _shape_invariant_to_type_spec(self, shape):
        if False:
            print('Hello World!')
        'Returns a TypeSpec given a shape invariant (used by `tf.while_loop`).\n\n    Args:\n      shape: A `tf.TensorShape` object.  The shape invariant for this\n        `CompositeTensor`, or `None` if a default shape invariant should be used\n        (based on the value of this `CompositeTensor`).\n\n    Returns:\n      A nested structure whose values are `tf.TensorShape` objects, specifying\n      the shape invariants for the tensors that comprise this `CompositeTensor`.\n    '
        raise NotImplementedError(f'{type(self).__name__}._shape_invariant_to_type_spec')

    def _consumers(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of `Operation`s that consume this `CompositeTensor`.\n\n    Returns:\n      A list of `Operation`s.\n\n    Raises:\n      RuntimeError: If this method is called while executing eagerly.\n    '
        consumers = nest.flatten([component.consumers() for component in nest.flatten(self, expand_composites=True) if getattr(component, 'graph', None) is not None])
        return list(set(consumers))

    def __tf_tracing_type__(self, context):
        if False:
            while True:
                i = 10
        return self._type_spec.__tf_tracing_type__(context)

    def _convert_variables_to_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        'Converts ResourceVariable components to Tensors.\n\n    Override this method to explicitly convert ResourceVariables embedded in the\n    CompositeTensor to Tensors. By default, it returns the CompositeTensor\n    unchanged.\n\n    Returns:\n      A CompositeTensor with all its ResourceVariable components converted to\n      Tensors.\n    '
        return self
_pywrap_utils.RegisterType('CompositeTensor', CompositeTensor)

def replace_composites_with_components(structure):
    if False:
        for i in range(10):
            print('nop')
    'Recursively replaces CompositeTensors with their components.\n\n  Args:\n    structure: A `nest`-compatible structure, possibly containing composite\n      tensors.\n\n  Returns:\n    A copy of `structure`, where each composite tensor has been replaced by\n    its components.  The result will contain no composite tensors.\n    Note that `nest.flatten(replace_composites_with_components(structure))`\n    returns the same value as `nest.flatten(structure)`.\n  '
    if isinstance(structure, CompositeTensor):
        return replace_composites_with_components(structure._type_spec._to_components(structure))
    elif not nest.is_nested(structure):
        return structure
    else:
        return nest.map_structure(replace_composites_with_components, structure, expand_composites=False)

def convert_variables_to_tensors(composite_tensor):
    if False:
        return 10
    return composite_tensor._convert_variables_to_tensors()