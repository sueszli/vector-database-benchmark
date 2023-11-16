"""Autograph specifc overrides for tf.data.ops."""
import functools
import numpy as np
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import cond
from tensorflow.python.util import nest

def _verify_spec_compatible(input_name, spec_name, input_, spec):
    if False:
        while True:
            i = 10
    'Verifies that a symbol has a type compatible vith a given spec.\n\n  Here, compatibility is viewed in the general TensorFlow sense: that the dtypes\n  are the same after implicit conversion, if both are tensors.\n\n  This verifier ensures consistent treatment of types across AutoGraph.\n\n  Args:\n    input_name: A name to use for `input_` in error messages.\n    spec_name: A name to use for `spec` in error messages.\n    input_: Any, value to verify.\n    spec: TypeSpec that `input_` must be compatible with.\n\n  Raises:\n    ValueError if the two types have been determined not to be compatible.\n  '
    assert isinstance(spec, tensor_spec.TensorSpec)
    if input is None:
        raise ValueError('{} cannot be None'.format(input_name))
    if isinstance(input_, (bool, int, float, str, np.ndarray)):
        input_ = tensor_conversion.convert_to_tensor_v2(input_)
    input_dtype = getattr(input_, 'dtype', None)
    if input_dtype != spec.dtype:
        input_dtype_str = 'no dtype' if input_dtype is None else str(input_dtype)
        raise TypeError('{} must have the same dtype as {}. Expected {}, got {}'.format(input_name, spec_name, spec.dtype, input_dtype_str))

def _verify_structure_compatible(input_name, spec_name, input_, spec):
    if False:
        while True:
            i = 10
    'Verifies that possibly-structured symbol has types compatible vith another.\n\n  See _verify_spec_compatible for a more concrete meaning of "compatible".\n  Unspec _verify_spec_compatible, which handles singular Tensor-spec objects,\n  verify_structures_compatible can process structures recognized by tf.nest.\n\n  Args:\n    input_name: A name to use for `input_` in error messages.\n    spec_name: A name to use for `spec` in error messages.\n    input_: Any, value to verify. May, but doesn\'t need to, be a structure.\n    spec: Any, value that `input_` must be compatible with. May, but doesn\'t\n      need to, be a structure.\n\n  Raises:\n    ValueError if the two types have been determined not to be compatible.\n  '
    try:
        nest.assert_same_structure(input_, spec, expand_composites=True)
    except (ValueError, TypeError) as e:
        raise TypeError('{} must have the same element structure as {}.\n\n{}'.format(input_name, spec_name, str(e))) from e
    nest.map_structure(functools.partial(_verify_spec_compatible, input_name, spec_name), input_, spec)

def _next_tf_iterator(iterator, default=py_builtins.UNSPECIFIED):
    if False:
        print('Hello World!')
    if default is py_builtins.UNSPECIFIED:
        return next(iterator)
    opt_iterate = iterator.get_next_as_optional()
    _verify_structure_compatible('the default argument', 'the iterate', default, iterator.element_spec)
    return cond.cond(opt_iterate.has_value(), opt_iterate.get_value, lambda : default)

def register_overrides():
    if False:
        print('Hello World!')
    py_builtins.next_registry.register(iterator_ops.OwnedIterator, _next_tf_iterator)
    control_flow.for_loop_registry.register(iterator_ops.OwnedIterator, control_flow._tf_iterator_for_stmt)