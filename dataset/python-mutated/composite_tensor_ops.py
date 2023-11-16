"""Operations for ExtensionTypes (aka Composite Tensors)."""
from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest

def composite_tensor_to_variants(value, type_spec=None, name=None):
    if False:
        return 10
    "Encodes `value` as a scalar variant tensor.\n\n  Args:\n    value: The `ExtensionType` value to encode.\n    type_spec: Information about the value's type that should be included in the\n      encoding.\n    name: Optional name for the operation.\n\n  Returns:\n    A Tensor with shape=`()` and dtype=`tf.variant`.\n\n  Raises:\n    ValueError: If `type_spec` is not compatible with `value`.\n  "
    if not isinstance(value, composite_tensor.CompositeTensor):
        raise TypeError(f'Expected `value` to be a CompositeTensor. Received {type(value)}.')
    if type_spec is None:
        type_spec = value._type_spec
    if not type_spec.is_compatible_with(value):
        raise ValueError(f'`type_spec` {type_spec} is not compatible with `value` {value!r}.')
    metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
    metadata.type_spec_proto.CopyFrom(nested_structure_coder.encode_structure(type_spec).type_spec_value)
    return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(components=nest.flatten(value, expand_composites=True), metadata=metadata.SerializeToString(), name=name)

def composite_tensor_from_variant(encoded, type_spec, name=None):
    if False:
        i = 10
        return i + 15
    'Returns the `ExtensionType` value encoded by a variant scalar tensor.\n\n  Args:\n    encoded: A Tensor returned by `composite_tensor_to_variants`.\n    type_spec: The `TypeSpec` of the original value.  This is used to determine\n      the number and types of the component tensors that comprise the decoded\n      value.  Must be compatible with the `TypeSpec` serilized in `encoded`.\n    name: Optional name for the operation.\n\n  Returns:\n    An `ExtensionType` value that is compatible with `TypeSpec`.\n\n  Raises:\n    TypeError: If `encoded` is not a Tensor with dtype=variant.\n    InvalidArgumentError: If `encoded` is not compatible with `type_spec`.\n  '
    if not isinstance(encoded, tensor.Tensor):
        raise TypeError(f'Expected `encoded` to be a Tensor, got {encoded!r}.')
    if encoded.dtype != dtypes.variant:
        raise TypeError(f'Expected `encoded` to have dtype=variant, got {encoded!r}.')
    encoded.shape.assert_is_compatible_with(())
    metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
    metadata.type_spec_proto.CopyFrom(nested_structure_coder.encode_structure(type_spec).type_spec_value)
    component_dtypes = [t.dtype for t in nest.flatten(type_spec, expand_composites=True)]
    components = gen_composite_tensor_ops.CompositeTensorVariantToComponents(encoded=encoded, metadata=metadata.SerializeToString(), Tcomponents=component_dtypes, name=name)
    return nest.pack_sequence_as(type_spec, components, expand_composites=True)

@ops.RegisterGradient('CompositeTensorVariantFromComponents')
def _composite_tensor_to_variants_grad(op, grad):
    if False:
        while True:
            i = 10
    return gen_composite_tensor_ops.CompositeTensorVariantToComponents(encoded=grad, metadata=op.get_attr('metadata'), Tcomponents=op.get_attr('Tcomponents'))

@ops.RegisterGradient('CompositeTensorVariantToComponents')
def _composite_tensor_from_variant_grad(op, *grad):
    if False:
        print('Hello World!')
    assert len(grad) == len(op.outputs)
    components = [op.outputs[i] if grad[i] is None else grad[i] for i in range(len(grad))]
    return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(components=components, metadata=op.get_attr('metadata'))