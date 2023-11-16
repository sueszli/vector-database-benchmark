"""Ops for compressing and uncompressing dataset elements."""
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def compress(element):
    if False:
        return 10
    'Compress a dataset element.\n\n  Args:\n    element: A nested structure of types supported by Tensorflow.\n\n  Returns:\n    A variant tensor representing the compressed element. This variant can be\n    passed to `uncompress` to get back the original element.\n  '
    element_spec = structure.type_spec_from_value(element)
    tensor_list = structure.to_tensor_list(element_spec, element)
    return ged_ops.compress_element(tensor_list)

def uncompress(element, output_spec):
    if False:
        i = 10
        return i + 15
    'Uncompress a compressed dataset element.\n\n  Args:\n    element: A scalar variant tensor to uncompress. The element should have been\n      created by calling `compress`.\n    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of\n      the uncompressed element.\n\n  Returns:\n    The uncompressed element.\n  '
    flat_types = structure.get_flat_tensor_types(output_spec)
    flat_shapes = structure.get_flat_tensor_shapes(output_spec)
    tensor_list = ged_ops.uncompress_element(element, output_types=flat_types, output_shapes=flat_shapes)
    return structure.from_tensor_list(output_spec, tensor_list)