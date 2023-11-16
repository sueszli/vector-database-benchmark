"""A set of utils for dealing with nested lists and tuples of Tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import tensorflow as tf
from tensorflow.python.util import nest

def map_nested(map_fn, nested):
    if False:
        print('Hello World!')
    "Executes map_fn on every element in a (potentially) nested structure.\n\n  Args:\n    map_fn: A callable to execute on each element in 'nested'.\n    nested: A potentially nested combination of sequence objects. Sequence\n      objects include tuples, lists, namedtuples, and all subclasses of\n      collections.Sequence except strings. See nest.is_sequence for details.\n      For example [1, ('hello', 4.3)] is a nested structure containing elements\n      1, 'hello', and 4.3.\n  Returns:\n    out_structure: A potentially nested combination of sequence objects with the\n      same structure as the 'nested' input argument. out_structure\n      contains the result of applying map_fn to each element in 'nested'. For\n      example map_nested(lambda x: x+1, [1, (3, 4.3)]) returns [2, (4, 5.3)].\n  "
    out = map(map_fn, nest.flatten(nested))
    return nest.pack_sequence_as(nested, out)

def tile_tensors(tensors, multiples):
    if False:
        return 10
    "Tiles a set of Tensors.\n\n  Args:\n    tensors: A potentially nested tuple or list of Tensors with rank\n      greater than or equal to the length of 'multiples'. The Tensors do not\n      need to have the same rank, but their rank must not be dynamic.\n    multiples: A python list of ints indicating how to tile each Tensor\n      in 'tensors'. Similar to the 'multiples' argument to tf.tile.\n  Returns:\n    tiled_tensors: A potentially nested tuple or list of Tensors with the same\n      structure as the 'tensors' input argument. Contains the result of\n      applying tf.tile to each Tensor in 'tensors'. When the rank of a Tensor\n      in 'tensors' is greater than the length of multiples, multiples is padded\n      at the end with 1s. For example when tiling a 4-dimensional Tensor with\n      multiples [3, 4], multiples would be padded to [3, 4, 1, 1] before tiling.\n  "

    def tile_fn(x):
        if False:
            i = 10
            return i + 15
        return tf.tile(x, multiples + [1] * (x.shape.ndims - len(multiples)))
    return map_nested(tile_fn, tensors)

def where_tensors(condition, x_tensors, y_tensors):
    if False:
        while True:
            i = 10
    "Performs a tf.where operation on a two sets of Tensors.\n\n  Args:\n    condition: The condition tensor to use for the where operation.\n    x_tensors: A potentially nested tuple or list of Tensors.\n    y_tensors: A potentially nested tuple or list of Tensors. Must have the\n    same structure as x_tensors.\n  Returns:\n    whered_tensors: A potentially nested tuple or list of Tensors with the\n      same structure as the 'tensors' input argument. Contains the result of\n      applying tf.where(condition, x, y) on each pair of elements in x_tensors\n      and y_tensors.\n  "
    flat_x = nest.flatten(x_tensors)
    flat_y = nest.flatten(y_tensors)
    result = [tf.where(condition, x, y) for (x, y) in itertools.izip(flat_x, flat_y)]
    return nest.pack_sequence_as(x_tensors, result)

def gather_tensors(tensors, indices):
    if False:
        while True:
            i = 10
    "Performs a tf.gather operation on a set of Tensors.\n\n  Args:\n    tensors: A potentially nested tuple or list of Tensors.\n    indices: The indices to use for the gather operation.\n  Returns:\n    gathered_tensors: A potentially nested tuple or list of Tensors with the\n      same structure as the 'tensors' input argument. Contains the result of\n      applying tf.gather(x, indices) on each element x in 'tensors'.\n  "
    return map_nested(lambda x: tf.gather(x, indices), tensors)

def tas_for_tensors(tensors, length, **kwargs):
    if False:
        print('Hello World!')
    "Unstacks a set of Tensors into TensorArrays.\n\n  Args:\n    tensors: A potentially nested tuple or list of Tensors with length in the\n      first dimension greater than or equal to the 'length' input argument.\n    length: The desired length of the TensorArrays.\n    **kwargs: Keyword args for TensorArray constructor.\n  Returns:\n    tensorarrays: A potentially nested tuple or list of TensorArrays with the\n      same structure as 'tensors'. Contains the result of unstacking each Tensor\n      in 'tensors'.\n  "

    def map_fn(x):
        if False:
            for i in range(10):
                print('nop')
        ta = tf.TensorArray(x.dtype, length, name=x.name.split(':')[0] + '_ta', **kwargs)
        return ta.unstack(x[:length, :])
    return map_nested(map_fn, tensors)

def read_tas(tas, index):
    if False:
        i = 10
        return i + 15
    "Performs a read operation on a set of TensorArrays.\n\n  Args:\n    tas: A potentially nested tuple or list of TensorArrays with length greater\n      than 'index'.\n    index: The location to read from.\n  Returns:\n    read_tensors: A potentially nested tuple or list of Tensors with the same\n      structure as the 'tas' input argument. Contains the result of\n      performing a read operation at 'index' on each TensorArray in 'tas'.\n  "
    return map_nested(lambda ta: ta.read(index), tas)