import copy
from collections import defaultdict
from collections.abc import Sequence
from uuid import uuid4
from weakref import WeakKeyDictionary
import numpy as np
import paddle
from paddle.pir.core import convert_np_dtype_to_dtype_
from ..base.data_feeder import check_dtype, convert_dtype
from ..base.framework import Block, Variable, _current_expected_place, in_dygraph_mode

def convert_to_list(value, n, name, dtype=int):
    if False:
        while True:
            i = 10
    '\n    Converts a single numerical type or iterable of numerical\n    types into a numerical type list.\n\n    Arguments:\n      value: The value to validate and convert. Could an int, or any iterable\n        of ints.\n      n: The size of the list to be returned.\n      name: The name of the argument being validated, e.g. "stride" or\n        "filter_size". This is only used to format error messages.\n      dtype: the numerical type of the element of the list to be returned.\n\n    Returns:\n      A list of n dtypes.\n\n    Raises:\n      ValueError: If something else than an int/long or iterable thereof was\n        passed.\n    '
    if isinstance(value, dtype):
        return [value] * n
    else:
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError('The ' + name + "'s type must be list or tuple. Received: " + str(value))
        if len(value_list) != n:
            raise ValueError('The ' + name + "'s length must be " + str(n) + '. Received: ' + str(value))
        for single_value in value_list:
            assert not isinstance(single_value, (Variable, paddle.pir.OpResult)), "Required numerical type with '%s', but received Tensor." % dtype
            try:
                dtype(single_value)
            except (ValueError, TypeError):
                raise ValueError('The ' + name + "'s type must be a list or tuple of " + str(n) + ' ' + str(dtype) + ' . Received: ' + str(value) + ' including element ' + str(single_value) + ' of type' + ' ' + str(type(single_value)))
        return value_list

def is_sequence(seq):
    if False:
        for i in range(10):
            print('nop')
    '\n    Whether `seq` is an entry or nested structure\n    '
    if isinstance(seq, dict):
        return True
    return isinstance(seq, Sequence) and (not isinstance(seq, str))

class UniqueIdMap(WeakKeyDictionary):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(self)
        self.data = defaultdict(uuid4)
uniqueidmap = UniqueIdMap()

def uniqueid(obj):
    if False:
        i = 10
        return i + 15
    if isinstance(obj, str):
        return (hash(obj),)
    elif isinstance(obj, list):
        return (id(obj),)
    else:
        return (uniqueidmap[obj].int,)

def _hash_with_id(*args):
    if False:
        i = 10
        return i + 15
    '\n    Return int hash value calculated by id(arg) or tuple(id1,id2, ...).\n    '
    assert len(args) > 0
    info = ()
    for v in args:
        info = info + uniqueid(v)
    return hash(info)

def _sorted(dict_):
    if False:
        i = 10
        return i + 15
    '\n    Returns a sorted list of the dict keys, with error if keys not sortable.\n    '
    try:
        return sorted(dict_.keys())
    except TypeError:
        raise TypeError('nest only supports dicts with sortable keys.')

def _yield_value(iterable):
    if False:
        i = 10
        return i + 15
    if isinstance(iterable, dict):
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        yield from iterable

def _yield_flat_nest(nest):
    if False:
        while True:
            i = 10
    for n in _yield_value(nest):
        if is_sequence(n):
            yield from _yield_flat_nest(n)
        else:
            yield n

def to_sequence(nest):
    if False:
        while True:
            i = 10
    if is_sequence(nest):
        return nest
    else:
        return [nest]

def flatten(nest):
    if False:
        return 10
    '\n        :alias_main: paddle.flatten\n        :alias: paddle.flatten,paddle.tensor.flatten,paddle.tensor.manipulation.flatten\n        :old_api: paddle.base.layers.flatten\n\n    Traverse all entries in the nested structure and put them into an list.\n    '
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]

def _sequence_like(instance, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert the sequence `args` to the same type as `instance`.\n    '
    if isinstance(instance, dict):
        result = dict(zip(_sorted(instance), args))
        return type(instance)(((key, result[key]) for key in instance.keys()))
    elif isinstance(instance, tuple) and hasattr(instance, '_fields') and isinstance(instance._fields, Sequence) and all((isinstance(f, str) for f in instance._fields)):
        return type(instance)(*args)
    else:
        return type(instance)(args)

def _packed_nest_with_indices(structure, flat, index):
    if False:
        print('Hello World!')
    '\n    Helper function for pack_sequence_as.\n    '
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            (new_index, child) = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)

def pack_sequence_as(structure, flat_sequence):
    if False:
        i = 10
        return i + 15
    '\n    Pack a given flattened sequence into a given structure.\n    '
    if not is_sequence(flat_sequence):
        raise TypeError('flat_sequence must be a sequence')
    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError('Structure is a scalar but len(flat_sequence) == %d > 1' % len(flat_sequence))
        return flat_sequence[0]
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError('Could not pack sequence. Structure had %d elements, but flat_sequence had %d elements.  Structure: %s, flat_sequence: %s.' % (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    (_, packed) = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)

def map_structure(func, *structure):
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply `func` to each entry in `structure` and return a new structure.\n    '
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure[0], [func(*x) for x in entries])

def hold_mutable_vars(structure):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns whether structure holds sequence like `list/dict`.\n    '
    for s in structure:
        if is_sequence(s):
            return True
    return False

def copy_mutable_vars(structure):
    if False:
        return 10
    '\n    Returns vars copied from sequence without mutable property.\n    '
    flat_structure = copy.copy(flatten(structure))
    return pack_sequence_as(structure, flat_structure)

def _recursive_assert_same_structure(nest1, nest2, check_types):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for `assert_same_structure`.\n    '
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(f"The two structures don't have the same nested structure.\n\nFirst structure: {nest1}\n\nSecond structure: {nest2}.")
    if not is_sequence_nest1:
        return
    if check_types:
        type_nest1 = type(nest1)
        type_nest2 = type(nest2)
        if type_nest1 != type_nest2:
            raise TypeError("The two structures don't have the same sequence type. First structure has type {}, while second structure has type {}.".format(type_nest1, type_nest2))
        if isinstance(nest1, dict):
            keys1 = set(nest1.keys())
            keys2 = set(nest2.keys())
            if keys1 != keys2:
                raise ValueError("The two dictionaries don't have the same set of keys. First structure has keys {}, while second structure has keys {}.".format(keys1, keys2))
    nest1_as_sequence = list(_yield_value(nest1))
    nest2_as_sequence = list(_yield_value(nest2))
    for (n1, n2) in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2, check_types)

def padding_to_same_structure(nest1, nest2, obj=None):
    if False:
        for i in range(10):
            print('nop')

    def _padding_to_same_structure_single(value, obj):
        if False:
            while True:
                i = 10

        def change_none_to_obj(x):
            if False:
                while True:
                    i = 10
            if x is None:
                return obj
            return x
        if is_sequence(value):
            value = pack_sequence_as(value, [change_none_to_obj(item) for item in flatten(value)])
        else:
            value = change_none_to_obj(value)
        return value
    nest1 = _padding_to_same_structure_single(nest1, obj)
    nest2 = _padding_to_same_structure_single(nest2, obj)
    return (nest1, nest2)

def assert_same_structure(nest1, nest2, check_types=True):
    if False:
        i = 10
        return i + 15
    '\n    Confirm two nested structures with the same structure.\n    '
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("The two structures don't have the same number of elements.\n\nFirst structure (%i elements): %s\n\nSecond structure (%i elements): %s" % (len_nest1, nest1, len_nest2, nest2))
    _recursive_assert_same_structure(nest1, nest2, check_types)

def _is_symmetric_padding(padding, data_dim):
    if False:
        print('Hello World!')
    '\n    Check whether padding is symmetrical.\n    '
    assert len(padding) == data_dim * 2 or len(padding) == data_dim
    is_sys = True
    if len(padding) == data_dim * 2:
        for i in range(data_dim):
            if padding[i * 2] != padding[i * 2 + 1]:
                is_sys = False
    return is_sys

def _contain_var(list_or_tuple):
    if False:
        print('Hello World!')
    '\n    Check whether list or tuple contains variable / OpResult.\n    '
    for item in list_or_tuple:
        if isinstance(item, (Variable, paddle.pir.OpResult)):
            return True
    return False

def get_int_tensor_list(ele_list, place=None, default_dtype='int64'):
    if False:
        while True:
            i = 10
    if place is None:
        place = _current_expected_place()
    int_tensor_list = []
    for ele in ele_list:
        if isinstance(ele, paddle.pir.OpResult):
            ele.stop_gradient = True
            if convert_dtype(ele.dtype) != default_dtype:
                ele = paddle.cast(x=ele, dtype=default_dtype)
            if ele.shape != []:
                ele = paddle.reshape(ele, [])
            int_tensor_list.append(ele)
        else:
            temp_out = paddle.full([], ele, convert_np_dtype_to_dtype_(np.dtype(default_dtype)), place)
            int_tensor_list.append(temp_out)
    return int_tensor_list

def get_shape_tensor_inputs(inputs, attrs, shape, op_type):
    if False:
        print('Hello World!')
    from paddle.tensor import fill_constant

    def _get_attr_shape(list_shape):
        if False:
            while True:
                i = 10
        attr_shape = []
        for (idx, dim) in enumerate(list_shape):
            if isinstance(dim, Variable):
                attr_shape.append(-1)
            else:
                attr_shape.append(dim)
        return attr_shape

    def _get_shape_tensor(list_shape):
        if False:
            print('Hello World!')
        shape_tensor_list = []
        for (idx, dim) in enumerate(list_shape):
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                check_dtype(dim.dtype, 'shape[' + str(idx) + ']', ['int32', 'int64'], op_type, '(When type of shape in' + op_type + 'is list or tuple.)')
                if convert_dtype(dim.dtype) == 'int64':
                    dim = paddle.cast(x=dim, dtype='int32')
                shape_tensor_list.append(dim)
            else:
                temp_out = fill_constant([], 'int32', dim, force_cpu=True)
                shape_tensor_list.append(temp_out)
        return shape_tensor_list
    if isinstance(shape, Variable):
        shape.stop_gradient = True
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant', '(When type of shape in' + op_type + ' is Variable.)')
        if convert_dtype(shape.dtype) == 'int64':
            shape = paddle.cast(shape, 'int32')
        inputs['ShapeTensor'] = shape
    elif isinstance(shape, (list, tuple)):
        attrs['shape'] = _get_attr_shape(shape)
        if _contain_var(shape):
            inputs['ShapeTensorList'] = _get_shape_tensor(shape)
    else:
        raise TypeError('Shape only supports Variable, or list, or tuple.')

def _convert_to_tensor_list(old_list, dtype='int32'):
    if False:
        return 10
    '\n    Converts all elements of a list to Variable / OpResult.\n    '
    from paddle.tensor import fill_constant
    new_list_tensor = []
    for ele in old_list:
        if isinstance(ele, (Variable, paddle.pir.OpResult)):
            ele.stop_gradient = True
            new_list_tensor.append(ele)
        else:
            assert isinstance(ele, int)
            temp_out = fill_constant([1], dtype, ele, force_cpu=True)
            new_list_tensor.append(temp_out)
    return new_list_tensor

def convert_shape_to_list(shape):
    if False:
        return 10
    '\n    Convert shape(list, tuple, variable) to list in imperative mode\n    '
    if isinstance(shape, (list, tuple)):
        shape = [x.item(0) if isinstance(x, Variable) else x for x in shape]
    elif in_dygraph_mode():
        shape = shape.astype(int).tolist()
    return shape

def check_shape(shape):
    if False:
        i = 10
        return i + 15
    '\n    Check shape type and shape elements type before passing it to fill_constant\n    '
    if isinstance(shape, Variable):
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant')
    else:
        for ele in shape:
            if not isinstance(ele, Variable):
                if ele < 0:
                    raise ValueError("All elements in ``shape`` must be positive when it's a list or tuple")
                if not isinstance(ele, int):
                    raise TypeError("All elements in ``shape`` must be integers when it's a list or tuple")

def try_set_static_shape_tensor(tensor, shape):
    if False:
        for i in range(10):
            print('nop')
    'Try to set static shape of tensor from a shape tensor.\n\n    For example,\n\n    import paddle\n    paddle.enable_static()\n    data = paddle.static.data(name="x", shape=[-1, 2], dtype=\'float32\')\n    shape = paddle.shape(data)  # shape should be [-1, 2] instead of [-1, -1]\n    x = paddle.uniform(shape)\n    print(x.shape)\n    # (-1, 2)\n\n    '
    if not in_dygraph_mode():
        if -1 in tensor.shape:
            if isinstance(shape, Variable):
                shape = try_get_constant_shape_from_tensor(shape)
                if shape:
                    tensor.desc.set_shape(shape)

def try_get_constant_shape_from_tensor(shape_tensor):
    if False:
        return 10
    'Try to get shape from a tensor with constant value.\n\n    For example,\n\n    import paddle\n    paddle.enable_static()\n    data = paddle.static.data(name="x", shape=[-1, 2], dtype=\'float32\')\n    shape = paddle.shape(data)  # shape should be [-1, 2] instead of [-1, -1]\n    x = paddle.uniform(shape)\n    print(x.shape)\n    # (-1, 2)\n\n    '
    if not in_dygraph_mode():
        try:
            if shape_tensor.op is not None:
                generate_op = shape_tensor.op
                if generate_op.type == 'shape':
                    var = shape_tensor.block.vars[generate_op.input_arg_names[0]]
                    return var.shape
        except:
            return None
        return None

def get_inputs_outputs_in_block(block):
    if False:
        print('Hello World!')
    '\n    Returns the inputs and outputs variable used in this block but not\n    created in this block.\n    '
    assert isinstance(block, Block), 'input non-Block argument for get_inputs_outputs_in_block.'
    assert block.parent_idx != -1, 'input block should be a sub-block, not main block.'
    inner_inputs = set()
    inner_outputs = set()
    for op in block.ops:
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if not block.has_var(in_var_name):
                    inner_inputs.add(in_var_name)
        for oname in op.output_names:
            for out_var_name in op.output(oname):
                if not block.has_var(out_var_name):
                    inner_outputs.add(out_var_name)
    return (inner_inputs, inner_outputs)