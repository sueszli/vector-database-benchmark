"""Class to hold a library of OpDefs and use it to create Brain operations."""
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib

def _Attr(op_def, name):
    if False:
        for i in range(10):
            print('nop')
    for attr in op_def.attr:
        if attr.name == name:
            return attr
    raise TypeError(f"Inconsistent OpDef for '{op_def.name}', missing attr '{name}'")

def _AttrValue(attr_protos, name, op_type_name):
    if False:
        return 10
    if name in attr_protos:
        return attr_protos[name]
    raise TypeError(f"Inconsistent OpDef for '{op_type_name}', missing attr '{name}' from '{attr_protos}'.")

def _SatisfiesTypeConstraint(dtype, attr_def, param_name):
    if False:
        for i in range(10):
            print('nop')
    if attr_def.HasField('allowed_values'):
        allowed_list = attr_def.allowed_values.list.type
        allowed_values = ', '.join((dtypes.as_dtype(x).name for x in allowed_list))
        if dtype not in allowed_list:
            raise TypeError(f"Value passed to parameter '{param_name}' has DataType {dtypes.as_dtype(dtype).name} not in list of allowed values: {allowed_values}")

def _SatisfiesLengthConstraint(length, attr_def, param_name, op_type_name):
    if False:
        while True:
            i = 10
    if attr_def.has_minimum and length < attr_def.minimum:
        raise ValueError(f"Attr '{param_name}' of '{op_type_name}' Op passed list of length {length} less than minimum {attr_def.minimum}.")

def _SatisfiesAllowedStringsConstraint(value, attr_def, arg_name, op_type_name):
    if False:
        return 10
    if value not in attr_def.allowed_values.list.s:
        allowed_values = '", "'.join(map(compat.as_text, attr_def.allowed_values.list.s))
        raise ValueError(f'''Attr '{arg_name}' of '{op_type_name}' Op passed string '{compat.as_text(value)}' not in: "{allowed_values}".''')

def _SatisfiesIntMinimumConstraint(value, attr_def, arg_name, op_type_name):
    if False:
        return 10
    if value < attr_def.minimum:
        raise ValueError(f"Attr '{arg_name}' of '{op_type_name}' Op passed {value} less than minimum {attr_def.minimum}.")

def _IsListParameter(arg):
    if False:
        for i in range(10):
            print('nop')
    if arg.number_attr:
        return True
    elif arg.type_list_attr:
        return True
    return False

def _NumTypeFields(arg):
    if False:
        i = 10
        return i + 15
    num = 0
    if arg.type != types_pb2.DT_INVALID:
        num += 1
    if arg.type_attr:
        num += 1
    if arg.type_list_attr:
        num += 1
    return num

def _IsListValue(v):
    if False:
        while True:
            i = 10
    return isinstance(v, (list, tuple))

def _Flatten(l):
    if False:
        for i in range(10):
            print('nop')
    'Converts [1, 2, [3, 4], [5]] to [1, 2, 3, 4, 5].'
    l_of_l = [x if _IsListValue(x) else [x] for x in l]
    return [item for sublist in l_of_l for item in sublist]

def _Restructure(l, structure):
    if False:
        print('Hello World!')
    'Returns the elements of list l structured according to the given structure.\n\n  A structure is represented by a list whose elements are either\n  `None` or a non-negative integer. `None` corresponds to a single\n  element in the output list, and an integer N corresponds to a nested\n  list of length N.\n\n  The function returns a data structure whose shape is given by\n  `structure`, and whose elements are taken from `l`. If `structure`\n  is a singleton, the function returns the single data structure\n  implied by the 0th element of `structure`. For example:\n\n      _Restructure(["foo", "bar", "baz", "qux"], [None, 2, None])\n        -> ["foo", ["bar", "baz"], "qux"]\n\n      _Restructure(["foo"], [None]) -> "foo"\n\n      _Restructure(["foo"], [1]) -> ["foo"]\n\n      _Restructure([], [0]) -> []\n\n  Args:\n    l: A list.\n    structure: A list whose elements are either `None` or a non-negative\n      integer.\n\n  Returns:\n    The elements of `l`, restructured according to `structure`. If\n    `structure` is a list of length 1, this function returns the\n    single data structure implied by `structure[0]`.\n\n  '
    result = []
    current_index = 0
    for element in structure:
        if element is None:
            result.append(l[current_index])
            current_index += 1
        else:
            result.append(l[current_index:current_index + element])
            current_index += element
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)

def _MakeFloat(v, arg_name):
    if False:
        print('Hello World!')
    if not isinstance(v, compat.real_types):
        raise TypeError(f"Expected float for argument '{arg_name}' not {repr(v)}.")
    return float(v)

def _MakeInt(v, arg_name):
    if False:
        i = 10
        return i + 15
    if isinstance(v, str):
        raise TypeError(f"Expected int for argument '{arg_name}' not {repr(v)}.")
    try:
        return int(v)
    except (ValueError, TypeError):
        raise TypeError(f"Expected int for argument '{arg_name}' not {repr(v)}.")

def _MakeStr(v, arg_name):
    if False:
        while True:
            i = 10
    if not isinstance(v, compat.bytes_or_text_types):
        raise TypeError(f"Expected string for argument '{arg_name}' not {repr(v)}.")
    return compat.as_bytes(v)

def _MakeBool(v, arg_name):
    if False:
        i = 10
        return i + 15
    if not isinstance(v, bool):
        raise TypeError(f"Expected bool for argument '{arg_name}' not {repr(v)}.")
    return v

def _MakeType(v, arg_name):
    if False:
        while True:
            i = 10
    try:
        v = dtypes.as_dtype(v).base_dtype
    except TypeError:
        raise TypeError(f"Expected DataType for argument '{arg_name}' not {repr(v)}.")
    return v.as_datatype_enum

def _MakeShape(v, arg_name):
    if False:
        i = 10
        return i + 15
    'Convert v into a TensorShapeProto.'
    if isinstance(v, tensor_shape_pb2.TensorShapeProto):
        for d in v.dim:
            if d.name:
                logging.warning('Warning: TensorShapeProto with a named dimension: %s', str(v))
                break
        return v
    try:
        return tensor_shape.as_shape(v).as_proto()
    except TypeError as e:
        raise TypeError(f'Error converting {repr(v)} (arg name = {arg_name}) to a TensorShape: {e}')
    except ValueError as e:
        raise TypeError(f'Error converting {repr(v)} (arg name = {arg_name}) to a TensorShape: {e}')

def _MakeTensor(v, arg_name):
    if False:
        for i in range(10):
            print('nop')
    'Ensure v is a TensorProto.'
    if isinstance(v, tensor_pb2.TensorProto):
        return v
    raise TypeError(f"Don't know how to convert {repr(v)} to a TensorProto for argument '{arg_name}'")

def _MakeFunc(v, arg_name):
    if False:
        i = 10
        return i + 15
    'Ensure v is a func.'
    if isinstance(v, attr_value_pb2.NameAttrList):
        return v
    if isinstance(v, compat.bytes_or_text_types):
        fn_attr = attr_value_pb2.NameAttrList(name=v)
    elif hasattr(v, 'add_to_graph'):
        v.add_to_graph(ops.get_default_graph())
        if hasattr(v, '_as_name_attr_list'):
            fn_attr = v._as_name_attr_list
        else:
            fn_attr = attr_value_pb2.NameAttrList(name=v.name)
    else:
        raise TypeError(f"Don't know how to convert {repr(v)} to a func for argument {arg_name}")
    return fn_attr

@tf_contextlib.contextmanager
def _MaybeColocateWith(inputs):
    if False:
        while True:
            i = 10
    'A context manager for (maybe) colocating with a list of input tensors.\n\n  Args:\n    inputs: A list of `Tensor` or `Operation` objects.\n\n  Returns:\n    A context manager.\n  '
    if not inputs:
        yield
    else:
        with ops.colocate_with(inputs[0]), _MaybeColocateWith(inputs[1:]):
            yield

def apply_op(op_type_name, name=None, **keywords):
    if False:
        while True:
            i = 10
    'Add a node invoking a registered Op to a graph.\n\n  Example usage:\n     # input1 and input2 can be Tensors or anything ops.convert_to_tensor()\n     # will convert to a Tensor.\n     op_def_library.apply_op("op", input1=input1, input2=input2)\n     # Can specify a node name.\n     op_def_library.apply_op("op", input1=input1, name="node_name")\n     # Must use keyword arguments, with the names specified in the OpDef.\n     op_def_library.apply_op("op", input_name=input, attr_name=attr)\n\n  All attrs must either be inferred from an input or specified.\n  (If inferred, the attr must not be specified.)  If an attr has a default\n  value specified in the Op\'s OpDef, then you may pass None as the value\n  of that attr to get the default.\n\n  Args:\n    op_type_name: string. Must match the name field of a registered Op.\n    name: string. Optional name of the created op.\n    **keywords: input Tensor and attr arguments specified by name, and optional\n      parameters to pass when constructing the Operation.\n\n  Returns:\n    The Tensor(s) representing the output of the operation, or the Operation\n    itself if there are no outputs.\n\n  Raises:\n    RuntimeError: On some errors.\n    TypeError: On some errors.\n    ValueError: On some errors.\n  '
    (output_structure, is_stateful, op, outputs) = _apply_op_helper(op_type_name, name, **keywords)
    if output_structure:
        res = _Restructure(ops.convert_n_to_tensor(outputs), output_structure)
        if isinstance(res, list) and (not res) and is_stateful:
            return op
        else:
            return res
    else:
        return op

def _ExtractAttrProto(op_type_name, op_def, attrs, attr_protos):
    if False:
        for i in range(10):
            print('nop')
    'Extracts `attr_protos`. For use in _apply_op_helper.'
    for attr_def in op_def.attr:
        key = attr_def.name
        value = attrs[key]
        if attr_def.HasField('default_value') and value is None:
            attr_value = attr_value_pb2.AttrValue()
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        attr_value = value_to_attr_value(value, attr_def.type, key)
        if attr_def.type.startswith('list('):
            _SatisfiesLengthConstraint(len(value), attr_def, key, op_type_name)
        if attr_def.HasField('allowed_values'):
            if attr_def.type == 'string':
                _SatisfiesAllowedStringsConstraint(attr_value.s, attr_def, key, op_type_name)
            elif attr_def.type == 'list(string)':
                for value in attr_value.list.s:
                    _SatisfiesAllowedStringsConstraint(value, attr_def, key, op_type_name)
        if attr_def.has_minimum and attr_def.type == 'int':
            _SatisfiesIntMinimumConstraint(attr_value.i, attr_def, key, op_type_name)
        if attr_def.type == 'type':
            _SatisfiesTypeConstraint(attr_value.type, attr_def, key)
        if attr_def.type == 'list(type)':
            for value in attr_value.list.type:
                _SatisfiesTypeConstraint(value, attr_def, key)
        attr_protos[key] = attr_value

def _ExtractOutputStructure(op_type_name, op_def, attr_protos, output_structure):
    if False:
        i = 10
        return i + 15
    'Extracts `output_structure`. For use in _apply_op_helper.'
    for arg in op_def.output_arg:
        if arg.number_attr:
            n = _AttrValue(attr_protos, arg.number_attr, op_type_name).i
            output_structure.append(n)
        elif arg.type_attr:
            t = _AttrValue(attr_protos, arg.type_attr, op_type_name)
            output_structure.append(None)
        elif arg.type_list_attr:
            t = _AttrValue(attr_protos, arg.type_list_attr, op_type_name)
            output_structure.append(len(t.list.type))
        else:
            output_structure.append(None)

def _CanExtractAttrsFastPath(op_def, keywords):
    if False:
        while True:
            i = 10
    'Check if the fast path for _apply_op_helper is applicable.'
    for input_arg in op_def.input_arg:
        value = keywords.get(input_arg.name, None)
        if not isinstance(value, tensor.Tensor):
            return False
    for attr_def in op_def.attr:
        if attr_def.type == 'func' or attr_def.type == 'list(func)':
            return False
    return True

def _CheckOpDeprecation(op_type_name, op_def, producer):
    if False:
        while True:
            i = 10
    'Checks if the op is deprecated.'
    deprecation_version = op_def.deprecation.version
    if deprecation_version and producer >= deprecation_version:
        raise NotImplementedError(f'Op {op_type_name} is not available in GraphDef version {producer}. It has been removed in version {deprecation_version}. {op_def.deprecation.explanation}.')

def _ExtractDefaultTypesAndAllowedTypes(op_def, default_type_attr_map, allowed_list_attr_map):
    if False:
        print('Hello World!')
    'Extracts the `default_type_attr_map` and `allowed_list_attr_map`.'
    for attr_def in op_def.attr:
        if attr_def.type != 'type':
            continue
        key = attr_def.name
        if attr_def.HasField('default_value'):
            default_type_attr_map[key] = dtypes.as_dtype(attr_def.default_value.type)
        if attr_def.HasField('allowed_values'):
            allowed_list_attr_map[key] = attr_def.allowed_values.list.type

def _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types):
    if False:
        for i in range(10):
            print('nop')
    'Extracts `attrs`, `inputs`, and `input_types` in _apply_op_helper.'
    inferred_from = {}
    for input_arg in op_def.input_arg:
        input_name = input_arg.name
        if input_name in keywords:
            values = keywords.pop(input_name)
        elif input_name + '_' in keywords:
            input_name += '_'
            values = keywords.pop(input_name)
        else:
            raise TypeError(f'No argument for input {input_name} found in {op_def}')
        if _IsListParameter(input_arg):
            if not _IsListValue(values):
                raise TypeError(f"Expected list for '{input_name}' argument to '{op_type_name}' Op, not {values}.")
            dtype = None
            default_dtype = None
            if input_arg.type != types_pb2.DT_INVALID:
                dtype = input_arg.type
            elif input_arg.number_attr:
                if input_arg.type_attr in attrs:
                    dtype = attrs[input_arg.type_attr]
                else:
                    for t in values:
                        if isinstance(t, tensor.Tensor):
                            dtype = t.dtype
                            break
                if dtype is None and input_arg.type_attr in default_type_attr_map:
                    default_dtype = default_type_attr_map[input_arg.type_attr]
            try:
                if not input_arg.is_ref and dtype:
                    dtype = dtypes.as_dtype(dtype).base_dtype
                values = ops.internal_convert_n_to_tensor(values, name=input_arg.name, dtype=dtype if dtype else None, preferred_dtype=default_dtype, as_ref=input_arg.is_ref)
                all_types = set((v.dtype.base_dtype for v in values))
                if input_arg.number_attr and len(all_types) > 1:
                    raise TypeError(f'Not all types matched for {input_arg.name} for {op_type_name}. Got {all_types}')
            except (TypeError, ValueError):
                observed_types = []
                for value in values:
                    try:
                        converted_value = ops.convert_to_tensor(value, as_ref=input_arg.is_ref)
                        observed_types.append(converted_value.dtype.base_dtype.name)
                    except (TypeError, ValueError):
                        observed_types.append('<NOT CONVERTIBLE TO TENSOR>')
                observed = ', '.join(observed_types)
                prefix = "Tensors in list passed to '%s' of '%s' Op have types [%s]" % (input_name, op_type_name, observed)
                if input_arg.number_attr:
                    if input_arg.type != types_pb2.DT_INVALID:
                        raise TypeError(f'{prefix} that do not match expected type {dtype.name}.')
                    elif input_arg.type_attr in attrs:
                        raise TypeError(f'{prefix} that do not match type {dtype.name} inferred from earlier arguments.')
                    else:
                        raise TypeError(f"{prefix} that don't all match.")
                else:
                    raise TypeError(f'{prefix} that are invalid. Tensors: {values}')
            types = [x.dtype for x in values]
            inputs.extend(values)
        else:
            dtype = None
            default_dtype = None
            allowed_list = None
            if input_arg.type != types_pb2.DT_INVALID:
                dtype = input_arg.type
            elif input_arg.type_attr in attrs:
                dtype = attrs[input_arg.type_attr]
            elif input_arg.type_attr in default_type_attr_map:
                default_dtype = default_type_attr_map[input_arg.type_attr]
                allowed_list = allowed_list_attr_map.get(input_arg.type_attr)
            try:
                if dtype is None and allowed_list:
                    inferred = None
                    try:
                        inferred = ops.convert_to_tensor(values, name=input_arg.name, as_ref=input_arg.is_ref)
                    except TypeError as err:
                        pass
                    if inferred is not None and inferred.dtype in allowed_list:
                        values = inferred
                    else:
                        values = ops.convert_to_tensor(values, name=input_arg.name, as_ref=input_arg.is_ref, preferred_dtype=default_dtype)
                else:
                    values = ops.convert_to_tensor(values, name=input_arg.name, dtype=dtype, as_ref=input_arg.is_ref, preferred_dtype=default_dtype)
            except TypeError as err:
                if dtype is None:
                    raise err
                else:
                    raise TypeError(f"Expected {dtypes.as_dtype(dtype).name} passed to parameter '{input_arg.name}' of op '{op_type_name}', got {repr(values)} of type '{type(values).__name__}' instead. Error: {err}")
            except ValueError:
                try:
                    observed = ops.convert_to_tensor(values, as_ref=input_arg.is_ref).dtype.name
                except ValueError as err:
                    raise ValueError(f"Tried to convert '{input_name}' to a tensor and failed. Error: {err}")
                prefix = "Input '%s' of '%s' Op has type %s that does not match" % (input_name, op_type_name, observed)
                if input_arg.type != types_pb2.DT_INVALID:
                    raise TypeError(f'{prefix} expected type of {dtypes.as_dtype(input_arg.type).name}.')
                else:
                    k = input_arg.type_attr
                    if k in default_type_attr_map:
                        if k not in attrs:
                            attrs[k] = default_type_attr_map[k]
                            if k not in inferred_from:
                                inferred_from[k] = 'Default in OpDef'
                    raise TypeError(f"{prefix} type {dtypes.as_dtype(attrs[input_arg.type_attr]).name} of argument '{inferred_from[input_arg.type_attr]}'.")
            types = [values.dtype]
            inputs.append(values)
        base_types = [x.base_dtype for x in types]
        if input_arg.number_attr:
            if input_arg.number_attr in attrs:
                if len(values) != attrs[input_arg.number_attr]:
                    raise ValueError(f"List argument '{input_name}' to '{op_type_name}' Op with length {len(values)} must match length {attrs[input_arg.number_attr]} of argument '{inferred_from[input_arg.number_attr]}'.")
            else:
                attrs[input_arg.number_attr] = len(values)
                inferred_from[input_arg.number_attr] = input_name
                num_attr = _Attr(op_def, input_arg.number_attr)
                if num_attr.has_minimum and len(values) < num_attr.minimum:
                    raise ValueError(f"List argument '{input_name}' to '{op_type_name}' Op with length {len(values)} shorter than minimum length {num_attr.minimum}.")
            if any((bt != base_types[0] for bt in base_types)):
                raise TypeError(f"All tensors passed to '{input_name}' of '{op_type_name}' Op must have the same type. Got {base_types} instead.")
            if input_arg.type != types_pb2.DT_INVALID:
                if base_types and base_types[0] != input_arg.type:
                    assert False, 'Unreachable'
            elif input_arg.type_attr in attrs:
                if base_types and base_types[0] != attrs[input_arg.type_attr]:
                    assert False, 'Unreachable'
            elif not base_types:
                if input_arg.type_attr not in default_type_attr_map:
                    raise TypeError(f"Don't know how to infer type variable from empty input list passed to input '{input_name}' of '{op_type_name}' Op.")
            else:
                attrs[input_arg.type_attr] = base_types[0]
                inferred_from[input_arg.type_attr] = input_name
                type_attr = _Attr(op_def, input_arg.type_attr)
                _SatisfiesTypeConstraint(base_types[0], type_attr, param_name=input_name)
        elif input_arg.type_attr:
            attr_value = base_types[0]
            if input_arg.type_attr in attrs:
                if attrs[input_arg.type_attr] != attr_value:
                    raise TypeError(f"Input '{input_name}' of '{op_type_name}' Op has type {dtypes.as_dtype(attr_value).name} that does not match type {dtypes.as_dtype(attrs[input_arg.type_attr]).name} of argument '{inferred_from[input_arg.type_attr]}'.")
            else:
                for base_type in base_types:
                    _SatisfiesTypeConstraint(base_type, _Attr(op_def, input_arg.type_attr), param_name=input_name)
                attrs[input_arg.type_attr] = attr_value
                inferred_from[input_arg.type_attr] = input_name
        elif input_arg.type_list_attr:
            attr_value = base_types
            if input_arg.type_list_attr in attrs:
                if attrs[input_arg.type_list_attr] != attr_value:
                    actual_types = ', '.join((dtypes.as_dtype(x).name for x in attr_value))
                    expected_types = ', '.join((dtypes.as_dtype(x).name for x in attrs[input_arg.type_list_attr]))
                    raise TypeError(f"Input '{input_name}' of '{op_type_name}' Op has type list of {actual_types} that does not match type list {expected_types} of argument '{inferred_from[input_arg.type_list_attr]}'.")
            else:
                for base_type in base_types:
                    _SatisfiesTypeConstraint(base_type, _Attr(op_def, input_arg.type_list_attr), param_name=input_name)
                attrs[input_arg.type_list_attr] = attr_value
                inferred_from[input_arg.type_list_attr] = input_name
        elif base_types[0] != input_arg.type:
            assert False, 'Unreachable'
        if input_arg.is_ref:
            if not all((x._is_ref_dtype for x in types)):
                raise TypeError(f"'{op_type_name}' Op requires that input '{input_name}' be a mutable tensor (e.g.: a tf.Variable)")
            input_types.extend(types)
        else:
            input_types.extend(base_types)

def _ExtractRemainingAttrs(op_type_name, op_def, keywords, default_type_attr_map, attrs):
    if False:
        i = 10
        return i + 15
    'Extracts the remaining attributes into `attrs` in _apply_op_helper.'
    for attr in op_def.attr:
        if attr.name in attrs:
            if attr.name in keywords:
                raise TypeError(f"Should not specify value for inferred attr '{attr.name}' for {op_type_name}.")
            continue
        if attr.name in keywords:
            attrs[attr.name] = keywords.pop(attr.name)
        elif attr.name + '_' in keywords:
            attrs[attr.name] = keywords.pop(attr.name + '_')
        elif attr.name in default_type_attr_map:
            attrs[attr.name] = default_type_attr_map[attr.name]
        else:
            raise TypeError(f'No argument found for attr {attr.name} for {op_type_name}')

def _GetOpDef(op_type_name, keywords):
    if False:
        print('Hello World!')
    'Returns the OpDef, Graph and Producer. For use in _apply_op_helper.'
    op_def = op_def_registry.get(op_type_name)
    if op_def is None:
        raise RuntimeError(f'Unrecognized Op name {op_type_name}')
    try:
        g = ops._get_graph_from_inputs(_Flatten(keywords.values()))
        producer = g.graph_def_versions.producer
    except AssertionError as e:
        raise RuntimeError(f"Cannot determine graph for Op '{op_type_name}' due to: {e.message}")
    return (op_def, g, producer)

def _CheckAllInputsUsed(op_type_name, keywords):
    if False:
        i = 10
        return i + 15
    'Ensures all inputs passed into _apply_op_helper were used.'
    if keywords:
        all_keywords = ', '.join(sorted(keywords.keys()))
        raise TypeError(f'{op_type_name} got unexpected keyword arguments: {all_keywords}.')

def _apply_op_helper(op_type_name, name=None, **keywords):
    if False:
        while True:
            i = 10
    'Implementation of apply_op that returns output_structure, op.'
    (op_def, g, producer) = _GetOpDef(op_type_name, keywords)
    name = name if name else op_type_name
    (attrs, attr_protos) = ({}, {})
    (default_type_attr_map, allowed_list_attr_map) = ({}, {})
    (inputs, input_types, output_structure) = ([], [], [])
    fallback = True
    if _CanExtractAttrsFastPath(op_def, keywords) and flags.config().graph_building_optimization.value():
        fallback = False
        (attr_protos, inputs, input_types, output_structure) = op_def_library_pybind.process_inputs(op_type_name, producer, keywords)
    if fallback:
        _CheckOpDeprecation(op_type_name, op_def, producer)
        _ExtractDefaultTypesAndAllowedTypes(op_def, default_type_attr_map, allowed_list_attr_map)
    with g.as_default(), ops.name_scope(name) as scope:
        if fallback:
            _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
            _ExtractRemainingAttrs(op_type_name, op_def, keywords, default_type_attr_map, attrs)
            _ExtractAttrProto(op_type_name, op_def, attrs, attr_protos)
            del attrs
            _ExtractOutputStructure(op_type_name, op_def, attr_protos, output_structure)
            _CheckAllInputsUsed(op_type_name, keywords)
        must_colocate_inputs = [val for (arg, val) in zip(op_def.input_arg, inputs) if arg.is_ref]
        with _MaybeColocateWith(must_colocate_inputs):
            op = g._create_op_internal(op_type_name, inputs, dtypes=None, name=scope, input_types=input_types, attrs=attr_protos, op_def=op_def)
        outputs = op.outputs
        if op_callbacks.should_invoke_op_callbacks():
            callback_outputs = op_callbacks.invoke_op_callbacks(op.node_def.op, tuple(op.inputs), attr_protos, tuple(outputs), op_name=op.name, graph=g)
            if callback_outputs is not None:
                outputs = callback_outputs
        return (output_structure, op_def.is_stateful, op, outputs)

def value_to_attr_value(value, attr_type, arg_name):
    if False:
        while True:
            i = 10
    'Encodes a Python value as an `AttrValue` proto message.\n\n  Args:\n    value: The value to convert.\n    attr_type: The value type (string) -- see the AttrValue proto definition for\n      valid strings.\n    arg_name: Argument name (for error messages).\n\n  Returns:\n    An AttrValue proto message that encodes `value`.\n  '
    attr_value = attr_value_pb2.AttrValue()
    if attr_type.startswith('list('):
        if not _IsListValue(value):
            raise TypeError(f'Expected list for attr {arg_name}, obtained {type(value).__name__} instead.')
    if attr_type == 'string':
        attr_value.s = _MakeStr(value, arg_name)
    elif attr_type == 'list(string)':
        attr_value.list.s.extend([_MakeStr(x, arg_name) for x in value])
    elif attr_type == 'int':
        attr_value.i = _MakeInt(value, arg_name)
    elif attr_type == 'list(int)':
        attr_value.list.i.extend([_MakeInt(x, arg_name) for x in value])
    elif attr_type == 'float':
        attr_value.f = _MakeFloat(value, arg_name)
    elif attr_type == 'list(float)':
        attr_value.list.f.extend([_MakeFloat(x, arg_name) for x in value])
    elif attr_type == 'bool':
        attr_value.b = _MakeBool(value, arg_name)
    elif attr_type == 'list(bool)':
        attr_value.list.b.extend([_MakeBool(x, arg_name) for x in value])
    elif attr_type == 'type':
        attr_value.type = _MakeType(value, arg_name)
    elif attr_type == 'list(type)':
        attr_value.list.type.extend([_MakeType(x, arg_name) for x in value])
    elif attr_type == 'shape':
        attr_value.shape.CopyFrom(_MakeShape(value, arg_name))
    elif attr_type == 'list(shape)':
        attr_value.list.shape.extend([_MakeShape(x, arg_name) for x in value])
    elif attr_type == 'tensor':
        attr_value.tensor.CopyFrom(_MakeTensor(value, arg_name))
    elif attr_type == 'list(tensor)':
        attr_value.list.tensor.extend([_MakeTensor(x, arg_name) for x in value])
    elif attr_type == 'func':
        attr_value.func.CopyFrom(_MakeFunc(value, arg_name))
    elif attr_type == 'list(func)':
        attr_value.list.func.extend([_MakeFunc(x, arg_name) for x in value])
    else:
        raise TypeError(f'Unrecognized Attr type {attr_type} for {arg_name}.')
    return attr_value
_pywrap_utils.RegisterPyObject('tf.dtypes.DType', dtypes.DType)
_pywrap_utils.RegisterPyObject('tf.dtypes.as_dtype', dtypes.as_dtype)
_pywrap_utils.RegisterPyObject('tf.TensorShape', tensor_shape.TensorShape)
_pywrap_utils.RegisterPyObject('tf.as_shape', tensor_shape.as_shape)
_pywrap_utils.RegisterPyObject('tf.TensorProto', tensor_pb2.TensorProto)
_pywrap_utils.RegisterPyObject('text_format.Parse', text_format.Parse)
_pywrap_utils.RegisterPyObject('tf.convert_to_tensor', ops.convert_to_tensor)