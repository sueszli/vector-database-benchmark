from coremltools.converters.mil.mil import types
from .var import InternalVar
from collections import OrderedDict

class InputSpec(object):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self._input_types = [(k, v) for (k, v) in kwargs.items()]
        self._ordered_dict = OrderedDict()
        for (k, v) in self._input_types:
            self._ordered_dict[k] = v

    def __add__(self, input_spec):
        if False:
            return 10
        self._input_types.extend(input_spec._input_types)
        for (k, v) in input_spec._input_types:
            self._ordered_dict[k] = v
        return self

    @property
    def input_types(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ordered dict[str, _InputType] (name, input_type)\n        '
        return self._ordered_dict

    def parse_inputs(self, kwargs):
        if False:
            return 10
        ' Parse and extract (name, value) pairs from kwargs according to the spec.\n\n        Args:\n            kwargs: must contain a Var compatible with\n                    compatible type for each\n                    1) required _InputType\n                    2) optional _InputType with default value\n\n        Return:\n            out: List[(name, Var or None)]\n                The list has the same length as the `input_types`.\n                `(k, None)` is in the list iff input_type of `k`\n                is optional, has no default value, and\n                `k` is not specified in the input.\n\n        Raise:\n            TypeError if value type is incompatible\n            ValueError if a require input is missing\n        '
        ret = []
        no_check_var_visibility = kwargs.get('no_check_var_visibility', False)
        for (name, input_type) in self.input_types.items():
            if name in kwargs:
                var = kwargs[name]
                if isinstance(var, InternalVar) or input_type.is_compatible(var):
                    ret.append((name, var))
                else:
                    msg = 'Input {} has type {} not compatible with expected type {}'.format(name, var.sym_type, input_type)
                    raise TypeError(msg)
            elif not input_type.optional or input_type.default:
                if no_check_var_visibility or isinstance(input_type, PyFunctionInputType):
                    continue
                raise ValueError('Input {} is required'.format(name))
            else:
                assert input_type.default is None
                ret.append((name, None))
        return ret

class _InputType(object):
    """
    (Untyped) input containing fundamental properties of all inputs to an
    Operation:
    """

    def __init__(self, const=False, default=None, optional=False):
        if False:
            print('Hello World!')
        '\n        const (bool):\n            True if the InputType has to be constant / materialized at compile time.\n            Const InputType is semantically equivalent to attribute. By\n            default False. Read-only.\n\n        optional (bool):\n            If default is not None, optional will be set to True\n\n        default:\n            Default value of optional input. InputType is optional if a default\n            is provided or optional == True.  default can be int, float,\n            string, np.ndarray etc depending on subclass.\n\n        Note: _InputType should not be directly instantiated. Only its subclasses may\n        be instantiated.\n        '
        self.default = default
        self.const = const
        self.optional = True if default is not None else optional

    def is_compatible(self, v):
        if False:
            while True:
                i = 10
        '\n        Return True if (possibly symbolic) value `v` is compatible. False\n        otherwise.\n\n        Inputs:\n\n        v (Var | ListVar | native python function): input\n\n        Comment: Define is_compatible as instance method to call proper subclass\n        methods.\n        '
        return self._is_compatible(v)

    def _is_compatible(self, v):
        if False:
            print('Hello World!')
        return True

    def _get_predefined_datatype(self):
        if False:
            while True:
                i = 10
        '\n        Override this function if datatype can be known without `_default` or\n        `_val`.\n        '
        return None

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return type(self).__name__

class ListInputType(_InputType):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ListInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            i = 10
            return i + 15
        return types.is_list(v.sym_type)

class ScalarOrTensorInputType(_InputType):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ScalarOrTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            while True:
                i = 10
        return types.is_scalar(v.dtype) or types.is_tensor(v.dtype)

class ListOrScalarOrTensorInputType(_InputType):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(ListOrScalarOrTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            return 10
        return types.is_list(v.sym_type) or types.is_scalar(v.dtype) or types.is_tensor(v.dtype)

class IntInputType(ScalarOrTensorInputType):
    """
    Int input with _sym_type == types.int32 or _sym_type == types.int64
    predefined to be types.int32 by default.

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(IntInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            return 10
        return v.dtype in {types.int32, types.int64}

    def _get_predefined_datatype(self):
        if False:
            while True:
                i = 10
        return types.int32

class BoolInputType(ScalarOrTensorInputType):
    """
    Int32 input, with _sym_type == types.int32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(BoolInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            while True:
                i = 10
        return v.dtype == types.bool

    def _get_predefined_datatype(self):
        if False:
            print('Hello World!')
        return types.bool

class FloatInputType(ScalarOrTensorInputType):
    """
    fp32 input, with _sym_type == types.fp32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(FloatInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            i = 10
            return i + 15
        return v.dtype == types.fp32

    def _get_predefined_datatype(self):
        if False:
            return 10
        return types.fp32

class IntOrFloatInputType(ScalarOrTensorInputType):
    """
    input with _sym_type == types.int32 or _sym_type == types.int64 or _sym_type == types.fp32
    predefined to be types.fp32 by default.
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        super(IntOrFloatInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            for i in range(10):
                print('nop')
        return v.dtype in {types.int32, types.int64, types.fp32}

    def _get_predefined_datatype(self):
        if False:
            for i in range(10):
                print('nop')
        return types.fp32

class TensorInputType(ScalarOrTensorInputType):
    """
    TensorInputType must be numpy ndarray of numeric types. Min rank = 1. (Use
    ScalarOrTensorInputType for possibly scalar input).
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(TensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            i = 10
            return i + 15
        return types.is_tensor(v.sym_type)

class IntTensorInputType(ScalarOrTensorInputType):
    """
    Tensor input with int values, _sym_type == types.int32 or
    _sym_type == types.int64

    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(IntTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            for i in range(10):
                print('nop')
        return types.is_tensor(v.sym_type) and v.dtype in {types.int32, types.int64}

class IntOrIntTensorInputType(ScalarOrTensorInputType):
    """
    builtins.in32 or Tensor with int values, _sym_type == builtins.int32 or
    _sym_type == builtins.int64

    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(IntOrIntTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            return 10
        return v.dtype in {types.int32, types.int64}

class BoolTensorInputType(ScalarOrTensorInputType):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(BoolTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            while True:
                i = 10
        return types.is_tensor(v.sym_type) and v.dtype == types.bool

class StringInputType(ScalarOrTensorInputType):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(StringInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            for i in range(10):
                print('nop')
        return types.is_str(v.sym_type)

class TupleInputType(_InputType):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(TupleInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            print('Hello World!')
        return isinstance(v, (tuple, list))

class InternalInputType(_InputType):
    """
    InternalInputType specifies input types outside of Program's type system.
    It allows ops to take, for example, python primitive types, instead of
    only the builtin types.
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(InternalInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        if False:
            while True:
                i = 10
        return True

class PyFunctionInputType(InternalInputType):
    """
    Native python function.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(PyFunctionInputType, self).__init__(**kwargs)

class InternalStringInputType(InternalInputType):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(InternalStringInputType, self).__init__(**kwargs)

class InternalScalarOrTensorInputType(InternalInputType):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(InternalScalarOrTensorInputType, self).__init__(**kwargs)