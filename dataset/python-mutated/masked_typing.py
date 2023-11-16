import operator
import numpy as np
from numba import types
from numba.core.extending import make_attribute_wrapper, models, register_model, typeof_impl
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate, ConcreteTemplate
from numba.core.typing.typeof import typeof
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.np.numpy_support import from_dtype
from cudf.core.missing import NA
from cudf.core.udf import api
from cudf.core.udf._ops import arith_ops, bitwise_ops, comparison_ops, unary_ops
from cudf.core.udf.strings_typing import StringView, UDFString, bool_binary_funcs, id_unary_funcs, int_binary_funcs, size_type, string_return_attrs, string_unary_funcs, string_view, udf_string
from cudf.utils.dtypes import DATETIME_TYPES, NUMERIC_TYPES, STRING_TYPES, TIMEDELTA_TYPES
SUPPORTED_NUMPY_TYPES = NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | STRING_TYPES
supported_type_str = '\n'.join(sorted(list(SUPPORTED_NUMPY_TYPES) + ['bool']))
_units = ['ns', 'ms', 'us', 's']
_datetime_cases = {types.NPDatetime(u) for u in _units}
_timedelta_cases = {types.NPTimedelta(u) for u in _units}
_supported_masked_types = types.integer_domain | types.real_domain | _datetime_cases | _timedelta_cases | {types.boolean} | {string_view, udf_string}
SUPPORTED_NUMBA_TYPES = (types.Number, types.Boolean, types.NPDatetime, types.NPTimedelta, StringView, UDFString)

def _format_error_string(err):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap an error message in newlines and color it red.\n    '
    return '\x1b[91m' + '\n' + err + '\n' + '\x1b[0m'

def _type_to_masked_type(t):
    if False:
        return 10
    if isinstance(t, SUPPORTED_NUMBA_TYPES):
        return t
    else:
        err = _format_error_string(f'Unsupported MaskedType. This is usually caused by attempting to use a column of unsupported dtype in a UDF. Supported dtypes are:\n{supported_type_str}')
        return types.Poison(err)

class MaskedType(types.Type):
    """
    A Numba type consisting of a value of some primitive type
    and a validity boolean, over which we can define math ops
    """

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value_type = _type_to_masked_type(value)
        super().__init__(name=f'Masked({self.value_type})')

    def __hash__(self):
        if False:
            return 10
        '\n        Needed so that numba caches type instances with different\n        `value_type` separately.\n        '
        return hash(repr(self))

    def unify(self, context, other):
        if False:
            return 10
        '\n        Often within a UDF an instance arises where a variable could\n        be a `MaskedType`, an `NAType`, or a literal based off\n        the data at runtime, for example the variable `ret` here:\n\n        def f(x):\n            if x == 1:\n                ret = x\n            elif x > 2:\n                ret = 1\n            else:\n                ret = cudf.NA\n            return ret\n\n        When numba analyzes this function it will eventually figure\n        out that the variable `ret` could be any of the three types\n        from above. This scenario will only work if numba knows how\n        to find some kind of common type between the possibilities,\n        and this function implements that - the goal is to return a\n        common type when comparing `self` to other.\n\n        '
        if isinstance(other, NAType):
            return self
        elif isinstance(other, MaskedType):
            return MaskedType(context.unify_pairs(self.value_type, other.value_type))
        unified = context.unify_pairs(self.value_type, other)
        if unified is None:
            return None
        return MaskedType(unified)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, MaskedType):
            return False
        return self.value_type == other.value_type

@typeof_impl.register(api.Masked)
def typeof_masked(val, c):
    if False:
        for i in range(10):
            print('nop')
    return MaskedType(typeof(val.value))

@cuda_decl_registry.register
class MaskedConstructor(ConcreteTemplate):
    key = api.Masked
    cases = [nb_signature(MaskedType(t), t, types.boolean) for t in _supported_masked_types]

@cuda_decl_registry.register_attr
class ClassesTemplate(AttributeTemplate):
    key = types.Module(api)

    def resolve_Masked(self, mod):
        if False:
            while True:
                i = 10
        return types.Function(MaskedConstructor)
cuda_decl_registry.register_global(api, types.Module(api))
cuda_decl_registry.register_global(api.Masked, types.Function(MaskedConstructor))
make_attribute_wrapper(MaskedType, 'value', 'value')
make_attribute_wrapper(MaskedType, 'valid', 'valid')

@register_model(MaskedType)
class MaskedModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            i = 10
            return i + 15
        members = [('value', fe_type.value_type), ('valid', types.bool_)]
        models.StructModel.__init__(self, dmm, fe_type, members)

class NAType(types.Type):
    """
    A type for handling ops against nulls
    Exists so we can:
    1. Teach numba that all occurrences of `cudf.NA` are
       to be read as instances of this type instead
    2. Define ops like `if x is cudf.NA` where `x` is of
       type `Masked` to mean `if x.valid is False`
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(name='NA')

    def unify(self, context, other):
        if False:
            print('Hello World!')
        '\n        Masked  <-> NA is deferred to MaskedType.unify()\n        Literal <-> NA -> Masked\n        '
        if isinstance(other, MaskedType):
            return None
        elif isinstance(other, NAType):
            return self
        else:
            return MaskedType(other)
na_type = NAType()

@typeof_impl.register(type(NA))
def typeof_na(val, c):
    if False:
        return 10
    '\n    Tie instances of _NAType (cudf.NA) to our NAType.\n    Effectively make it so numba sees `cudf.NA` as an\n    instance of this NAType -> handle it accordingly.\n    '
    return na_type
register_model(NAType)(models.OpaqueModel)

class MaskedScalarArithOp(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            print('Hello World!')
        '\n        Typing for `Masked` <op> `Masked`\n        Numba expects a valid numba type to be returned if typing is successful\n        else `None` signifies the error state (this pattern is commonly used\n        in Numba)\n        '
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            return_type = self.context.resolve_function_type(self.key, (args[0].value_type, args[1].value_type), kws).return_type
            return nb_signature(MaskedType(return_type), args[0], args[1])

class MaskedScalarUnaryOp(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            print('Hello World!')
        if len(args) == 1 and isinstance(args[0], MaskedType):
            return_type = self.context.resolve_function_type(self.key, (args[0].value_type,), kws).return_type
            return nb_signature(MaskedType(return_type), args[0])

class MaskedScalarNullOp(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            return 10
        '\n        Typing for `Masked` + `NA`\n        Handles situations like `x + cudf.NA`\n        '
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            return nb_signature(args[0], args[0], na_type)
        elif isinstance(args[0], NAType) and isinstance(args[1], MaskedType):
            return nb_signature(args[1], na_type, args[1])

class MaskedScalarScalarOp(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        '\n        Typing for `Masked` <op> a scalar (and vice-versa).\n        handles situations like `x + 1`\n        '
        to_resolve_types = None
        if isinstance(args[0], MaskedType) and isinstance(args[1], SUPPORTED_NUMBA_TYPES):
            to_resolve_types = (args[0].value_type, args[1])
        elif isinstance(args[0], SUPPORTED_NUMBA_TYPES) and isinstance(args[1], MaskedType):
            to_resolve_types = (args[1].value_type, args[0])
        else:
            return None
        return_type = self.context.resolve_function_type(self.key, to_resolve_types, kws).return_type
        return nb_signature(MaskedType(return_type), args[0], args[1])

@cuda_decl_registry.register_global(operator.is_)
class MaskedScalarIsNull(AbstractTemplate):
    """
    Typing for `Masked is cudf.NA`
    """

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            return nb_signature(types.boolean, args[0], na_type)
        elif isinstance(args[1], MaskedType) and isinstance(args[0], NAType):
            return nb_signature(types.boolean, na_type, args[1])

@cuda_decl_registry.register_global(operator.truth)
class MaskedScalarTruth(AbstractTemplate):
    """
    Typing for `if Masked`
    Used for `if x > y`
    The truthiness of a MaskedType shall be the truthiness
    of the `value` stored therein
    """

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        if isinstance(args[0], MaskedType):
            return nb_signature(types.boolean, MaskedType(types.boolean))

@cuda_decl_registry.register_global(float)
class MaskedScalarFloatCast(AbstractTemplate):
    """
    Typing for float(Masked)
    returns the result of calling "float" on the input
    TODO: retains the validity of the input rather than
    raising as in float(pd.NA)
    """

    def generic(self, args, kws):
        if False:
            return 10
        if isinstance(args[0], MaskedType):
            return nb_signature(MaskedType(types.float64), args[0])

@cuda_decl_registry.register_global(int)
class MaskedScalarIntCast(AbstractTemplate):
    """
    Typing for int(Masked)
    returns the result of calling "int" on the input
    TODO: retains the validity of the input rather than
    raising as in int(pd.NA)
    """

    def generic(self, args, kws):
        if False:
            return 10
        if isinstance(args[0], MaskedType):
            return nb_signature(MaskedType(types.int64), args[0])

@cuda_decl_registry.register_global(abs)
class MaskedScalarAbsoluteValue(AbstractTemplate):
    """
    Typing for the builtin function abs. Returns the same
    type as input except for boolean values which are converted
    to integer.

    This follows the expected result from the builtin abs function
    which differs from numpy - np.abs returns a bool whereas abs
    itself performs the cast.
    """

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        if isinstance(args[0], MaskedType):
            if isinstance(args[0].value_type, (StringView, UDFString)):
                return
            else:
                return_type = self.context.resolve_function_type(self.key, (args[0].value_type,), kws).return_type
                if return_type in types.signed_domain:
                    return_type = from_dtype(np.dtype('u' + return_type.name))
                return nb_signature(MaskedType(return_type), args[0])

@cuda_decl_registry.register_global(api.pack_return)
class UnpackReturnToMasked(AbstractTemplate):
    """
    Turn a returned MaskedType into its value and validity
    or turn a scalar into the tuple (scalar, True).
    """

    def generic(self, args, kws):
        if False:
            i = 10
            return i + 15
        if isinstance(args[0], MaskedType):
            return nb_signature(args[0], args[0])
        elif isinstance(args[0], SUPPORTED_NUMBA_TYPES):
            return_type = MaskedType(args[0])
            return nb_signature(return_type, args[0])
for binary_op in arith_ops + bitwise_ops + comparison_ops:
    cuda_decl_registry.register_global(binary_op)(MaskedScalarArithOp)
    cuda_decl_registry.register_global(binary_op)(MaskedScalarNullOp)
    cuda_decl_registry.register_global(binary_op)(MaskedScalarScalarOp)
for unary_op in unary_ops:
    cuda_decl_registry.register_global(unary_op)(MaskedScalarUnaryOp)

def _is_valid_string_arg(ty):
    if False:
        while True:
            i = 10
    return isinstance(ty, MaskedType) and isinstance(ty.value_type, (StringView, UDFString)) or isinstance(ty, types.StringLiteral)

def register_masked_string_function(func):
    if False:
        i = 10
        return i + 15
    "\n    Helper function wrapping numba's low level extension API. Provides\n    the boilerplate needed to associate a signature with a function or\n    operator to be overloaded.\n    "

    def deco(generic):
        if False:
            i = 10
            return i + 15

        class MaskedStringFunction(AbstractTemplate):
            pass
        MaskedStringFunction.generic = generic
        cuda_decl_registry.register_global(func)(MaskedStringFunction)
    return deco

@register_masked_string_function(len)
def len_typing(self, args, kws):
    if False:
        i = 10
        return i + 15
    if isinstance(args[0], MaskedType) and isinstance(args[0].value_type, (StringView, UDFString)):
        return nb_signature(MaskedType(size_type), MaskedType(string_view))
    elif isinstance(args[0], types.StringLiteral) and len(args) == 1:
        return nb_signature(size_type, args[0])

@register_masked_string_function(operator.add)
def concat_typing(self, args, kws):
    if False:
        while True:
            i = 10
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(MaskedType(udf_string), MaskedType(string_view), MaskedType(string_view))

@register_masked_string_function(operator.contains)
def contains_typing(self, args, kws):
    if False:
        return 10
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(MaskedType(types.boolean), MaskedType(string_view), MaskedType(string_view))

class MaskedStringViewCmpOp(AbstractTemplate):
    """
    return the boolean result of `cmpop` between to strings
    since the typing is the same for every comparison operator,
    we can reuse this class for all of them.
    """

    def generic(self, args, kws):
        if False:
            return 10
        if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
            return nb_signature(MaskedType(types.boolean), MaskedType(string_view), MaskedType(string_view))
for op in comparison_ops:
    cuda_decl_registry.register_global(op)(MaskedStringViewCmpOp)

def create_masked_binary_attr(attrname, retty):
    if False:
        return 10
    "\n    Helper function wrapping numba's low level extension API. Provides\n    the boilerplate needed to register a binary function of two masked\n    string objects as an attribute of one, e.g. `string.func(other)`.\n    "

    class MaskedStringViewBinaryAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            if False:
                for i in range(10):
                    print('nop')
            return nb_signature(MaskedType(retty), MaskedType(string_view), recvr=self.this)

    def attr(self, mod):
        if False:
            while True:
                i = 10
        return types.BoundFunction(MaskedStringViewBinaryAttr, MaskedType(string_view))
    return attr

def create_masked_unary_attr(attrname, retty):
    if False:
        while True:
            i = 10
    "\n    Helper function wrapping numba's low level extension API. Provides\n    the boilerplate needed to register a unary function of a masked\n    string object as an attribute, e.g. `string.func()`.\n    "

    class MaskedStringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            if False:
                print('Hello World!')
            return nb_signature(MaskedType(retty), recvr=self.this)

    def attr(self, mod):
        if False:
            while True:
                i = 10
        return types.BoundFunction(MaskedStringViewIdentifierAttr, MaskedType(string_view))
    return attr

class MaskedStringViewCount(AbstractTemplate):
    key = 'MaskedType.count'

    def generic(self, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return nb_signature(MaskedType(size_type), MaskedType(string_view), recvr=self.this)

class MaskedStringViewReplace(AbstractTemplate):
    key = 'MaskedType.replace'

    def generic(self, args, kws):
        if False:
            i = 10
            return i + 15
        return nb_signature(MaskedType(udf_string), MaskedType(string_view), MaskedType(string_view), recvr=self.this)

class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_replace(self, mod):
        if False:
            for i in range(10):
                print('nop')
        return types.BoundFunction(MaskedStringViewReplace, MaskedType(string_view))

    def resolve_count(self, mod):
        if False:
            return 10
        return types.BoundFunction(MaskedStringViewCount, MaskedType(string_view))

    def resolve_value(self, mod):
        if False:
            return 10
        return string_view

    def resolve_valid(self, mod):
        if False:
            for i in range(10):
                print('nop')
        return types.boolean
for func in bool_binary_funcs:
    setattr(MaskedStringViewAttrs, f'resolve_{func}', create_masked_binary_attr(f'MaskedType.{func}', types.boolean))
for func in int_binary_funcs:
    setattr(MaskedStringViewAttrs, f'resolve_{func}', create_masked_binary_attr(f'MaskedType.{func}', size_type))
for func in string_return_attrs:
    setattr(MaskedStringViewAttrs, f'resolve_{func}', create_masked_binary_attr(f'MaskedType.{func}', udf_string))
for func in id_unary_funcs:
    setattr(MaskedStringViewAttrs, f'resolve_{func}', create_masked_unary_attr(f'MaskedType.{func}', types.boolean))
for func in string_unary_funcs:
    setattr(MaskedStringViewAttrs, f'resolve_{func}', create_masked_unary_attr(f'MaskedType.{func}', udf_string))

class MaskedUDFStringAttrs(MaskedStringViewAttrs):
    key = MaskedType(udf_string)

    def resolve_value(self, mod):
        if False:
            i = 10
            return i + 15
        return udf_string
cuda_decl_registry.register_attr(MaskedStringViewAttrs)
cuda_decl_registry.register_attr(MaskedUDFStringAttrs)