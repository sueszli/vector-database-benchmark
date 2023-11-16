import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import AttributeTemplate, AbstractTemplate, CallableTemplate, Registry, signature
from numba.np.numpy_support import ufunc_find_matching_loop, supported_ufunc_loop, as_dtype, from_dtype, as_dtype, resolve_output_type, carray, farray, _ufunc_loop_sig
from numba.core.errors import TypingError, NumbaPerformanceWarning, NumbaTypeError, NumbaAssertionError
from numba import pndindex
registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr

class Numpy_rules_ufunc(AbstractTemplate):

    @classmethod
    def _handle_inputs(cls, ufunc, args, kws):
        if False:
            return 10
        '\n        Process argument types to a given *ufunc*.\n        Returns a (base types, explicit outputs, ndims, layout) tuple where:\n        - `base types` is a tuple of scalar types for each input\n        - `explicit outputs` is a tuple of explicit output types (arrays)\n        - `ndims` is the number of dimensions of the loop and also of\n          any outputs, explicit or implicit\n        - `layout` is the layout for any implicit output to be allocated\n        '
        nin = ufunc.nin
        nout = ufunc.nout
        nargs = ufunc.nargs
        assert nargs == nin + nout
        if len(args) < nin:
            msg = "ufunc '{0}': not enough arguments ({1} found, {2} required)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nin))
        if len(args) > nargs:
            msg = "ufunc '{0}': too many arguments ({1} found, {2} maximum)"
            raise TypingError(msg=msg.format(ufunc.__name__, len(args), nargs))
        args = [a.as_array if isinstance(a, types.ArrayCompatible) else a for a in args]
        arg_ndims = [a.ndim if isinstance(a, types.ArrayCompatible) else 0 for a in args]
        ndims = max(arg_ndims)
        explicit_outputs = args[nin:]
        if not all((d == ndims for d in arg_ndims[nin:])):
            msg = "ufunc '{0}' called with unsuitable explicit output arrays."
            raise TypingError(msg=msg.format(ufunc.__name__))
        if not all((isinstance(output, types.ArrayCompatible) for output in explicit_outputs)):
            msg = "ufunc '{0}' called with an explicit output that is not an array"
            raise TypingError(msg=msg.format(ufunc.__name__))
        if not all((output.mutable for output in explicit_outputs)):
            msg = "ufunc '{0}' called with an explicit output that is read-only"
            raise TypingError(msg=msg.format(ufunc.__name__))
        base_types = [x.dtype if isinstance(x, types.ArrayCompatible) else x for x in args]
        layout = None
        if ndims > 0 and len(explicit_outputs) < ufunc.nout:
            layout = 'C'
            layouts = [x.layout if isinstance(x, types.ArrayCompatible) else '' for x in args]
            if 'C' not in layouts and 'F' in layouts:
                layout = 'F'
        return (base_types, explicit_outputs, ndims, layout)

    @property
    def ufunc(self):
        if False:
            return 10
        return self.key

    def generic(self, args, kws):
        if False:
            return 10
        args = [x.type if isinstance(x, types.Optional) else x for x in args]
        ufunc = self.ufunc
        (base_types, explicit_outputs, ndims, layout) = self._handle_inputs(ufunc, args, kws)
        ufunc_loop = ufunc_find_matching_loop(ufunc, base_types)
        if ufunc_loop is None:
            raise TypingError("can't resolve ufunc {0} for types {1}".format(ufunc.__name__, args))
        if not supported_ufunc_loop(ufunc, ufunc_loop):
            msg = "ufunc '{0}' using the loop '{1}' not supported in this mode"
            raise TypingError(msg=msg.format(ufunc.__name__, ufunc_loop.ufunc_sig))
        explicit_outputs_np = [as_dtype(tp.dtype) for tp in explicit_outputs]
        if not all((np.can_cast(fromty, toty, 'unsafe') for (fromty, toty) in zip(ufunc_loop.numpy_outputs, explicit_outputs_np))):
            msg = "ufunc '{0}' can't cast result to explicit result type"
            raise TypingError(msg=msg.format(ufunc.__name__))
        out = list(explicit_outputs)
        implicit_output_count = ufunc.nout - len(explicit_outputs)
        if implicit_output_count > 0:
            ret_tys = ufunc_loop.outputs[-implicit_output_count:]
            if ndims > 0:
                assert layout is not None
                array_ufunc_type = None
                for a in args:
                    if hasattr(a, '__array_ufunc__'):
                        array_ufunc_type = a
                        break
                output_type = types.Array
                if array_ufunc_type is not None:
                    output_type = array_ufunc_type.__array_ufunc__(ufunc, '__call__', *args, **kws)
                    if output_type is NotImplemented:
                        msg = f'unsupported use of ufunc {ufunc} on {array_ufunc_type}'
                        raise NumbaTypeError(msg)
                    elif not issubclass(output_type, types.Array):
                        msg = f'ufunc {ufunc} on {array_ufunc_type}cannot return non-array {output_type}'
                        raise TypeError(msg)
                ret_tys = [output_type(dtype=ret_ty, ndim=ndims, layout=layout) for ret_ty in ret_tys]
                ret_tys = [resolve_output_type(self.context, args, ret_ty) for ret_ty in ret_tys]
            out.extend(ret_tys)
        return _ufunc_loop_sig(out, args)

class NumpyRulesArrayOperator(Numpy_rules_ufunc):
    _op_map = {operator.add: 'add', operator.sub: 'subtract', operator.mul: 'multiply', operator.truediv: 'true_divide', operator.floordiv: 'floor_divide', operator.mod: 'remainder', operator.pow: 'power', operator.lshift: 'left_shift', operator.rshift: 'right_shift', operator.and_: 'bitwise_and', operator.or_: 'bitwise_or', operator.xor: 'bitwise_xor', operator.eq: 'equal', operator.gt: 'greater', operator.ge: 'greater_equal', operator.lt: 'less', operator.le: 'less_equal', operator.ne: 'not_equal'}

    @property
    def ufunc(self):
        if False:
            print('Hello World!')
        return getattr(np, self._op_map[self.key])

    @classmethod
    def install_operations(cls):
        if False:
            print('Hello World!')
        for (op, ufunc_name) in cls._op_map.items():
            infer_global(op)(type('NumpyRulesArrayOperator_' + ufunc_name, (cls,), dict(key=op)))

    def generic(self, args, kws):
        if False:
            return 10
        'Overloads and calls base class generic() method, returning\n        None if a TypingError occurred.\n\n        Returning None for operators is important since operators are\n        heavily overloaded, and by suppressing type errors, we allow\n        type inference to check other possibilities before giving up\n        (particularly user-defined operators).\n        '
        try:
            sig = super(NumpyRulesArrayOperator, self).generic(args, kws)
        except TypingError:
            return None
        if sig is None:
            return None
        args = sig.args
        if not any((isinstance(arg, types.ArrayCompatible) for arg in args)):
            return None
        return sig
_binop_map = NumpyRulesArrayOperator._op_map

class NumpyRulesInplaceArrayOperator(NumpyRulesArrayOperator):
    _op_map = {operator.iadd: 'add', operator.isub: 'subtract', operator.imul: 'multiply', operator.itruediv: 'true_divide', operator.ifloordiv: 'floor_divide', operator.imod: 'remainder', operator.ipow: 'power', operator.ilshift: 'left_shift', operator.irshift: 'right_shift', operator.iand: 'bitwise_and', operator.ior: 'bitwise_or', operator.ixor: 'bitwise_xor'}

    def generic(self, args, kws):
        if False:
            for i in range(10):
                print('nop')
        (lhs, rhs) = args
        if not isinstance(lhs, types.ArrayCompatible):
            return
        args = args + (lhs,)
        sig = super(NumpyRulesInplaceArrayOperator, self).generic(args, kws)
        assert len(sig.args) == 3
        real_sig = signature(sig.return_type, *sig.args[:2])
        return real_sig

class NumpyRulesUnaryArrayOperator(NumpyRulesArrayOperator):
    _op_map = {operator.pos: 'positive', operator.neg: 'negative', operator.invert: 'invert'}

    def generic(self, args, kws):
        if False:
            for i in range(10):
                print('nop')
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.ArrayCompatible):
            return super(NumpyRulesUnaryArrayOperator, self).generic(args, kws)
math_operations = ['add', 'subtract', 'multiply', 'logaddexp', 'logaddexp2', 'true_divide', 'floor_divide', 'negative', 'positive', 'power', 'float_power', 'remainder', 'fmod', 'absolute', 'rint', 'sign', 'conjugate', 'exp', 'exp2', 'log', 'log2', 'log10', 'expm1', 'log1p', 'sqrt', 'square', 'cbrt', 'reciprocal', 'divide', 'mod', 'divmod', 'abs', 'fabs', 'gcd', 'lcm']
trigonometric_functions = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2', 'hypot', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'deg2rad', 'rad2deg', 'degrees', 'radians']
bit_twiddling_functions = ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'left_shift', 'right_shift', 'bitwise_not']
comparison_functions = ['greater', 'greater_equal', 'less', 'less_equal', 'not_equal', 'equal', 'logical_and', 'logical_or', 'logical_xor', 'logical_not', 'maximum', 'minimum', 'fmax', 'fmin']
floating_functions = ['isfinite', 'isinf', 'isnan', 'signbit', 'copysign', 'nextafter', 'modf', 'ldexp', 'frexp', 'floor', 'ceil', 'trunc', 'spacing']
logic_functions = ['isnat']
_unsupported = set(['frexp', 'modf'])

def register_numpy_ufunc(name, register_global=infer_global):
    if False:
        while True:
            i = 10
    func = getattr(np, name)

    class typing_class(Numpy_rules_ufunc):
        key = func
    typing_class.__name__ = 'resolve_{0}'.format(name)
    aliases = ('abs', 'bitwise_not', 'divide', 'abs')
    if name not in aliases:
        register_global(func, types.Function(typing_class))
all_ufuncs = sum([math_operations, trigonometric_functions, bit_twiddling_functions, comparison_functions, floating_functions, logic_functions], [])
supported_ufuncs = [x for x in all_ufuncs if x not in _unsupported]
for func in supported_ufuncs:
    register_numpy_ufunc(func)
all_ufuncs = [getattr(np, name) for name in all_ufuncs]
supported_ufuncs = [getattr(np, name) for name in supported_ufuncs]
NumpyRulesUnaryArrayOperator.install_operations()
NumpyRulesArrayOperator.install_operations()
NumpyRulesInplaceArrayOperator.install_operations()
supported_array_operators = set(NumpyRulesUnaryArrayOperator._op_map.keys()).union(NumpyRulesArrayOperator._op_map.keys()).union(NumpyRulesInplaceArrayOperator._op_map.keys())
del _unsupported

class Numpy_method_redirection(AbstractTemplate):
    """
    A template redirecting a Numpy global function (e.g. np.sum) to an
    array method of the same name (e.g. ndarray.sum).
    """
    prefer_literal = True

    def generic(self, args, kws):
        if False:
            i = 10
            return i + 15
        pysig = None
        if kws:
            if self.method_name == 'sum':
                if 'axis' in kws and 'dtype' not in kws:

                    def sum_stub(arr, axis):
                        if False:
                            while True:
                                i = 10
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' not in kws:

                    def sum_stub(arr, dtype):
                        if False:
                            while True:
                                i = 10
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' in kws:

                    def sum_stub(arr, axis, dtype):
                        if False:
                            print('Hello World!')
                        pass
                    pysig = utils.pysignature(sum_stub)
            elif self.method_name == 'argsort':

                def argsort_stub(arr, kind='quicksort'):
                    if False:
                        return 10
                    pass
                pysig = utils.pysignature(argsort_stub)
            else:
                fmt = "numba doesn't support kwarg for {}"
                raise TypingError(fmt.format(self.method_name))
        arr = args[0]
        meth_ty = self.context.resolve_getattr(arr, self.method_name)
        meth_sig = self.context.resolve_function_type(meth_ty, args[1:], kws)
        if meth_sig is not None:
            return meth_sig.as_function().replace(pysig=pysig)

def _numpy_redirect(fname):
    if False:
        i = 10
        return i + 15
    numpy_function = getattr(np, fname)
    cls = type('Numpy_redirect_{0}'.format(fname), (Numpy_method_redirection,), dict(key=numpy_function, method_name=fname))
    infer_global(numpy_function, types.Function(cls))
for func in ['sum', 'argsort', 'nonzero', 'ravel']:
    _numpy_redirect(func)
np_types = set((getattr(np, str(nb_type)) for nb_type in types.number_domain))
np_types.add(np.bool_)
np_types.add(np.intc)
np_types.add(np.intp)
np_types.add(np.uintc)
np_types.add(np.uintp)

def register_number_classes(register_global):
    if False:
        for i in range(10):
            print('nop')
    for np_type in np_types:
        nb_type = getattr(types, np_type.__name__)
        register_global(np_type, types.NumberClass(nb_type))
register_number_classes(infer_global)

def parse_shape(shape):
    if False:
        while True:
            i = 10
    '\n    Given a shape, return the number of dimensions.\n    '
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        int_tys = (types.Integer, types.IntEnumMember)
        if all((isinstance(s, int_tys) for s in shape)):
            ndim = len(shape)
    return ndim

def parse_dtype(dtype):
    if False:
        return 10
    '\n    Return the dtype of a type, if it is either a DtypeSpec (used for most\n    dtypes) or a TypeRef (used for record types).\n    '
    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype
    elif isinstance(dtype, types.TypeRef):
        return dtype.instance_type
    elif isinstance(dtype, types.StringLiteral):
        dtstr = dtype.literal_value
        try:
            dt = np.dtype(dtstr)
        except TypeError:
            msg = f"Invalid NumPy dtype specified: '{dtstr}'"
            raise TypingError(msg)
        return from_dtype(dt)

def _parse_nested_sequence(context, typ):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse a (possibly 0d) nested sequence type.\n    A (ndim, dtype) tuple is returned.  Note the sequence may still be\n    heterogeneous, as long as it converts to the given dtype.\n    '
    if isinstance(typ, (types.Buffer,)):
        raise TypingError('%s not allowed in a homogeneous sequence' % typ)
    elif isinstance(typ, (types.Sequence,)):
        (n, dtype) = _parse_nested_sequence(context, typ.dtype)
        return (n + 1, dtype)
    elif isinstance(typ, (types.BaseTuple,)):
        if typ.count == 0:
            return (1, types.float64)
        (n, dtype) = _parse_nested_sequence(context, typ[0])
        dtypes = [dtype]
        for i in range(1, typ.count):
            (_n, dtype) = _parse_nested_sequence(context, typ[i])
            if _n != n:
                raise TypingError('type %s does not have a regular shape' % (typ,))
            dtypes.append(dtype)
        dtype = context.unify_types(*dtypes)
        if dtype is None:
            raise TypingError('cannot convert %s to a homogeneous type' % typ)
        return (n + 1, dtype)
    else:
        as_dtype(typ)
        return (0, typ)

def _infer_dtype_from_inputs(inputs):
    if False:
        print('Hello World!')
    return dtype

def _homogeneous_dims(context, func_name, arrays):
    if False:
        print('Hello World!')
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            msg = f'{func_name}(): all the input arrays must have same number of dimensions'
            raise NumbaTypeError(msg)
    return ndim

def _sequence_of_arrays(context, func_name, arrays, dim_chooser=_homogeneous_dims):
    if False:
        while True:
            i = 10
    if not isinstance(arrays, types.BaseTuple) or not len(arrays) or (not all((isinstance(a, types.Array) for a in arrays))):
        raise TypeError('%s(): expecting a non-empty tuple of arrays, got %s' % (func_name, arrays))
    ndim = dim_chooser(context, func_name, arrays)
    dtype = context.unify_types(*(a.dtype for a in arrays))
    if dtype is None:
        raise TypeError('%s(): input arrays must have compatible dtypes' % func_name)
    return (dtype, ndim)

def _choose_concatenation_layout(arrays):
    if False:
        i = 10
        return i + 15
    return 'F' if all((a.layout == 'F' for a in arrays)) else 'C'

class BaseStackTemplate(CallableTemplate):

    def generic(self):
        if False:
            while True:
                i = 10

        def typer(arrays):
            if False:
                i = 10
                return i + 15
            (dtype, ndim) = _sequence_of_arrays(self.context, self.func_name, arrays)
            ndim = max(ndim, self.ndim_min)
            layout = _choose_concatenation_layout(arrays)
            return types.Array(dtype, ndim, layout)
        return typer

class MatMulTyperMixin(object):

    def matmul_typer(self, a, b, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Typer function for Numpy matrix multiplication.\n        '
        if not isinstance(a, types.Array) or not isinstance(b, types.Array):
            return
        if not all((x.ndim in (1, 2) for x in (a, b))):
            raise TypingError('%s only supported on 1-D and 2-D arrays' % (self.func_name,))
        ndims = set([a.ndim, b.ndim])
        if ndims == set([2]):
            out_ndim = 2
        elif ndims == set([1, 2]):
            out_ndim = 1
        elif ndims == set([1]):
            out_ndim = 0
        if out is not None:
            if out_ndim == 0:
                raise TypeError('explicit output unsupported for vector * vector')
            elif out.ndim != out_ndim:
                raise TypeError('explicit output has incorrect dimensionality')
            if not isinstance(out, types.Array) or out.layout != 'C':
                raise TypeError('output must be a C-contiguous array')
            all_args = (a, b, out)
        else:
            all_args = (a, b)
        if not (config.DISABLE_PERFORMANCE_WARNINGS or all((x.layout in 'CF' for x in (a, b)))):
            msg = '%s is faster on contiguous arrays, called on %s' % (self.func_name, (a, b))
            warnings.warn(NumbaPerformanceWarning(msg))
        if not all((x.dtype == a.dtype for x in all_args)):
            raise TypingError('%s arguments must all have the same dtype' % (self.func_name,))
        if not isinstance(a.dtype, (types.Float, types.Complex)):
            raise TypingError('%s only supported on float and complex arrays' % (self.func_name,))
        if out:
            return out
        elif out_ndim > 0:
            return types.Array(a.dtype, out_ndim, 'C')
        else:
            return a.dtype

def _check_linalg_matrix(a, func_name):
    if False:
        while True:
            i = 10
    if not isinstance(a, types.Array):
        return
    if not a.ndim == 2:
        raise TypingError('np.linalg.%s() only supported on 2-D arrays' % func_name)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError('np.linalg.%s() only supported on float and complex arrays' % func_name)

@infer_global(np.ndenumerate)
class NdEnumerate(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        assert not kws
        (arr,) = args
        if isinstance(arr, types.Array):
            enumerate_type = types.NumpyNdEnumerateType(arr)
            return signature(enumerate_type, *args)

@infer_global(np.nditer)
class NdIter(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            for i in range(10):
                print('nop')
        assert not kws
        if len(args) != 1:
            return
        (arrays,) = args
        if isinstance(arrays, types.BaseTuple):
            if not arrays:
                return
            arrays = list(arrays)
        else:
            arrays = [arrays]
        nditerty = types.NumpyNdIterType(arrays)
        return signature(nditerty, *args)

@infer_global(pndindex)
@infer_global(np.ndindex)
class NdIndex(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.BaseTuple):
            tup = args[0]
            if tup.count > 0 and (not isinstance(tup, types.UniTuple)):
                return
            shape = list(tup)
        else:
            shape = args
        if all((isinstance(x, types.Integer) for x in shape)):
            iterator_type = types.NumpyNdIndexType(len(shape))
            return signature(iterator_type, *args)