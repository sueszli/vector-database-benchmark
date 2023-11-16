"""
Implementation of functions in the Numpy package.
"""
import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import ufunc_find_matching_loop, select_array_wrapper, from_dtype, _ufunc_loop_sig
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
registry = Registry('npyimpl')

class _ScalarIndexingHelper(object):

    def update_indices(self, loop_indices, name):
        if False:
            print('Hello World!')
        pass

    def as_values(self):
        if False:
            while True:
                i = 10
        pass

class _ScalarHelper(object):
    """Helper class to handle scalar arguments (and result).
    Note that store_data is only used when generating code for
    a scalar ufunc and to write the output value.

    For loading, the value is directly used without having any
    kind of indexing nor memory backing it up. This is the use
    for input arguments.

    For storing, a variable is created in the stack where the
    value will be written.

    Note that it is not supported (as it is unneeded for our
    current use-cases) reading back a stored value. This class
    will always "load" the original value it got at its creation.
    """

    def __init__(self, ctxt, bld, val, ty):
        if False:
            print('Hello World!')
        self.context = ctxt
        self.builder = bld
        self.val = val
        self.base_type = ty
        intpty = ctxt.get_value_type(types.intp)
        self.shape = [ir.Constant(intpty, 1)]
        lty = ctxt.get_data_type(ty) if ty != types.boolean else ir.IntType(1)
        self._ptr = cgutils.alloca_once(bld, lty)

    def create_iter_indices(self):
        if False:
            while True:
                i = 10
        return _ScalarIndexingHelper()

    def load_data(self, indices):
        if False:
            for i in range(10):
                print('nop')
        return self.val

    def store_data(self, indices, val):
        if False:
            for i in range(10):
                print('nop')
        self.builder.store(val, self._ptr)

    @property
    def return_val(self):
        if False:
            for i in range(10):
                print('nop')
        return self.builder.load(self._ptr)

class _ArrayIndexingHelper(namedtuple('_ArrayIndexingHelper', ('array', 'indices'))):

    def update_indices(self, loop_indices, name):
        if False:
            while True:
                i = 10
        bld = self.array.builder
        intpty = self.array.context.get_value_type(types.intp)
        ONE = ir.Constant(ir.IntType(intpty.width), 1)
        indices = loop_indices[len(loop_indices) - len(self.indices):]
        for (src, dst, dim) in zip(indices, self.indices, self.array.shape):
            cond = bld.icmp_unsigned('>', dim, ONE)
            with bld.if_then(cond):
                bld.store(src, dst)

    def as_values(self):
        if False:
            i = 10
            return i + 15
        '\n        The indexing helper is built using alloca for each value, so it\n        actually contains pointers to the actual indices to load. Note\n        that update_indices assumes the same. This method returns the\n        indices as values\n        '
        bld = self.array.builder
        return [bld.load(index) for index in self.indices]

class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder', 'shape', 'strides', 'data', 'layout', 'base_type', 'ndim', 'return_val'))):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.
    """

    def create_iter_indices(self):
        if False:
            while True:
                i = 10
        intpty = self.context.get_value_type(types.intp)
        ZERO = ir.Constant(ir.IntType(intpty.width), 0)
        indices = []
        for i in range(self.ndim):
            x = cgutils.alloca_once(self.builder, ir.IntType(intpty.width))
            self.builder.store(ZERO, x)
            indices.append(x)
        return _ArrayIndexingHelper(self, indices)

    def _load_effective_address(self, indices):
        if False:
            return 10
        return cgutils.get_item_pointer2(self.context, self.builder, data=self.data, shape=self.shape, strides=self.strides, layout=self.layout, inds=indices)

    def load_data(self, indices):
        if False:
            print('Hello World!')
        model = self.context.data_model_manager[self.base_type]
        ptr = self._load_effective_address(indices)
        return model.load_from_data_pointer(self.builder, ptr)

    def store_data(self, indices, value):
        if False:
            i = 10
            return i + 15
        ctx = self.context
        bld = self.builder
        store_value = ctx.get_value_as_data(bld, self.base_type, value)
        assert ctx.get_data_type(self.base_type) == store_value.type
        bld.store(store_value, self._load_effective_address(indices))

def _prepare_argument(ctxt, bld, inp, tyinp, where='input operand'):
    if False:
        return 10
    'returns an instance of the appropriate Helper (either\n    _ScalarHelper or _ArrayHelper) class to handle the argument.\n    using the polymorphic interface of the Helper classes, scalar\n    and array cases can be handled with the same code'
    if isinstance(tyinp, types.Optional):
        oty = tyinp
        tyinp = tyinp.type
        inp = ctxt.cast(bld, inp, oty, tyinp)
    if isinstance(tyinp, types.ArrayCompatible):
        ary = ctxt.make_array(tyinp)(ctxt, bld, inp)
        shape = cgutils.unpack_tuple(bld, ary.shape, tyinp.ndim)
        strides = cgutils.unpack_tuple(bld, ary.strides, tyinp.ndim)
        return _ArrayHelper(ctxt, bld, shape, strides, ary.data, tyinp.layout, tyinp.dtype, tyinp.ndim, inp)
    elif types.unliteral(tyinp) in types.number_domain | {types.boolean} or isinstance(tyinp, types.scalars._NPDatetimeBase):
        return _ScalarHelper(ctxt, bld, inp, tyinp)
    else:
        raise NotImplementedError('unsupported type for {0}: {1}'.format(where, str(tyinp)))
_broadcast_onto_sig = types.intp(types.intp, types.CPointer(types.intp), types.intp, types.CPointer(types.intp))

def _broadcast_onto(src_ndim, src_shape, dest_ndim, dest_shape):
    if False:
        while True:
            i = 10
    'Low-level utility function used in calculating a shape for\n    an implicit output array.  This function assumes that the\n    destination shape is an LLVM pointer to a C-style array that was\n    already initialized to a size of one along all axes.\n\n    Returns an integer value:\n    >= 1  :  Succeeded.  Return value should equal the number of dimensions in\n             the destination shape.\n    0     :  Failed to broadcast because source shape is larger than the\n             destination shape (this case should be weeded out at type\n             checking).\n    < 0   :  Failed to broadcast onto destination axis, at axis number ==\n             -(return_value + 1).\n    '
    if src_ndim > dest_ndim:
        return 0
    else:
        src_index = 0
        dest_index = dest_ndim - src_ndim
        while src_index < src_ndim:
            src_dim_size = src_shape[src_index]
            dest_dim_size = dest_shape[dest_index]
            if dest_dim_size != 1:
                if src_dim_size != dest_dim_size and src_dim_size != 1:
                    return -(dest_index + 1)
            elif src_dim_size != 1:
                dest_shape[dest_index] = src_dim_size
            src_index += 1
            dest_index += 1
    return dest_index

def _build_array(context, builder, array_ty, input_types, inputs):
    if False:
        for i in range(10):
            print('nop')
    'Utility function to handle allocation of an implicit output array\n    given the target context, builder, output array type, and a list of\n    _ArrayHelper instances.\n    '
    input_types = [x.type if isinstance(x, types.Optional) else x for x in input_types]
    intp_ty = context.get_value_type(types.intp)

    def make_intp_const(val):
        if False:
            for i in range(10):
                print('nop')
        return context.get_constant(types.intp, val)
    ZERO = make_intp_const(0)
    ONE = make_intp_const(1)
    src_shape = cgutils.alloca_once(builder, intp_ty, array_ty.ndim, 'src_shape')
    dest_ndim = make_intp_const(array_ty.ndim)
    dest_shape = cgutils.alloca_once(builder, intp_ty, array_ty.ndim, 'dest_shape')
    dest_shape_addrs = tuple((cgutils.gep_inbounds(builder, dest_shape, index) for index in range(array_ty.ndim)))
    for dest_shape_addr in dest_shape_addrs:
        builder.store(ONE, dest_shape_addr)
    for (arg_number, arg) in enumerate(inputs):
        if not hasattr(arg, 'ndim'):
            continue
        arg_ndim = make_intp_const(arg.ndim)
        for index in range(arg.ndim):
            builder.store(arg.shape[index], cgutils.gep_inbounds(builder, src_shape, index))
        arg_result = context.compile_internal(builder, _broadcast_onto, _broadcast_onto_sig, [arg_ndim, src_shape, dest_ndim, dest_shape])
        with cgutils.if_unlikely(builder, builder.icmp_signed('<', arg_result, ONE)):
            msg = 'unable to broadcast argument %d to output array' % (arg_number,)
            loc = errors.loc_info.get('loc', None)
            if loc is not None:
                msg += '\nFile "%s", line %d, ' % (loc.filename, loc.line)
            context.call_conv.return_user_exc(builder, ValueError, (msg,))
    real_array_ty = array_ty.as_array
    dest_shape_tup = tuple((builder.load(dest_shape_addr) for dest_shape_addr in dest_shape_addrs))
    array_val = arrayobj._empty_nd_impl(context, builder, real_array_ty, dest_shape_tup)
    array_wrapper_index = select_array_wrapper(input_types)
    array_wrapper_ty = input_types[array_wrapper_index]
    try:
        array_wrap = context.get_function('__array_wrap__', array_ty(array_wrapper_ty, real_array_ty))
    except NotImplementedError:
        if array_wrapper_ty.array_priority != types.Array.array_priority:
            raise
        out_val = array_val._getvalue()
    else:
        wrap_args = (inputs[array_wrapper_index].return_val, array_val._getvalue())
        out_val = array_wrap(builder, wrap_args)
    ndim = array_ty.ndim
    shape = cgutils.unpack_tuple(builder, array_val.shape, ndim)
    strides = cgutils.unpack_tuple(builder, array_val.strides, ndim)
    return _ArrayHelper(context, builder, shape, strides, array_val.data, array_ty.layout, array_ty.dtype, ndim, out_val)

def _unpack_output_types(ufunc, sig):
    if False:
        i = 10
        return i + 15
    if ufunc.nout == 1:
        return [sig.return_type]
    else:
        return list(sig.return_type)

def _unpack_output_values(ufunc, builder, values):
    if False:
        i = 10
        return i + 15
    if ufunc.nout == 1:
        return [values]
    else:
        return cgutils.unpack_tuple(builder, values)

def _pack_output_values(ufunc, context, builder, typ, values):
    if False:
        i = 10
        return i + 15
    if ufunc.nout == 1:
        return values[0]
    else:
        return context.make_tuple(builder, typ, values)

def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):
    if False:
        for i in range(10):
            print('nop')
    arguments = [_prepare_argument(context, builder, arg, tyarg) for (arg, tyarg) in zip(args, sig.args)]
    if len(arguments) < ufunc.nin:
        raise RuntimeError('Not enough inputs to {}, expected {} got {}'.format(ufunc.__name__, ufunc.nin, len(arguments)))
    for (out_i, ret_ty) in enumerate(_unpack_output_types(ufunc, sig)):
        if ufunc.nin + out_i >= len(arguments):
            if isinstance(ret_ty, types.ArrayCompatible):
                output = _build_array(context, builder, ret_ty, sig.args, arguments)
            else:
                output = _prepare_argument(context, builder, ir.Constant(context.get_value_type(ret_ty), None), ret_ty)
            arguments.append(output)
        elif context.enable_nrt:
            context.nrt.incref(builder, ret_ty, args[ufunc.nin + out_i])
    inputs = arguments[:ufunc.nin]
    outputs = arguments[ufunc.nin:]
    assert len(outputs) == ufunc.nout
    outer_sig = _ufunc_loop_sig([a.base_type for a in outputs], [a.base_type for a in inputs])
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)
    indices = [inp.create_iter_indices() for inp in inputs]
    loopshape = outputs[0].shape
    input_layouts = [inp.layout for inp in inputs if isinstance(inp, _ArrayHelper)]
    num_c_layout = len([x for x in input_layouts if x == 'C'])
    num_f_layout = len([x for x in input_layouts if x == 'F'])
    if num_f_layout > num_c_layout:
        order = 'F'
    else:
        order = 'C'
    with cgutils.loop_nest(builder, loopshape, intp=intpty, order=order) as loop_indices:
        vals_in = []
        for (i, (index, arg)) in enumerate(zip(indices, inputs)):
            index.update_indices(loop_indices, i)
            vals_in.append(arg.load_data(index.as_values()))
        vals_out = _unpack_output_values(ufunc, builder, kernel.generate(*vals_in))
        for (val_out, output) in zip(vals_out, outputs):
            output.store_data(loop_indices, val_out)
    out = _pack_output_values(ufunc, context, builder, sig.return_type, [o.return_val for o in outputs])
    return impl_ret_new_ref(context, builder, sig.return_type, out)

class _Kernel(object):

    def __init__(self, context, builder, outer_sig):
        if False:
            return 10
        self.context = context
        self.builder = builder
        self.outer_sig = outer_sig

    def cast(self, val, fromty, toty):
        if False:
            print('Hello World!')
        'Numpy uses cast semantics that are different from standard Python\n        (for example, it does allow casting from complex to float).\n\n        This method acts as a patch to context.cast so that it allows\n        complex to real/int casts.\n\n        '
        if isinstance(fromty, types.Complex) and (not isinstance(toty, types.Complex)):
            newty = fromty.underlying_float
            attr = self.context.get_getattr(fromty, 'real')
            val = attr(self.context, self.builder, fromty, val, 'real')
            fromty = newty
        return self.context.cast(self.builder, val, fromty, toty)

def _ufunc_db_function(ufunc):
    if False:
        while True:
            i = 10
    "Use the ufunc loop type information to select the code generation\n    function from the table provided by the dict_of_kernels. The dict\n    of kernels maps the loop identifier to a function with the\n    following signature: (context, builder, signature, args).\n\n    The loop type information has the form 'AB->C'. The letters to the\n    left of '->' are the input types (specified as NumPy letter\n    types).  The letters to the right of '->' are the output\n    types. There must be 'ufunc.nin' letters to the left of '->', and\n    'ufunc.nout' letters to the right.\n\n    For example, a binary float loop resulting in a float, will have\n    the following signature: 'ff->f'.\n\n    A given ufunc implements many loops. The list of loops implemented\n    for a given ufunc can be accessed using the 'types' attribute in\n    the ufunc object. The NumPy machinery selects the first loop that\n    fits a given calling signature (in our case, what we call the\n    outer_sig). This logic is mimicked by 'ufunc_find_matching_loop'.\n    "

    class _KernelImpl(_Kernel):

        def __init__(self, context, builder, outer_sig):
            if False:
                for i in range(10):
                    print('nop')
            super(_KernelImpl, self).__init__(context, builder, outer_sig)
            loop = ufunc_find_matching_loop(ufunc, outer_sig.args + tuple(_unpack_output_types(ufunc, outer_sig)))
            self.fn = context.get_ufunc_info(ufunc).get(loop.ufunc_sig)
            self.inner_sig = _ufunc_loop_sig(loop.outputs, loop.inputs)
            if self.fn is None:
                msg = "Don't know how to lower ufunc '{0}' for loop '{1}'"
                raise NotImplementedError(msg.format(ufunc.__name__, loop))

        def generate(self, *args):
            if False:
                print('Hello World!')
            isig = self.inner_sig
            osig = self.outer_sig
            cast_args = [self.cast(val, inty, outty) for (val, inty, outty) in zip(args, osig.args, isig.args)]
            with force_error_model(self.context, 'numpy'):
                res = self.fn(self.context, self.builder, isig, cast_args)
            dmm = self.context.data_model_manager
            res = dmm[isig.return_type].from_return(self.builder, res)
            return self.cast(res, isig.return_type, osig.return_type)
    return _KernelImpl

def register_ufunc_kernel(ufunc, kernel, lower):
    if False:
        for i in range(10):
            print('nop')

    def do_ufunc(context, builder, sig, args):
        if False:
            return 10
        return numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel)
    _any = types.Any
    in_args = (_any,) * ufunc.nin
    for n_explicit_out in range(ufunc.nout + 1):
        out_args = (types.Array,) * n_explicit_out
        lower(ufunc, *in_args, *out_args)(do_ufunc)
    return kernel

def register_unary_operator_kernel(operator, ufunc, kernel, lower, inplace=False):
    if False:
        while True:
            i = 10
    assert not inplace

    def lower_unary_operator(context, builder, sig, args):
        if False:
            print('Hello World!')
        return numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel)
    _arr_kind = types.Array
    lower(operator, _arr_kind)(lower_unary_operator)

def register_binary_operator_kernel(op, ufunc, kernel, lower, inplace=False):
    if False:
        return 10

    def lower_binary_operator(context, builder, sig, args):
        if False:
            return 10
        return numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel)

    def lower_inplace_operator(context, builder, sig, args):
        if False:
            print('Hello World!')
        args = tuple(args) + (args[0],)
        sig = typing.signature(sig.return_type, *sig.args + (sig.args[0],))
        return numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel)
    _any = types.Any
    _arr_kind = types.Array
    formal_sigs = [(_arr_kind, _arr_kind), (_any, _arr_kind), (_arr_kind, _any)]
    for sig in formal_sigs:
        if not inplace:
            lower(op, *sig)(lower_binary_operator)
        else:
            lower(op, *sig)(lower_inplace_operator)

@registry.lower(operator.pos, types.Array)
def array_positive_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    'Lowering function for +(array) expressions.  Defined here\n    (numba.targets.npyimpl) since the remaining array-operator\n    lowering functions are also registered in this module.\n    '

    class _UnaryPositiveKernel(_Kernel):

        def generate(self, *args):
            if False:
                i = 10
                return i + 15
            [val] = args
            return val
    return numpy_ufunc_kernel(context, builder, sig, args, np.positive, _UnaryPositiveKernel)

def register_ufuncs(ufuncs, lower):
    if False:
        return 10
    kernels = {}
    for ufunc in ufuncs:
        db_func = _ufunc_db_function(ufunc)
        kernels[ufunc] = register_ufunc_kernel(ufunc, db_func, lower)
    for _op_map in (npydecl.NumpyRulesUnaryArrayOperator._op_map, npydecl.NumpyRulesArrayOperator._op_map):
        for (operator, ufunc_name) in _op_map.items():
            ufunc = getattr(np, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                register_unary_operator_kernel(operator, ufunc, kernel, lower)
            elif ufunc.nin == 2:
                register_binary_operator_kernel(operator, ufunc, kernel, lower)
            else:
                raise RuntimeError("There shouldn't be any non-unary or binary operators")
    for _op_map in (npydecl.NumpyRulesInplaceArrayOperator._op_map,):
        for (operator, ufunc_name) in _op_map.items():
            ufunc = getattr(np, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                register_unary_operator_kernel(operator, ufunc, kernel, lower, inplace=True)
            elif ufunc.nin == 2:
                register_binary_operator_kernel(operator, ufunc, kernel, lower, inplace=True)
            else:
                raise RuntimeError("There shouldn't be any non-unary or binary operators")
register_ufuncs(ufunc_db.get_ufuncs(), registry.lower)

@intrinsic
def _make_dtype_object(typingctx, desc):
    if False:
        while True:
            i = 10
    'Given a string or NumberClass description *desc*, returns the dtype object.\n    '

    def from_nb_type(nb_type):
        if False:
            return 10
        return_type = types.DType(nb_type)
        sig = return_type(desc)

        def codegen(context, builder, signature, args):
            if False:
                return 10
            return context.get_dummy_value()
        return (sig, codegen)
    if isinstance(desc, types.Literal):
        nb_type = from_dtype(np.dtype(desc.literal_value))
        return from_nb_type(nb_type)
    elif isinstance(desc, types.functions.NumberClass):
        thestr = str(desc.dtype)
        nb_type = from_dtype(np.dtype(thestr))
        return from_nb_type(nb_type)

@overload(np.dtype)
def numpy_dtype(desc):
    if False:
        print('Hello World!')
    'Provide an implementation so that numpy.dtype function can be lowered.\n    '
    if isinstance(desc, (types.Literal, types.functions.NumberClass)):

        def imp(desc):
            if False:
                for i in range(10):
                    print('nop')
            return _make_dtype_object(desc)
        return imp
    else:
        raise errors.NumbaTypeError('unknown dtype descriptor: {}'.format(desc))