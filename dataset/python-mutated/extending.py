import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.templates import infer, infer_getattr
from numba.core.imputils import lower_builtin, lower_getattr, lower_getattr_generic, lower_setattr, lower_setattr_generic, lower_cast
from numba.core.datamodel import models
from numba.core.datamodel import register_default as register_model
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba._helperlib import _import_cython_function
from numba.core.serialize import ReduceMixin

def type_callable(func):
    if False:
        i = 10
        return i + 15
    "\n    Decorate a function as implementing typing for the callable *func*.\n    *func* can be a callable object (probably a global) or a string\n    denoting a built-in operation (such 'getitem' or '__array_wrap__')\n    "
    from numba.core.typing.templates import CallableTemplate, infer, infer_global
    if not callable(func) and (not isinstance(func, str)):
        raise TypeError('`func` should be a function or string')
    try:
        func_name = func.__name__
    except AttributeError:
        func_name = str(func)

    def decorate(typing_func):
        if False:
            while True:
                i = 10

        def generic(self):
            if False:
                print('Hello World!')
            return typing_func(self.context)
        name = '%s_CallableTemplate' % (func_name,)
        bases = (CallableTemplate,)
        class_dict = dict(key=func, generic=generic)
        template = type(name, bases, class_dict)
        infer(template)
        if callable(func):
            infer_global(func, types.Function(template))
        return typing_func
    return decorate
_overload_default_jit_options = {'no_cpython_wrapper': True, 'nopython': True}

def overload(func, jit_options={}, strict=True, inline='never', prefer_literal=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    A decorator marking the decorated function as typing and implementing\n    *func* in nopython mode.\n\n    The decorated function will have the same formal parameters as *func*\n    and be passed the Numba types of those parameters.  It should return\n    a function implementing *func* for the given types.\n\n    Here is an example implementing len() for tuple types::\n\n        @overload(len)\n        def tuple_len(seq):\n            if isinstance(seq, types.BaseTuple):\n                n = len(seq)\n                def len_impl(seq):\n                    return n\n                return len_impl\n\n    Compiler options can be passed as an dictionary using the **jit_options**\n    argument.\n\n    Overloading strictness (that the typing and implementing signatures match)\n    is enforced by the **strict** keyword argument, it is recommended that this\n    is set to True (default).\n\n    To handle a function that accepts imprecise types, an overload\n    definition can return 2-tuple of ``(signature, impl_function)``, where\n    the ``signature`` is a ``typing.Signature`` specifying the precise\n    signature to be used; and ``impl_function`` is the same implementation\n    function as in the simple case.\n\n    If the kwarg inline determines whether the overload is inlined in the\n    calling function and can be one of three values:\n    * 'never' (default) - the overload is never inlined.\n    * 'always' - the overload is always inlined.\n    * a function that takes two arguments, both of which are instances of a\n      namedtuple with fields:\n        * func_ir\n        * typemap\n        * calltypes\n        * signature\n      The first argument holds the information from the caller, the second\n      holds the information from the callee. The function should return Truthy\n      to determine whether to inline, this essentially permitting custom\n      inlining rules (typical use might be cost models).\n\n    The *prefer_literal* option allows users to control if literal types should\n    be tried first or last. The default (`False`) is to use non-literal types.\n    Implementations that can specialize based on literal values should set the\n    option to `True`. Note, this option maybe expanded in the near future to\n    allow for more control (e.g. disabling non-literal types).\n\n    **kwargs prescribes additional arguments passed through to the overload\n    template. The only accepted key at present is 'target' which is a string\n    corresponding to the target that this overload should be bound against.\n    "
    from numba.core.typing.templates import make_overload_template, infer_global
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)

    def decorate(overload_func):
        if False:
            for i in range(10):
                print('nop')
        template = make_overload_template(func, overload_func, opts, strict, inline, prefer_literal, **kwargs)
        infer(template)
        if callable(func):
            infer_global(func, types.Function(template))
        return overload_func
    return decorate

def register_jitable(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Register a regular python function that can be executed by the python\n    interpreter and can be compiled into a nopython function when referenced\n    by other jit'ed functions.  Can be used as::\n\n        @register_jitable\n        def foo(x, y):\n            return x + y\n\n    Or, with compiler options::\n\n        @register_jitable(_nrt=False) # disable runtime allocation\n        def foo(x, y):\n            return x + y\n\n    "

    def wrap(fn):
        if False:
            print('Hello World!')
        inline = kwargs.pop('inline', 'never')

        @overload(fn, jit_options=kwargs, inline=inline, strict=False)
        def ov_wrap(*args, **kwargs):
            if False:
                print('Hello World!')
            return fn
        return fn
    if kwargs:
        return wrap
    else:
        return wrap(*args)

def overload_attribute(typ, attr, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    A decorator marking the decorated function as typing and implementing\n    attribute *attr* for the given Numba type in nopython mode.\n\n    *kwargs* are passed to the underlying `@overload` call.\n\n    Here is an example implementing .nbytes for array types::\n\n        @overload_attribute(types.Array, 'nbytes')\n        def array_nbytes(arr):\n            def get(arr):\n                return arr.size * arr.itemsize\n            return get\n    "
    from numba.core.typing.templates import make_overload_attribute_template

    def decorate(overload_func):
        if False:
            return 10
        template = make_overload_attribute_template(typ, attr, overload_func, inline=kwargs.get('inline', 'never'))
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func
    return decorate

def _overload_method_common(typ, attr, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Common code for overload_method and overload_classmethod\n    '
    from numba.core.typing.templates import make_overload_method_template

    def decorate(overload_func):
        if False:
            i = 10
            return i + 15
        copied_kwargs = kwargs.copy()
        template = make_overload_method_template(typ, attr, overload_func, inline=copied_kwargs.pop('inline', 'never'), prefer_literal=copied_kwargs.pop('prefer_literal', False), **copied_kwargs)
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func
    return decorate

def overload_method(typ, attr, **kwargs):
    if False:
        return 10
    "\n    A decorator marking the decorated function as typing and implementing\n    method *attr* for the given Numba type in nopython mode.\n\n    *kwargs* are passed to the underlying `@overload` call.\n\n    Here is an example implementing .take() for array types::\n\n        @overload_method(types.Array, 'take')\n        def array_take(arr, indices):\n            if isinstance(indices, types.Array):\n                def take_impl(arr, indices):\n                    n = indices.shape[0]\n                    res = np.empty(n, arr.dtype)\n                    for i in range(n):\n                        res[i] = arr[indices[i]]\n                    return res\n                return take_impl\n    "
    return _overload_method_common(typ, attr, **kwargs)

def overload_classmethod(typ, attr, **kwargs):
    if False:
        return 10
    '\n    A decorator marking the decorated function as typing and implementing\n    classmethod *attr* for the given Numba type in nopython mode.\n\n\n    Similar to ``overload_method``.\n\n\n    Here is an example implementing a classmethod on the Array type to call\n    ``np.arange()``::\n\n        @overload_classmethod(types.Array, "make")\n        def ov_make(cls, nitems):\n            def impl(cls, nitems):\n                return np.arange(nitems)\n            return impl\n\n    The above code will allow the following to work in jit-compiled code::\n\n        @njit\n        def foo(n):\n            return types.Array.make(n)\n    '
    return _overload_method_common(types.TypeRef(typ), attr, **kwargs)

def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    if False:
        while True:
            i = 10
    "\n    Make an automatic attribute wrapper exposing member named *struct_attr*\n    as a read-only attribute named *python_attr*.\n    The given *typeclass*'s model must be a StructModel subclass.\n    "
    from numba.core.typing.templates import AttributeTemplate
    from numba.core.datamodel import default_manager
    from numba.core.datamodel.models import StructModel
    from numba.core.imputils import impl_ret_borrowed
    from numba.core import cgutils
    if not isinstance(typeclass, type) or not issubclass(typeclass, types.Type):
        raise TypeError('typeclass should be a Type subclass, got %s' % (typeclass,))

    def get_attr_fe_type(typ):
        if False:
            while True:
                i = 10
        '\n        Get the Numba type of member *struct_attr* in *typ*.\n        '
        model = default_manager.lookup(typ)
        if not isinstance(model, StructModel):
            raise TypeError('make_struct_attribute_wrapper() needs a type with a StructModel, but got %s' % (model,))
        return model.get_member_fe_type(struct_attr)

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if False:
                print('Hello World!')
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @lower_getattr(typeclass, python_attr)
    def struct_getattr_impl(context, builder, typ, val):
        if False:
            while True:
                i = 10
        val = cgutils.create_struct_proxy(typ)(context, builder, value=val)
        attrty = get_attr_fe_type(typ)
        attrval = getattr(val, struct_attr)
        return impl_ret_borrowed(context, builder, attrty, attrval)

class _Intrinsic(ReduceMixin):
    """
    Dummy callable for intrinsic
    """
    _memo = weakref.WeakValueDictionary()
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)
    __uuid = None

    def __init__(self, name, defn, prefer_literal=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._ctor_kwargs = kwargs
        self._name = name
        self._defn = defn
        self._prefer_literal = prefer_literal
        functools.update_wrapper(self, defn)

    @property
    def _uuid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An instance-specific UUID, to avoid multiple deserializations of\n        a given instance.\n\n        Note this is lazily-generated, for performance reasons.\n        '
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid1())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        if False:
            print('Hello World!')
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self
        self._recent.append(self)

    def _register(self):
        if False:
            return 10
        from numba.core.typing.templates import make_intrinsic_template, infer_global
        template = make_intrinsic_template(self, self._defn, self._name, prefer_literal=self._prefer_literal, kwargs=self._ctor_kwargs)
        infer(template)
        infer_global(self, types.Function(template))

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        This is only defined to pretend to be a callable from CPython.\n        '
        msg = '{0} is not usable in pure-python'.format(self)
        raise NotImplementedError(msg)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<intrinsic {0}>'.format(self._name)

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _reduce_states(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        NOTE: part of ReduceMixin protocol\n        '
        return dict(uuid=self._uuid, name=self._name, defn=self._defn)

    @classmethod
    def _rebuild(cls, uuid, name, defn):
        if False:
            i = 10
            return i + 15
        '\n        NOTE: part of ReduceMixin protocol\n        '
        try:
            return cls._memo[uuid]
        except KeyError:
            llc = cls(name=name, defn=defn)
            llc._register()
            llc._set_uuid(uuid)
            return llc

def intrinsic(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    A decorator marking the decorated function as typing and implementing\n    *func* in nopython mode using the llvmlite IRBuilder API.  This is an escape\n    hatch for expert users to build custom LLVM IR that will be inlined to\n    the caller.\n\n    The first argument to *func* is the typing context.  The rest of the\n    arguments corresponds to the type of arguments of the decorated function.\n    These arguments are also used as the formal argument of the decorated\n    function.  If *func* has the signature ``foo(typing_context, arg0, arg1)``,\n    the decorated function will have the signature ``foo(arg0, arg1)``.\n\n    The return values of *func* should be a 2-tuple of expected type signature,\n    and a code-generation function that will passed to ``lower_builtin``.\n    For unsupported operation, return None.\n\n    Here is an example implementing a ``cast_int_to_byte_ptr`` that cast\n    any integer to a byte pointer::\n\n        @intrinsic\n        def cast_int_to_byte_ptr(typingctx, src):\n            # check for accepted types\n            if isinstance(src, types.Integer):\n                # create the expected type signature\n                result_type = types.CPointer(types.uint8)\n                sig = result_type(types.uintp)\n                # defines the custom code generation\n                def codegen(context, builder, signature, args):\n                    # llvm IRBuilder code here\n                    [src] = args\n                    rtype = signature.return_type\n                    llrtype = context.get_value_type(rtype)\n                    return builder.inttoptr(src, llrtype)\n                return sig, codegen\n    '

    def _intrinsic(func):
        if False:
            i = 10
            return i + 15
        name = getattr(func, '__name__', str(func))
        llc = _Intrinsic(name, func, **kwargs)
        llc._register()
        return llc
    if not kwargs:
        return _intrinsic(*args)
    else:

        def wrapper(func):
            if False:
                i = 10
                return i + 15
            return _intrinsic(func)
        return wrapper

def get_cython_function_address(module_name, function_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the address of a Cython function.\n\n    Args\n    ----\n    module_name:\n        Name of the Cython module\n    function_name:\n        Name of the Cython function\n\n    Returns\n    -------\n    A Python int containing the address of the function\n\n    '
    return _import_cython_function(module_name, function_name)

def include_path():
    if False:
        print('Hello World!')
    'Returns the C include directory path.\n    '
    include_dir = os.path.dirname(os.path.dirname(numba.__file__))
    path = os.path.abspath(include_dir)
    return path

def sentry_literal_args(pysig, literal_args, args, kwargs):
    if False:
        i = 10
        return i + 15
    'Ensures that the given argument types (in *args* and *kwargs*) are\n    literally typed for a function with the python signature *pysig* and the\n    list of literal argument names in *literal_args*.\n\n    Alternatively, this is the same as::\n\n        SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)\n    '
    boundargs = pysig.bind(*args, **kwargs)
    request_pos = set()
    missing = False
    for (i, (k, v)) in enumerate(boundargs.arguments.items()):
        if k in literal_args:
            request_pos.add(i)
            if not isinstance(v, types.Literal):
                missing = True
    if missing:
        e = errors.ForceLiteralArg(request_pos)

        def folded(args, kwargs):
            if False:
                print('Hello World!')
            out = pysig.bind(*args, **kwargs).arguments.values()
            return tuple(out)
        raise e.bind_fold_arguments(folded)

class SentryLiteralArgs(collections.namedtuple('_SentryLiteralArgs', ['literal_args'])):
    """
    Parameters
    ----------
    literal_args : Sequence[str]
        A sequence of names for literal arguments

    Examples
    --------

    The following line:

    >>> SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)

    is equivalent to:

    >>> sentry_literal_args(pysig, literal_args, args, kwargs)
    """

    def for_function(self, func):
        if False:
            while True:
                i = 10
        'Bind the sentry to the signature of *func*.\n\n        Parameters\n        ----------\n        func : Function\n            A python function.\n\n        Returns\n        -------\n        obj : BoundLiteralArgs\n        '
        return self.for_pysig(utils.pysignature(func))

    def for_pysig(self, pysig):
        if False:
            i = 10
            return i + 15
        'Bind the sentry to the given signature *pysig*.\n\n        Parameters\n        ----------\n        pysig : inspect.Signature\n\n\n        Returns\n        -------\n        obj : BoundLiteralArgs\n        '
        return BoundLiteralArgs(pysig=pysig, literal_args=self.literal_args)

class BoundLiteralArgs(collections.namedtuple('BoundLiteralArgs', ['pysig', 'literal_args'])):
    """
    This class is usually created by SentryLiteralArgs.
    """

    def bind(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Bind to argument types.\n        '
        return sentry_literal_args(self.pysig, self.literal_args, args, kwargs)

def is_jitted(function):
    if False:
        for i in range(10):
            print('nop')
    'Returns True if a function is wrapped by one of the Numba @jit\n    decorators, for example: numba.jit, numba.njit\n\n    The purpose of this function is to provide a means to check if a function is\n    already JIT decorated.\n    '
    from numba.core.dispatcher import Dispatcher
    return isinstance(function, Dispatcher)