"""
Function descriptors.
"""
from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module

def default_mangler(name, argtypes, *, abi_tags=(), uid=None):
    if False:
        i = 10
        return i + 15
    return itanium_mangler.mangle(name, argtypes, abi_tags=abi_tags, uid=uid)

def qualifying_prefix(modname, qualname):
    if False:
        print('Hello World!')
    '\n    Returns a new string that is used for the first half of the mangled name.\n    '
    return '{}.{}'.format(modname, qualname) if modname else qualname

class FunctionDescriptor(object):
    """
    Base class for function descriptors: an object used to carry
    useful metadata about a natively callable function.

    Note that while `FunctionIdentity` denotes a Python function
    which is being concretely compiled by Numba, `FunctionDescriptor`
    may be more "abstract": e.g. a function decorated with `@generated_jit`.
    """
    __slots__ = ('native', 'modname', 'qualname', 'doc', 'typemap', 'calltypes', 'args', 'kws', 'restype', 'argtypes', 'mangled_name', 'unique_name', 'env_name', 'global_dict', 'inline', 'noalias', 'abi_tags', 'uid')

    def __init__(self, native, modname, qualname, unique_name, doc, typemap, restype, calltypes, args, kws, mangler=None, argtypes=None, inline=False, noalias=False, env_name=None, global_dict=None, abi_tags=(), uid=None):
        if False:
            i = 10
            return i + 15
        self.native = native
        self.modname = modname
        self.global_dict = global_dict
        self.qualname = qualname
        self.unique_name = unique_name
        self.doc = doc
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        if argtypes is not None:
            assert isinstance(argtypes, tuple), argtypes
            self.argtypes = argtypes
        else:
            self.argtypes = tuple((self.typemap['arg.' + a] for a in args))
        mangler = default_mangler if mangler is None else mangler
        qualprefix = qualifying_prefix(self.modname, self.qualname)
        self.uid = uid
        self.mangled_name = mangler(qualprefix, self.argtypes, abi_tags=abi_tags, uid=uid)
        if env_name is None:
            env_name = mangler('.NumbaEnv.{}'.format(qualprefix), self.argtypes, abi_tags=abi_tags, uid=uid)
        self.env_name = env_name
        self.inline = inline
        self.noalias = noalias
        self.abi_tags = abi_tags

    def lookup_globals(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the global dictionary of the function.\n        It may not match the Module's globals if the function is created\n        dynamically (i.e. exec)\n        "
        return self.global_dict or self.lookup_module().__dict__

    def lookup_module(self):
        if False:
            return 10
        "\n        Return the module in which this function is supposed to exist.\n        This may be a dummy module if the function was dynamically\n        generated or the module can't be found.\n        "
        if self.modname == _dynamic_modname:
            return _dynamic_module
        else:
            try:
                return importlib.import_module(self.modname)
            except ImportError:
                return _dynamic_module

    def lookup_function(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the original function object described by this object.\n        '
        return getattr(self.lookup_module(), self.qualname)

    @property
    def llvm_func_name(self):
        if False:
            return 10
        '\n        The LLVM-registered name for the raw function.\n        '
        return self.mangled_name

    @property
    def llvm_cpython_wrapper_name(self):
        if False:
            return 10
        '\n        The LLVM-registered name for a CPython-compatible wrapper of the\n        raw function (i.e. a PyCFunctionWithKeywords).\n        '
        return itanium_mangler.prepend_namespace(self.mangled_name, ns='cpython')

    @property
    def llvm_cfunc_wrapper_name(self):
        if False:
            i = 10
            return i + 15
        '\n        The LLVM-registered name for a C-compatible wrapper of the\n        raw function.\n        '
        return 'cfunc.' + self.mangled_name

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<function descriptor %r>' % self.unique_name

    @classmethod
    def _get_function_info(cls, func_ir):
        if False:
            return 10
        '\n        Returns\n        -------\n        qualname, unique_name, modname, doc, args, kws, globals\n\n        ``unique_name`` must be a unique name.\n        '
        func = func_ir.func_id.func
        qualname = func_ir.func_id.func_qualname
        modname = func.__module__
        doc = func.__doc__ or ''
        args = tuple(func_ir.arg_names)
        kws = ()
        global_dict = None
        if modname is None:
            modname = _dynamic_modname
            global_dict = func_ir.func_id.func.__globals__
        unique_name = func_ir.func_id.unique_name
        return (qualname, unique_name, modname, doc, args, kws, global_dict)

    @classmethod
    def _from_python_function(cls, func_ir, typemap, restype, calltypes, native, mangler=None, inline=False, noalias=False, abi_tags=()):
        if False:
            print('Hello World!')
        (qualname, unique_name, modname, doc, args, kws, global_dict) = cls._get_function_info(func_ir)
        self = cls(native, modname, qualname, unique_name, doc, typemap, restype, calltypes, args, kws, mangler=mangler, inline=inline, noalias=noalias, global_dict=global_dict, abi_tags=abi_tags, uid=func_ir.func_id.unique_id)
        return self

class PythonFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for Numba-compiled functions.
    """
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, func_ir, typemap, restype, calltypes, mangler, inline, noalias, abi_tags):
        if False:
            return 10
        '\n        Build a FunctionDescriptor for a given specialization of a Python\n        function (in nopython mode).\n        '
        return cls._from_python_function(func_ir, typemap, restype, calltypes, native=True, mangler=mangler, inline=inline, noalias=noalias, abi_tags=abi_tags)

    @classmethod
    def from_object_mode_function(cls, func_ir):
        if False:
            while True:
                i = 10
        '\n        Build a FunctionDescriptor for an object mode variant of a Python\n        function.\n        '
        typemap = defaultdict(lambda : types.pyobject)
        calltypes = typemap.copy()
        restype = types.pyobject
        return cls._from_python_function(func_ir, typemap, restype, calltypes, native=False)

class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__ = ()

    def __init__(self, name, restype, argtypes):
        if False:
            while True:
                i = 10
        args = ['arg%d' % i for i in range(len(argtypes))]

        def mangler(a, x, abi_tags, uid=None):
            if False:
                i = 10
                return i + 15
            return a
        super(ExternalFunctionDescriptor, self).__init__(native=True, modname=None, qualname=name, unique_name=name, doc='', typemap=None, restype=restype, calltypes=None, args=args, kws=None, mangler=mangler, argtypes=argtypes)