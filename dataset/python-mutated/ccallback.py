"""
Implementation of compiled C callbacks (@cfunc).
"""
import ctypes
from functools import cached_property
from numba.core import compiler, registry
from numba.core.caching import NullCache, FunctionCache
from numba.core.dispatcher import _FunctionCompiler
from numba.core.typing import signature
from numba.core.typing.ctypes_utils import to_ctypes
from numba.core.compiler_lock import global_compiler_lock

class _CFuncCompiler(_FunctionCompiler):

    def _customize_flags(self, flags):
        if False:
            print('Hello World!')
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = False
        flags.no_compile = True
        flags.enable_pyobject = False
        if flags.force_pyobject:
            raise NotImplementedError('object mode not allowed in C callbacks')
        return flags

class CFunc(object):
    """
    A compiled C callback, as created by the @cfunc decorator.
    """
    _targetdescr = registry.cpu_target

    def __init__(self, pyfunc, sig, locals, options, pipeline_class=compiler.Compiler):
        if False:
            return 10
        (args, return_type) = sig
        if return_type is None:
            raise TypeError('C callback needs an explicit return type')
        self.__name__ = pyfunc.__name__
        self.__qualname__ = getattr(pyfunc, '__qualname__', self.__name__)
        self.__wrapped__ = pyfunc
        self._pyfunc = pyfunc
        self._sig = signature(return_type, *args)
        self._compiler = _CFuncCompiler(pyfunc, self._targetdescr, options, locals, pipeline_class=pipeline_class)
        self._wrapper_name = None
        self._wrapper_address = None
        self._cache = NullCache()
        self._cache_hits = 0

    def enable_caching(self):
        if False:
            print('Hello World!')
        self._cache = FunctionCache(self._pyfunc)

    @global_compiler_lock
    def compile(self):
        if False:
            while True:
                i = 10
        cres = self._cache.load_overload(self._sig, self._targetdescr.target_context)
        if cres is None:
            cres = self._compile_uncached()
            self._cache.save_overload(self._sig, cres)
        else:
            self._cache_hits += 1
        self._library = cres.library
        self._wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
        self._wrapper_address = self._library.get_pointer_to_function(self._wrapper_name)

    def _compile_uncached(self):
        if False:
            print('Hello World!')
        sig = self._sig
        return self._compiler.compile(sig.args, sig.return_type)

    @property
    def native_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The process-wide symbol the C callback is exposed as.\n        '
        return self._wrapper_name

    @property
    def address(self):
        if False:
            while True:
                i = 10
        '\n        The address of the C callback.\n        '
        return self._wrapper_address

    @cached_property
    def cffi(self):
        if False:
            i = 10
            return i + 15
        '\n        A cffi function pointer representing the C callback.\n        '
        import cffi
        ffi = cffi.FFI()
        return ffi.cast('void *', self.address)

    @cached_property
    def ctypes(self):
        if False:
            i = 10
            return i + 15
        '\n        A ctypes function object representing the C callback.\n        '
        ctypes_args = [to_ctypes(ty) for ty in self._sig.args]
        ctypes_restype = to_ctypes(self._sig.return_type)
        functype = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_args)
        return functype(self.address)

    def inspect_llvm(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the LLVM IR of the C callback definition.\n        '
        return self._library.get_llvm_str()

    @property
    def cache_hits(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cache_hits

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Numba C callback %r>' % (self.__qualname__,)

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._pyfunc(*args, **kwargs)