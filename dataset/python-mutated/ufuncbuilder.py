import inspect
import warnings
from contextlib import contextmanager
from numba.core import config, targetconfig
from numba.core.decorators import jit
from numba.core.descriptors import TargetDescriptor
from numba.core.extending import is_jitted
from numba.core.errors import NumbaDeprecationWarning
from numba.core.options import TargetOptions, include_default_options
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry
from numba.core import utils, types, serialize, compiler, sigutils
from numba.np.numpy_support import as_dtype
from numba.np.ufunc import _internal
from numba.np.ufunc.sigparse import parse_signature
from numba.np.ufunc.wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.core.caching import FunctionCache, NullCache
from numba.core.compiler_lock import global_compiler_lock
_options_mixin = include_default_options('nopython', 'forceobj', 'boundscheck', 'fastmath', 'target_backend', 'writable_args')

class UFuncTargetOptions(_options_mixin, TargetOptions):

    def finalize(self, flags, options):
        if False:
            i = 10
            return i + 15
        if not flags.is_set('enable_pyobject'):
            flags.enable_pyobject = True
        if not flags.is_set('enable_looplift'):
            flags.enable_looplift = True
        flags.inherit_if_not_set('nrt', default=True)
        if not flags.is_set('debuginfo'):
            flags.debuginfo = config.DEBUGINFO_DEFAULT
        if not flags.is_set('boundscheck'):
            flags.boundscheck = flags.debuginfo
        flags.enable_pyobject_looplift = True
        flags.inherit_if_not_set('fastmath')

class UFuncTarget(TargetDescriptor):
    options = UFuncTargetOptions

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('ufunc')

    @property
    def typing_context(self):
        if False:
            for i in range(10):
                print('nop')
        return cpu_target.typing_context

    @property
    def target_context(self):
        if False:
            print('Hello World!')
        return cpu_target.target_context
ufunc_target = UFuncTarget()

class UFuncDispatcher(serialize.ReduceMixin):
    """
    An object handling compilation of various signatures for a ufunc.
    """
    targetdescr = ufunc_target

    def __init__(self, py_func, locals={}, targetoptions={}):
        if False:
            while True:
                i = 10
        self.py_func = py_func
        self.overloads = utils.UniqueDict()
        self.targetoptions = targetoptions
        self.locals = locals
        self.cache = NullCache()

    def _reduce_states(self):
        if False:
            i = 10
            return i + 15
        '\n        NOTE: part of ReduceMixin protocol\n        '
        return dict(pyfunc=self.py_func, locals=self.locals, targetoptions=self.targetoptions)

    @classmethod
    def _rebuild(cls, pyfunc, locals, targetoptions):
        if False:
            i = 10
            return i + 15
        '\n        NOTE: part of ReduceMixin protocol\n        '
        return cls(py_func=pyfunc, locals=locals, targetoptions=targetoptions)

    def enable_caching(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache = FunctionCache(self.py_func)

    def compile(self, sig, locals={}, **targetoptions):
        if False:
            i = 10
            return i + 15
        locs = self.locals.copy()
        locs.update(locals)
        topt = self.targetoptions.copy()
        topt.update(targetoptions)
        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)
        flags.no_cpython_wrapper = True
        flags.error_model = 'numpy'
        flags.enable_looplift = False
        return self._compile_core(sig, flags, locals)

    def _compile_core(self, sig, flags, locals):
        if False:
            print('Hello World!')
        '\n        Trigger the compiler on the core function or load a previously\n        compiled version from the cache.  Returns the CompileResult.\n        '
        typingctx = self.targetdescr.typing_context
        targetctx = self.targetdescr.target_context

        @contextmanager
        def store_overloads_on_success():
            if False:
                return 10
            try:
                yield
            except Exception:
                raise
            else:
                exists = self.overloads.get(cres.signature)
                if exists is None:
                    self.overloads[cres.signature] = cres
        with global_compiler_lock:
            with targetconfig.ConfigStack().enter(flags.copy()):
                with store_overloads_on_success():
                    cres = self.cache.load_overload(sig, targetctx)
                    if cres is not None:
                        return cres
                    (args, return_type) = sigutils.normalize_signature(sig)
                    cres = compiler.compile_extra(typingctx, targetctx, self.py_func, args=args, return_type=return_type, flags=flags, locals=locals)
                    self.cache.save_overload(sig, cres)
                    return cres
dispatcher_registry[target_registry['npyufunc']] = UFuncDispatcher

def _compile_element_wise_function(nb_func, targetoptions, sig):
    if False:
        print('Hello World!')
    cres = nb_func.compile(sig, **targetoptions)
    (args, return_type) = sigutils.normalize_signature(sig)
    return (cres, args, return_type)

def _finalize_ufunc_signature(cres, args, return_type):
    if False:
        for i in range(10):
            print('nop')
    "Given a compilation result, argument types, and a return type,\n    build a valid Numba signature after validating that it doesn't\n    violate the constraints for the compilation mode.\n    "
    if return_type is None:
        if cres.objectmode:
            raise TypeError('return type must be specified for object mode')
        else:
            return_type = cres.signature.return_type
    assert return_type != types.pyobject
    return return_type(*args)

def _build_element_wise_ufunc_wrapper(cres, signature):
    if False:
        return 10
    'Build a wrapper for the ufunc loop entry point given by the\n    compilation result object, using the element-wise signature.\n    '
    ctx = cres.target_context
    library = cres.library
    fname = cres.fndesc.llvm_func_name
    with global_compiler_lock:
        info = build_ufunc_wrapper(library, ctx, fname, signature, cres.objectmode, cres)
        ptr = info.library.get_pointer_to_function(info.name)
    dtypenums = [as_dtype(a).num for a in signature.args]
    dtypenums.append(as_dtype(signature.return_type).num)
    return (dtypenums, ptr, cres.environment)
_identities = {0: _internal.PyUFunc_Zero, 1: _internal.PyUFunc_One, None: _internal.PyUFunc_None, 'reorderable': _internal.PyUFunc_ReorderableNone}

def parse_identity(identity):
    if False:
        return 10
    '\n    Parse an identity value and return the corresponding low-level value\n    for Numpy.\n    '
    try:
        identity = _identities[identity]
    except KeyError:
        raise ValueError('Invalid identity value %r' % (identity,))
    return identity

@contextmanager
def _suppress_deprecation_warning_nopython_not_supplied():
    if False:
        for i in range(10):
            print('nop')
    'This suppresses the NumbaDeprecationWarning that occurs through the use\n    of `jit` without the `nopython` kwarg. This use of `jit` occurs in a few\n    places in the `{g,}ufunc` mechanism in Numba, predominantly to wrap the\n    "kernel" function.'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NumbaDeprecationWarning, message=".*The 'nopython' keyword argument was not supplied*")
        yield

class _BaseUFuncBuilder(object):

    def add(self, sig=None):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'targetoptions'):
            targetoptions = self.targetoptions
        else:
            targetoptions = self.nb_func.targetoptions
        (cres, args, return_type) = _compile_element_wise_function(self.nb_func, targetoptions, sig)
        sig = self._finalize_signature(cres, args, return_type)
        self._sigs.append(sig)
        self._cres[sig] = cres
        return cres

    def disable_compile(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disable the compilation of new signatures at call time.\n        '

class UFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, identity=None, cache=False, targetoptions={}):
        if False:
            while True:
                i = 10
        if is_jitted(py_func):
            py_func = py_func.py_func
        self.py_func = py_func
        self.identity = parse_identity(identity)
        with _suppress_deprecation_warning_nopython_not_supplied():
            self.nb_func = jit(_target='npyufunc', cache=cache, **targetoptions)(py_func)
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        if False:
            return 10
        'Slated for deprecation, use ufuncbuilder._finalize_ufunc_signature()\n        instead.\n        '
        return _finalize_ufunc_signature(cres, args, return_type)

    def build_ufunc(self):
        if False:
            print('Hello World!')
        with global_compiler_lock:
            dtypelist = []
            ptrlist = []
            if not self.nb_func:
                raise TypeError('No definition')
            keepalive = []
            cres = None
            for sig in self._sigs:
                cres = self._cres[sig]
                (dtypenums, ptr, env) = self.build(cres, sig)
                dtypelist.append(dtypenums)
                ptrlist.append(int(ptr))
                keepalive.append((cres.library, env))
            datlist = [None] * len(ptrlist)
            if cres is None:
                argspec = inspect.getfullargspec(self.py_func)
                inct = len(argspec.args)
            else:
                inct = len(cres.signature.args)
            outct = 1
            ufunc = _internal.fromfunc(self.py_func.__name__, self.py_func.__doc__, ptrlist, dtypelist, inct, outct, datlist, keepalive, self.identity)
            return ufunc

    def build(self, cres, signature):
        if False:
            while True:
                i = 10
        'Slated for deprecation, use\n        ufuncbuilder._build_element_wise_ufunc_wrapper().\n        '
        return _build_element_wise_ufunc_wrapper(cres, signature)

class GUFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, signature, identity=None, cache=False, targetoptions={}, writable_args=()):
        if False:
            return 10
        self.py_func = py_func
        self.identity = parse_identity(identity)
        with _suppress_deprecation_warning_nopython_not_supplied():
            self.nb_func = jit(_target='npyufunc', cache=cache)(py_func)
        self.signature = signature
        (self.sin, self.sout) = parse_signature(signature)
        self.targetoptions = targetoptions
        self.cache = cache
        self._sigs = []
        self._cres = {}
        transform_arg = _get_transform_arg(py_func)
        self.writable_args = tuple([transform_arg(a) for a in writable_args])

    def _finalize_signature(self, cres, args, return_type):
        if False:
            return 10
        if not cres.objectmode and cres.signature.return_type != types.void:
            raise TypeError('gufunc kernel must have void return type')
        if return_type is None:
            return_type = types.void
        return return_type(*args)

    @global_compiler_lock
    def build_ufunc(self):
        if False:
            for i in range(10):
                print('nop')
        type_list = []
        func_list = []
        if not self.nb_func:
            raise TypeError('No definition')
        keepalive = []
        for sig in self._sigs:
            cres = self._cres[sig]
            (dtypenums, ptr, env) = self.build(cres)
            type_list.append(dtypenums)
            func_list.append(int(ptr))
            keepalive.append((cres.library, env))
        datalist = [None] * len(func_list)
        nin = len(self.sin)
        nout = len(self.sout)
        ufunc = _internal.fromfunc(self.py_func.__name__, self.py_func.__doc__, func_list, type_list, nin, nout, datalist, keepalive, self.identity, self.signature, self.writable_args)
        return ufunc

    def build(self, cres):
        if False:
            return 10
        '\n        Returns (dtype numbers, function ptr, EnvironmentObject)\n        '
        signature = cres.signature
        info = build_gufunc_wrapper(self.py_func, cres, self.sin, self.sout, cache=self.cache, is_parfors=False)
        env = info.env
        ptr = info.library.get_pointer_to_function(info.name)
        dtypenums = []
        for a in signature.args:
            if isinstance(a, types.Array):
                ty = a.dtype
            else:
                ty = a
            dtypenums.append(as_dtype(ty).num)
        return (dtypenums, ptr, env)

def _get_transform_arg(py_func):
    if False:
        return 10
    'Return function that transform arg into index'
    args = inspect.getfullargspec(py_func).args
    pos_by_arg = {arg: i for (i, arg) in enumerate(args)}

    def transform_arg(arg):
        if False:
            return 10
        if isinstance(arg, int):
            return arg
        try:
            return pos_by_arg[arg]
        except KeyError:
            msg = f'Specified writable arg {arg} not found in arg list {args} for function {py_func.__qualname__}'
            raise RuntimeError(msg)
    return transform_arg