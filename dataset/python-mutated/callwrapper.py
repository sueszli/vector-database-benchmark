from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils

class _ArgManager(object):
    """
    A utility class to handle argument unboxing and cleanup
    """

    def __init__(self, context, builder, api, env_manager, endblk, nargs):
        if False:
            for i in range(10):
                print('nop')
        self.context = context
        self.builder = builder
        self.api = api
        self.env_manager = env_manager
        self.arg_count = 0
        self.cleanups = []
        self.nextblk = endblk

    def add_arg(self, obj, ty):
        if False:
            return 10
        '\n        Unbox argument and emit code that handles any error during unboxing.\n        Args are cleaned up in reverse order of the parameter list, and\n        cleanup begins as soon as unboxing of any argument fails. E.g. failure\n        on arg2 will result in control flow going through:\n\n            arg2.err -> arg1.err -> arg0.err -> arg.end (returns)\n        '
        native = self.api.to_native_value(ty, obj)
        with cgutils.if_unlikely(self.builder, native.is_error):
            self.builder.branch(self.nextblk)

        def cleanup_arg():
            if False:
                return 10
            self.api.reflect_native_value(ty, native.value, self.env_manager)
            if native.cleanup is not None:
                native.cleanup()
            if self.context.enable_nrt:
                self.context.nrt.decref(self.builder, ty, native.value)
        self.cleanups.append(cleanup_arg)
        cleanupblk = self.builder.append_basic_block('arg%d.err' % self.arg_count)
        with self.builder.goto_block(cleanupblk):
            cleanup_arg()
            self.builder.branch(self.nextblk)
        self.nextblk = cleanupblk
        self.arg_count += 1
        return native.value

    def emit_cleanup(self):
        if False:
            i = 10
            return i + 15
        '\n        Emit the cleanup code after returning from the wrapped function.\n        '
        for dtor in self.cleanups:
            dtor()

class _GilManager(object):
    """
    A utility class to handle releasing the GIL and then re-acquiring it
    again.
    """

    def __init__(self, builder, api, argman):
        if False:
            return 10
        self.builder = builder
        self.api = api
        self.argman = argman
        self.thread_state = api.save_thread()

    def emit_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        self.api.restore_thread(self.thread_state)
        self.argman.emit_cleanup()

class PyCallWrapper(object):

    def __init__(self, context, module, func, fndesc, env, call_helper, release_gil):
        if False:
            while True:
                i = 10
        self.context = context
        self.module = module
        self.func = func
        self.fndesc = fndesc
        self.env = env
        self.release_gil = release_gil

    def build(self):
        if False:
            return 10
        wrapname = self.fndesc.llvm_cpython_wrapper_name
        pyobj = self.context.get_argument_type(types.pyobject)
        wrapty = llvmlite.ir.FunctionType(pyobj, [pyobj, pyobj, pyobj])
        wrapper = llvmlite.ir.Function(self.module, wrapty, name=wrapname)
        builder = IRBuilder(wrapper.append_basic_block('entry'))
        (closure, args, kws) = wrapper.args
        closure.name = 'py_closure'
        args.name = 'py_args'
        kws.name = 'py_kws'
        api = self.context.get_python_api(builder)
        self.build_wrapper(api, builder, closure, args, kws)
        return (wrapper, api)

    def build_wrapper(self, api, builder, closure, args, kws):
        if False:
            i = 10
            return i + 15
        nargs = len(self.fndesc.argtypes)
        objs = [api.alloca_obj() for _ in range(nargs)]
        parseok = api.unpack_tuple(args, self.fndesc.qualname, nargs, nargs, *objs)
        pred = builder.icmp_unsigned('==', parseok, Constant(parseok.type, None))
        with cgutils.if_unlikely(builder, pred):
            builder.ret(api.get_null_object())
        endblk = builder.append_basic_block('arg.end')
        with builder.goto_block(endblk):
            builder.ret(api.get_null_object())
        env_manager = self.get_env(api, builder)
        cleanup_manager = _ArgManager(self.context, builder, api, env_manager, endblk, nargs)
        innerargs = []
        for (obj, ty) in zip(objs, self.fndesc.argtypes):
            if isinstance(ty, types.Omitted):
                innerargs.append(None)
            else:
                val = cleanup_manager.add_arg(builder.load(obj), ty)
                innerargs.append(val)
        if self.release_gil:
            cleanup_manager = _GilManager(builder, api, cleanup_manager)
        (status, retval) = self.context.call_conv.call_function(builder, self.func, self.fndesc.restype, self.fndesc.argtypes, innerargs, attrs=('noinline',))
        self.debug_print(builder, '# callwrapper: emit_cleanup')
        cleanup_manager.emit_cleanup()
        self.debug_print(builder, '# callwrapper: emit_cleanup end')
        with builder.if_then(status.is_ok, likely=True):
            with builder.if_then(status.is_none):
                api.return_none()
            retty = self._simplified_return_type()
            obj = api.from_native_return(retty, retval, env_manager)
            builder.ret(obj)
        self.context.call_conv.raise_error(builder, api, status)
        builder.ret(api.get_null_object())

    def get_env(self, api, builder):
        if False:
            for i in range(10):
                print('nop')
        'Get the Environment object which is declared as a global\n        in the module of the wrapped function.\n        '
        envname = self.context.get_env_name(self.fndesc)
        gvptr = self.context.declare_env_global(builder.module, envname)
        envptr = builder.load(gvptr)
        env_body = self.context.get_env_body(builder, envptr)
        api.emit_environment_sentry(envptr, return_pyobject=True, debug_msg=self.fndesc.env_name)
        env_manager = api.get_env_manager(self.env, env_body, envptr)
        return env_manager

    def _simplified_return_type(self):
        if False:
            while True:
                i = 10
        '\n        The NPM callconv has already converted simplified optional types.\n        We can simply use the value type from it.\n        '
        restype = self.fndesc.restype
        if isinstance(restype, types.Optional):
            return restype.type
        else:
            return restype

    def debug_print(self, builder, msg):
        if False:
            return 10
        if config.DEBUG_JIT:
            self.context.debug_print(builder, 'DEBUGJIT: {0}'.format(msg))