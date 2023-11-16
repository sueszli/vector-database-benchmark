from llvmlite import ir
from numba.core import cgutils, types
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs
registry = Registry()
lower = registry.lower

def libdevice_implement(func, retty, nbargs):
    if False:
        while True:
            i = 10

    def core(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        lmod = builder.module
        fretty = context.get_value_type(retty)
        fargtys = [context.get_value_type(arg.ty) for arg in nbargs]
        fnty = ir.FunctionType(fretty, fargtys)
        fn = cgutils.get_or_insert_function(lmod, fnty, func)
        return builder.call(fn, args)
    key = getattr(libdevice, func[5:])
    argtys = [arg.ty for arg in args if not arg.is_ptr]
    lower(key, *argtys)(core)

def libdevice_implement_multiple_returns(func, retty, prototype_args):
    if False:
        while True:
            i = 10
    sig = libdevicefuncs.create_signature(retty, prototype_args)
    nb_retty = sig.return_type

    def core(context, builder, sig, args):
        if False:
            return 10
        lmod = builder.module
        fargtys = []
        for arg in prototype_args:
            ty = context.get_value_type(arg.ty)
            if arg.is_ptr:
                ty = ty.as_pointer()
            fargtys.append(ty)
        fretty = context.get_value_type(retty)
        fnty = ir.FunctionType(fretty, fargtys)
        fn = cgutils.get_or_insert_function(lmod, fnty, func)
        actual_args = []
        virtual_args = []
        arg_idx = 0
        for arg in prototype_args:
            if arg.is_ptr:
                tmp_arg = cgutils.alloca_once(builder, context.get_value_type(arg.ty))
                actual_args.append(tmp_arg)
                virtual_args.append(tmp_arg)
            else:
                actual_args.append(args[arg_idx])
                arg_idx += 1
        ret = builder.call(fn, actual_args)
        tuple_args = []
        if retty != types.void:
            tuple_args.append(ret)
        for arg in virtual_args:
            tuple_args.append(builder.load(arg))
        if isinstance(nb_retty, types.UniTuple):
            return cgutils.pack_array(builder, tuple_args)
        else:
            return cgutils.pack_struct(builder, tuple_args)
    key = getattr(libdevice, func[5:])
    lower(key, *sig.args)(core)
for (func, (retty, args)) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        libdevice_implement_multiple_returns(func, retty, args)
    else:
        libdevice_implement(func, retty, args)