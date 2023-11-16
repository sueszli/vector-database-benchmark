"""
Core Implementations for Generator/BitGenerator Models.
"""
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import intrinsic, make_attribute_wrapper, models, overload, register_jitable, register_model
from numba import float32

@register_model(types.NumPyRandomBitGeneratorType)
class NumPyRngBitGeneratorModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            for i in range(10):
                print('nop')
        members = [('parent', types.pyobject), ('state_address', types.uintp), ('state', types.uintp), ('fnptr_next_uint64', types.uintp), ('fnptr_next_uint32', types.uintp), ('fnptr_next_double', types.uintp), ('bit_generator', types.uintp)]
        super(NumPyRngBitGeneratorModel, self).__init__(dmm, fe_type, members)
_bit_gen_type = types.NumPyRandomBitGeneratorType('bit_generator')

@register_model(types.NumPyRandomGeneratorType)
class NumPyRandomGeneratorTypeModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            return 10
        members = [('bit_generator', _bit_gen_type), ('meminfo', types.MemInfoPointer(types.voidptr)), ('parent', types.pyobject)]
        super(NumPyRandomGeneratorTypeModel, self).__init__(dmm, fe_type, members)
make_attribute_wrapper(types.NumPyRandomGeneratorType, 'bit_generator', 'bit_generator')

def _generate_next_binding(overloadable_function, return_type):
    if False:
        return 10
    '\n        Generate the overloads for "next_(some type)" functions.\n    '

    @intrinsic
    def intrin_NumPyRandomBitGeneratorType_next_ty(tyctx, inst):
        if False:
            for i in range(10):
                print('nop')
        sig = return_type(inst)

        def codegen(cgctx, builder, sig, llargs):
            if False:
                i = 10
                return i + 15
            name = overloadable_function.__name__
            struct_ptr = cgutils.create_struct_proxy(inst)(cgctx, builder, value=llargs[0])
            state = struct_ptr.state
            next_double_addr = getattr(struct_ptr, f'fnptr_{name}')
            ll_void_ptr_t = cgctx.get_value_type(types.voidptr)
            ll_return_t = cgctx.get_value_type(return_type)
            ll_uintp_t = cgctx.get_value_type(types.uintp)
            next_fn_fnptr = builder.inttoptr(next_double_addr, ll_void_ptr_t)
            fnty = ir.FunctionType(ll_return_t, (ll_uintp_t,))
            next_fn = cgutils.get_or_insert_function(builder.module, fnty, name)
            fnptr_as_fntype = builder.bitcast(next_fn_fnptr, next_fn.type)
            ret = builder.call(fnptr_as_fntype, (state,))
            return ret
        return (sig, codegen)

    @overload(overloadable_function)
    def ol_next_ty(bitgen):
        if False:
            while True:
                i = 10
        if isinstance(bitgen, types.NumPyRandomBitGeneratorType):

            def impl(bitgen):
                if False:
                    print('Hello World!')
                return intrin_NumPyRandomBitGeneratorType_next_ty(bitgen)
            return impl

def next_double(bitgen):
    if False:
        while True:
            i = 10
    return bitgen.ctypes.next_double(bitgen.ctypes.state)

def next_uint32(bitgen):
    if False:
        for i in range(10):
            print('nop')
    return bitgen.ctypes.next_uint32(bitgen.ctypes.state)

def next_uint64(bitgen):
    if False:
        print('Hello World!')
    return bitgen.ctypes.next_uint64(bitgen.ctypes.state)
_generate_next_binding(next_double, types.double)
_generate_next_binding(next_uint32, types.uint32)
_generate_next_binding(next_uint64, types.uint64)

@register_jitable
def next_float(bitgen):
    if False:
        while True:
            i = 10
    return float32(float32(next_uint32(bitgen) >> 8) * float32(1.0) / float32(16777216.0))