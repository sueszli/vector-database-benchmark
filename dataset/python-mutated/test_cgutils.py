import contextlib
import ctypes
import struct
import sys
import llvmlite.ir as ir
import numpy as np
import unittest
from numba.core import types, typing, cgutils, cpu
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, run_in_subprocess
machine_int = ir.IntType(types.intp.bitwidth)

def machine_const(n):
    if False:
        return 10
    return ir.Constant(machine_int, n)

class StructureTestCase(TestCase):

    def setUp(self):
        if False:
            return 10
        typing_context = typing.Context()
        self.context = cpu.CPUContext(typing_context)

    @contextlib.contextmanager
    def compile_function(self, nargs):
        if False:
            i = 10
            return i + 15
        llvm_fnty = ir.FunctionType(machine_int, [machine_int] * nargs)
        ctypes_fnty = ctypes.CFUNCTYPE(ctypes.c_size_t, *(ctypes.c_size_t,) * nargs)
        module = self.context.create_module('')
        function = cgutils.get_or_insert_function(module, llvm_fnty, self.id())
        assert function.is_declaration
        entry_block = function.append_basic_block('entry')
        builder = ir.IRBuilder(entry_block)
        first = [True]

        @global_compiler_lock
        def call_func(*args):
            if False:
                print('Hello World!')
            codegen = self.context.codegen()
            library = codegen.create_library('test_module.%s' % self.id())
            library.add_ir_module(module)
            cptr = library.get_pointer_to_function(function.name)
            cfunc = ctypes_fnty(cptr)
            return cfunc(*args)
        yield (self.context, builder, function.args, call_func)

    def get_bytearray_addr(self, ba):
        if False:
            print('Hello World!')
        assert isinstance(ba, bytearray)
        ba_as_string = ctypes.pythonapi.PyByteArray_AsString
        ba_as_string.argtypes = [ctypes.py_object]
        ba_as_string.restype = ctypes.c_void_p
        return ba_as_string(ba)

    def test_compile_function(self):
        if False:
            i = 10
            return i + 15
        with self.compile_function(2) as (context, builder, args, call):
            res = builder.add(args[0], args[1])
            builder.ret(res)
        self.assertEqual(call(5, -2), 3)
        self.assertEqual(call(4, 2), 6)

    @contextlib.contextmanager
    def run_struct_access(self, struct_class, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        with self.compile_function(1) as (context, builder, args, call):
            inst = struct_class(context, builder)
            sptr = builder.add(args[0], machine_const(offset))
            sptr = builder.inttoptr(sptr, ir.PointerType(inst._type))
            inst = struct_class(context, builder, ref=sptr)
            yield (context, builder, args, inst)
            builder.ret(ir.Constant(machine_int, 0))
        call(self.get_bytearray_addr(buf))

    @contextlib.contextmanager
    def run_simple_struct_test(self, struct_class, struct_fmt, struct_args):
        if False:
            i = 10
            return i + 15
        buf = bytearray(b'!') * 40
        expected = buf[:]
        offset = 8
        with self.run_struct_access(struct_class, buf, offset) as (context, builder, args, inst):
            yield (context, builder, inst)
        self.assertNotEqual(buf, expected)
        struct.pack_into(struct_fmt, expected, offset, *struct_args)
        self.assertEqual(buf, expected)

    def test_int_fields(self):
        if False:
            while True:
                i = 10

        class S(cgutils.Structure):
            _fields = [('a', types.int32), ('b', types.uint16)]
        fmt = '=iH'
        with self.run_simple_struct_test(S, fmt, (305419896, 43981)) as (context, builder, inst):
            inst.a = ir.Constant(ir.IntType(32), 305419896)
            inst.b = ir.Constant(ir.IntType(16), 43981)

    def test_float_fields(self):
        if False:
            while True:
                i = 10

        class S(cgutils.Structure):
            _fields = [('a', types.float64), ('b', types.float32)]
        fmt = '=df'
        with self.run_simple_struct_test(S, fmt, (1.23, 4.56)) as (context, builder, inst):
            inst.a = ir.Constant(ir.DoubleType(), 1.23)
            inst.b = ir.Constant(ir.FloatType(), 4.56)

class TestCGContext(TestCase):
    """Tests for code generation context functionality"""

    def test_printf(self):
        if False:
            for i in range(10):
                print('nop')
        value = 123456
        code = f'if 1:\n        from numba import njit, types\n        from numba.extending import intrinsic\n\n        @intrinsic\n        def printf(tyctx, int_arg):\n            sig = types.void(int_arg)\n            def codegen(cgctx, builder, sig, llargs):\n                cgctx.printf(builder, "%d\\n", *llargs)\n            return sig, codegen\n\n        @njit\n        def foo():\n            printf({value})\n\n        foo()\n        '
        (out, _) = run_in_subprocess(code)
        self.assertIn(str(value), out.decode())
if __name__ == '__main__':
    unittest.main()