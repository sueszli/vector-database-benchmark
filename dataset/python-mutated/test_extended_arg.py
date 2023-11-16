import unittest
import dis
import struct
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase

class TestExtendedArg(TestCase):
    """
    Test support for the EXTENDED_ARG opcode.
    """
    bytecode_len = 255

    def get_extended_arg_load_const(self):
        if False:
            return 10
        '\n        Get a function with a EXTENDED_ARG opcode before a LOAD_CONST opcode.\n        '

        def f():
            if False:
                i = 10
                return i + 15
            x = 5
            return x
        b = bytearray(f.__code__.co_code)
        consts = f.__code__.co_consts
        bytecode_format = '<BB'
        consts = consts + (None,) * self.bytecode_len + (42,)
        if utils.PYVERSION >= (3, 11):
            offset = 2
        else:
            offset = 0
        packed_extend_arg = struct.pack(bytecode_format, dis.EXTENDED_ARG, 1)
        b[:] = b[:offset] + packed_extend_arg + b[offset:]
        f.__code__ = f.__code__.replace(co_code=bytes(b), co_consts=consts)
        return f

    def test_extended_arg_load_const(self):
        if False:
            print('Hello World!')
        pyfunc = self.get_extended_arg_load_const()
        self.assertGreater(len(pyfunc.__code__.co_consts), self.bytecode_len)
        self.assertPreciseEqual(pyfunc(), 42)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)
if __name__ == '__main__':
    unittest.main()