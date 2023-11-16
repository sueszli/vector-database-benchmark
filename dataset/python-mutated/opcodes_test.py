import pycnite.types
from pytype.pyc import opcodes
import unittest

class _TestBase(unittest.TestCase):
    """Base class for all opcodes.dis testing."""

    def dis(self, data, **kwargs):
        if False:
            return 10
        'Return the opcodes from disassembling a code sequence.'
        defaults = {'co_code': data, 'co_argcount': 0, 'co_posonlyargcount': 0, 'co_kwonlyargcount': 0, 'co_nlocals': 0, 'co_stacksize': 0, 'co_flags': 0, 'co_consts': [], 'co_names': [], 'co_varnames': [], 'co_filename': '', 'co_name': '', 'co_firstlineno': 0, 'co_lnotab': [], 'co_freevars': [], 'co_cellvars': [], 'python_version': self.python_version}
        defaults.update(kwargs)
        code = pycnite.types.CodeType38(**defaults)
        return opcodes.dis(code)

    def assertSimple(self, opcode, name):
        if False:
            print('Hello World!')
        'Assert that a single opcode byte disassembles to the given name.'
        self.assertName([opcode], name)

    def assertName(self, code, name):
        if False:
            print('Hello World!')
        'Assert that the first disassembled opcode has the given name.'
        self.assertEqual(self.dis(code)[0].name, name)

    def assertDisassembly(self, code, expected):
        if False:
            while True:
                i = 10
        'Assert that an extended code sequence has the expected disassembly.'
        ops = self.dis(code)
        self.assertEqual(len(ops), len(expected))
        for (o, e) in zip(ops, expected):
            if len(e) == 1:
                self.assertEqual(e, (o.name,))
            else:
                self.assertEqual(e, (o.name, o.arg))

    def assertLineNumbers(self, code, co_lnotab, expected):
        if False:
            print('Hello World!')
        'Assert that the opcodes have the expected line numbers.'
        ops = self.dis(code, co_lnotab=bytes(co_lnotab), co_firstlineno=1)
        self.assertEqual(len(ops), len(expected))
        for (o, e) in zip(ops, expected):
            self.assertEqual(e, o.line)

class CommonTest(_TestBase):
    """Test bytecodes that are common to multiple Python versions."""
    python_version = (3, 10)

    def test_pop_top(self):
        if False:
            print('Hello World!')
        self.assertSimple(1, 'POP_TOP')

    def test_store_name(self):
        if False:
            return 10
        self.assertName([90, 0], 'STORE_NAME')

    def test_for_iter(self):
        if False:
            i = 10
            return i + 15
        self.assertName([93, 0, 9], 'FOR_ITER')

    def test_extended_disassembly(self):
        if False:
            return 10
        code = [124, 0, 124, 0, 23, 1, 124, 0, 124, 0, 20, 1, 124, 0, 124, 0, 22, 1, 124, 0, 124, 0, 27, 1, 100, 0, 83, 0]
        expected = [('LOAD_FAST', 0), ('LOAD_FAST', 0), ('BINARY_ADD',), ('LOAD_FAST', 0), ('LOAD_FAST', 0), ('BINARY_MULTIPLY',), ('LOAD_FAST', 0), ('LOAD_FAST', 0), ('BINARY_MODULO',), ('LOAD_FAST', 0), ('LOAD_FAST', 0), ('BINARY_TRUE_DIVIDE',), ('LOAD_CONST', 0), ('RETURN_VALUE',)]
        self.assertDisassembly(code, expected)

class Python38Test(_TestBase):
    python_version = (3, 8, 0)

    def test_non_monotonic_line_numbers(self):
        if False:
            print('Hello World!')
        code = [101, 0, 100, 0, 100, 1, 131, 2, 83, 0]
        expected = [('LOAD_NAME', 0), ('LOAD_CONST', 0), ('LOAD_CONST', 1), ('CALL_FUNCTION', 2), ('RETURN_VALUE',)]
        self.assertDisassembly(code, expected)
        lnotab = [2, 1, 2, 1, 2, 254]
        self.assertLineNumbers(code, lnotab, [1, 2, 3, 1, 1])

class ExceptionBitmaskTest(unittest.TestCase):
    """Tests for opcodes._get_exception_bitmask."""

    def assertBitmask(self, *, offset_to_op, exc_ranges, expected_bitmask):
        if False:
            while True:
                i = 10
        bitmask = bin(opcodes._get_exception_bitmask(offset_to_op, exc_ranges))
        self.assertEqual(bitmask, expected_bitmask)

    def test_one_exception_range(self):
        if False:
            return 10
        self.assertBitmask(offset_to_op={1: None, 5: None, 8: None, 13: None}, exc_ranges={4: 10}, expected_bitmask='0b11111110000')

    def test_multiple_exception_ranges(self):
        if False:
            i = 10
            return i + 15
        self.assertBitmask(offset_to_op={1: None, 3: None, 5: None, 7: None, 9: None}, exc_ranges={1: 4, 7: 9}, expected_bitmask='0b1110011110')

    def test_length_one_range(self):
        if False:
            i = 10
            return i + 15
        self.assertBitmask(offset_to_op={0: None, 3: None, 6: None, 7: None, 12: None}, exc_ranges={0: 0, 6: 6, 7: 7, 12: 12}, expected_bitmask='0b1000011000001')

    def test_overlapping_ranges(self):
        if False:
            print('Hello World!')
        self.assertBitmask(offset_to_op={1: None, 5: None, 8: None, 13: None}, exc_ranges={1: 5, 4: 9}, expected_bitmask='0b1111111110')

    def test_no_exception(self):
        if False:
            print('Hello World!')
        self.assertBitmask(offset_to_op={1: None, 5: None, 8: None, 13: None}, exc_ranges={}, expected_bitmask='0b0')
if __name__ == '__main__':
    unittest.main()