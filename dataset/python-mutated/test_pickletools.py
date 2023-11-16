import pickle
import pickletools
from test import support
from test.pickletester import AbstractPickleTests
import doctest
import unittest

class OptimizedPickleTests(AbstractPickleTests, unittest.TestCase):

    def dumps(self, arg, proto=None, **kwargs):
        if False:
            return 10
        return pickletools.optimize(pickle.dumps(arg, proto, **kwargs))

    def loads(self, buf, **kwds):
        if False:
            i = 10
            return i + 15
        return pickle.loads(buf, **kwds)
    test_pickle_to_2x = None
    test_framed_write_sizes_with_delayed_writer = None

    def test_optimize_long_binget(self):
        if False:
            while True:
                i = 10
        data = [str(i) for i in range(257)]
        data.append(data[-1])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(data, proto)
            unpickled = pickle.loads(pickled)
            self.assertEqual(unpickled, data)
            self.assertIs(unpickled[-1], unpickled[-2])
            pickled2 = pickletools.optimize(pickled)
            unpickled2 = pickle.loads(pickled2)
            self.assertEqual(unpickled2, data)
            self.assertIs(unpickled2[-1], unpickled2[-2])
            self.assertNotIn(pickle.LONG_BINGET, pickled2)
            self.assertNotIn(pickle.LONG_BINPUT, pickled2)

    def test_optimize_binput_and_memoize(self):
        if False:
            return 10
        pickled = b'\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00]\x94(\x8c\x04spamq\x01\x8c\x03ham\x94h\x02e.'
        self.assertIn(pickle.BINPUT, pickled)
        unpickled = pickle.loads(pickled)
        self.assertEqual(unpickled, ['spam', 'ham', 'ham'])
        self.assertIs(unpickled[1], unpickled[2])
        pickled2 = pickletools.optimize(pickled)
        unpickled2 = pickle.loads(pickled2)
        self.assertEqual(unpickled2, ['spam', 'ham', 'ham'])
        self.assertIs(unpickled2[1], unpickled2[2])
        self.assertNotIn(pickle.BINPUT, pickled2)

class MiscTestCase(unittest.TestCase):

    def test__all__(self):
        if False:
            i = 10
            return i + 15
        not_exported = {'bytes_types', 'UP_TO_NEWLINE', 'TAKEN_FROM_ARGUMENT1', 'TAKEN_FROM_ARGUMENT4', 'TAKEN_FROM_ARGUMENT4U', 'TAKEN_FROM_ARGUMENT8U', 'ArgumentDescriptor', 'read_uint1', 'read_uint2', 'read_int4', 'read_uint4', 'read_uint8', 'read_stringnl', 'read_stringnl_noescape', 'read_stringnl_noescape_pair', 'read_string1', 'read_string4', 'read_bytes1', 'read_bytes4', 'read_bytes8', 'read_bytearray8', 'read_unicodestringnl', 'read_unicodestring1', 'read_unicodestring4', 'read_unicodestring8', 'read_decimalnl_short', 'read_decimalnl_long', 'read_floatnl', 'read_float8', 'read_long1', 'read_long4', 'uint1', 'uint2', 'int4', 'uint4', 'uint8', 'stringnl', 'stringnl_noescape', 'stringnl_noescape_pair', 'string1', 'string4', 'bytes1', 'bytes4', 'bytes8', 'bytearray8', 'unicodestringnl', 'unicodestring1', 'unicodestring4', 'unicodestring8', 'decimalnl_short', 'decimalnl_long', 'floatnl', 'float8', 'long1', 'long4', 'StackObject', 'pyint', 'pylong', 'pyinteger_or_bool', 'pybool', 'pyfloat', 'pybytes_or_str', 'pystring', 'pybytes', 'pybytearray', 'pyunicode', 'pynone', 'pytuple', 'pylist', 'pydict', 'pyset', 'pyfrozenset', 'pybuffer', 'anyobject', 'markobject', 'stackslice', 'OpcodeInfo', 'opcodes', 'code2op'}
        support.check__all__(self, pickletools, not_exported=not_exported)

def load_tests(loader, tests, pattern):
    if False:
        for i in range(10):
            print('nop')
    tests.addTest(doctest.DocTestSuite(pickletools))
    return tests
if __name__ == '__main__':
    unittest.main()