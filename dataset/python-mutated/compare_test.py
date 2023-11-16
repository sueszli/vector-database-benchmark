"""Tests for python.util.protobuf.compare."""
import copy
import re
import sys
import textwrap
import six
from google.protobuf import text_format
from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.protobuf import compare_test_pb2

def LargePbs(*args):
    if False:
        return 10
    'Converts ASCII string Large PBs to messages.'
    return [text_format.Merge(arg, compare_test_pb2.Large()) for arg in args]

class ProtoEqTest(googletest.TestCase):

    def assertNotEquals(self, a, b):
        if False:
            return 10
        'Asserts that ProtoEq says a != b.'
        (a, b) = LargePbs(a, b)
        googletest.TestCase.assertEqual(self, compare.ProtoEq(a, b), False)

    def assertEqual(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Asserts that ProtoEq says a == b.'
        (a, b) = LargePbs(a, b)
        googletest.TestCase.assertEqual(self, compare.ProtoEq(a, b), True)

    def testPrimitives(self):
        if False:
            for i in range(10):
                print('nop')
        googletest.TestCase.assertEqual(self, True, compare.ProtoEq('a', 'a'))
        googletest.TestCase.assertEqual(self, False, compare.ProtoEq('b', 'a'))

    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('', '')

    def testPrimitiveFields(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual('string_: "a"', '')
        self.assertEqual('string_: "a"', 'string_: "a"')
        self.assertNotEqual('string_: "b"', 'string_: "a"')
        self.assertNotEqual('string_: "ab"', 'string_: "aa"')
        self.assertNotEqual('int64_: 0', '')
        self.assertEqual('int64_: 0', 'int64_: 0')
        self.assertNotEqual('int64_: -1', '')
        self.assertNotEqual('int64_: 1', 'int64_: 0')
        self.assertNotEqual('int64_: 0', 'int64_: -1')
        self.assertNotEqual('float_: 0.0', '')
        self.assertEqual('float_: 0.0', 'float_: 0.0')
        self.assertNotEqual('float_: -0.1', '')
        self.assertNotEqual('float_: 3.14', 'float_: 0')
        self.assertNotEqual('float_: 0', 'float_: -0.1')
        self.assertEqual('float_: -0.1', 'float_: -0.1')
        self.assertNotEqual('bool_: true', '')
        self.assertNotEqual('bool_: false', '')
        self.assertNotEqual('bool_: true', 'bool_: false')
        self.assertEqual('bool_: false', 'bool_: false')
        self.assertEqual('bool_: true', 'bool_: true')
        self.assertNotEqual('enum_: A', '')
        self.assertNotEqual('enum_: B', 'enum_: A')
        self.assertNotEqual('enum_: C', 'enum_: B')
        self.assertEqual('enum_: C', 'enum_: C')

    def testRepeatedPrimitives(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual('int64s: 0', '')
        self.assertEqual('int64s: 0', 'int64s: 0')
        self.assertNotEqual('int64s: 1', 'int64s: 0')
        self.assertNotEqual('int64s: 0 int64s: 0', '')
        self.assertNotEqual('int64s: 0 int64s: 0', 'int64s: 0')
        self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0')
        self.assertNotEqual('int64s: 0 int64s: 1', 'int64s: 0')
        self.assertNotEqual('int64s: 1', 'int64s: 0 int64s: 2')
        self.assertNotEqual('int64s: 2 int64s: 0', 'int64s: 1')
        self.assertEqual('int64s: 0 int64s: 0', 'int64s: 0 int64s: 0')
        self.assertEqual('int64s: 0 int64s: 1', 'int64s: 0 int64s: 1')
        self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 0')
        self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 1')
        self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 2')
        self.assertNotEqual('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0')
        self.assertNotEqual('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0 int64s: 2')

    def testMessage(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual('small <>', '')
        self.assertEqual('small <>', 'small <>')
        self.assertNotEqual('small < strings: "a" >', '')
        self.assertNotEqual('small < strings: "a" >', 'small <>')
        self.assertEqual('small < strings: "a" >', 'small < strings: "a" >')
        self.assertNotEqual('small < strings: "b" >', 'small < strings: "a" >')
        self.assertNotEqual('small < strings: "a" strings: "b" >', 'small < strings: "a" >')
        self.assertNotEqual('string_: "a"', 'small <>')
        self.assertNotEqual('string_: "a"', 'small < strings: "b" >')
        self.assertNotEqual('string_: "a"', 'small < strings: "b" strings: "c" >')
        self.assertNotEqual('string_: "a" small <>', 'small <>')
        self.assertNotEqual('string_: "a" small <>', 'small < strings: "b" >')
        self.assertEqual('string_: "a" small <>', 'string_: "a" small <>')
        self.assertNotEqual('string_: "a" small < strings: "a" >', 'string_: "a" small <>')
        self.assertEqual('string_: "a" small < strings: "a" >', 'string_: "a" small < strings: "a" >')
        self.assertNotEqual('string_: "a" small < strings: "a" >', 'int64_: 1 small < strings: "a" >')
        self.assertNotEqual('string_: "a" small < strings: "a" >', 'int64_: 1')
        self.assertNotEqual('string_: "a"', 'int64_: 1 small < strings: "a" >')
        self.assertNotEqual('string_: "a" int64_: 0 small < strings: "a" >', 'int64_: 1 small < strings: "a" >')
        self.assertNotEqual('string_: "a" int64_: 1 small < strings: "a" >', 'string_: "a" int64_: 0 small < strings: "a" >')
        self.assertEqual('string_: "a" int64_: 0 small < strings: "a" >', 'string_: "a" int64_: 0 small < strings: "a" >')

    def testNestedMessage(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual('medium <>', '')
        self.assertEqual('medium <>', 'medium <>')
        self.assertNotEqual('medium < smalls <> >', 'medium <>')
        self.assertEqual('medium < smalls <> >', 'medium < smalls <> >')
        self.assertNotEqual('medium < smalls <> smalls <> >', 'medium < smalls <> >')
        self.assertEqual('medium < smalls <> smalls <> >', 'medium < smalls <> smalls <> >')
        self.assertNotEqual('medium < int32s: 0 >', 'medium < smalls <> >')
        self.assertNotEqual('medium < smalls < strings: "a"> >', 'medium < smalls <> >')

    def testTagOrder(self):
        if False:
            print('Hello World!')
        'Tests that different fields are ordered by tag number.\n\n    For reference, here are the relevant tag numbers from compare_test.proto:\n      optional string string_ = 1;\n      optional int64 int64_ = 2;\n      optional float float_ = 3;\n      optional Small small = 8;\n      optional Medium medium = 7;\n      optional Small small = 8;\n    '
        self.assertNotEqual('string_: "a"                      ', '             int64_: 1            ')
        self.assertNotEqual('string_: "a" int64_: 2            ', '             int64_: 1            ')
        self.assertNotEqual('string_: "b" int64_: 1            ', 'string_: "a" int64_: 2            ')
        self.assertEqual('string_: "a" int64_: 1            ', 'string_: "a" int64_: 1            ')
        self.assertNotEqual('string_: "a" int64_: 1 float_: 0.0', 'string_: "a" int64_: 1            ')
        self.assertEqual('string_: "a" int64_: 1 float_: 0.0', 'string_: "a" int64_: 1 float_: 0.0')
        self.assertNotEqual('string_: "a" int64_: 1 float_: 0.1', 'string_: "a" int64_: 1 float_: 0.0')
        self.assertNotEqual('string_: "a" int64_: 2 float_: 0.0', 'string_: "a" int64_: 1 float_: 0.1')
        self.assertNotEqual('string_: "a"                      ', '             int64_: 1 float_: 0.1')
        self.assertNotEqual('string_: "a"           float_: 0.0', '             int64_: 1            ')
        self.assertNotEqual('string_: "b"           float_: 0.0', 'string_: "a" int64_: 1            ')
        self.assertNotEqual('string_: "a"', 'small < strings: "a" >')
        self.assertNotEqual('string_: "a" small < strings: "a" >', 'small < strings: "b" >')
        self.assertNotEqual('string_: "a" small < strings: "b" >', 'string_: "a" small < strings: "a" >')
        self.assertEqual('string_: "a" small < strings: "a" >', 'string_: "a" small < strings: "a" >')
        self.assertNotEqual('string_: "a" medium <>', 'string_: "a" small < strings: "a" >')
        self.assertNotEqual('string_: "a" medium < smalls <> >', 'string_: "a" small < strings: "a" >')
        self.assertNotEqual('medium <>', 'small < strings: "a" >')
        self.assertNotEqual('medium <> small <>', 'small < strings: "a" >')
        self.assertNotEqual('medium < smalls <> >', 'small < strings: "a" >')
        self.assertNotEqual('medium < smalls < strings: "a" > >', 'small < strings: "b" >')

    def testIsClose(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(compare.isClose(1, 1, 1e-10))
        self.assertTrue(compare.isClose(65061.042, 65061.0322, 1e-05))
        self.assertFalse(compare.isClose(65061.042, 65061.0322, 1e-07))

    def testIsCloseNan(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(compare.isClose(float('nan'), float('nan'), 1e-10))
        self.assertFalse(compare.isClose(float('nan'), 1, 1e-10))
        self.assertFalse(compare.isClose(1, float('nan'), 1e-10))
        self.assertFalse(compare.isClose(float('nan'), float('inf'), 1e-10))

    def testIsCloseInf(self):
        if False:
            print('Hello World!')
        self.assertTrue(compare.isClose(float('inf'), float('inf'), 1e-10))
        self.assertTrue(compare.isClose(float('-inf'), float('-inf'), 1e-10))
        self.assertFalse(compare.isClose(float('-inf'), float('inf'), 1e-10))
        self.assertFalse(compare.isClose(float('inf'), 1, 1e-10))
        self.assertFalse(compare.isClose(1, float('inf'), 1e-10))

    def testIsCloseSubnormal(self):
        if False:
            print('Hello World!')
        x = sys.float_info.min * sys.float_info.epsilon
        self.assertTrue(compare.isClose(x, x, 1e-10))
        self.assertFalse(compare.isClose(x, 1, 1e-10))

class NormalizeNumbersTest(googletest.TestCase):
    """Tests for NormalizeNumberFields()."""

    def testNormalizesInts(self):
        if False:
            return 10
        pb = compare_test_pb2.Large(int64_=4)
        compare.NormalizeNumberFields(pb)
        self.assertIsInstance(pb.int64_, six.integer_types)
        pb.int64_ = 4
        compare.NormalizeNumberFields(pb)
        self.assertIsInstance(pb.int64_, six.integer_types)
        pb.int64_ = 9999999999999999
        compare.NormalizeNumberFields(pb)
        self.assertIsInstance(pb.int64_, six.integer_types)

    def testNormalizesRepeatedInts(self):
        if False:
            print('Hello World!')
        pb = compare_test_pb2.Large(int64s=[1, 400, 999999999999999])
        compare.NormalizeNumberFields(pb)
        self.assertIsInstance(pb.int64s[0], six.integer_types)
        self.assertIsInstance(pb.int64s[1], six.integer_types)
        self.assertIsInstance(pb.int64s[2], six.integer_types)

    def testNormalizesFloats(self):
        if False:
            for i in range(10):
                print('nop')
        pb1 = compare_test_pb2.Large(float_=1.2314352351231)
        pb2 = compare_test_pb2.Large(float_=1.231435)
        self.assertNotEqual(pb1.float_, pb2.float_)
        compare.NormalizeNumberFields(pb1)
        compare.NormalizeNumberFields(pb2)
        self.assertEqual(pb1.float_, pb2.float_)

    def testNormalizesRepeatedFloats(self):
        if False:
            return 10
        pb = compare_test_pb2.Large(medium=compare_test_pb2.Medium(floats=[0.111111111, 0.111111]))
        compare.NormalizeNumberFields(pb)
        for value in pb.medium.floats:
            self.assertAlmostEqual(0.111111, value)

    def testNormalizesDoubles(self):
        if False:
            for i in range(10):
                print('nop')
        pb1 = compare_test_pb2.Large(double_=1.2314352351231)
        pb2 = compare_test_pb2.Large(double_=1.2314352)
        self.assertNotEqual(pb1.double_, pb2.double_)
        compare.NormalizeNumberFields(pb1)
        compare.NormalizeNumberFields(pb2)
        self.assertEqual(pb1.double_, pb2.double_)

    def testNormalizesMaps(self):
        if False:
            print('Hello World!')
        pb = compare_test_pb2.WithMap()
        pb.value_message[4].strings.extend(['a', 'b', 'c'])
        pb.value_string['d'] = 'e'
        compare.NormalizeNumberFields(pb)

class AssertTest(googletest.TestCase):
    """Tests assertProtoEqual()."""

    def assertProtoEqual(self, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(a, six.string_types) and isinstance(b, six.string_types):
            (a, b) = LargePbs(a, b)
        compare.assertProtoEqual(self, a, b, **kwargs)

    def assertAll(self, a, **kwargs):
        if False:
            print('Hello World!')
        'Checks that all possible asserts pass.'
        self.assertProtoEqual(a, a, **kwargs)

    def assertSameNotEqual(self, a, b):
        if False:
            print('Hello World!')
        'Checks that assertProtoEqual() fails.'
        self.assertRaises(AssertionError, self.assertProtoEqual, a, b)

    def assertNone(self, a, b, message, **kwargs):
        if False:
            return 10
        'Checks that all possible asserts fail with the given message.'
        message = re.escape(textwrap.dedent(message))
        self.assertRaisesRegex(AssertionError, message, self.assertProtoEqual, a, b, **kwargs)

    def testCheckInitialized(self):
        if False:
            while True:
                i = 10
        a = compare_test_pb2.Labeled(optional=1)
        self.assertNone(a, a, 'Initialization errors: ', check_initialized=True)
        self.assertAll(a, check_initialized=False)
        b = copy.deepcopy(a)
        a.required = 2
        self.assertNone(a, b, 'Initialization errors: ', check_initialized=True)
        self.assertNone(a, b, '\n                    - required: 2\n                      optional: 1\n                    ', check_initialized=False)
        a = compare_test_pb2.Labeled(required=2)
        self.assertAll(a, check_initialized=True)
        self.assertAll(a, check_initialized=False)
        b = copy.deepcopy(a)
        b.required = 3
        message = '\n              - required: 2\n              ?           ^\n              + required: 3\n              ?           ^\n              '
        self.assertNone(a, b, message, check_initialized=True)
        self.assertNone(a, b, message, check_initialized=False)

    def testAssertEqualWithStringArg(self):
        if False:
            i = 10
            return i + 15
        pb = compare_test_pb2.Large(string_='abc', float_=1.234)
        compare.assertProtoEqual(self, "\n          string_: 'abc'\n          float_: 1.234\n        ", pb)

    def testNormalizesNumbers(self):
        if False:
            print('Hello World!')
        pb1 = compare_test_pb2.Large(int64_=4)
        pb2 = compare_test_pb2.Large(int64_=4)
        compare.assertProtoEqual(self, pb1, pb2)

    def testNormalizesFloat(self):
        if False:
            print('Hello World!')
        pb1 = compare_test_pb2.Large(double_=4.0)
        pb2 = compare_test_pb2.Large(double_=4)
        compare.assertProtoEqual(self, pb1, pb2, normalize_numbers=True)

    def testLargeProtoData(self):
        if False:
            for i in range(10):
                print('nop')
        number_of_entries = 2 ** 13
        string_value = 'dummystr'
        pb1_txt = 'strings: "dummystr"\n' * number_of_entries
        pb2 = compare_test_pb2.Small(strings=[string_value] * number_of_entries)
        compare.assertProtoEqual(self, pb1_txt, pb2)
        with self.assertRaises(AssertionError):
            compare.assertProtoEqual(self, pb1_txt + 'strings: "Should fail."', pb2)

    def testPrimitives(self):
        if False:
            print('Hello World!')
        self.assertAll('string_: "x"')
        self.assertNone('string_: "x"', 'string_: "y"', '\n                    - string_: "x"\n                    ?           ^\n                    + string_: "y"\n                    ?           ^\n                    ')

    def testRepeatedPrimitives(self):
        if False:
            while True:
                i = 10
        self.assertAll('int64s: 0 int64s: 1')
        self.assertSameNotEqual('int64s: 0 int64s: 1', 'int64s: 1 int64s: 0')
        self.assertSameNotEqual('int64s: 0 int64s: 1 int64s: 2', 'int64s: 2 int64s: 1 int64s: 0')
        self.assertSameNotEqual('int64s: 0', 'int64s: 0 int64s: 0')
        self.assertSameNotEqual('int64s: 0 int64s: 1', 'int64s: 1 int64s: 0 int64s: 1')
        self.assertNone('int64s: 0', 'int64s: 0 int64s: 2', '\n                      int64s: 0\n                    + int64s: 2\n                    ')
        self.assertNone('int64s: 0 int64s: 1', 'int64s: 0 int64s: 2', '\n                      int64s: 0\n                    - int64s: 1\n                    ?         ^\n                    + int64s: 2\n                    ?         ^\n                    ')

    def testMessage(self):
        if False:
            return 10
        self.assertAll('medium: {}')
        self.assertAll('medium: { smalls: {} }')
        self.assertAll('medium: { int32s: 1 smalls: {} }')
        self.assertAll('medium: { smalls: { strings: "x" } }')
        self.assertAll('medium: { smalls: { strings: "x" } } small: { strings: "y" }')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" strings: "y" } }', 'medium: { smalls: { strings: "y" strings: "x" } }')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" } smalls: { strings: "y" } }', 'medium: { smalls: { strings: "y" } smalls: { strings: "x" } }')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" strings: "y" strings: "x" } }', 'medium: { smalls: { strings: "y" strings: "x" } }')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" } int32s: 0 }', 'medium: { int32s: 0 smalls: { strings: "x" } int32s: 0 }')
        self.assertNone('medium: {}', 'medium: { smalls: { strings: "x" } }', '\n                      medium {\n                    +   smalls {\n                    +     strings: "x"\n                    +   }\n                      }\n                    ')
        self.assertNone('medium: { smalls: { strings: "x" } }', 'medium: { smalls: {} }', '\n                      medium {\n                        smalls {\n                    -     strings: "x"\n                        }\n                      }\n                    ')
        self.assertNone('medium: { int32s: 0 }', 'medium: { int32s: 1 }', '\n                      medium {\n                    -   int32s: 0\n                    ?           ^\n                    +   int32s: 1\n                    ?           ^\n                      }\n                    ')

    def testMsgPassdown(self):
        if False:
            return 10
        self.assertRaisesRegex(AssertionError, 'test message passed down', self.assertProtoEqual, 'medium: {}', 'medium: { smalls: { strings: "x" } }', msg='test message passed down')

    def testRepeatedMessage(self):
        if False:
            while True:
                i = 10
        self.assertAll('medium: { smalls: {} smalls: {} }')
        self.assertAll('medium: { smalls: { strings: "x" } } medium: {}')
        self.assertAll('medium: { smalls: { strings: "x" } } medium: { int32s: 0 }')
        self.assertAll('medium: { smalls: {} smalls: { strings: "x" } } small: {}')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" } smalls: {} }', 'medium: { smalls: {} smalls: { strings: "x" } }')
        self.assertSameNotEqual('medium: { smalls: {} }', 'medium: { smalls: {} smalls: {} }')
        self.assertSameNotEqual('medium: { smalls: {} smalls: {} } medium: {}', 'medium: {} medium: {} medium: { smalls: {} }')
        self.assertSameNotEqual('medium: { smalls: { strings: "x" } smalls: {} }', 'medium: { smalls: {} smalls: { strings: "x" } smalls: {} }')
        self.assertNone('medium: {}', 'medium: {} medium { smalls: {} }', '\n                      medium {\n                    +   smalls {\n                    +   }\n                      }\n                    ')
        self.assertNone('medium: { smalls: {} smalls: { strings: "x" } }', 'medium: { smalls: {} smalls: { strings: "y" } }', '\n                      medium {\n                        smalls {\n                        }\n                        smalls {\n                    -     strings: "x"\n                    ?               ^\n                    +     strings: "y"\n                    ?               ^\n                        }\n                      }\n                    ')

class MixinTests(compare.ProtoAssertions, googletest.TestCase):

    def testAssertEqualWithStringArg(self):
        if False:
            print('Hello World!')
        pb = compare_test_pb2.Large(string_='abc', float_=1.234)
        self.assertProtoEqual("\n          string_: 'abc'\n          float_: 1.234\n        ", pb)
if __name__ == '__main__':
    googletest.main()