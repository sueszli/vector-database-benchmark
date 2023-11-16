"""Tests python protocol buffers against the golden message.

Note that the golden messages exercise every known field type, thus this
test ends up exercising and verifying nearly all of the parsing and
serialization code in the whole library.

TODO(kenton):  Merge with wire_format_test?  It doesn't make a whole lot of
sense to call this a test of the "message" module, which only declares an
abstract interface.
"""
__author__ = 'gps@google.com (Gregory P. Smith)'
import copy
import math
import operator
import pickle
import unittest
from google.protobuf import unittest_import_pb2
from google.protobuf import unittest_pb2
from google.protobuf.internal import api_implementation
from google.protobuf.internal import test_util
from google.protobuf import message

def isnan(val):
    if False:
        while True:
            i = 10
    return val != val

def isinf(val):
    if False:
        return 10
    return not isnan(val) and isnan(val * 0)

def IsPosInf(val):
    if False:
        i = 10
        return i + 15
    return isinf(val) and val > 0

def IsNegInf(val):
    if False:
        return 10
    return isinf(val) and val < 0

class MessageTest(unittest.TestCase):

    def testGoldenMessage(self):
        if False:
            i = 10
            return i + 15
        golden_data = test_util.GoldenFile('golden_message').read()
        golden_message = unittest_pb2.TestAllTypes()
        golden_message.ParseFromString(golden_data)
        test_util.ExpectAllFieldsSet(self, golden_message)
        self.assertEqual(golden_data, golden_message.SerializeToString())
        golden_copy = copy.deepcopy(golden_message)
        self.assertEqual(golden_data, golden_copy.SerializeToString())

    def testGoldenExtensions(self):
        if False:
            while True:
                i = 10
        golden_data = test_util.GoldenFile('golden_message').read()
        golden_message = unittest_pb2.TestAllExtensions()
        golden_message.ParseFromString(golden_data)
        all_set = unittest_pb2.TestAllExtensions()
        test_util.SetAllExtensions(all_set)
        self.assertEquals(all_set, golden_message)
        self.assertEqual(golden_data, golden_message.SerializeToString())
        golden_copy = copy.deepcopy(golden_message)
        self.assertEqual(golden_data, golden_copy.SerializeToString())

    def testGoldenPackedMessage(self):
        if False:
            while True:
                i = 10
        golden_data = test_util.GoldenFile('golden_packed_fields_message').read()
        golden_message = unittest_pb2.TestPackedTypes()
        golden_message.ParseFromString(golden_data)
        all_set = unittest_pb2.TestPackedTypes()
        test_util.SetAllPackedFields(all_set)
        self.assertEquals(all_set, golden_message)
        self.assertEqual(golden_data, all_set.SerializeToString())
        golden_copy = copy.deepcopy(golden_message)
        self.assertEqual(golden_data, golden_copy.SerializeToString())

    def testGoldenPackedExtensions(self):
        if False:
            for i in range(10):
                print('nop')
        golden_data = test_util.GoldenFile('golden_packed_fields_message').read()
        golden_message = unittest_pb2.TestPackedExtensions()
        golden_message.ParseFromString(golden_data)
        all_set = unittest_pb2.TestPackedExtensions()
        test_util.SetAllPackedExtensions(all_set)
        self.assertEquals(all_set, golden_message)
        self.assertEqual(golden_data, all_set.SerializeToString())
        golden_copy = copy.deepcopy(golden_message)
        self.assertEqual(golden_data, golden_copy.SerializeToString())

    def testPickleSupport(self):
        if False:
            while True:
                i = 10
        golden_data = test_util.GoldenFile('golden_message').read()
        golden_message = unittest_pb2.TestAllTypes()
        golden_message.ParseFromString(golden_data)
        pickled_message = pickle.dumps(golden_message)
        unpickled_message = pickle.loads(pickled_message)
        self.assertEquals(unpickled_message, golden_message)

    def testPickleIncompleteProto(self):
        if False:
            while True:
                i = 10
        golden_message = unittest_pb2.TestRequired(a=1)
        pickled_message = pickle.dumps(golden_message)
        unpickled_message = pickle.loads(pickled_message)
        self.assertEquals(unpickled_message, golden_message)
        self.assertEquals(unpickled_message.a, 1)
        self.assertRaises(message.EncodeError, unpickled_message.SerializeToString)

    def testPositiveInfinity(self):
        if False:
            return 10
        golden_data = ']\x00\x00\x80\x7fa\x00\x00\x00\x00\x00\x00ð\x7fÍ\x02\x00\x00\x80\x7fÑ\x02\x00\x00\x00\x00\x00\x00ð\x7f'
        golden_message = unittest_pb2.TestAllTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(IsPosInf(golden_message.optional_float))
        self.assertTrue(IsPosInf(golden_message.optional_double))
        self.assertTrue(IsPosInf(golden_message.repeated_float[0]))
        self.assertTrue(IsPosInf(golden_message.repeated_double[0]))
        self.assertEqual(golden_data, golden_message.SerializeToString())

    def testNegativeInfinity(self):
        if False:
            return 10
        golden_data = ']\x00\x00\x80ÿa\x00\x00\x00\x00\x00\x00ðÿÍ\x02\x00\x00\x80ÿÑ\x02\x00\x00\x00\x00\x00\x00ðÿ'
        golden_message = unittest_pb2.TestAllTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(IsNegInf(golden_message.optional_float))
        self.assertTrue(IsNegInf(golden_message.optional_double))
        self.assertTrue(IsNegInf(golden_message.repeated_float[0]))
        self.assertTrue(IsNegInf(golden_message.repeated_double[0]))
        self.assertEqual(golden_data, golden_message.SerializeToString())

    def testNotANumber(self):
        if False:
            i = 10
            return i + 15
        golden_data = ']\x00\x00À\x7fa\x00\x00\x00\x00\x00\x00ø\x7fÍ\x02\x00\x00À\x7fÑ\x02\x00\x00\x00\x00\x00\x00ø\x7f'
        golden_message = unittest_pb2.TestAllTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(isnan(golden_message.optional_float))
        self.assertTrue(isnan(golden_message.optional_double))
        self.assertTrue(isnan(golden_message.repeated_float[0]))
        self.assertTrue(isnan(golden_message.repeated_double[0]))
        serialized = golden_message.SerializeToString()
        message = unittest_pb2.TestAllTypes()
        message.ParseFromString(serialized)
        self.assertTrue(isnan(message.optional_float))
        self.assertTrue(isnan(message.optional_double))
        self.assertTrue(isnan(message.repeated_float[0]))
        self.assertTrue(isnan(message.repeated_double[0]))

    def testPositiveInfinityPacked(self):
        if False:
            print('Hello World!')
        golden_data = '¢\x06\x04\x00\x00\x80\x7fª\x06\x08\x00\x00\x00\x00\x00\x00ð\x7f'
        golden_message = unittest_pb2.TestPackedTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(IsPosInf(golden_message.packed_float[0]))
        self.assertTrue(IsPosInf(golden_message.packed_double[0]))
        self.assertEqual(golden_data, golden_message.SerializeToString())

    def testNegativeInfinityPacked(self):
        if False:
            for i in range(10):
                print('nop')
        golden_data = '¢\x06\x04\x00\x00\x80ÿª\x06\x08\x00\x00\x00\x00\x00\x00ðÿ'
        golden_message = unittest_pb2.TestPackedTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(IsNegInf(golden_message.packed_float[0]))
        self.assertTrue(IsNegInf(golden_message.packed_double[0]))
        self.assertEqual(golden_data, golden_message.SerializeToString())

    def testNotANumberPacked(self):
        if False:
            return 10
        golden_data = '¢\x06\x04\x00\x00À\x7fª\x06\x08\x00\x00\x00\x00\x00\x00ø\x7f'
        golden_message = unittest_pb2.TestPackedTypes()
        golden_message.ParseFromString(golden_data)
        self.assertTrue(isnan(golden_message.packed_float[0]))
        self.assertTrue(isnan(golden_message.packed_double[0]))
        serialized = golden_message.SerializeToString()
        message = unittest_pb2.TestPackedTypes()
        message.ParseFromString(serialized)
        self.assertTrue(isnan(message.packed_float[0]))
        self.assertTrue(isnan(message.packed_double[0]))

    def testExtremeFloatValues(self):
        if False:
            while True:
                i = 10
        message = unittest_pb2.TestAllTypes()
        kMostPosExponentNoSigBits = math.pow(2, 127)
        message.optional_float = kMostPosExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == kMostPosExponentNoSigBits)
        kMostPosExponentOneSigBit = 1.5 * math.pow(2, 127)
        message.optional_float = kMostPosExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == kMostPosExponentOneSigBit)
        message.optional_float = -kMostPosExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == -kMostPosExponentNoSigBits)
        message.optional_float = -kMostPosExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == -kMostPosExponentOneSigBit)
        kMostNegExponentNoSigBits = math.pow(2, -127)
        message.optional_float = kMostNegExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == kMostNegExponentNoSigBits)
        kMostNegExponentOneSigBit = 1.5 * math.pow(2, -127)
        message.optional_float = kMostNegExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == kMostNegExponentOneSigBit)
        message.optional_float = -kMostNegExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == -kMostNegExponentNoSigBits)
        message.optional_float = -kMostNegExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_float == -kMostNegExponentOneSigBit)

    def testExtremeDoubleValues(self):
        if False:
            print('Hello World!')
        message = unittest_pb2.TestAllTypes()
        kMostPosExponentNoSigBits = math.pow(2, 1023)
        message.optional_double = kMostPosExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == kMostPosExponentNoSigBits)
        kMostPosExponentOneSigBit = 1.5 * math.pow(2, 1023)
        message.optional_double = kMostPosExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == kMostPosExponentOneSigBit)
        message.optional_double = -kMostPosExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == -kMostPosExponentNoSigBits)
        message.optional_double = -kMostPosExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == -kMostPosExponentOneSigBit)
        kMostNegExponentNoSigBits = math.pow(2, -1023)
        message.optional_double = kMostNegExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == kMostNegExponentNoSigBits)
        kMostNegExponentOneSigBit = 1.5 * math.pow(2, -1023)
        message.optional_double = kMostNegExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == kMostNegExponentOneSigBit)
        message.optional_double = -kMostNegExponentNoSigBits
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == -kMostNegExponentNoSigBits)
        message.optional_double = -kMostNegExponentOneSigBit
        message.ParseFromString(message.SerializeToString())
        self.assertTrue(message.optional_double == -kMostNegExponentOneSigBit)

    def testSortingRepeatedScalarFieldsDefaultComparator(self):
        if False:
            for i in range(10):
                print('nop')
        'Check some different types with the default comparator.'
        message = unittest_pb2.TestAllTypes()
        message.repeated_int32.append(1)
        message.repeated_int32.append(3)
        message.repeated_int32.append(2)
        message.repeated_int32.sort()
        self.assertEqual(message.repeated_int32[0], 1)
        self.assertEqual(message.repeated_int32[1], 2)
        self.assertEqual(message.repeated_int32[2], 3)
        message.repeated_float.append(1.1)
        message.repeated_float.append(1.3)
        message.repeated_float.append(1.2)
        message.repeated_float.sort()
        self.assertAlmostEqual(message.repeated_float[0], 1.1)
        self.assertAlmostEqual(message.repeated_float[1], 1.2)
        self.assertAlmostEqual(message.repeated_float[2], 1.3)
        message.repeated_string.append('a')
        message.repeated_string.append('c')
        message.repeated_string.append('b')
        message.repeated_string.sort()
        self.assertEqual(message.repeated_string[0], 'a')
        self.assertEqual(message.repeated_string[1], 'b')
        self.assertEqual(message.repeated_string[2], 'c')
        message.repeated_bytes.append('a')
        message.repeated_bytes.append('c')
        message.repeated_bytes.append('b')
        message.repeated_bytes.sort()
        self.assertEqual(message.repeated_bytes[0], 'a')
        self.assertEqual(message.repeated_bytes[1], 'b')
        self.assertEqual(message.repeated_bytes[2], 'c')

    def testSortingRepeatedScalarFieldsCustomComparator(self):
        if False:
            i = 10
            return i + 15
        'Check some different types with custom comparator.'
        message = unittest_pb2.TestAllTypes()
        message.repeated_int32.append(-3)
        message.repeated_int32.append(-2)
        message.repeated_int32.append(-1)
        message.repeated_int32.sort(lambda x, y: cmp(abs(x), abs(y)))
        self.assertEqual(message.repeated_int32[0], -1)
        self.assertEqual(message.repeated_int32[1], -2)
        self.assertEqual(message.repeated_int32[2], -3)
        message.repeated_string.append('aaa')
        message.repeated_string.append('bb')
        message.repeated_string.append('c')
        message.repeated_string.sort(lambda x, y: cmp(len(x), len(y)))
        self.assertEqual(message.repeated_string[0], 'c')
        self.assertEqual(message.repeated_string[1], 'bb')
        self.assertEqual(message.repeated_string[2], 'aaa')

    def testSortingRepeatedCompositeFieldsCustomComparator(self):
        if False:
            print('Hello World!')
        'Check passing a custom comparator to sort a repeated composite field.'
        message = unittest_pb2.TestAllTypes()
        message.repeated_nested_message.add().bb = 1
        message.repeated_nested_message.add().bb = 3
        message.repeated_nested_message.add().bb = 2
        message.repeated_nested_message.add().bb = 6
        message.repeated_nested_message.add().bb = 5
        message.repeated_nested_message.add().bb = 4
        message.repeated_nested_message.sort(lambda x, y: cmp(x.bb, y.bb))
        self.assertEqual(message.repeated_nested_message[0].bb, 1)
        self.assertEqual(message.repeated_nested_message[1].bb, 2)
        self.assertEqual(message.repeated_nested_message[2].bb, 3)
        self.assertEqual(message.repeated_nested_message[3].bb, 4)
        self.assertEqual(message.repeated_nested_message[4].bb, 5)
        self.assertEqual(message.repeated_nested_message[5].bb, 6)

    def testRepeatedCompositeFieldSortArguments(self):
        if False:
            i = 10
            return i + 15
        'Check sorting a repeated composite field using list.sort() arguments.'
        message = unittest_pb2.TestAllTypes()
        get_bb = operator.attrgetter('bb')
        cmp_bb = lambda a, b: cmp(a.bb, b.bb)
        message.repeated_nested_message.add().bb = 1
        message.repeated_nested_message.add().bb = 3
        message.repeated_nested_message.add().bb = 2
        message.repeated_nested_message.add().bb = 6
        message.repeated_nested_message.add().bb = 5
        message.repeated_nested_message.add().bb = 4
        message.repeated_nested_message.sort(key=get_bb)
        self.assertEqual([k.bb for k in message.repeated_nested_message], [1, 2, 3, 4, 5, 6])
        message.repeated_nested_message.sort(key=get_bb, reverse=True)
        self.assertEqual([k.bb for k in message.repeated_nested_message], [6, 5, 4, 3, 2, 1])
        message.repeated_nested_message.sort(sort_function=cmp_bb)
        self.assertEqual([k.bb for k in message.repeated_nested_message], [1, 2, 3, 4, 5, 6])
        message.repeated_nested_message.sort(cmp=cmp_bb, reverse=True)
        self.assertEqual([k.bb for k in message.repeated_nested_message], [6, 5, 4, 3, 2, 1])

    def testRepeatedScalarFieldSortArguments(self):
        if False:
            for i in range(10):
                print('nop')
        'Check sorting a scalar field using list.sort() arguments.'
        message = unittest_pb2.TestAllTypes()
        abs_cmp = lambda a, b: cmp(abs(a), abs(b))
        message.repeated_int32.append(-3)
        message.repeated_int32.append(-2)
        message.repeated_int32.append(-1)
        message.repeated_int32.sort(key=abs)
        self.assertEqual(list(message.repeated_int32), [-1, -2, -3])
        message.repeated_int32.sort(key=abs, reverse=True)
        self.assertEqual(list(message.repeated_int32), [-3, -2, -1])
        message.repeated_int32.sort(sort_function=abs_cmp)
        self.assertEqual(list(message.repeated_int32), [-1, -2, -3])
        message.repeated_int32.sort(cmp=abs_cmp, reverse=True)
        self.assertEqual(list(message.repeated_int32), [-3, -2, -1])
        len_cmp = lambda a, b: cmp(len(a), len(b))
        message.repeated_string.append('aaa')
        message.repeated_string.append('bb')
        message.repeated_string.append('c')
        message.repeated_string.sort(key=len)
        self.assertEqual(list(message.repeated_string), ['c', 'bb', 'aaa'])
        message.repeated_string.sort(key=len, reverse=True)
        self.assertEqual(list(message.repeated_string), ['aaa', 'bb', 'c'])
        message.repeated_string.sort(sort_function=len_cmp)
        self.assertEqual(list(message.repeated_string), ['c', 'bb', 'aaa'])
        message.repeated_string.sort(cmp=len_cmp, reverse=True)
        self.assertEqual(list(message.repeated_string), ['aaa', 'bb', 'c'])

    def testParsingMerge(self):
        if False:
            return 10
        'Check the merge behavior when a required or optional field appears\n    multiple times in the input.'
        messages = [unittest_pb2.TestAllTypes(), unittest_pb2.TestAllTypes(), unittest_pb2.TestAllTypes()]
        messages[0].optional_int32 = 1
        messages[1].optional_int64 = 2
        messages[2].optional_int32 = 3
        messages[2].optional_string = 'hello'
        merged_message = unittest_pb2.TestAllTypes()
        merged_message.optional_int32 = 3
        merged_message.optional_int64 = 2
        merged_message.optional_string = 'hello'
        generator = unittest_pb2.TestParsingMerge.RepeatedFieldsGenerator()
        generator.field1.extend(messages)
        generator.field2.extend(messages)
        generator.field3.extend(messages)
        generator.ext1.extend(messages)
        generator.ext2.extend(messages)
        generator.group1.add().field1.MergeFrom(messages[0])
        generator.group1.add().field1.MergeFrom(messages[1])
        generator.group1.add().field1.MergeFrom(messages[2])
        generator.group2.add().field1.MergeFrom(messages[0])
        generator.group2.add().field1.MergeFrom(messages[1])
        generator.group2.add().field1.MergeFrom(messages[2])
        data = generator.SerializeToString()
        parsing_merge = unittest_pb2.TestParsingMerge()
        parsing_merge.ParseFromString(data)
        self.assertEqual(parsing_merge.required_all_types, merged_message)
        self.assertEqual(parsing_merge.optional_all_types, merged_message)
        self.assertEqual(parsing_merge.optionalgroup.optional_group_all_types, merged_message)
        self.assertEqual(parsing_merge.Extensions[unittest_pb2.TestParsingMerge.optional_ext], merged_message)
        self.assertEqual(len(parsing_merge.repeated_all_types), 3)
        self.assertEqual(len(parsing_merge.repeatedgroup), 3)
        self.assertEqual(len(parsing_merge.Extensions[unittest_pb2.TestParsingMerge.repeated_ext]), 3)

    def testSortEmptyRepeatedCompositeContainer(self):
        if False:
            for i in range(10):
                print('nop')
        'Exercise a scenario that has led to segfaults in the past.\n    '
        m = unittest_pb2.TestAllTypes()
        m.repeated_nested_message.sort()
if __name__ == '__main__':
    unittest.main()