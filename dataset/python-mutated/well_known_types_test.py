"""Test for google.protobuf.internal.well_known_types."""
__author__ = 'jieluo@google.com (Jie Luo)'
from datetime import datetime
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import unittest_pb2
from google.protobuf.internal import any_test_pb2
from google.protobuf.internal import test_util
from google.protobuf.internal import well_known_types
from google.protobuf import descriptor
from google.protobuf import text_format

class TimeUtilTestBase(unittest.TestCase):

    def CheckTimestampConversion(self, message, text):
        if False:
            print('Hello World!')
        self.assertEqual(text, message.ToJsonString())
        parsed_message = timestamp_pb2.Timestamp()
        parsed_message.FromJsonString(text)
        self.assertEqual(message, parsed_message)

    def CheckDurationConversion(self, message, text):
        if False:
            i = 10
            return i + 15
        self.assertEqual(text, message.ToJsonString())
        parsed_message = duration_pb2.Duration()
        parsed_message.FromJsonString(text)
        self.assertEqual(message, parsed_message)

class TimeUtilTest(TimeUtilTestBase):

    def testTimestampSerializeAndParse(self):
        if False:
            return 10
        message = timestamp_pb2.Timestamp()
        message.seconds = 0
        message.nanos = 0
        self.CheckTimestampConversion(message, '1970-01-01T00:00:00Z')
        message.nanos = 10000000
        self.CheckTimestampConversion(message, '1970-01-01T00:00:00.010Z')
        message.nanos = 10000
        self.CheckTimestampConversion(message, '1970-01-01T00:00:00.000010Z')
        message.nanos = 10
        self.CheckTimestampConversion(message, '1970-01-01T00:00:00.000000010Z')
        message.seconds = -62135596800
        message.nanos = 0
        self.CheckTimestampConversion(message, '0001-01-01T00:00:00Z')
        message.seconds = 253402300799
        message.nanos = 999999999
        self.CheckTimestampConversion(message, '9999-12-31T23:59:59.999999999Z')
        message.seconds = -1
        self.CheckTimestampConversion(message, '1969-12-31T23:59:59.999999999Z')
        message.FromJsonString('1970-01-01T00:00:00.1Z')
        self.assertEqual(0, message.seconds)
        self.assertEqual(100000000, message.nanos)
        message.FromJsonString('1970-01-01T00:00:00-08:00')
        self.assertEqual(8 * 3600, message.seconds)
        self.assertEqual(0, message.nanos)

    def testDurationSerializeAndParse(self):
        if False:
            print('Hello World!')
        message = duration_pb2.Duration()
        message.seconds = 0
        message.nanos = 0
        self.CheckDurationConversion(message, '0s')
        message.nanos = 10000000
        self.CheckDurationConversion(message, '0.010s')
        message.nanos = 10000
        self.CheckDurationConversion(message, '0.000010s')
        message.nanos = 10
        self.CheckDurationConversion(message, '0.000000010s')
        message.seconds = 315576000000
        message.nanos = 999999999
        self.CheckDurationConversion(message, '315576000000.999999999s')
        message.seconds = -315576000000
        message.nanos = -999999999
        self.CheckDurationConversion(message, '-315576000000.999999999s')
        message.FromJsonString('0.1s')
        self.assertEqual(100000000, message.nanos)
        message.FromJsonString('0.0000001s')
        self.assertEqual(100, message.nanos)

    def testTimestampIntegerConversion(self):
        if False:
            return 10
        message = timestamp_pb2.Timestamp()
        message.FromNanoseconds(1)
        self.assertEqual('1970-01-01T00:00:00.000000001Z', message.ToJsonString())
        self.assertEqual(1, message.ToNanoseconds())
        message.FromNanoseconds(-1)
        self.assertEqual('1969-12-31T23:59:59.999999999Z', message.ToJsonString())
        self.assertEqual(-1, message.ToNanoseconds())
        message.FromMicroseconds(1)
        self.assertEqual('1970-01-01T00:00:00.000001Z', message.ToJsonString())
        self.assertEqual(1, message.ToMicroseconds())
        message.FromMicroseconds(-1)
        self.assertEqual('1969-12-31T23:59:59.999999Z', message.ToJsonString())
        self.assertEqual(-1, message.ToMicroseconds())
        message.FromMilliseconds(1)
        self.assertEqual('1970-01-01T00:00:00.001Z', message.ToJsonString())
        self.assertEqual(1, message.ToMilliseconds())
        message.FromMilliseconds(-1)
        self.assertEqual('1969-12-31T23:59:59.999Z', message.ToJsonString())
        self.assertEqual(-1, message.ToMilliseconds())
        message.FromSeconds(1)
        self.assertEqual('1970-01-01T00:00:01Z', message.ToJsonString())
        self.assertEqual(1, message.ToSeconds())
        message.FromSeconds(-1)
        self.assertEqual('1969-12-31T23:59:59Z', message.ToJsonString())
        self.assertEqual(-1, message.ToSeconds())
        message.FromNanoseconds(1999)
        self.assertEqual(1, message.ToMicroseconds())
        message.FromNanoseconds(-1999)
        self.assertEqual(-2, message.ToMicroseconds())

    def testDurationIntegerConversion(self):
        if False:
            return 10
        message = duration_pb2.Duration()
        message.FromNanoseconds(1)
        self.assertEqual('0.000000001s', message.ToJsonString())
        self.assertEqual(1, message.ToNanoseconds())
        message.FromNanoseconds(-1)
        self.assertEqual('-0.000000001s', message.ToJsonString())
        self.assertEqual(-1, message.ToNanoseconds())
        message.FromMicroseconds(1)
        self.assertEqual('0.000001s', message.ToJsonString())
        self.assertEqual(1, message.ToMicroseconds())
        message.FromMicroseconds(-1)
        self.assertEqual('-0.000001s', message.ToJsonString())
        self.assertEqual(-1, message.ToMicroseconds())
        message.FromMilliseconds(1)
        self.assertEqual('0.001s', message.ToJsonString())
        self.assertEqual(1, message.ToMilliseconds())
        message.FromMilliseconds(-1)
        self.assertEqual('-0.001s', message.ToJsonString())
        self.assertEqual(-1, message.ToMilliseconds())
        message.FromSeconds(1)
        self.assertEqual('1s', message.ToJsonString())
        self.assertEqual(1, message.ToSeconds())
        message.FromSeconds(-1)
        self.assertEqual('-1s', message.ToJsonString())
        self.assertEqual(-1, message.ToSeconds())
        message.FromNanoseconds(1999)
        self.assertEqual(1, message.ToMicroseconds())
        message.FromNanoseconds(-1999)
        self.assertEqual(-1, message.ToMicroseconds())

    def testDatetimeConverison(self):
        if False:
            for i in range(10):
                print('nop')
        message = timestamp_pb2.Timestamp()
        dt = datetime(1970, 1, 1)
        message.FromDatetime(dt)
        self.assertEqual(dt, message.ToDatetime())
        message.FromMilliseconds(1999)
        self.assertEqual(datetime(1970, 1, 1, 0, 0, 1, 999000), message.ToDatetime())

    def testTimedeltaConversion(self):
        if False:
            print('Hello World!')
        message = duration_pb2.Duration()
        message.FromNanoseconds(1999999999)
        td = message.ToTimedelta()
        self.assertEqual(1, td.seconds)
        self.assertEqual(999999, td.microseconds)
        message.FromNanoseconds(-1999999999)
        td = message.ToTimedelta()
        self.assertEqual(-1, td.days)
        self.assertEqual(86398, td.seconds)
        self.assertEqual(1, td.microseconds)
        message.FromMicroseconds(-1)
        td = message.ToTimedelta()
        self.assertEqual(-1, td.days)
        self.assertEqual(86399, td.seconds)
        self.assertEqual(999999, td.microseconds)
        converted_message = duration_pb2.Duration()
        converted_message.FromTimedelta(td)
        self.assertEqual(message, converted_message)

    def testInvalidTimestamp(self):
        if False:
            while True:
                i = 10
        message = timestamp_pb2.Timestamp()
        self.assertRaisesRegexp(ValueError, "time data '10000-01-01T00:00:00' does not match format '%Y-%m-%dT%H:%M:%S'", message.FromJsonString, '10000-01-01T00:00:00.00Z')
        self.assertRaisesRegexp(well_known_types.ParseError, 'nanos 0123456789012 more than 9 fractional digits.', message.FromJsonString, '1970-01-01T00:00:00.0123456789012Z')
        self.assertRaisesRegexp(well_known_types.ParseError, 'Invalid timezone offset value: \\+08.', message.FromJsonString, '1972-01-01T01:00:00.01+08')
        self.assertRaisesRegexp(ValueError, 'year is out of range', message.FromJsonString, '0000-01-01T00:00:00Z')
        message.seconds = 253402300800
        self.assertRaisesRegexp(OverflowError, 'date value out of range', message.ToJsonString)

    def testInvalidDuration(self):
        if False:
            i = 10
            return i + 15
        message = duration_pb2.Duration()
        self.assertRaisesRegexp(well_known_types.ParseError, 'Duration must end with letter "s": 1.', message.FromJsonString, '1')
        self.assertRaisesRegexp(well_known_types.ParseError, "Couldn't parse duration: 1...2s.", message.FromJsonString, '1...2s')
        text = '-315576000001.000000000s'
        self.assertRaisesRegexp(well_known_types.Error, 'Duration is not valid\\: Seconds -315576000001 must be in range \\[-315576000000\\, 315576000000\\].', message.FromJsonString, text)
        text = '315576000001.000000000s'
        self.assertRaisesRegexp(well_known_types.Error, 'Duration is not valid\\: Seconds 315576000001 must be in range \\[-315576000000\\, 315576000000\\].', message.FromJsonString, text)
        message.seconds = -315576000001
        message.nanos = 0
        self.assertRaisesRegexp(well_known_types.Error, 'Duration is not valid\\: Seconds -315576000001 must be in range \\[-315576000000\\, 315576000000\\].', message.ToJsonString)

class FieldMaskTest(unittest.TestCase):

    def testStringFormat(self):
        if False:
            while True:
                i = 10
        mask = field_mask_pb2.FieldMask()
        self.assertEqual('', mask.ToJsonString())
        mask.paths.append('foo')
        self.assertEqual('foo', mask.ToJsonString())
        mask.paths.append('bar')
        self.assertEqual('foo,bar', mask.ToJsonString())
        mask.FromJsonString('')
        self.assertEqual('', mask.ToJsonString())
        mask.FromJsonString('foo')
        self.assertEqual(['foo'], mask.paths)
        mask.FromJsonString('foo,bar')
        self.assertEqual(['foo', 'bar'], mask.paths)
        mask.Clear()
        mask.paths.append('foo_bar')
        self.assertEqual('fooBar', mask.ToJsonString())
        mask.paths.append('bar_quz')
        self.assertEqual('fooBar,barQuz', mask.ToJsonString())
        mask.FromJsonString('')
        self.assertEqual('', mask.ToJsonString())
        mask.FromJsonString('fooBar')
        self.assertEqual(['foo_bar'], mask.paths)
        mask.FromJsonString('fooBar,barQuz')
        self.assertEqual(['foo_bar', 'bar_quz'], mask.paths)

    def testDescriptorToFieldMask(self):
        if False:
            print('Hello World!')
        mask = field_mask_pb2.FieldMask()
        msg_descriptor = unittest_pb2.TestAllTypes.DESCRIPTOR
        mask.AllFieldsFromDescriptor(msg_descriptor)
        self.assertEqual(75, len(mask.paths))
        self.assertTrue(mask.IsValidForDescriptor(msg_descriptor))
        for field in msg_descriptor.fields:
            self.assertTrue(field.name in mask.paths)
        mask.paths.append('optional_nested_message.bb')
        self.assertTrue(mask.IsValidForDescriptor(msg_descriptor))
        mask.paths.append('repeated_nested_message.bb')
        self.assertFalse(mask.IsValidForDescriptor(msg_descriptor))

    def testCanonicalFrom(self):
        if False:
            for i in range(10):
                print('nop')
        mask = field_mask_pb2.FieldMask()
        out_mask = field_mask_pb2.FieldMask()
        mask.FromJsonString('baz.quz,bar,foo')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('bar,baz.quz,foo', out_mask.ToJsonString())
        mask.FromJsonString('foo,bar,foo')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('bar,foo', out_mask.ToJsonString())
        mask.FromJsonString('foo.b1,bar.b1,foo.b2,bar')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('bar,foo.b1,foo.b2', out_mask.ToJsonString())
        mask.FromJsonString('foo.bar.baz1,foo.bar.baz2.quz,foo.bar.baz2')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('foo.bar.baz1,foo.bar.baz2', out_mask.ToJsonString())
        mask.FromJsonString('foo.bar.baz1,foo.bar.baz2,foo.bar.baz2.quz')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('foo.bar.baz1,foo.bar.baz2', out_mask.ToJsonString())
        mask.FromJsonString('foo.bar.baz1,foo.bar.baz2,foo.bar.baz2.quz,foo.bar')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('foo.bar', out_mask.ToJsonString())
        mask.FromJsonString('foo.bar.baz1,foo.bar.baz2,foo.bar.baz2.quz,foo')
        out_mask.CanonicalFormFromMask(mask)
        self.assertEqual('foo', out_mask.ToJsonString())

    def testUnion(self):
        if False:
            for i in range(10):
                print('nop')
        mask1 = field_mask_pb2.FieldMask()
        mask2 = field_mask_pb2.FieldMask()
        out_mask = field_mask_pb2.FieldMask()
        mask1.FromJsonString('foo,baz')
        mask2.FromJsonString('bar,quz')
        out_mask.Union(mask1, mask2)
        self.assertEqual('bar,baz,foo,quz', out_mask.ToJsonString())
        mask1.FromJsonString('foo,baz.bb')
        mask2.FromJsonString('baz.bb,quz')
        out_mask.Union(mask1, mask2)
        self.assertEqual('baz.bb,foo,quz', out_mask.ToJsonString())
        mask1.FromJsonString('foo.bar.baz,quz')
        mask2.FromJsonString('foo.bar,bar')
        out_mask.Union(mask1, mask2)
        self.assertEqual('bar,foo.bar,quz', out_mask.ToJsonString())

    def testIntersect(self):
        if False:
            while True:
                i = 10
        mask1 = field_mask_pb2.FieldMask()
        mask2 = field_mask_pb2.FieldMask()
        out_mask = field_mask_pb2.FieldMask()
        mask1.FromJsonString('foo,baz')
        mask2.FromJsonString('bar,quz')
        out_mask.Intersect(mask1, mask2)
        self.assertEqual('', out_mask.ToJsonString())
        mask1.FromJsonString('foo,baz.bb')
        mask2.FromJsonString('baz.bb,quz')
        out_mask.Intersect(mask1, mask2)
        self.assertEqual('baz.bb', out_mask.ToJsonString())
        mask1.FromJsonString('foo.bar.baz,quz')
        mask2.FromJsonString('foo.bar,bar')
        out_mask.Intersect(mask1, mask2)
        self.assertEqual('foo.bar.baz', out_mask.ToJsonString())
        mask1.FromJsonString('foo.bar,bar')
        mask2.FromJsonString('foo.bar.baz,quz')
        out_mask.Intersect(mask1, mask2)
        self.assertEqual('foo.bar.baz', out_mask.ToJsonString())

    def testMergeMessage(self):
        if False:
            while True:
                i = 10
        src = unittest_pb2.TestAllTypes()
        test_util.SetAllFields(src)
        for field in src.DESCRIPTOR.fields:
            if field.containing_oneof:
                continue
            field_name = field.name
            dst = unittest_pb2.TestAllTypes()
            mask = field_mask_pb2.FieldMask()
            mask.paths.append(field_name)
            mask.MergeMessage(src, dst)
            msg = unittest_pb2.TestAllTypes()
            if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                repeated_src = getattr(src, field_name)
                repeated_msg = getattr(msg, field_name)
                if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                    for item in repeated_src:
                        repeated_msg.add().CopyFrom(item)
                else:
                    repeated_msg.extend(repeated_src)
            elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                getattr(msg, field_name).CopyFrom(getattr(src, field_name))
            else:
                setattr(msg, field_name, getattr(src, field_name))
            self.assertEqual(msg, dst)
        nested_src = unittest_pb2.NestedTestAllTypes()
        nested_dst = unittest_pb2.NestedTestAllTypes()
        nested_src.child.payload.optional_int32 = 1234
        nested_src.child.child.payload.optional_int32 = 5678
        mask = field_mask_pb2.FieldMask()
        mask.FromJsonString('child.payload')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(1234, nested_dst.child.payload.optional_int32)
        self.assertEqual(0, nested_dst.child.child.payload.optional_int32)
        mask.FromJsonString('child.child.payload')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(1234, nested_dst.child.payload.optional_int32)
        self.assertEqual(5678, nested_dst.child.child.payload.optional_int32)
        nested_dst.Clear()
        mask.FromJsonString('child.child.payload')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(0, nested_dst.child.payload.optional_int32)
        self.assertEqual(5678, nested_dst.child.child.payload.optional_int32)
        nested_dst.Clear()
        mask.FromJsonString('child')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(1234, nested_dst.child.payload.optional_int32)
        self.assertEqual(5678, nested_dst.child.child.payload.optional_int32)
        nested_dst.Clear()
        nested_dst.child.payload.optional_int64 = 4321
        mask.FromJsonString('child.payload')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(1234, nested_dst.child.payload.optional_int32)
        self.assertEqual(4321, nested_dst.child.payload.optional_int64)
        mask.FromJsonString('child.payload')
        mask.MergeMessage(nested_src, nested_dst, True, False)
        self.assertEqual(1234, nested_dst.child.payload.optional_int32)
        self.assertEqual(0, nested_dst.child.payload.optional_int64)
        nested_dst.payload.optional_int32 = 1234
        self.assertTrue(nested_dst.HasField('payload'))
        mask.FromJsonString('payload')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertTrue(nested_dst.HasField('payload'))
        nested_dst.Clear()
        nested_dst.payload.optional_int32 = 1234
        mask.FromJsonString('payload')
        mask.MergeMessage(nested_src, nested_dst, True, False)
        self.assertFalse(nested_dst.HasField('payload'))
        nested_src.payload.repeated_int32.append(1234)
        nested_dst.payload.repeated_int32.append(5678)
        mask.FromJsonString('payload.repeatedInt32')
        mask.MergeMessage(nested_src, nested_dst)
        self.assertEqual(2, len(nested_dst.payload.repeated_int32))
        self.assertEqual(5678, nested_dst.payload.repeated_int32[0])
        self.assertEqual(1234, nested_dst.payload.repeated_int32[1])
        mask.FromJsonString('payload.repeatedInt32')
        mask.MergeMessage(nested_src, nested_dst, False, True)
        self.assertEqual(1, len(nested_dst.payload.repeated_int32))
        self.assertEqual(1234, nested_dst.payload.repeated_int32[0])

    def testSnakeCaseToCamelCase(self):
        if False:
            while True:
                i = 10
        self.assertEqual('fooBar', well_known_types._SnakeCaseToCamelCase('foo_bar'))
        self.assertEqual('FooBar', well_known_types._SnakeCaseToCamelCase('_foo_bar'))
        self.assertEqual('foo3Bar', well_known_types._SnakeCaseToCamelCase('foo3_bar'))
        self.assertRaisesRegexp(well_known_types.Error, 'Fail to print FieldMask to Json string: Path name Foo must not contain uppercase letters.', well_known_types._SnakeCaseToCamelCase, 'Foo')
        self.assertRaisesRegexp(well_known_types.Error, 'Fail to print FieldMask to Json string: The character after a "_" must be a lowercase letter in path name foo__bar.', well_known_types._SnakeCaseToCamelCase, 'foo__bar')
        self.assertRaisesRegexp(well_known_types.Error, 'Fail to print FieldMask to Json string: The character after a "_" must be a lowercase letter in path name foo_3bar.', well_known_types._SnakeCaseToCamelCase, 'foo_3bar')
        self.assertRaisesRegexp(well_known_types.Error, 'Fail to print FieldMask to Json string: Trailing "_" in path name foo_bar_.', well_known_types._SnakeCaseToCamelCase, 'foo_bar_')

    def testCamelCaseToSnakeCase(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('foo_bar', well_known_types._CamelCaseToSnakeCase('fooBar'))
        self.assertEqual('_foo_bar', well_known_types._CamelCaseToSnakeCase('FooBar'))
        self.assertEqual('foo3_bar', well_known_types._CamelCaseToSnakeCase('foo3Bar'))
        self.assertRaisesRegexp(well_known_types.ParseError, 'Fail to parse FieldMask: Path name foo_bar must not contain "_"s.', well_known_types._CamelCaseToSnakeCase, 'foo_bar')

class StructTest(unittest.TestCase):

    def testStruct(self):
        if False:
            i = 10
            return i + 15
        struct = struct_pb2.Struct()
        struct_class = struct.__class__
        struct['key1'] = 5
        struct['key2'] = 'abc'
        struct['key3'] = True
        struct.get_or_create_struct('key4')['subkey'] = 11.0
        struct_list = struct.get_or_create_list('key5')
        struct_list.extend([6, 'seven', True, False, None])
        struct_list.add_struct()['subkey2'] = 9
        self.assertTrue(isinstance(struct, well_known_types.Struct))
        self.assertEqual(5, struct['key1'])
        self.assertEqual('abc', struct['key2'])
        self.assertIs(True, struct['key3'])
        self.assertEqual(11, struct['key4']['subkey'])
        inner_struct = struct_class()
        inner_struct['subkey2'] = 9
        self.assertEqual([6, 'seven', True, False, None, inner_struct], list(struct['key5'].items()))
        serialized = struct.SerializeToString()
        struct2 = struct_pb2.Struct()
        struct2.ParseFromString(serialized)
        self.assertEqual(struct, struct2)
        self.assertTrue(isinstance(struct2, well_known_types.Struct))
        self.assertEqual(5, struct2['key1'])
        self.assertEqual('abc', struct2['key2'])
        self.assertIs(True, struct2['key3'])
        self.assertEqual(11, struct2['key4']['subkey'])
        self.assertEqual([6, 'seven', True, False, None, inner_struct], list(struct2['key5'].items()))
        struct_list = struct2['key5']
        self.assertEqual(6, struct_list[0])
        self.assertEqual('seven', struct_list[1])
        self.assertEqual(True, struct_list[2])
        self.assertEqual(False, struct_list[3])
        self.assertEqual(None, struct_list[4])
        self.assertEqual(inner_struct, struct_list[5])
        struct_list[1] = 7
        self.assertEqual(7, struct_list[1])
        struct_list.add_list().extend([1, 'two', True, False, None])
        self.assertEqual([1, 'two', True, False, None], list(struct_list[6].items()))
        text_serialized = str(struct)
        struct3 = struct_pb2.Struct()
        text_format.Merge(text_serialized, struct3)
        self.assertEqual(struct, struct3)
        struct.get_or_create_struct('key3')['replace'] = 12
        self.assertEqual(12, struct['key3']['replace'])

class AnyTest(unittest.TestCase):

    def testAnyMessage(self):
        if False:
            i = 10
            return i + 15
        msg = any_test_pb2.TestAny()
        msg_descriptor = msg.DESCRIPTOR
        all_types = unittest_pb2.TestAllTypes()
        all_descriptor = all_types.DESCRIPTOR
        all_types.repeated_string.append(u'üꜟ')
        msg.value.Pack(all_types)
        self.assertEqual(msg.value.type_url, 'type.googleapis.com/%s' % all_descriptor.full_name)
        self.assertEqual(msg.value.value, all_types.SerializeToString())
        self.assertTrue(msg.value.Is(all_descriptor))
        self.assertFalse(msg.value.Is(msg_descriptor))
        unpacked_message = unittest_pb2.TestAllTypes()
        self.assertTrue(msg.value.Unpack(unpacked_message))
        self.assertEqual(all_types, unpacked_message)
        self.assertFalse(msg.value.Unpack(msg))
        try:
            msg.Pack(all_types)
        except AttributeError:
            pass
        else:
            raise AttributeError('%s should not have Pack method.' % msg_descriptor.full_name)

    def testMessageName(self):
        if False:
            print('Hello World!')
        submessage = any_test_pb2.TestAny()
        submessage.int_value = 12345
        msg = any_pb2.Any()
        msg.Pack(submessage)
        self.assertEqual(msg.TypeName(), 'google.protobuf.internal.TestAny')

    def testPackWithCustomTypeUrl(self):
        if False:
            i = 10
            return i + 15
        submessage = any_test_pb2.TestAny()
        submessage.int_value = 12345
        msg = any_pb2.Any()
        msg.Pack(submessage, 'type.myservice.com')
        self.assertEqual(msg.type_url, 'type.myservice.com/%s' % submessage.DESCRIPTOR.full_name)
        msg.Pack(submessage, 'type.myservice.com/')
        self.assertEqual(msg.type_url, 'type.myservice.com/%s' % submessage.DESCRIPTOR.full_name)
        msg.Pack(submessage, '')
        self.assertEqual(msg.type_url, '/%s' % submessage.DESCRIPTOR.full_name)
        unpacked_message = any_test_pb2.TestAny()
        self.assertTrue(msg.Unpack(unpacked_message))
        self.assertEqual(submessage, unpacked_message)
if __name__ == '__main__':
    unittest.main()