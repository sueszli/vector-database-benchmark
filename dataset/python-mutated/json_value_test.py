"""Unit tests for the json_value module."""
import unittest
from apache_beam.internal.gcp.json_value import from_json_value
from apache_beam.internal.gcp.json_value import to_json_value
from apache_beam.options.value_provider import RuntimeValueProvider
from apache_beam.options.value_provider import StaticValueProvider
try:
    from apitools.base.py.extra_types import JsonValue
except ImportError:
    JsonValue = None

@unittest.skipIf(JsonValue is None, 'GCP dependencies are not installed')
class JsonValueTest(unittest.TestCase):

    def test_string_to(self):
        if False:
            print('Hello World!')
        self.assertEqual(JsonValue(string_value='abc'), to_json_value('abc'))

    def test_bytes_to(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(JsonValue(string_value='abc'), to_json_value(b'abc'))

    def test_true_to(self):
        if False:
            while True:
                i = 10
        self.assertEqual(JsonValue(boolean_value=True), to_json_value(True))

    def test_false_to(self):
        if False:
            while True:
                i = 10
        self.assertEqual(JsonValue(boolean_value=False), to_json_value(False))

    def test_int_to(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(JsonValue(integer_value=14), to_json_value(14))

    def test_float_to(self):
        if False:
            print('Hello World!')
        self.assertEqual(JsonValue(double_value=2.75), to_json_value(2.75))

    def test_static_value_provider_to(self):
        if False:
            while True:
                i = 10
        svp = StaticValueProvider(str, 'abc')
        self.assertEqual(JsonValue(string_value=svp.value), to_json_value(svp))

    def test_runtime_value_provider_to(self):
        if False:
            print('Hello World!')
        RuntimeValueProvider.set_runtime_options(None)
        rvp = RuntimeValueProvider('arg', 123, int)
        self.assertEqual(JsonValue(is_null=True), to_json_value(rvp))
        RuntimeValueProvider.set_runtime_options(None)

    def test_none_to(self):
        if False:
            print('Hello World!')
        self.assertEqual(JsonValue(is_null=True), to_json_value(None))

    def test_string_from(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('WXYZ', from_json_value(to_json_value('WXYZ')))

    def test_true_from(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(True, from_json_value(to_json_value(True)))

    def test_false_from(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(False, from_json_value(to_json_value(False)))

    def test_int_from(self):
        if False:
            while True:
                i = 10
        self.assertEqual(-27, from_json_value(to_json_value(-27)))

    def test_float_from(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(4.5, from_json_value(to_json_value(4.5)))

    def test_with_type(self):
        if False:
            while True:
                i = 10
        rt = from_json_value(to_json_value('abcd', with_type=True))
        self.assertEqual('http://schema.org/Text', rt['@type'])
        self.assertEqual('abcd', rt['value'])

    def test_none_from(self):
        if False:
            return 10
        self.assertIsNone(from_json_value(to_json_value(None)))

    def test_large_integer(self):
        if False:
            print('Hello World!')
        num = 1 << 35
        self.assertEqual(num, from_json_value(to_json_value(num)))

    def test_long_value(self):
        if False:
            return 10
        num = 1 << 63 - 1
        self.assertEqual(num, from_json_value(to_json_value(num)))

    def test_too_long_value(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            to_json_value(1 << 64)
if __name__ == '__main__':
    unittest.main()