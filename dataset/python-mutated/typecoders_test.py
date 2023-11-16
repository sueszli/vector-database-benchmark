"""Unit tests for the typecoders module."""
import unittest
from apache_beam.coders import coders
from apache_beam.coders import typecoders
from apache_beam.internal import pickler
from apache_beam.typehints import typehints

class CustomClass(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.number = n

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.number == other.number

    def __hash__(self):
        if False:
            return 10
        return self.number

class CustomCoder(coders.Coder):

    def encode(self, value):
        if False:
            while True:
                i = 10
        return str(value.number).encode('ASCII')

    def decode(self, encoded):
        if False:
            while True:
                i = 10
        return CustomClass(int(encoded))

    def is_deterministic(self):
        if False:
            i = 10
            return i + 15
        return True

class TypeCodersTest(unittest.TestCase):

    def test_register_non_type_coder(self):
        if False:
            for i in range(10):
                print('nop')
        coder = CustomCoder()
        with self.assertRaisesRegex(TypeError, 'Coder registration requires a coder class object. Received %r instead.' % coder):
            typecoders.registry.register_coder(CustomClass, coder)

    def test_get_coder_with_custom_coder(self):
        if False:
            while True:
                i = 10
        typecoders.registry.register_coder(CustomClass, CustomCoder)
        self.assertEqual(CustomCoder, typecoders.registry.get_coder(CustomClass).__class__)

    def test_get_coder_with_composite_custom_coder(self):
        if False:
            return 10
        typecoders.registry.register_coder(CustomClass, CustomCoder)
        coder = typecoders.registry.get_coder(typehints.KV[CustomClass, str])
        revived_coder = pickler.loads(pickler.dumps(coder))
        self.assertEqual((CustomClass(123), 'abc'), revived_coder.decode(revived_coder.encode((CustomClass(123), 'abc'))))

    def test_get_coder_with_standard_coder(self):
        if False:
            print('Hello World!')
        self.assertEqual(coders.BytesCoder, typecoders.registry.get_coder(bytes).__class__)

    def test_fallbackcoder(self):
        if False:
            for i in range(10):
                print('nop')
        coder = typecoders.registry.get_coder(typehints.Any)
        self.assertEqual(('abc', 123), coder.decode(coder.encode(('abc', 123))))

    def test_get_coder_can_be_pickled(self):
        if False:
            while True:
                i = 10
        coder = typecoders.registry.get_coder(typehints.Tuple[str, int])
        revived_coder = pickler.loads(pickler.dumps(coder))
        self.assertEqual(('abc', 123), revived_coder.decode(revived_coder.encode(('abc', 123))))

    def test_standard_int_coder(self):
        if False:
            for i in range(10):
                print('nop')
        real_coder = typecoders.registry.get_coder(int)
        expected_coder = coders.VarIntCoder()
        self.assertEqual(real_coder.encode(1028), expected_coder.encode(1028))
        self.assertEqual(1028, real_coder.decode(real_coder.encode(1028)))
        self.assertEqual(real_coder.encode(4415293752324), expected_coder.encode(4415293752324))
        self.assertEqual(4415293752324, real_coder.decode(real_coder.encode(4415293752324)))

    def test_standard_str_coder(self):
        if False:
            for i in range(10):
                print('nop')
        real_coder = typecoders.registry.get_coder(bytes)
        expected_coder = coders.BytesCoder()
        self.assertEqual(real_coder.encode(b'abc'), expected_coder.encode(b'abc'))
        self.assertEqual(b'abc', real_coder.decode(real_coder.encode(b'abc')))

    def test_standard_bool_coder(self):
        if False:
            print('Hello World!')
        real_coder = typecoders.registry.get_coder(bool)
        expected_coder = coders.BooleanCoder()
        self.assertEqual(real_coder.encode(True), expected_coder.encode(True))
        self.assertEqual(True, real_coder.decode(real_coder.encode(True)))
        self.assertEqual(real_coder.encode(False), expected_coder.encode(False))
        self.assertEqual(False, real_coder.decode(real_coder.encode(False)))

    def test_iterable_coder(self):
        if False:
            for i in range(10):
                print('nop')
        real_coder = typecoders.registry.get_coder(typehints.Iterable[bytes])
        expected_coder = coders.IterableCoder(coders.BytesCoder())
        values = [b'abc', b'xyz']
        self.assertEqual(expected_coder, real_coder)
        self.assertEqual(real_coder.encode(values), expected_coder.encode(values))

    @unittest.skip('https://github.com/apache/beam/issues/21658')
    def test_list_coder(self):
        if False:
            return 10
        real_coder = typecoders.registry.get_coder(typehints.List[bytes])
        expected_coder = coders.IterableCoder(coders.BytesCoder())
        values = [b'abc', b'xyz']
        self.assertEqual(expected_coder, real_coder)
        self.assertEqual(real_coder.encode(values), expected_coder.encode(values))
        self.assertIs(list, type(expected_coder.decode(expected_coder.encode(values))))

    def test_nullable_coder(self):
        if False:
            print('Hello World!')
        expected_coder = coders.NullableCoder(coders.BytesCoder())
        real_coder = typecoders.registry.get_coder(typehints.Optional[bytes])
        self.assertEqual(expected_coder, real_coder)
        self.assertEqual(expected_coder.encode(None), real_coder.encode(None))
        self.assertEqual(expected_coder.encode(b'abc'), real_coder.encode(b'abc'))
if __name__ == '__main__':
    unittest.main()