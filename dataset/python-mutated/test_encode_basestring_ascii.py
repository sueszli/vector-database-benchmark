from collections import OrderedDict
from test.test_json import PyTest, CTest
from test.support import bigaddrspacetest
CASES = [('/\\"쫾몾ꮘﳞ볚\uef4a\x08\x0c\n\r\t`1~!@#$%^&*()_+-=[]{}|;:\',./<>?', '"/\\\\\\"\\ucafe\\ubabe\\uab98\\ufcde\\ubcda\\uef4a\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:\',./<>?"'), ('ģ䕧覫췯ꯍ\uef4a', '"\\u0123\\u4567\\u89ab\\ucdef\\uabcd\\uef4a"'), ('controls', '"controls"'), ('\x08\x0c\n\r\t', '"\\b\\f\\n\\r\\t"'), ('{"object with 1 member":["array with 1 element"]}', '"{\\"object with 1 member\\":[\\"array with 1 element\\"]}"'), (' s p a c e d ', '" s p a c e d "'), ('𝄠', '"\\ud834\\udd20"'), ('αΩ', '"\\u03b1\\u03a9"'), ("`1~!@#$%^&*()_+-={':[,]}|;.</>?", '"`1~!@#$%^&*()_+-={\':[,]}|;.</>?"'), ('\x08\x0c\n\r\t', '"\\b\\f\\n\\r\\t"'), ('ģ䕧覫췯ꯍ\uef4a', '"\\u0123\\u4567\\u89ab\\ucdef\\uabcd\\uef4a"')]

class TestEncodeBasestringAscii:

    def test_encode_basestring_ascii(self):
        if False:
            i = 10
            return i + 15
        fname = self.json.encoder.encode_basestring_ascii.__name__
        for (input_string, expect) in CASES:
            result = self.json.encoder.encode_basestring_ascii(input_string)
            self.assertEqual(result, expect, '{0!r} != {1!r} for {2}({3!r})'.format(result, expect, fname, input_string))

    def test_ordered_dict(self):
        if False:
            print('Hello World!')
        items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
        s = self.dumps(OrderedDict(items))
        self.assertEqual(s, '{"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}')

    def test_sorted_dict(self):
        if False:
            print('Hello World!')
        items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
        s = self.dumps(dict(items), sort_keys=True)
        self.assertEqual(s, '{"five": 5, "four": 4, "one": 1, "three": 3, "two": 2}')

class TestPyEncodeBasestringAscii(TestEncodeBasestringAscii, PyTest):
    pass

class TestCEncodeBasestringAscii(TestEncodeBasestringAscii, CTest):

    @bigaddrspacetest
    def test_overflow(self):
        if False:
            i = 10
            return i + 15
        size = 2 ** 32 // 6 + 1
        s = '\x00' * size
        with self.assertRaises(OverflowError):
            self.json.encoder.encode_basestring_ascii(s)