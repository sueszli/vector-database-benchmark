import unittest
from nyaa import bencode

class TestBencode(unittest.TestCase):

    def test_pairwise(self):
        if False:
            return 10
        initial = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        for (index, values) in enumerate(bencode._pairwise(initial)):
            self.assertEqual(values, expected[index])
        initial = [0, 1, 2, 3, 4]
        expected = [(0, 1), (2, 3), 4]
        for (index, values) in enumerate(bencode._pairwise(initial)):
            self.assertEqual(values, expected[index])
        initial = b'012345'
        expected = [(48, 49), (50, 51), (52, 53)]
        for (index, values) in enumerate(bencode._pairwise(initial)):
            self.assertEqual(values, expected[index])

    def test_encode(self):
        if False:
            for i in range(10):
                print('nop')
        exception_test_cases = [(None, bencode.BencodeException, 'Unsupported type'), (1.6, bencode.BencodeException, 'Unsupported type')]
        test_cases = [(100, b'i100e'), (-5, b'i-5e'), ('test', b'4:test'), (b'test', b'4:test'), (['test', 100], b'l4:testi100ee'), ({'numbers': [1, 2], 'hello': 'world'}, b'd5:hello5:world7:numbersli1ei2eee')]
        for (raw, raised_exception, expected_result_regexp) in exception_test_cases:
            self.assertRaisesRegexp(raised_exception, expected_result_regexp, bencode.encode, raw)
        for (raw, expected_result) in test_cases:
            self.assertEqual(bencode.encode(raw), expected_result)

    def test_decode(self):
        if False:
            for i in range(10):
                print('nop')
        exception_test_cases = [(b'l4:hey', bencode.MalformedBencodeException, 'Read only \\d+ bytes, \\d+ wanted'), (b'ie', bencode.MalformedBencodeException, 'Unable to parse int'), (b'i64', bencode.MalformedBencodeException, 'EOF, expecting more integer'), (b'', bencode.MalformedBencodeException, 'EOF, expecting kind'), (b'i6-4', bencode.MalformedBencodeException, 'Unexpected input while reading an integer'), (b'4#string', bencode.MalformedBencodeException, 'Unexpected input while reading string length'), (b'4', bencode.MalformedBencodeException, 'EOF, expecting more string len'), (b'$:string', bencode.MalformedBencodeException, 'Unexpected data type'), (b'd5:world7:numbersli1ei2eee', bencode.MalformedBencodeException, 'Uneven amount of key/value pairs')]
        test_cases = [(b'i100e', 100), (b'i-5e', -5), ('4:test', b'test'), (b'4:test', b'test'), (b'15:thisisalongone!', b'thisisalongone!'), (b'l4:testi100ee', [b'test', 100]), (b'd5:hello5:world7:numbersli1ei2eee', {'hello': b'world', 'numbers': [1, 2]})]
        for (raw, raised_exception, expected_result_regexp) in exception_test_cases:
            self.assertRaisesRegexp(raised_exception, expected_result_regexp, bencode.decode, raw)
        for (raw, expected_result) in test_cases:
            self.assertEqual(bencode.decode(raw), expected_result)
if __name__ == '__main__':
    unittest.main()