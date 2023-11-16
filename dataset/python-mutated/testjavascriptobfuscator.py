"""Tests for JavaScriptObfuscator unpacker."""
import unittest
from jsbeautifier.unpackers.javascriptobfuscator import unpack, detect, smartsplit

class TestJavascriptObfuscator(unittest.TestCase):
    """JavascriptObfuscator.com test case."""

    def test_smartsplit(self):
        if False:
            return 10
        'Test smartsplit() function.'
        split = smartsplit

        def equals(data, result):
            if False:
                i = 10
                return i + 15
            return self.assertEqual(split(data), result)
        equals('', [])
        equals('"a", "b"', ['"a"', '"b"'])
        equals('"aaa","bbbb"', ['"aaa"', '"bbbb"'])
        equals('"a", "b\\""', ['"a"', '"b\\""'])

    def test_detect(self):
        if False:
            print('Hello World!')
        'Test detect() function.'

        def positive(source):
            if False:
                print('Hello World!')
            return self.assertTrue(detect(source))

        def negative(source):
            if False:
                return 10
            return self.assertFalse(detect(source))
        negative('')
        negative('abcd')
        negative('var _0xaaaa')
        positive('var _0xaaaa = ["a", "b"]')
        positive('var _0xaaaa=["a", "b"]')
        positive('var _0x1234=["a","b"]')

    def test_unpack(self):
        if False:
            i = 10
            return i + 15
        'Test unpack() function.'

        def decodeto(ob, original):
            if False:
                for i in range(10):
                    print('nop')
            return self.assertEqual(unpack(ob), original)
        decodeto('var _0x8df3=[];var a=10;', 'var a=10;')
        decodeto('var _0xb2a7=["t\'est"];var i;for(i=0;i<10;++i){alert(_0xb2a7[0]);} ;', 'var i;for(i=0;i<10;++i){alert("t\'est");} ;')
if __name__ == '__main__':
    unittest.main()