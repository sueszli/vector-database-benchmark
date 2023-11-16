import sys
import unittest
import Cython.Compiler.StringEncoding as StringEncoding

class StringEncodingTest(unittest.TestCase):
    """
    Test the StringEncoding module.
    """

    def test_string_contains_lone_surrogates(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(StringEncoding.string_contains_lone_surrogates(u'abc'))
        self.assertFalse(StringEncoding.string_contains_lone_surrogates(u'ꯍ'))
        self.assertFalse(StringEncoding.string_contains_lone_surrogates(u'☃'))
        if sys.version_info[0] != 2:
            self.assertTrue(StringEncoding.string_contains_lone_surrogates(u'\ud800\udfff'))
        obfuscated_surrogate_pair = (u'\udfff' + '\ud800')[::-1]
        if sys.version_info[0] == 2 and sys.maxunicode == 65565:
            self.assertFalse(StringEncoding.string_contains_lone_surrogates(obfuscated_surrogate_pair))
        else:
            self.assertTrue(StringEncoding.string_contains_lone_surrogates(obfuscated_surrogate_pair))
        self.assertTrue(StringEncoding.string_contains_lone_surrogates(u'\ud800'))
        self.assertTrue(StringEncoding.string_contains_lone_surrogates(u'\udfff'))
        self.assertTrue(StringEncoding.string_contains_lone_surrogates(u'\udfff\ud800'))
        self.assertTrue(StringEncoding.string_contains_lone_surrogates(u'\ud800x\udfff'))

    def test_string_contains_surrogates(self):
        if False:
            return 10
        self.assertFalse(StringEncoding.string_contains_surrogates(u'abc'))
        self.assertFalse(StringEncoding.string_contains_surrogates(u'ꯍ'))
        self.assertFalse(StringEncoding.string_contains_surrogates(u'☃'))
        self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800'))
        self.assertTrue(StringEncoding.string_contains_surrogates(u'\udfff'))
        self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800\udfff'))
        self.assertTrue(StringEncoding.string_contains_surrogates(u'\udfff\ud800'))
        self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800x\udfff'))