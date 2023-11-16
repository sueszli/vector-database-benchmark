""" Python character mapping codec test

This uses the test codec in testcodec.py and thus also tests the
encodings package lookup scheme.

Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright 2000 Guido van Rossum.

"""
import unittest
import codecs

def codec_search_function(encoding):
    if False:
        while True:
            i = 10
    if encoding == 'testcodec':
        from test import testcodec
        return tuple(testcodec.getregentry())
    return None
codecname = 'testcodec'

class CharmapCodecTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        codecs.register(codec_search_function)
        self.addCleanup(codecs.unregister, codec_search_function)

    def test_constructorx(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(b'abc', codecname), 'abc')
        self.assertEqual(str(b'xdef', codecname), 'abcdef')
        self.assertEqual(str(b'defx', codecname), 'defabc')
        self.assertEqual(str(b'dxf', codecname), 'dabcf')
        self.assertEqual(str(b'dxfx', codecname), 'dabcfabc')

    def test_encodex(self):
        if False:
            print('Hello World!')
        self.assertEqual('abc'.encode(codecname), b'abc')
        self.assertEqual('xdef'.encode(codecname), b'abcdef')
        self.assertEqual('defx'.encode(codecname), b'defabc')
        self.assertEqual('dxf'.encode(codecname), b'dabcf')
        self.assertEqual('dxfx'.encode(codecname), b'dabcfabc')

    def test_constructory(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(b'ydef', codecname), 'def')
        self.assertEqual(str(b'defy', codecname), 'def')
        self.assertEqual(str(b'dyf', codecname), 'df')
        self.assertEqual(str(b'dyfy', codecname), 'df')

    def test_maptoundefined(self):
        if False:
            print('Hello World!')
        self.assertRaises(UnicodeError, str, b'abc\x01', codecname)
if __name__ == '__main__':
    unittest.main()