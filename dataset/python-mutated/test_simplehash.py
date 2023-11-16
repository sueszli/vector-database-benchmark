import os
import unittest
from golem.core.simplehash import SimpleHash
from golem.testutils import TempDirFixture

class TestSimpleHash(unittest.TestCase):

    def testBase64(self):
        if False:
            while True:
                i = 10
        dec = b'Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.'
        enc = b'TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz\nIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg\ndGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu\ndWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo\nZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=\n'
        enc2 = SimpleHash.base64_encode(dec)
        self.assertEqual(enc, enc2)
        dec2 = SimpleHash.base64_decode(enc)
        self.assertEqual(dec, dec2)

    def testHash(self):
        if False:
            while True:
                i = 10
        ex1 = b''
        hex1 = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
        b641 = b'2jmj7l5rSw0yVb/vlWAYkK/YBwk=\n'
        hash1 = b'\xda9\xa3\xee^kK\r2U\xbf\xef\x95`\x18\x90\xaf\xd8\x07\t'
        ex2 = b'The quick brown fox jumps over the lazy dog'
        hex2 = '2fd4e1c67a2d28fced849ee1bb76e7391b93eb12'
        b642 = b'L9ThxnotKPzthJ7hu3bnORuT6xI=\n'
        hash2 = b'/\xd4\xe1\xc6z-(\xfc\xed\x84\x9e\xe1\xbbv\xe79\x1b\x93\xeb\x12'
        self.assertEqual(hash1, SimpleHash.hash(ex1))
        self.assertEqual(hash2, SimpleHash.hash(ex2))
        self.assertEqual(hex1, SimpleHash.hash_hex(ex1))
        self.assertEqual(hex2, SimpleHash.hash_hex(ex2))
        self.assertEqual(b641, SimpleHash.hash_base64(ex1))
        self.assertEqual(b642, SimpleHash.hash_base64(ex2))

class TestFileHash(TempDirFixture):

    def test(self):
        if False:
            i = 10
            return i + 15
        file_path = os.path.join(self.path, 'file.txt')
        with open(file_path, 'wb') as out:
            out.write(b'The quick brown fox jumps over the lazy dog\n')
        b64 = b'vkF3aLXDxcHZvLLnwRkZbddrVXA=\n'
        self.assertEqual(b64, SimpleHash.hash_file_base64(file_path))