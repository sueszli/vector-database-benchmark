import binascii
import functools
import hmac
import hashlib
import unittest
import unittest.mock
import warnings
from test.support import hashlib_helper, check_disallow_instantiation
from _operator import _compare_digest as operator_compare_digest
try:
    import _hashlib as _hashopenssl
    from _hashlib import HMAC as C_HMAC
    from _hashlib import hmac_new as c_hmac_new
    from _hashlib import compare_digest as openssl_compare_digest
except ImportError:
    _hashopenssl = None
    C_HMAC = None
    c_hmac_new = None
    openssl_compare_digest = None
try:
    import _sha256 as sha256_module
except ImportError:
    sha256_module = None

def ignore_warning(func):
    if False:
        i = 10
        return i + 15

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            return func(*args, **kwargs)
    return wrapper

class TestVectorsTestCase(unittest.TestCase):

    def assert_hmac_internals(self, h, digest, hashname, digest_size, block_size):
        if False:
            print('Hello World!')
        self.assertEqual(h.hexdigest().upper(), digest.upper())
        self.assertEqual(h.digest(), binascii.unhexlify(digest))
        self.assertEqual(h.name, f'hmac-{hashname}')
        self.assertEqual(h.digest_size, digest_size)
        self.assertEqual(h.block_size, block_size)

    def assert_hmac(self, key, data, digest, hashfunc, hashname, digest_size, block_size):
        if False:
            while True:
                i = 10
        h = hmac.HMAC(key, data, digestmod=hashfunc)
        self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
        h = hmac.HMAC(key, data, digestmod=hashname)
        self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
        h = hmac.HMAC(key, digestmod=hashname)
        h2 = h.copy()
        h2.update(b'test update')
        h.update(data)
        self.assertEqual(h.hexdigest().upper(), digest.upper())
        h = hmac.new(key, data, digestmod=hashname)
        self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
        h = hmac.new(key, None, digestmod=hashname)
        h.update(data)
        self.assertEqual(h.hexdigest().upper(), digest.upper())
        h = hmac.new(key, digestmod=hashname)
        h.update(data)
        self.assertEqual(h.hexdigest().upper(), digest.upper())
        h = hmac.new(key, data, digestmod=hashfunc)
        self.assertEqual(h.hexdigest().upper(), digest.upper())
        self.assertEqual(hmac.digest(key, data, digest=hashname), binascii.unhexlify(digest))
        self.assertEqual(hmac.digest(key, data, digest=hashfunc), binascii.unhexlify(digest))
        h = hmac.HMAC.__new__(hmac.HMAC)
        h._init_old(key, data, digestmod=hashname)
        self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
        if c_hmac_new is not None:
            h = c_hmac_new(key, data, digestmod=hashname)
            self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
            h = c_hmac_new(key, digestmod=hashname)
            h2 = h.copy()
            h2.update(b'test update')
            h.update(data)
            self.assertEqual(h.hexdigest().upper(), digest.upper())
            func = getattr(_hashopenssl, f'openssl_{hashname}')
            h = c_hmac_new(key, data, digestmod=func)
            self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)
            h = hmac.HMAC.__new__(hmac.HMAC)
            h._init_hmac(key, data, digestmod=hashname)
            self.assert_hmac_internals(h, digest, hashname, digest_size, block_size)

    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_md5_vectors(self):
        if False:
            while True:
                i = 10

        def md5test(key, data, digest):
            if False:
                i = 10
                return i + 15
            self.assert_hmac(key, data, digest, hashfunc=hashlib.md5, hashname='md5', digest_size=16, block_size=64)
        md5test(b'\x0b' * 16, b'Hi There', '9294727A3638BB1C13F48EF8158BFC9D')
        md5test(b'Jefe', b'what do ya want for nothing?', '750c783e6ab0b503eaa86e310a5db738')
        md5test(b'\xaa' * 16, b'\xdd' * 50, '56be34521d144c88dbb8c733f0e8b3f6')
        md5test(bytes(range(1, 26)), b'\xcd' * 50, '697eaf0aca3a3aea3a75164746ffaa79')
        md5test(b'\x0c' * 16, b'Test With Truncation', '56461ef2342edc00f9bab995690efd4c')
        md5test(b'\xaa' * 80, b'Test Using Larger Than Block-Size Key - Hash Key First', '6b1ab7fe4bd7bf8f0b62e6ce61b9d0cd')
        md5test(b'\xaa' * 80, b'Test Using Larger Than Block-Size Key and Larger Than One Block-Size Data', '6f630fad67cda0ee1fb1f562db3aa53e')

    @hashlib_helper.requires_hashdigest('sha1', openssl=True)
    def test_sha_vectors(self):
        if False:
            while True:
                i = 10

        def shatest(key, data, digest):
            if False:
                i = 10
                return i + 15
            self.assert_hmac(key, data, digest, hashfunc=hashlib.sha1, hashname='sha1', digest_size=20, block_size=64)
        shatest(b'\x0b' * 20, b'Hi There', 'b617318655057264e28bc0b6fb378c8ef146be00')
        shatest(b'Jefe', b'what do ya want for nothing?', 'effcdf6ae5eb2fa2d27416d5f184df9c259a7c79')
        shatest(b'\xaa' * 20, b'\xdd' * 50, '125d7342b9ac11cd91a39af48aa17b4f63f175d3')
        shatest(bytes(range(1, 26)), b'\xcd' * 50, '4c9007f4026250c6bc8414f9bf50c86c2d7235da')
        shatest(b'\x0c' * 20, b'Test With Truncation', '4c1a03424b55e07fe7f27be1d58bb9324a9a5a04')
        shatest(b'\xaa' * 80, b'Test Using Larger Than Block-Size Key - Hash Key First', 'aa4ae5e15272d00e95705637ce8a3b55ed402112')
        shatest(b'\xaa' * 80, b'Test Using Larger Than Block-Size Key and Larger Than One Block-Size Data', 'e8e99d0f45237d786d6bbaa7965c7808bbff1a91')

    def _rfc4231_test_cases(self, hashfunc, hash_name, digest_size, block_size):
        if False:
            print('Hello World!')

        def hmactest(key, data, hexdigests):
            if False:
                for i in range(10):
                    print('nop')
            digest = hexdigests[hashfunc]
            self.assert_hmac(key, data, digest, hashfunc=hashfunc, hashname=hash_name, digest_size=digest_size, block_size=block_size)
        hmactest(key=b'\x0b' * 20, data=b'Hi There', hexdigests={hashlib.sha224: '896fb1128abbdf196832107cd49df33f47b4b1169912ba4f53684b22', hashlib.sha256: 'b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7', hashlib.sha384: 'afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6', hashlib.sha512: '87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854'})
        hmactest(key=b'Jefe', data=b'what do ya want for nothing?', hexdigests={hashlib.sha224: 'a30e01098bc6dbbf45690f3a7e9e6d0f8bbea2a39e6148008fd05e44', hashlib.sha256: '5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843', hashlib.sha384: 'af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649', hashlib.sha512: '164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737'})
        hmactest(key=b'\xaa' * 20, data=b'\xdd' * 50, hexdigests={hashlib.sha224: '7fb3cb3588c6c1f6ffa9694d7d6ad2649365b0c1f65d69d1ec8333ea', hashlib.sha256: '773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe', hashlib.sha384: '88062608d3e6ad8a0aa2ace014c8a86f0aa635d947ac9febe83ef4e55966144b2a5ab39dc13814b94e3ab6e101a34f27', hashlib.sha512: 'fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb'})
        hmactest(key=bytes((x for x in range(1, 25 + 1))), data=b'\xcd' * 50, hexdigests={hashlib.sha224: '6c11506874013cac6a2abc1bb382627cec6a90d86efc012de7afec5a', hashlib.sha256: '82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b', hashlib.sha384: '3e8a69b7783c25851933ab6290af6ca77a9981480850009cc5577c6e1f573b4e6801dd23c4a7d679ccf8a386c674cffb', hashlib.sha512: 'b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3dba91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd'})
        hmactest(key=b'\xaa' * 131, data=b'Test Using Larger Than Block-Size Key - Hash Key First', hexdigests={hashlib.sha224: '95e9a0db962095adaebe9b2d6f0dbce2d499f112f2d2b7273fa6870e', hashlib.sha256: '60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54', hashlib.sha384: '4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952', hashlib.sha512: '80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598'})
        hmactest(key=b'\xaa' * 131, data=b'This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.', hexdigests={hashlib.sha224: '3a854166ac5d9f023f54d517d0b39dbd946770db9c2b95c9f6f565d1', hashlib.sha256: '9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2', hashlib.sha384: '6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e', hashlib.sha512: 'e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58'})

    @hashlib_helper.requires_hashdigest('sha224', openssl=True)
    def test_sha224_rfc4231(self):
        if False:
            for i in range(10):
                print('nop')
        self._rfc4231_test_cases(hashlib.sha224, 'sha224', 28, 64)

    @hashlib_helper.requires_hashdigest('sha256', openssl=True)
    def test_sha256_rfc4231(self):
        if False:
            for i in range(10):
                print('nop')
        self._rfc4231_test_cases(hashlib.sha256, 'sha256', 32, 64)

    @hashlib_helper.requires_hashdigest('sha384', openssl=True)
    def test_sha384_rfc4231(self):
        if False:
            i = 10
            return i + 15
        self._rfc4231_test_cases(hashlib.sha384, 'sha384', 48, 128)

    @hashlib_helper.requires_hashdigest('sha512', openssl=True)
    def test_sha512_rfc4231(self):
        if False:
            return 10
        self._rfc4231_test_cases(hashlib.sha512, 'sha512', 64, 128)

    @hashlib_helper.requires_hashdigest('sha256')
    def test_legacy_block_size_warnings(self):
        if False:
            print('Hello World!')

        class MockCrazyHash(object):
            """Ain't no block_size attribute here."""

            def __init__(self, *args):
                if False:
                    print('Hello World!')
                self._x = hashlib.sha256(*args)
                self.digest_size = self._x.digest_size

            def update(self, v):
                if False:
                    for i in range(10):
                        print('nop')
                self._x.update(v)

            def digest(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._x.digest()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                hmac.HMAC(b'a', b'b', digestmod=MockCrazyHash)
                self.fail('Expected warning about missing block_size')
            MockCrazyHash.block_size = 1
            with self.assertRaises(RuntimeWarning):
                hmac.HMAC(b'a', b'b', digestmod=MockCrazyHash)
                self.fail('Expected warning about small block_size')

    def test_with_digestmod_no_default(self):
        if False:
            return 10
        'The digestmod parameter is required as of Python 3.8.'
        with self.assertRaisesRegex(TypeError, 'required.*digestmod'):
            key = b'\x0b' * 16
            data = b'Hi There'
            hmac.HMAC(key, data, digestmod=None)
        with self.assertRaisesRegex(TypeError, 'required.*digestmod'):
            hmac.new(key, data)
        with self.assertRaisesRegex(TypeError, 'required.*digestmod'):
            hmac.HMAC(key, msg=data, digestmod='')

class ConstructorTestCase(unittest.TestCase):
    expected = '6c845b47f52b3b47f6590c502db7825aad757bf4fadc8fa972f7cd2e76a5bdeb'

    @hashlib_helper.requires_hashdigest('sha256')
    def test_normal(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            hmac.HMAC(b'key', digestmod='sha256')
        except Exception:
            self.fail('Standard constructor call raised exception.')

    @hashlib_helper.requires_hashdigest('sha256')
    def test_with_str_key(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            h = hmac.HMAC('key', digestmod='sha256')

    @hashlib_helper.requires_hashdigest('sha256')
    def test_dot_new_with_str_key(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            h = hmac.new('key', digestmod='sha256')

    @hashlib_helper.requires_hashdigest('sha256')
    def test_withtext(self):
        if False:
            print('Hello World!')
        try:
            h = hmac.HMAC(b'key', b'hash this!', digestmod='sha256')
        except Exception:
            self.fail('Constructor call with text argument raised exception.')
        self.assertEqual(h.hexdigest(), self.expected)

    @hashlib_helper.requires_hashdigest('sha256')
    def test_with_bytearray(self):
        if False:
            i = 10
            return i + 15
        try:
            h = hmac.HMAC(bytearray(b'key'), bytearray(b'hash this!'), digestmod='sha256')
        except Exception:
            self.fail('Constructor call with bytearray arguments raised exception.')
        self.assertEqual(h.hexdigest(), self.expected)

    @hashlib_helper.requires_hashdigest('sha256')
    def test_with_memoryview_msg(self):
        if False:
            return 10
        try:
            h = hmac.HMAC(b'key', memoryview(b'hash this!'), digestmod='sha256')
        except Exception:
            self.fail('Constructor call with memoryview msg raised exception.')
        self.assertEqual(h.hexdigest(), self.expected)

    @hashlib_helper.requires_hashdigest('sha256')
    def test_withmodule(self):
        if False:
            print('Hello World!')
        try:
            h = hmac.HMAC(b'key', b'', hashlib.sha256)
        except Exception:
            self.fail('Constructor call with hashlib.sha256 raised exception.')

    @unittest.skipUnless(C_HMAC is not None, 'need _hashlib')
    def test_internal_types(self):
        if False:
            return 10
        check_disallow_instantiation(self, C_HMAC)
        with self.assertRaisesRegex(TypeError, 'immutable type'):
            C_HMAC.value = None

    @unittest.skipUnless(sha256_module is not None, 'need _sha256')
    def test_with_sha256_module(self):
        if False:
            i = 10
            return i + 15
        h = hmac.HMAC(b'key', b'hash this!', digestmod=sha256_module.sha256)
        self.assertEqual(h.hexdigest(), self.expected)
        self.assertEqual(h.name, 'hmac-sha256')
        digest = hmac.digest(b'key', b'hash this!', sha256_module.sha256)
        self.assertEqual(digest, binascii.unhexlify(self.expected))

class SanityTestCase(unittest.TestCase):

    @hashlib_helper.requires_hashdigest('sha256')
    def test_exercise_all_methods(self):
        if False:
            i = 10
            return i + 15
        try:
            h = hmac.HMAC(b'my secret key', digestmod='sha256')
            h.update(b'compute the hash of this text!')
            h.digest()
            h.hexdigest()
            h.copy()
        except Exception:
            self.fail('Exception raised during normal usage of HMAC class.')

class CopyTestCase(unittest.TestCase):

    @hashlib_helper.requires_hashdigest('sha256')
    def test_attributes_old(self):
        if False:
            while True:
                i = 10
        h1 = hmac.HMAC.__new__(hmac.HMAC)
        h1._init_old(b'key', b'msg', digestmod='sha256')
        h2 = h1.copy()
        self.assertEqual(type(h1._inner), type(h2._inner), "Types of inner don't match.")
        self.assertEqual(type(h1._outer), type(h2._outer), "Types of outer don't match.")

    @hashlib_helper.requires_hashdigest('sha256')
    def test_realcopy_old(self):
        if False:
            print('Hello World!')
        h1 = hmac.HMAC.__new__(hmac.HMAC)
        h1._init_old(b'key', b'msg', digestmod='sha256')
        h2 = h1.copy()
        self.assertTrue(id(h1) != id(h2), 'No real copy of the HMAC instance.')
        self.assertTrue(id(h1._inner) != id(h2._inner), "No real copy of the attribute 'inner'.")
        self.assertTrue(id(h1._outer) != id(h2._outer), "No real copy of the attribute 'outer'.")
        self.assertIs(h1._hmac, None)

    @unittest.skipIf(_hashopenssl is None, 'test requires _hashopenssl')
    @hashlib_helper.requires_hashdigest('sha256')
    def test_realcopy_hmac(self):
        if False:
            return 10
        h1 = hmac.HMAC.__new__(hmac.HMAC)
        h1._init_hmac(b'key', b'msg', digestmod='sha256')
        h2 = h1.copy()
        self.assertTrue(id(h1._hmac) != id(h2._hmac))

    @hashlib_helper.requires_hashdigest('sha256')
    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        h1 = hmac.HMAC(b'key', digestmod='sha256')
        h1.update(b'some random text')
        h2 = h1.copy()
        self.assertEqual(h1.digest(), h2.digest(), "Digest of copy doesn't match original digest.")
        self.assertEqual(h1.hexdigest(), h2.hexdigest(), "Hexdigest of copy doesn't match original hexdigest.")

    @hashlib_helper.requires_hashdigest('sha256')
    def test_equality_new(self):
        if False:
            for i in range(10):
                print('nop')
        h1 = hmac.new(b'key', digestmod='sha256')
        h1.update(b'some random text')
        h2 = h1.copy()
        self.assertTrue(id(h1) != id(h2), 'No real copy of the HMAC instance.')
        self.assertEqual(h1.digest(), h2.digest(), "Digest of copy doesn't match original digest.")
        self.assertEqual(h1.hexdigest(), h2.hexdigest(), "Hexdigest of copy doesn't match original hexdigest.")

class CompareDigestTestCase(unittest.TestCase):

    def test_hmac_compare_digest(self):
        if False:
            while True:
                i = 10
        self._test_compare_digest(hmac.compare_digest)
        if openssl_compare_digest is not None:
            self.assertIs(hmac.compare_digest, openssl_compare_digest)
        else:
            self.assertIs(hmac.compare_digest, operator_compare_digest)

    def test_operator_compare_digest(self):
        if False:
            while True:
                i = 10
        self._test_compare_digest(operator_compare_digest)

    @unittest.skipIf(openssl_compare_digest is None, 'test requires _hashlib')
    def test_openssl_compare_digest(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_compare_digest(openssl_compare_digest)

    def _test_compare_digest(self, compare_digest):
        if False:
            return 10
        (a, b) = (100, 200)
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (100, b'foobar')
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (b'foobar', 200)
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = ('foobar', b'foobar')
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (b'foobar', 'foobar')
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (b'foobar', b'foo')
        self.assertFalse(compare_digest(a, b))
        (a, b) = (b'\xde\xad\xbe\xef', b'\xde\xad')
        self.assertFalse(compare_digest(a, b))
        (a, b) = (b'foobar', b'foobaz')
        self.assertFalse(compare_digest(a, b))
        (a, b) = (b'\xde\xad\xbe\xef', b'\xab\xad\x1d\xea')
        self.assertFalse(compare_digest(a, b))
        (a, b) = (b'foobar', b'foobar')
        self.assertTrue(compare_digest(a, b))
        (a, b) = (b'\xde\xad\xbe\xef', b'\xde\xad\xbe\xef')
        self.assertTrue(compare_digest(a, b))
        (a, b) = (bytearray(b'foobar'), bytearray(b'foobar'))
        self.assertTrue(compare_digest(a, b))
        (a, b) = (bytearray(b'foobar'), bytearray(b'foo'))
        self.assertFalse(compare_digest(a, b))
        (a, b) = (bytearray(b'foobar'), bytearray(b'foobaz'))
        self.assertFalse(compare_digest(a, b))
        (a, b) = (bytearray(b'foobar'), b'foobar')
        self.assertTrue(compare_digest(a, b))
        self.assertTrue(compare_digest(b, a))
        (a, b) = (bytearray(b'foobar'), b'foo')
        self.assertFalse(compare_digest(a, b))
        self.assertFalse(compare_digest(b, a))
        (a, b) = (bytearray(b'foobar'), b'foobaz')
        self.assertFalse(compare_digest(a, b))
        self.assertFalse(compare_digest(b, a))
        (a, b) = ('foobar', 'foobar')
        self.assertTrue(compare_digest(a, b))
        (a, b) = ('foo', 'foobar')
        self.assertFalse(compare_digest(a, b))
        (a, b) = ('foobar', 'foobaz')
        self.assertFalse(compare_digest(a, b))
        (a, b) = ('foobar', b'foobar')
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (b'foobar', 'foobar')
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (b'foobar', 1)
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = (100, 200)
        self.assertRaises(TypeError, compare_digest, a, b)
        (a, b) = ('fooä', 'fooä')
        self.assertRaises(TypeError, compare_digest, a, b)

        class mystr(str):

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                return False
        (a, b) = (mystr('foobar'), mystr('foobar'))
        self.assertTrue(compare_digest(a, b))
        (a, b) = (mystr('foobar'), 'foobar')
        self.assertTrue(compare_digest(a, b))
        (a, b) = (mystr('foobar'), mystr('foobaz'))
        self.assertFalse(compare_digest(a, b))

        class mybytes(bytes):

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        (a, b) = (mybytes(b'foobar'), mybytes(b'foobar'))
        self.assertTrue(compare_digest(a, b))
        (a, b) = (mybytes(b'foobar'), b'foobar')
        self.assertTrue(compare_digest(a, b))
        (a, b) = (mybytes(b'foobar'), mybytes(b'foobaz'))
        self.assertFalse(compare_digest(a, b))
if __name__ == '__main__':
    unittest.main()