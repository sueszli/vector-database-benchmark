"""Tests for utf8 caching."""
from bzrlib import cache_utf8
from bzrlib.tests import TestCase

class TestEncodeCache(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestEncodeCache, self).setUp()
        cache_utf8.clear_encoding_cache()
        self.addCleanup(cache_utf8.clear_encoding_cache)

    def check_encode(self, rev_id):
        if False:
            i = 10
            return i + 15
        rev_id_utf8 = rev_id.encode('utf-8')
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id_utf8, cache_utf8.encode(rev_id))
        self.assertTrue(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertTrue(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id, cache_utf8.decode(rev_id_utf8))
        cache_utf8.clear_encoding_cache()
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)

    def check_decode(self, rev_id):
        if False:
            return 10
        rev_id_utf8 = rev_id.encode('utf-8')
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id, cache_utf8.decode(rev_id_utf8))
        self.assertTrue(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertTrue(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
        self.assertEqual(rev_id_utf8, cache_utf8.encode(rev_id))
        cache_utf8.clear_encoding_cache()
        self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
        self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)

    def test_ascii(self):
        if False:
            while True:
                i = 10
        self.check_decode(u'all_ascii_characters123123123')
        self.check_encode(u'all_ascii_characters123123123')

    def test_unicode(self):
        if False:
            while True:
                i = 10
        self.check_encode(u'some_µ_unicode_å_chars')
        self.check_decode(u'some_µ_unicode_å_chars')

    def test_cached_unicode(self):
        if False:
            while True:
                i = 10
        x = u'µyy' + u'åzz'
        y = u'µyy' + u'åzz'
        self.assertFalse(x is y)
        xp = cache_utf8.get_cached_unicode(x)
        yp = cache_utf8.get_cached_unicode(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)

    def test_cached_utf8(self):
        if False:
            for i in range(10):
                print('nop')
        x = u'µyyåzz'.encode('utf8')
        y = u'µyyåzz'.encode('utf8')
        self.assertFalse(x is y)
        xp = cache_utf8.get_cached_utf8(x)
        yp = cache_utf8.get_cached_utf8(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)

    def test_cached_ascii(self):
        if False:
            i = 10
            return i + 15
        x = '%s %s' % ('simple', 'text')
        y = '%s %s' % ('simple', 'text')
        self.assertFalse(x is y)
        xp = cache_utf8.get_cached_ascii(x)
        yp = cache_utf8.get_cached_ascii(y)
        self.assertIs(xp, x)
        self.assertIs(xp, yp)
        uni_x = cache_utf8.decode(x)
        self.assertEqual(u'simple text', uni_x)
        self.assertIsInstance(uni_x, unicode)
        utf8_x = cache_utf8.encode(uni_x)
        self.assertIs(utf8_x, x)

    def test_decode_with_None(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(None, cache_utf8._utf8_decode_with_None(None))
        self.assertEqual(u'foo', cache_utf8._utf8_decode_with_None('foo'))
        self.assertEqual(u'fµ', cache_utf8._utf8_decode_with_None('fÂµ'))