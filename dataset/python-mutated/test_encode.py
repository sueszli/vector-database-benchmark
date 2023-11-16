import unittest
from pyramid.util import text_

class UrlEncodeTests(unittest.TestCase):

    def _callFUT(self, query, doseq=False, **kw):
        if False:
            return 10
        from pyramid.encode import urlencode
        return urlencode(query, doseq, **kw)

    def test_ascii_only(self):
        if False:
            print('Hello World!')
        result = self._callFUT([('a', 1), ('b', 2)])
        self.assertEqual(result, 'a=1&b=2')

    def test_unicode_key(self):
        if False:
            print('Hello World!')
        la = text_(b'LaPe\xc3\xb1a', 'utf-8')
        result = self._callFUT([(la, 1), ('b', 2)])
        self.assertEqual(result, 'LaPe%C3%B1a=1&b=2')

    def test_unicode_val_single(self):
        if False:
            while True:
                i = 10
        la = text_(b'LaPe\xc3\xb1a', 'utf-8')
        result = self._callFUT([('a', la), ('b', 2)])
        self.assertEqual(result, 'a=LaPe%C3%B1a&b=2')

    def test_unicode_val_multiple(self):
        if False:
            while True:
                i = 10
        la = [text_(b'LaPe\xc3\xb1a', 'utf-8')] * 2
        result = self._callFUT([('a', la), ('b', 2)], doseq=True)
        self.assertEqual(result, 'a=LaPe%C3%B1a&a=LaPe%C3%B1a&b=2')

    def test_int_val_multiple(self):
        if False:
            return 10
        s = [1, 2]
        result = self._callFUT([('a', s)], doseq=True)
        self.assertEqual(result, 'a=1&a=2')

    def test_with_spaces(self):
        if False:
            return 10
        result = self._callFUT([('a', '123 456')], doseq=True)
        self.assertEqual(result, 'a=123+456')

    def test_dict(self):
        if False:
            i = 10
            return i + 15
        result = self._callFUT({'a': 1})
        self.assertEqual(result, 'a=1')

    def test_None_value(self):
        if False:
            while True:
                i = 10
        result = self._callFUT([('a', None)])
        self.assertEqual(result, 'a=')

    def test_None_value_with_prefix(self):
        if False:
            return 10
        result = self._callFUT([('a', '1'), ('b', None)])
        self.assertEqual(result, 'a=1&b=')

    def test_None_value_with_prefix_values(self):
        if False:
            for i in range(10):
                print('nop')
        result = self._callFUT([('a', '1'), ('b', None), ('c', None)])
        self.assertEqual(result, 'a=1&b=&c=')

    def test_quote_via(self):
        if False:
            return 10

        def my_quoter(value):
            if False:
                print('Hello World!')
            return 'xxx' + value
        result = self._callFUT([('a', '1'), ('b', None), ('c', None)], quote_via=my_quoter)
        self.assertEqual(result, 'xxxa=xxx1&xxxb=&xxxc=')

class URLQuoteTests(unittest.TestCase):

    def _callFUT(self, val, safe=''):
        if False:
            return 10
        from pyramid.encode import url_quote
        return url_quote(val, safe)

    def test_it_bytes(self):
        if False:
            while True:
                i = 10
        la = b'La/Pe\xc3\xb1a'
        result = self._callFUT(la)
        self.assertEqual(result, 'La%2FPe%C3%B1a')

    def test_it_native(self):
        if False:
            return 10
        la = text_(b'La/Pe\xc3\xb1a', 'utf-8')
        result = self._callFUT(la)
        self.assertEqual(result, 'La%2FPe%C3%B1a')

    def test_it_with_safe(self):
        if False:
            print('Hello World!')
        la = b'La/Pe\xc3\xb1a'
        result = self._callFUT(la, '/')
        self.assertEqual(result, 'La/Pe%C3%B1a')

    def test_it_with_nonstr_nonbinary(self):
        if False:
            for i in range(10):
                print('nop')
        la = None
        result = self._callFUT(la, '/')
        self.assertEqual(result, 'None')