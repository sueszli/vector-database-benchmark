import unittest
import tornado
from tornado.escape import utf8, xhtml_escape, xhtml_unescape, url_escape, url_unescape, to_unicode, json_decode, json_encode, squeeze, recursive_unicode
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any
linkify_tests = [('hello http://world.com/!', {}, 'hello <a href="http://world.com/">http://world.com/</a>!'), ('hello http://world.com/with?param=true&stuff=yes', {}, 'hello <a href="http://world.com/with?param=true&amp;stuff=yes">http://world.com/with?param=true&amp;stuff=yes</a>'), ('http://url.com/w(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', {}, '<a href="http://url.com/w">http://url.com/w</a>(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'), ('http://url.com/withmany.......................................', {}, '<a href="http://url.com/withmany">http://url.com/withmany</a>.......................................'), ('http://url.com/withmany((((((((((((((((((((((((((((((((((a)', {}, '<a href="http://url.com/withmany">http://url.com/withmany</a>((((((((((((((((((((((((((((((((((a)'), ('http://foo.com/blah_blah', {}, '<a href="http://foo.com/blah_blah">http://foo.com/blah_blah</a>'), ('http://foo.com/blah_blah/', {}, '<a href="http://foo.com/blah_blah/">http://foo.com/blah_blah/</a>'), ('(Something like http://foo.com/blah_blah)', {}, '(Something like <a href="http://foo.com/blah_blah">http://foo.com/blah_blah</a>)'), ('http://foo.com/blah_blah_(wikipedia)', {}, '<a href="http://foo.com/blah_blah_(wikipedia)">http://foo.com/blah_blah_(wikipedia)</a>'), ('http://foo.com/blah_(blah)_(wikipedia)_blah', {}, '<a href="http://foo.com/blah_(blah)_(wikipedia)_blah">http://foo.com/blah_(blah)_(wikipedia)_blah</a>'), ('(Something like http://foo.com/blah_blah_(wikipedia))', {}, '(Something like <a href="http://foo.com/blah_blah_(wikipedia)">http://foo.com/blah_blah_(wikipedia)</a>)'), ('http://foo.com/blah_blah.', {}, '<a href="http://foo.com/blah_blah">http://foo.com/blah_blah</a>.'), ('http://foo.com/blah_blah/.', {}, '<a href="http://foo.com/blah_blah/">http://foo.com/blah_blah/</a>.'), ('<http://foo.com/blah_blah>', {}, '&lt;<a href="http://foo.com/blah_blah">http://foo.com/blah_blah</a>&gt;'), ('<http://foo.com/blah_blah/>', {}, '&lt;<a href="http://foo.com/blah_blah/">http://foo.com/blah_blah/</a>&gt;'), ('http://foo.com/blah_blah,', {}, '<a href="http://foo.com/blah_blah">http://foo.com/blah_blah</a>,'), ('http://www.example.com/wpstyle/?p=364.', {}, '<a href="http://www.example.com/wpstyle/?p=364">http://www.example.com/wpstyle/?p=364</a>.'), ('rdar://1234', {'permitted_protocols': ['http', 'rdar']}, '<a href="rdar://1234">rdar://1234</a>'), ('rdar:/1234', {'permitted_protocols': ['rdar']}, '<a href="rdar:/1234">rdar:/1234</a>'), ('http://userid:password@example.com:8080', {}, '<a href="http://userid:password@example.com:8080">http://userid:password@example.com:8080</a>'), ('http://userid@example.com', {}, '<a href="http://userid@example.com">http://userid@example.com</a>'), ('http://userid@example.com:8080', {}, '<a href="http://userid@example.com:8080">http://userid@example.com:8080</a>'), ('http://userid:password@example.com', {}, '<a href="http://userid:password@example.com">http://userid:password@example.com</a>'), ('message://%3c330e7f8409726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e', {'permitted_protocols': ['http', 'message']}, '<a href="message://%3c330e7f8409726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e">message://%3c330e7f8409726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e</a>'), ('http://➡.ws/䨹', {}, '<a href="http://➡.ws/䨹">http://➡.ws/䨹</a>'), ('<tag>http://example.com</tag>', {}, '&lt;tag&gt;<a href="http://example.com">http://example.com</a>&lt;/tag&gt;'), ('Just a www.example.com link.', {}, 'Just a <a href="http://www.example.com">www.example.com</a> link.'), ('Just a www.example.com link.', {'require_protocol': True}, 'Just a www.example.com link.'), ('A http://reallylong.com/link/that/exceedsthelenglimit.html', {'require_protocol': True, 'shorten': True}, 'A <a href="http://reallylong.com/link/that/exceedsthelenglimit.html" title="http://reallylong.com/link/that/exceedsthelenglimit.html">http://reallylong.com/link...</a>'), ('A http://reallylongdomainnamethatwillbetoolong.com/hi!', {'shorten': True}, 'A <a href="http://reallylongdomainnamethatwillbetoolong.com/hi" title="http://reallylongdomainnamethatwillbetoolong.com/hi">http://reallylongdomainnametha...</a>!'), ('A file:///passwords.txt and http://web.com link', {}, 'A file:///passwords.txt and <a href="http://web.com">http://web.com</a> link'), ('A file:///passwords.txt and http://web.com link', {'permitted_protocols': ['file']}, 'A <a href="file:///passwords.txt">file:///passwords.txt</a> and http://web.com link'), ('www.external-link.com', {'extra_params': 'rel="nofollow" class="external"'}, '<a href="http://www.external-link.com" rel="nofollow" class="external">www.external-link.com</a>'), ('www.external-link.com and www.internal-link.com/blogs extra', {'extra_params': lambda href: 'class="internal"' if href.startswith('http://www.internal-link.com') else 'rel="nofollow" class="external"'}, '<a href="http://www.external-link.com" rel="nofollow" class="external">www.external-link.com</a> and <a href="http://www.internal-link.com/blogs" class="internal">www.internal-link.com/blogs</a> extra'), ('www.external-link.com', {'extra_params': lambda href: '    rel="nofollow" class="external"  '}, '<a href="http://www.external-link.com" rel="nofollow" class="external">www.external-link.com</a>')]

class EscapeTestCase(unittest.TestCase):

    def test_linkify(self):
        if False:
            return 10
        for (text, kwargs, html) in linkify_tests:
            linked = tornado.escape.linkify(text, **kwargs)
            self.assertEqual(linked, html)

    def test_xhtml_escape(self):
        if False:
            while True:
                i = 10
        tests = [('<foo>', '&lt;foo&gt;'), ('<foo>', '&lt;foo&gt;'), (b'<foo>', b'&lt;foo&gt;'), ('<>&"\'', '&lt;&gt;&amp;&quot;&#x27;'), ('&amp;', '&amp;amp;'), ('<é>', '&lt;é&gt;'), (b'<\xc3\xa9>', b'&lt;\xc3\xa9&gt;')]
        for (unescaped, escaped) in tests:
            self.assertEqual(utf8(xhtml_escape(unescaped)), utf8(escaped))
            self.assertEqual(utf8(unescaped), utf8(xhtml_unescape(escaped)))

    def test_xhtml_unescape_numeric(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [('foo&#32;bar', 'foo bar'), ('foo&#x20;bar', 'foo bar'), ('foo&#X20;bar', 'foo bar'), ('foo&#xabc;bar', 'foo઼bar'), ('foo&#xyz;bar', 'foo&#xyz;bar'), ('foo&#;bar', 'foo&#;bar'), ('foo&#x;bar', 'foo&#x;bar')]
        for (escaped, unescaped) in tests:
            self.assertEqual(unescaped, xhtml_unescape(escaped))

    def test_url_escape_unicode(self):
        if False:
            while True:
                i = 10
        tests = [('é'.encode('utf8'), '%C3%A9'), ('é'.encode('latin1'), '%E9'), ('é', '%C3%A9')]
        for (unescaped, escaped) in tests:
            self.assertEqual(url_escape(unescaped), escaped)

    def test_url_unescape_unicode(self):
        if False:
            i = 10
            return i + 15
        tests = [('%C3%A9', 'é', 'utf8'), ('%C3%A9', 'Ã©', 'latin1'), ('%C3%A9', utf8('é'), None)]
        for (escaped, unescaped, encoding) in tests:
            self.assertEqual(url_unescape(to_unicode(escaped), encoding), unescaped)
            self.assertEqual(url_unescape(utf8(escaped), encoding), unescaped)

    def test_url_escape_quote_plus(self):
        if False:
            while True:
                i = 10
        unescaped = '+ #%'
        plus_escaped = '%2B+%23%25'
        escaped = '%2B%20%23%25'
        self.assertEqual(url_escape(unescaped), plus_escaped)
        self.assertEqual(url_escape(unescaped, plus=False), escaped)
        self.assertEqual(url_unescape(plus_escaped), unescaped)
        self.assertEqual(url_unescape(escaped, plus=False), unescaped)
        self.assertEqual(url_unescape(plus_escaped, encoding=None), utf8(unescaped))
        self.assertEqual(url_unescape(escaped, encoding=None, plus=False), utf8(unescaped))

    def test_escape_return_types(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(type(xhtml_escape('foo')), str)
        self.assertEqual(type(xhtml_escape('foo')), unicode_type)

    def test_json_decode(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(json_decode(b'"foo"'), 'foo')
        self.assertEqual(json_decode('"foo"'), 'foo')
        self.assertEqual(json_decode(utf8('"é"')), 'é')

    def test_json_encode(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(json_decode(json_encode('é')), 'é')
        if bytes is str:
            self.assertEqual(json_decode(json_encode(utf8('é'))), 'é')
            self.assertRaises(UnicodeDecodeError, json_encode, b'\xe9')

    def test_squeeze(self):
        if False:
            return 10
        self.assertEqual(squeeze('sequences     of    whitespace   chars'), 'sequences of whitespace chars')

    def test_recursive_unicode(self):
        if False:
            i = 10
            return i + 15
        tests = {'dict': {b'foo': b'bar'}, 'list': [b'foo', b'bar'], 'tuple': (b'foo', b'bar'), 'bytes': b'foo'}
        self.assertEqual(recursive_unicode(tests['dict']), {'foo': 'bar'})
        self.assertEqual(recursive_unicode(tests['list']), ['foo', 'bar'])
        self.assertEqual(recursive_unicode(tests['tuple']), ('foo', 'bar'))
        self.assertEqual(recursive_unicode(tests['bytes']), 'foo')