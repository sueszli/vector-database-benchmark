import unittest
from r2.lib.utils import UrlParser
from r2.tests import RedditTestCase
from pylons import app_globals as g

class TestIsRedditURL(RedditTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch_g(offsite_subdomains=['blog'])

    def _is_safe_reddit_url(self, url, subreddit=None):
        if False:
            for i in range(10):
                print('nop')
        web_safe = UrlParser(url).is_web_safe_url()
        return web_safe and UrlParser(url).is_reddit_url(subreddit)

    def assertIsSafeRedditUrl(self, url, subreddit=None):
        if False:
            while True:
                i = 10
        self.assertTrue(self._is_safe_reddit_url(url, subreddit))

    def assertIsNotSafeRedditUrl(self, url, subreddit=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self._is_safe_reddit_url(url, subreddit))

    def test_normal_urls(self):
        if False:
            i = 10
            return i + 15
        self.assertIsSafeRedditUrl('https://%s/' % g.domain)
        self.assertIsSafeRedditUrl('https://en.%s/' % g.domain)
        self.assertIsSafeRedditUrl('https://foobar.baz.%s/quux/?a' % g.domain)
        self.assertIsSafeRedditUrl('#anchorage')
        self.assertIsSafeRedditUrl('?path_relative_queries')
        self.assertIsSafeRedditUrl('/')
        self.assertIsSafeRedditUrl('/cats')
        self.assertIsSafeRedditUrl('/cats/')
        self.assertIsSafeRedditUrl('/cats/#maru')
        self.assertIsSafeRedditUrl('//foobaz.%s/aa/baz#quux' % g.domain)
        self.assertIsSafeRedditUrl('path_relative_subpath.com')
        self.assertIsNotSafeRedditUrl('http://blog.%s/' % g.domain)
        self.assertIsNotSafeRedditUrl('http://foo.blog.%s/' % g.domain)

    def test_incorrect_anchoring(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNotSafeRedditUrl('http://www.%s.whatever.com/' % g.domain)

    def test_protocol_relative(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNotSafeRedditUrl('//foobaz.example.com/aa/baz#quux')

    def test_weird_protocols(self):
        if False:
            while True:
                i = 10
        self.assertIsNotSafeRedditUrl('javascript://%s/%%0d%%0aalert(1)' % g.domain)
        self.assertIsNotSafeRedditUrl('hackery:whatever')

    def test_http_auth(self):
        if False:
            return 10
        self.assertIsNotSafeRedditUrl('http://foo:bar@/example.com/')

    def test_browser_quirks(self):
        if False:
            print('Hello World!')
        self.assertIsNotSafeRedditUrl('/\x00/example.com')
        self.assertIsNotSafeRedditUrl('\t//example.com')
        self.assertIsNotSafeRedditUrl(' http://example.com/')
        self.assertIsNotSafeRedditUrl('////example.com/')
        self.assertIsNotSafeRedditUrl('//////example.com/')
        self.assertIsNotSafeRedditUrl('http:///example.com/')
        self.assertIsNotSafeRedditUrl('/\\example.com/')
        self.assertIsNotSafeRedditUrl('http://\\\\example.com\\a.%s/foo' % g.domain)
        self.assertIsNotSafeRedditUrl('///\\example.com/')
        self.assertIsNotSafeRedditUrl('\\\\example.com')
        self.assertIsNotSafeRedditUrl('/\x00//\\example.com/')
        self.assertIsNotSafeRedditUrl('\tjavascript://%s/%%0d%%0aalert(1)' % g.domain)
        self.assertIsNotSafeRedditUrl('http://\texample.com\\%s/foo' % g.domain)

    def test_url_mutation(self):
        if False:
            return 10
        u = UrlParser('http://example.com/')
        u.hostname = g.domain
        self.assertTrue(u.is_reddit_url())
        u = UrlParser('http://%s/' % g.domain)
        u.hostname = 'example.com'
        self.assertFalse(u.is_reddit_url())

    def test_nbsp_allowances(self):
        if False:
            while True:
                i = 10
        self.assertIsNotSafeRedditUrl('http://\xa0.%s/' % g.domain)
        self.assertIsNotSafeRedditUrl('\xa0http://%s/' % g.domain)
        self.assertIsSafeRedditUrl('http://%s/\xa0' % g.domain)
        self.assertIsSafeRedditUrl('/foo/bar/\xa0baz')
        self.assertIsNotSafeRedditUrl(u'http://\xa0.%s/' % g.domain)
        self.assertIsNotSafeRedditUrl(u'\xa0http://%s/' % g.domain)
        self.assertIsSafeRedditUrl(u'http://%s/\xa0' % g.domain)
        self.assertIsSafeRedditUrl(u'/foo/bar/\xa0baz')

class TestSwitchSubdomainByExtension(RedditTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.patch_g(domain='reddit.com', domain_prefix='www')

    def test_normal_urls(self):
        if False:
            i = 10
            return i + 15
        u = UrlParser('http://www.reddit.com/r/redditdev')
        u.switch_subdomain_by_extension('compact')
        result = u.unparse()
        self.assertEquals('http://i.reddit.com/r/redditdev', result)
        u = UrlParser(result)
        u.switch_subdomain_by_extension('mobile')
        result = u.unparse()
        self.assertEquals('http://simple.reddit.com/r/redditdev', result)

    def test_default_prefix(self):
        if False:
            i = 10
            return i + 15
        u = UrlParser('http://i.reddit.com/r/redditdev')
        u.switch_subdomain_by_extension()
        self.assertEquals('http://www.reddit.com/r/redditdev', u.unparse())
        u = UrlParser('http://i.reddit.com/r/redditdev')
        u.switch_subdomain_by_extension('does-not-exist')
        self.assertEquals('http://www.reddit.com/r/redditdev', u.unparse())

class TestPathExtension(unittest.TestCase):

    def test_no_path(self):
        if False:
            for i in range(10):
                print('nop')
        u = UrlParser('http://example.com')
        self.assertEquals('', u.path_extension())

    def test_directory(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/')
        self.assertEquals('', u.path_extension())
        u = UrlParser('http://example.com/foo/')
        self.assertEquals('', u.path_extension())

    def test_no_extension(self):
        if False:
            return 10
        u = UrlParser('http://example.com/a')
        self.assertEquals('', u.path_extension())

    def test_root_file(self):
        if False:
            print('Hello World!')
        u = UrlParser('http://example.com/a.jpg')
        self.assertEquals('jpg', u.path_extension())

    def test_nested_file(self):
        if False:
            i = 10
            return i + 15
        u = UrlParser('http://example.com/foo/a.jpg')
        self.assertEquals('jpg', u.path_extension())

    def test_empty_extension(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/a.')
        self.assertEquals('', u.path_extension())

    def test_two_extensions(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/a.jpg.exe')
        self.assertEquals('exe', u.path_extension())

    def test_only_extension(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/.bashrc')
        self.assertEquals('bashrc', u.path_extension())

class TestEquality(unittest.TestCase):

    def test_different_objects(self):
        if False:
            i = 10
            return i + 15
        u = UrlParser('http://example.com')
        self.assertNotEquals(u, None)

    def test_different_protocols(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com')
        u2 = UrlParser('https://example.com')
        self.assertNotEquals(u, u2)

    def test_different_domains(self):
        if False:
            i = 10
            return i + 15
        u = UrlParser('http://example.com')
        u2 = UrlParser('http://example.org')
        self.assertNotEquals(u, u2)

    def test_different_ports(self):
        if False:
            for i in range(10):
                print('nop')
        u = UrlParser('http://example.com')
        u2 = UrlParser('http://example.com:8000')
        u3 = UrlParser('http://example.com:8008')
        self.assertNotEquals(u, u2)
        self.assertNotEquals(u2, u3)

    def test_different_paths(self):
        if False:
            return 10
        u = UrlParser('http://example.com')
        u2 = UrlParser('http://example.com/a')
        u3 = UrlParser('http://example.com/b')
        self.assertNotEquals(u, u2)
        self.assertNotEquals(u2, u3)

    def test_different_params(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/')
        u2 = UrlParser('http://example.com/;foo')
        u3 = UrlParser('http://example.com/;bar')
        self.assertNotEquals(u, u2)
        self.assertNotEquals(u2, u3)

    def test_different_queries(self):
        if False:
            print('Hello World!')
        u = UrlParser('http://example.com/')
        u2 = UrlParser('http://example.com/?foo')
        u3 = UrlParser('http://example.com/?foo=bar')
        self.assertNotEquals(u, u2)
        self.assertNotEquals(u2, u3)

    def test_different_fragments(self):
        if False:
            for i in range(10):
                print('nop')
        u = UrlParser('http://example.com/')
        u2 = UrlParser('http://example.com/#foo')
        u3 = UrlParser('http://example.com/#bar')
        self.assertNotEquals(u, u2)
        self.assertNotEquals(u2, u3)

    def test_same_url(self):
        if False:
            for i in range(10):
                print('nop')
        u = UrlParser('http://example.com:8000/a;b?foo=bar&bar=baz#spam')
        u2 = UrlParser('http://example.com:8000/a;b?bar=baz&foo=bar#spam')
        self.assertEquals(u, u2)
        u3 = UrlParser('')
        u3.scheme = 'http'
        u3.hostname = 'example.com'
        u3.port = 8000
        u3.path = '/a'
        u3.params = 'b'
        u3.update_query(foo='bar', bar='baz')
        u3.fragment = 'spam'
        self.assertEquals(u, u3)

    def test_integer_query_params(self):
        if False:
            while True:
                i = 10
        u = UrlParser('http://example.com/?page=1234')
        u2 = UrlParser('http://example.com/')
        u2.update_query(page=1234)
        self.assertEquals(u, u2)

    def test_unicode_query_params(self):
        if False:
            return 10
        u = UrlParser(u'http://example.com/?page=ｕｎｉｃｏｄｅ：（')
        u2 = UrlParser('http://example.com/')
        u2.update_query(page=u'ｕｎｉｃｏｄｅ：（')
        self.assertEquals(u, u2)