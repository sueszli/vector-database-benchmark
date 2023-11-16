import io
import os
import threading
import unittest
import urllib.robotparser
from test import support
from test.support import socket_helper
from test.support import threading_helper
from http.server import BaseHTTPRequestHandler, HTTPServer

class BaseRobotTest:
    robots_txt = ''
    agent = 'test_robotparser'
    good = []
    bad = []
    site_maps = None

    def setUp(self):
        if False:
            print('Hello World!')
        lines = io.StringIO(self.robots_txt).readlines()
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.parse(lines)

    def get_agent_and_url(self, url):
        if False:
            return 10
        if isinstance(url, tuple):
            (agent, url) = url
            return (agent, url)
        return (self.agent, url)

    def test_good_urls(self):
        if False:
            i = 10
            return i + 15
        for url in self.good:
            (agent, url) = self.get_agent_and_url(url)
            with self.subTest(url=url, agent=agent):
                self.assertTrue(self.parser.can_fetch(agent, url))

    def test_bad_urls(self):
        if False:
            return 10
        for url in self.bad:
            (agent, url) = self.get_agent_and_url(url)
            with self.subTest(url=url, agent=agent):
                self.assertFalse(self.parser.can_fetch(agent, url))

    def test_site_maps(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.parser.site_maps(), self.site_maps)

class UserAgentWildcardTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nDisallow: /cyberworld/map/ # This is an infinite virtual URL space\nDisallow: /tmp/ # these will soon disappear\nDisallow: /foo.html\n    '
    good = ['/', '/test.html']
    bad = ['/cyberworld/map/index.html', '/tmp/xxx', '/foo.html']

class CrawlDelayAndCustomAgentTest(BaseRobotTest, unittest.TestCase):
    robots_txt = '# robots.txt for http://www.example.com/\n\nUser-agent: *\nCrawl-delay: 1\nRequest-rate: 3/15\nDisallow: /cyberworld/map/ # This is an infinite virtual URL space\n\n# Cybermapper knows where to go.\nUser-agent: cybermapper\nDisallow:\n    '
    good = ['/', '/test.html', ('cybermapper', '/cyberworld/map/index.html')]
    bad = ['/cyberworld/map/index.html']

class SitemapTest(BaseRobotTest, unittest.TestCase):
    robots_txt = '# robots.txt for http://www.example.com/\n\nUser-agent: *\nSitemap: http://www.gstatic.com/s2/sitemaps/profiles-sitemap.xml\nSitemap: http://www.google.com/hostednews/sitemap_index.xml\nRequest-rate: 3/15\nDisallow: /cyberworld/map/ # This is an infinite virtual URL space\n\n    '
    good = ['/', '/test.html']
    bad = ['/cyberworld/map/index.html']
    site_maps = ['http://www.gstatic.com/s2/sitemaps/profiles-sitemap.xml', 'http://www.google.com/hostednews/sitemap_index.xml']

class RejectAllRobotsTest(BaseRobotTest, unittest.TestCase):
    robots_txt = '# go away\nUser-agent: *\nDisallow: /\n    '
    good = []
    bad = ['/cyberworld/map/index.html', '/', '/tmp/']

class BaseRequestRateTest(BaseRobotTest):
    request_rate = None
    crawl_delay = None

    def test_request_rate(self):
        if False:
            print('Hello World!')
        parser = self.parser
        for url in self.good + self.bad:
            (agent, url) = self.get_agent_and_url(url)
            with self.subTest(url=url, agent=agent):
                self.assertEqual(parser.crawl_delay(agent), self.crawl_delay)
                parsed_request_rate = parser.request_rate(agent)
                self.assertEqual(parsed_request_rate, self.request_rate)
                if self.request_rate is not None:
                    self.assertIsInstance(parsed_request_rate, urllib.robotparser.RequestRate)
                    self.assertEqual(parsed_request_rate.requests, self.request_rate.requests)
                    self.assertEqual(parsed_request_rate.seconds, self.request_rate.seconds)

class EmptyFileTest(BaseRequestRateTest, unittest.TestCase):
    robots_txt = ''
    good = ['/foo']

class CrawlDelayAndRequestRateTest(BaseRequestRateTest, unittest.TestCase):
    robots_txt = 'User-agent: figtree\nCrawl-delay: 3\nRequest-rate: 9/30\nDisallow: /tmp\nDisallow: /a%3cd.html\nDisallow: /a%2fb.html\nDisallow: /%7ejoe/index.html\n    '
    agent = 'figtree'
    request_rate = urllib.robotparser.RequestRate(9, 30)
    crawl_delay = 3
    good = [('figtree', '/foo.html')]
    bad = ['/tmp', '/tmp.html', '/tmp/a.html', '/a%3cd.html', '/a%3Cd.html', '/a%2fb.html', '/~joe/index.html']

class DifferentAgentTest(CrawlDelayAndRequestRateTest):
    agent = 'FigTree Robot libwww-perl/5.04'

class InvalidRequestRateTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nDisallow: /tmp/\nDisallow: /a%3Cd.html\nDisallow: /a/b.html\nDisallow: /%7ejoe/index.html\nCrawl-delay: 3\nRequest-rate: 9/banana\n    '
    good = ['/tmp']
    bad = ['/tmp/', '/tmp/a.html', '/a%3cd.html', '/a%3Cd.html', '/a/b.html', '/%7Ejoe/index.html']
    crawl_delay = 3

class InvalidCrawlDelayTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-Agent: *\nDisallow: /.\nCrawl-delay: pears\n    '
    good = ['/foo.html']
    bad = []

class AnotherInvalidRequestRateTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: Googlebot\nAllow: /folder1/myfile.html\nDisallow: /folder1/\nRequest-rate: whale/banana\n    '
    agent = 'Googlebot'
    good = ['/folder1/myfile.html']
    bad = ['/folder1/anotherfile.html']

class UserAgentOrderingTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: Googlebot\nDisallow: /\n\nUser-agent: Googlebot-Mobile\nAllow: /\n    '
    agent = 'Googlebot'
    bad = ['/something.jpg']

class UserAgentGoogleMobileTest(UserAgentOrderingTest):
    agent = 'Googlebot-Mobile'

class GoogleURLOrderingTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: Googlebot\nAllow: /folder1/myfile.html\nDisallow: /folder1/\n    '
    agent = 'googlebot'
    good = ['/folder1/myfile.html']
    bad = ['/folder1/anotherfile.html']

class DisallowQueryStringTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nDisallow: /some/path?name=value\n    '
    good = ['/some/path']
    bad = ['/some/path?name=value']

class UseFirstUserAgentWildcardTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nDisallow: /some/path\n\nUser-agent: *\nDisallow: /another/path\n    '
    good = ['/another/path']
    bad = ['/some/path']

class EmptyQueryStringTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nAllow: /some/path?\nDisallow: /another/path?\n    '
    good = ['/some/path?']
    bad = ['/another/path?']

class DefaultEntryTest(BaseRequestRateTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nCrawl-delay: 1\nRequest-rate: 3/15\nDisallow: /cyberworld/map/\n    '
    request_rate = urllib.robotparser.RequestRate(3, 15)
    crawl_delay = 1
    good = ['/', '/test.html']
    bad = ['/cyberworld/map/index.html']

class StringFormattingTest(BaseRobotTest, unittest.TestCase):
    robots_txt = 'User-agent: *\nCrawl-delay: 1\nRequest-rate: 3/15\nDisallow: /cyberworld/map/ # This is an infinite virtual URL space\n\n# Cybermapper knows where to go.\nUser-agent: cybermapper\nDisallow: /some/path\n    '
    expected_output = 'User-agent: cybermapper\nDisallow: /some/path\n\nUser-agent: *\nCrawl-delay: 1\nRequest-rate: 3/15\nDisallow: /cyberworld/map/'

    def test_string_formatting(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(self.parser), self.expected_output)

class RobotHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_error(403, 'Forbidden access')

    def log_message(self, format, *args):
        if False:
            while True:
                i = 10
        pass

class PasswordProtectedSiteTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.addCleanup(urllib.request.urlcleanup)
        self.server = HTTPServer((socket_helper.HOST, 0), RobotHandler)
        self.t = threading.Thread(name='HTTPServer serving', target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.t.daemon = True
        self.t.start()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.server.shutdown()
        self.t.join()
        self.server.server_close()

    @threading_helper.reap_threads
    def testPasswordProtectedSite(self):
        if False:
            for i in range(10):
                print('nop')
        addr = self.server.server_address
        url = 'http://' + socket_helper.HOST + ':' + str(addr[1])
        robots_url = url + '/robots.txt'
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(url)
        parser.read()
        self.assertFalse(parser.can_fetch('*', robots_url))

class NetworkTestCase(unittest.TestCase):
    base_url = 'http://www.pythontest.net/'
    robots_txt = '{}elsewhere/robots.txt'.format(base_url)

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        support.requires('network')
        with socket_helper.transient_internet(cls.base_url):
            cls.parser = urllib.robotparser.RobotFileParser(cls.robots_txt)
            cls.parser.read()

    def url(self, path):
        if False:
            for i in range(10):
                print('nop')
        return '{}{}{}'.format(self.base_url, path, '/' if not os.path.splitext(path)[1] else '')

    def test_basic(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.parser.disallow_all)
        self.assertFalse(self.parser.allow_all)
        self.assertGreater(self.parser.mtime(), 0)
        self.assertFalse(self.parser.crawl_delay('*'))
        self.assertFalse(self.parser.request_rate('*'))

    def test_can_fetch(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.parser.can_fetch('*', self.url('elsewhere')))
        self.assertFalse(self.parser.can_fetch('Nutch', self.base_url))
        self.assertFalse(self.parser.can_fetch('Nutch', self.url('brian')))
        self.assertFalse(self.parser.can_fetch('Nutch', self.url('webstats')))
        self.assertFalse(self.parser.can_fetch('*', self.url('webstats')))
        self.assertTrue(self.parser.can_fetch('*', self.base_url))

    def test_read_404(self):
        if False:
            print('Hello World!')
        parser = urllib.robotparser.RobotFileParser(self.url('i-robot.txt'))
        parser.read()
        self.assertTrue(parser.allow_all)
        self.assertFalse(parser.disallow_all)
        self.assertEqual(parser.mtime(), 0)
        self.assertIsNone(parser.crawl_delay('*'))
        self.assertIsNone(parser.request_rate('*'))
if __name__ == '__main__':
    unittest.main()