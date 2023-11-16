from twisted.trial import unittest

def reppy_available():
    if False:
        while True:
            i = 10
    try:
        from reppy.robots import Robots
    except ImportError:
        return False
    return True

def rerp_available():
    if False:
        while True:
            i = 10
    try:
        from robotexclusionrulesparser import RobotExclusionRulesParser
    except ImportError:
        return False
    return True

def protego_available():
    if False:
        while True:
            i = 10
    try:
        from protego import Protego
    except ImportError:
        return False
    return True

class BaseRobotParserTest:

    def _setUp(self, parser_cls):
        if False:
            while True:
                i = 10
        self.parser_cls = parser_cls

    def test_allowed(self):
        if False:
            for i in range(10):
                print('nop')
        robotstxt_robotstxt_body = 'User-agent: * \nDisallow: /disallowed \nAllow: /allowed \nCrawl-delay: 10'.encode('utf-8')
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertTrue(rp.allowed('https://www.site.local/allowed', '*'))
        self.assertFalse(rp.allowed('https://www.site.local/disallowed', '*'))

    def test_allowed_wildcards(self):
        if False:
            return 10
        robotstxt_robotstxt_body = 'User-agent: first\n                                Disallow: /disallowed/*/end$\n\n                                User-agent: second\n                                Allow: /*allowed\n                                Disallow: /\n                                '.encode('utf-8')
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertTrue(rp.allowed('https://www.site.local/disallowed', 'first'))
        self.assertFalse(rp.allowed('https://www.site.local/disallowed/xyz/end', 'first'))
        self.assertFalse(rp.allowed('https://www.site.local/disallowed/abc/end', 'first'))
        self.assertTrue(rp.allowed('https://www.site.local/disallowed/xyz/endinglater', 'first'))
        self.assertTrue(rp.allowed('https://www.site.local/allowed', 'second'))
        self.assertTrue(rp.allowed('https://www.site.local/is_still_allowed', 'second'))
        self.assertTrue(rp.allowed('https://www.site.local/is_allowed_too', 'second'))

    def test_length_based_precedence(self):
        if False:
            i = 10
            return i + 15
        robotstxt_robotstxt_body = 'User-agent: * \nDisallow: / \nAllow: /page'.encode('utf-8')
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertTrue(rp.allowed('https://www.site.local/page', '*'))

    def test_order_based_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        robotstxt_robotstxt_body = 'User-agent: * \nDisallow: / \nAllow: /page'.encode('utf-8')
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertFalse(rp.allowed('https://www.site.local/page', '*'))

    def test_empty_response(self):
        if False:
            return 10
        "empty response should equal 'allow all'"
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=b'')
        self.assertTrue(rp.allowed('https://site.local/', '*'))
        self.assertTrue(rp.allowed('https://site.local/', 'chrome'))
        self.assertTrue(rp.allowed('https://site.local/index.html', '*'))
        self.assertTrue(rp.allowed('https://site.local/disallowed', '*'))

    def test_garbage_response(self):
        if False:
            while True:
                i = 10
        "garbage response should be discarded, equal 'allow all'"
        robotstxt_robotstxt_body = b'GIF89a\xd3\x00\xfe\x00\xa2'
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertTrue(rp.allowed('https://site.local/', '*'))
        self.assertTrue(rp.allowed('https://site.local/', 'chrome'))
        self.assertTrue(rp.allowed('https://site.local/index.html', '*'))
        self.assertTrue(rp.allowed('https://site.local/disallowed', '*'))

    def test_unicode_url_and_useragent(self):
        if False:
            return 10
        robotstxt_robotstxt_body = '\n        User-Agent: *\n        Disallow: /admin/\n        Disallow: /static/\n        # taken from https://en.wikipedia.org/robots.txt\n        Disallow: /wiki/K%C3%A4ytt%C3%A4j%C3%A4:\n        Disallow: /wiki/Käyttäjä:\n\n        User-Agent: UnicödeBöt\n        Disallow: /some/randome/page.html'.encode('utf-8')
        rp = self.parser_cls.from_crawler(crawler=None, robotstxt_body=robotstxt_robotstxt_body)
        self.assertTrue(rp.allowed('https://site.local/', '*'))
        self.assertFalse(rp.allowed('https://site.local/admin/', '*'))
        self.assertFalse(rp.allowed('https://site.local/static/', '*'))
        self.assertTrue(rp.allowed('https://site.local/admin/', 'UnicödeBöt'))
        self.assertFalse(rp.allowed('https://site.local/wiki/K%C3%A4ytt%C3%A4j%C3%A4:', '*'))
        self.assertFalse(rp.allowed('https://site.local/wiki/Käyttäjä:', '*'))
        self.assertTrue(rp.allowed('https://site.local/some/randome/page.html', '*'))
        self.assertFalse(rp.allowed('https://site.local/some/randome/page.html', 'UnicödeBöt'))

class PythonRobotParserTest(BaseRobotParserTest, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from scrapy.robotstxt import PythonRobotParser
        super()._setUp(PythonRobotParser)

    def test_length_based_precedence(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('RobotFileParser does not support length based directives precedence.')

    def test_allowed_wildcards(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('RobotFileParser does not support wildcards.')

class ReppyRobotParserTest(BaseRobotParserTest, unittest.TestCase):
    if not reppy_available():
        skip = 'Reppy parser is not installed'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from scrapy.robotstxt import ReppyRobotParser
        super()._setUp(ReppyRobotParser)

    def test_order_based_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('Reppy does not support order based directives precedence.')

class RerpRobotParserTest(BaseRobotParserTest, unittest.TestCase):
    if not rerp_available():
        skip = 'Rerp parser is not installed'

    def setUp(self):
        if False:
            while True:
                i = 10
        from scrapy.robotstxt import RerpRobotParser
        super()._setUp(RerpRobotParser)

    def test_length_based_precedence(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('Rerp does not support length based directives precedence.')

class ProtegoRobotParserTest(BaseRobotParserTest, unittest.TestCase):
    if not protego_available():
        skip = 'Protego parser is not installed'

    def setUp(self):
        if False:
            print('Hello World!')
        from scrapy.robotstxt import ProtegoRobotParser
        super()._setUp(ProtegoRobotParser)

    def test_order_based_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('Protego does not support order based directives precedence.')