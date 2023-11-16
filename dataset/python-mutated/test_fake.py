import unittest
import pytest
from fake_useragent import VERSION, FakeUserAgent, UserAgent, settings

class TestFake(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def test_fake_init(self):
        if False:
            print('Hello World!')
        ua = UserAgent()
        self.assertTrue(ua.chrome)
        self.assertIsInstance(ua.chrome, str)
        self.assertTrue(ua.edge)
        self.assertIsInstance(ua.edge, str)
        self.assertTrue(ua['internet explorer'])
        self.assertIsInstance(ua['internet explorer'], str)
        self.assertTrue(ua.firefox)
        self.assertIsInstance(ua.firefox, str)
        self.assertTrue(ua.safari)
        self.assertIsInstance(ua.safari, str)
        self.assertTrue(ua.random)
        self.assertIsInstance(ua.random, str)
        self.assertTrue(ua.getChrome)
        self.assertIsInstance(ua.getChrome, dict)
        self.assertTrue(ua.getEdge)
        self.assertIsInstance(ua.getEdge, dict)
        self.assertTrue(ua.getFirefox)
        self.assertIsInstance(ua.getFirefox, dict)
        self.assertTrue(ua.getSafari)
        self.assertIsInstance(ua.getSafari, dict)
        self.assertTrue(ua.getRandom)
        self.assertIsInstance(ua.getRandom, dict)

    def test_fake_probe_user_agent_browsers(self):
        if False:
            return 10
        ua = UserAgent()
        ua.edge
        ua.google
        ua.chrome
        ua.googlechrome
        ua.google_chrome
        ua['google chrome']
        ua.firefox
        ua.ff
        ua.safari
        ua.random
        ua['random']
        ua.getEdge
        ua.getChrome
        ua.getFirefox
        ua.getSafari
        ua.getRandom

    def test_fake_data_browser_type(self):
        if False:
            for i in range(10):
                print('nop')
        ua = UserAgent()
        assert isinstance(ua.data_browsers, list)

    def test_fake_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        fallback = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        ua = UserAgent()
        self.assertEqual(ua.non_existing, fallback)
        self.assertEqual(ua['non_existing'], fallback)

    def test_fake_fallback_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        fallback = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        ua = UserAgent()
        self.assertIsInstance(ua.getBrowser('non_existing'), dict)
        self.assertEqual(ua.getBrowser('non_existing').get('useragent'), fallback)

    def test_fake_fallback_str_types(self):
        if False:
            return 10
        with pytest.raises(AssertionError):
            UserAgent(fallback=True)

    def test_fake_browser_str_or_list_types(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AssertionError):
            UserAgent(browsers=52)

    def test_fake_os_str_or_list_types(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AssertionError):
            UserAgent(os=23.4)

    def test_fake_percentage_float_types(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(AssertionError):
            UserAgent(min_percentage='')

    def test_fake_safe_attrs_iterable_str_types(self):
        if False:
            print('Hello World!')
        with pytest.raises(AssertionError):
            UserAgent(safe_attrs={})
        with pytest.raises(AssertionError):
            UserAgent(safe_attrs=[66])

    def test_fake_safe_attrs(self):
        if False:
            return 10
        ua = UserAgent(safe_attrs=('__injections__',))
        with pytest.raises(AttributeError):
            ua.__injections__

    def test_fake_version(self):
        if False:
            while True:
                i = 10
        assert VERSION == settings.__version__

    def test_fake_aliases(self):
        if False:
            for i in range(10):
                print('nop')
        assert FakeUserAgent is UserAgent