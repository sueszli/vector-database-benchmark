import logging
import os
import platform
import signal
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List
import pytest
from packaging.version import parse as parse_version
from pexpect.popen_spawn import PopenSpawn
from pytest import mark, raises
from twisted.internet import defer
from twisted.trial import unittest
from w3lib import __version__ as w3lib_version
from zope.interface.exceptions import MultipleInvalid
import scrapy
from scrapy.crawler import Crawler, CrawlerProcess, CrawlerRunner
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.extensions import telnet
from scrapy.extensions.throttle import AutoThrottle
from scrapy.settings import Settings, default_settings
from scrapy.spiderloader import SpiderLoader
from scrapy.utils.log import configure_logging, get_scrapy_root_handler
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.test import get_crawler
from tests.mockserver import MockServer, get_mockserver_env

class BaseCrawlerTest(unittest.TestCase):

    def assertOptionIsDefault(self, settings, key):
        if False:
            while True:
                i = 10
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings[key], getattr(default_settings, key))

class CrawlerTestCase(BaseCrawlerTest):

    def test_populate_spidercls_settings(self):
        if False:
            while True:
                i = 10
        spider_settings = {'TEST1': 'spider', 'TEST2': 'spider'}
        project_settings = {'TEST1': 'project', 'TEST3': 'project'}

        class CustomSettingsSpider(DefaultSpider):
            custom_settings = spider_settings
        settings = Settings()
        settings.setdict(project_settings, priority='project')
        crawler = Crawler(CustomSettingsSpider, settings)
        crawler._apply_settings()
        self.assertEqual(crawler.settings.get('TEST1'), 'spider')
        self.assertEqual(crawler.settings.get('TEST2'), 'spider')
        self.assertEqual(crawler.settings.get('TEST3'), 'project')
        self.assertFalse(settings.frozen)
        self.assertTrue(crawler.settings.frozen)

    def test_crawler_accepts_dict(self):
        if False:
            for i in range(10):
                print('nop')
        crawler = get_crawler(DefaultSpider, {'foo': 'bar'})
        self.assertEqual(crawler.settings['foo'], 'bar')
        self.assertOptionIsDefault(crawler.settings, 'RETRY_ENABLED')

    def test_crawler_accepts_None(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            crawler = Crawler(DefaultSpider)
        self.assertOptionIsDefault(crawler.settings, 'RETRY_ENABLED')

    def test_crawler_rejects_spider_objects(self):
        if False:
            i = 10
            return i + 15
        with raises(ValueError):
            Crawler(DefaultSpider())

    @defer.inlineCallbacks
    def test_crawler_crawl_twice_deprecated(self):
        if False:
            return 10
        crawler = Crawler(NoRequestsSpider)
        yield crawler.crawl()
        with pytest.warns(ScrapyDeprecationWarning, match='Running Crawler.crawl\\(\\) more than once is deprecated'):
            yield crawler.crawl()

class SpiderSettingsTestCase(unittest.TestCase):

    def test_spider_custom_settings(self):
        if False:
            return 10

        class MySpider(scrapy.Spider):
            name = 'spider'
            custom_settings = {'AUTOTHROTTLE_ENABLED': True}
        crawler = get_crawler(MySpider)
        enabled_exts = [e.__class__ for e in crawler.extensions.middlewares]
        self.assertIn(AutoThrottle, enabled_exts)

class CrawlerLoggingTestCase(unittest.TestCase):

    def test_no_root_handler_installed(self):
        if False:
            return 10
        handler = get_scrapy_root_handler()
        if handler is not None:
            logging.root.removeHandler(handler)

        class MySpider(scrapy.Spider):
            name = 'spider'
        get_crawler(MySpider)
        assert get_scrapy_root_handler() is None

    def test_spider_custom_settings_log_level(self):
        if False:
            for i in range(10):
                print('nop')
        log_file = Path(self.mktemp())
        log_file.write_text('previous message\n', encoding='utf-8')

        class MySpider(scrapy.Spider):
            name = 'spider'
            custom_settings = {'LOG_LEVEL': 'INFO', 'LOG_FILE': str(log_file), 'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7', 'TELNETCONSOLE_ENABLED': telnet.TWISTED_CONCH_AVAILABLE}
        configure_logging()
        self.assertEqual(get_scrapy_root_handler().level, logging.DEBUG)
        crawler = get_crawler(MySpider)
        self.assertEqual(get_scrapy_root_handler().level, logging.INFO)
        info_count = crawler.stats.get_value('log_count/INFO')
        logging.debug('debug message')
        logging.info('info message')
        logging.warning('warning message')
        logging.error('error message')
        logged = log_file.read_text(encoding='utf-8')
        self.assertIn('previous message', logged)
        self.assertNotIn('debug message', logged)
        self.assertIn('info message', logged)
        self.assertIn('warning message', logged)
        self.assertIn('error message', logged)
        self.assertEqual(crawler.stats.get_value('log_count/ERROR'), 1)
        self.assertEqual(crawler.stats.get_value('log_count/WARNING'), 1)
        self.assertEqual(crawler.stats.get_value('log_count/INFO') - info_count, 1)
        self.assertEqual(crawler.stats.get_value('log_count/DEBUG', 0), 0)

    def test_spider_custom_settings_log_append(self):
        if False:
            for i in range(10):
                print('nop')
        log_file = Path(self.mktemp())
        log_file.write_text('previous message\n', encoding='utf-8')

        class MySpider(scrapy.Spider):
            name = 'spider'
            custom_settings = {'LOG_FILE': str(log_file), 'LOG_FILE_APPEND': False, 'TELNETCONSOLE_ENABLED': telnet.TWISTED_CONCH_AVAILABLE}
        configure_logging()
        get_crawler(MySpider)
        logging.debug('debug message')
        logged = log_file.read_text(encoding='utf-8')
        self.assertNotIn('previous message', logged)
        self.assertIn('debug message', logged)

class SpiderLoaderWithWrongInterface:

    def unneeded_method(self):
        if False:
            while True:
                i = 10
        pass

class CustomSpiderLoader(SpiderLoader):
    pass

class CrawlerRunnerTestCase(BaseCrawlerTest):

    def test_spider_manager_verify_interface(self):
        if False:
            print('Hello World!')
        settings = Settings({'SPIDER_LOADER_CLASS': SpiderLoaderWithWrongInterface})
        self.assertRaises(MultipleInvalid, CrawlerRunner, settings)

    def test_crawler_runner_accepts_dict(self):
        if False:
            for i in range(10):
                print('nop')
        runner = CrawlerRunner({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_runner_accepts_None(self):
        if False:
            print('Hello World!')
        runner = CrawlerRunner()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

class CrawlerProcessTest(BaseCrawlerTest):

    def test_crawler_process_accepts_dict(self):
        if False:
            return 10
        runner = CrawlerProcess({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_process_accepts_None(self):
        if False:
            print('Hello World!')
        runner = CrawlerProcess()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

class ExceptionSpider(scrapy.Spider):
    name = 'exception'

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise ValueError('Exception in from_crawler method')

class NoRequestsSpider(scrapy.Spider):
    name = 'no_request'

    def start_requests(self):
        if False:
            for i in range(10):
                print('nop')
        return []

@mark.usefixtures('reactor_pytest')
class CrawlerRunnerHasSpider(unittest.TestCase):

    def _runner(self):
        if False:
            while True:
                i = 10
        return CrawlerRunner({'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'})

    @defer.inlineCallbacks
    def test_crawler_runner_bootstrap_successful(self):
        if False:
            print('Hello World!')
        runner = self._runner()
        yield runner.crawl(NoRequestsSpider)
        self.assertFalse(runner.bootstrap_failed)

    @defer.inlineCallbacks
    def test_crawler_runner_bootstrap_successful_for_several(self):
        if False:
            return 10
        runner = self._runner()
        yield runner.crawl(NoRequestsSpider)
        yield runner.crawl(NoRequestsSpider)
        self.assertFalse(runner.bootstrap_failed)

    @defer.inlineCallbacks
    def test_crawler_runner_bootstrap_failed(self):
        if False:
            return 10
        runner = self._runner()
        try:
            yield runner.crawl(ExceptionSpider)
        except ValueError:
            pass
        else:
            self.fail('Exception should be raised from spider')
        self.assertTrue(runner.bootstrap_failed)

    @defer.inlineCallbacks
    def test_crawler_runner_bootstrap_failed_for_several(self):
        if False:
            for i in range(10):
                print('nop')
        runner = self._runner()
        try:
            yield runner.crawl(ExceptionSpider)
        except ValueError:
            pass
        else:
            self.fail('Exception should be raised from spider')
        yield runner.crawl(NoRequestsSpider)
        self.assertTrue(runner.bootstrap_failed)

    @defer.inlineCallbacks
    def test_crawler_runner_asyncio_enabled_true(self):
        if False:
            return 10
        if self.reactor_pytest == 'asyncio':
            CrawlerRunner(settings={'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor', 'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'})
        else:
            msg = 'The installed reactor \\(.*?\\) does not match the requested one \\(.*?\\)'
            with self.assertRaisesRegex(Exception, msg):
                runner = CrawlerRunner(settings={'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor', 'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'})
                yield runner.crawl(NoRequestsSpider)

class ScriptRunnerMixin:
    script_dir: Path
    cwd = os.getcwd()

    def get_script_args(self, script_name: str, *script_args: str) -> List[str]:
        if False:
            print('Hello World!')
        script_path = self.script_dir / script_name
        return [sys.executable, str(script_path)] + list(script_args)

    def run_script(self, script_name: str, *script_args: str) -> str:
        if False:
            i = 10
            return i + 15
        args = self.get_script_args(script_name, *script_args)
        p = subprocess.Popen(args, env=get_mockserver_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        return stderr.decode('utf-8')

class CrawlerProcessSubprocess(ScriptRunnerMixin, unittest.TestCase):
    script_dir = Path(__file__).parent.resolve() / 'CrawlerProcess'

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('simple.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    def test_multi(self):
        if False:
            print('Hello World!')
        log = self.run_script('multi.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertNotIn('ReactorAlreadyInstalledError', log)

    def test_reactor_default(self):
        if False:
            return 10
        log = self.run_script('reactor_default.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertNotIn('ReactorAlreadyInstalledError', log)

    def test_reactor_default_twisted_reactor_select(self):
        if False:
            i = 10
            return i + 15
        log = self.run_script('reactor_default_twisted_reactor_select.py')
        if platform.system() in ['Windows', 'Darwin']:
            self.assertIn('Spider closed (finished)', log)
        else:
            self.assertNotIn('Spider closed (finished)', log)
            self.assertIn('does not match the requested one (twisted.internet.selectreactor.SelectReactor)', log)

    def test_reactor_select(self):
        if False:
            return 10
        log = self.run_script('reactor_select.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('ReactorAlreadyInstalledError', log)

    def test_reactor_select_twisted_reactor_select(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('reactor_select_twisted_reactor_select.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('ReactorAlreadyInstalledError', log)

    def test_reactor_select_subclass_twisted_reactor_select(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('reactor_select_subclass_twisted_reactor_select.py')
        self.assertNotIn('Spider closed (finished)', log)
        self.assertIn('does not match the requested one (twisted.internet.selectreactor.SelectReactor)', log)

    def test_asyncio_enabled_no_reactor(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('asyncio_enabled_no_reactor.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    def test_asyncio_enabled_reactor(self):
        if False:
            i = 10
            return i + 15
        log = self.run_script('asyncio_enabled_reactor.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    @mark.skipif(parse_version(w3lib_version) >= parse_version('2.0.0'), reason='w3lib 2.0.0 and later do not allow invalid domains.')
    def test_ipv6_default_name_resolver(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('default_name_resolver.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn("'downloader/exception_type_count/twisted.internet.error.DNSLookupError': 1,", log)
        self.assertIn('twisted.internet.error.DNSLookupError: DNS lookup failed: no results for hostname lookup: ::1.', log)

    def test_caching_hostname_resolver_ipv6(self):
        if False:
            while True:
                i = 10
        log = self.run_script('caching_hostname_resolver_ipv6.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertNotIn('twisted.internet.error.DNSLookupError', log)

    def test_caching_hostname_resolver_finite_execution(self):
        if False:
            while True:
                i = 10
        with MockServer() as mock_server:
            http_address = mock_server.http_address.replace('0.0.0.0', '127.0.0.1')
            log = self.run_script('caching_hostname_resolver.py', http_address)
            self.assertIn('Spider closed (finished)', log)
            self.assertNotIn('ERROR: Error downloading', log)
            self.assertNotIn('TimeoutError', log)
            self.assertNotIn('twisted.internet.error.DNSLookupError', log)

    def test_twisted_reactor_select(self):
        if False:
            while True:
                i = 10
        log = self.run_script('twisted_reactor_select.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.selectreactor.SelectReactor', log)

    @mark.skipif(platform.system() == 'Windows', reason='PollReactor is not supported on Windows')
    def test_twisted_reactor_poll(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('twisted_reactor_poll.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.pollreactor.PollReactor', log)

    def test_twisted_reactor_asyncio(self):
        if False:
            return 10
        log = self.run_script('twisted_reactor_asyncio.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    def test_twisted_reactor_asyncio_custom_settings(self):
        if False:
            while True:
                i = 10
        log = self.run_script('twisted_reactor_custom_settings.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    def test_twisted_reactor_asyncio_custom_settings_same(self):
        if False:
            while True:
                i = 10
        log = self.run_script('twisted_reactor_custom_settings_same.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)

    def test_twisted_reactor_asyncio_custom_settings_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('twisted_reactor_custom_settings_conflict.py')
        self.assertIn('Using reactor: twisted.internet.selectreactor.SelectReactor', log)
        self.assertIn('(twisted.internet.selectreactor.SelectReactor) does not match the requested one', log)

    @mark.requires_uvloop
    def test_custom_loop_asyncio(self):
        if False:
            i = 10
            return i + 15
        log = self.run_script('asyncio_custom_loop.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertIn('Using asyncio event loop: uvloop.Loop', log)

    @mark.requires_uvloop
    def test_custom_loop_asyncio_deferred_signal(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('asyncio_deferred_signal.py', 'uvloop.Loop')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertIn('Using asyncio event loop: uvloop.Loop', log)
        self.assertIn('async pipeline opened!', log)

    @mark.requires_uvloop
    def test_asyncio_enabled_reactor_same_loop(self):
        if False:
            print('Hello World!')
        log = self.run_script('asyncio_enabled_reactor_same_loop.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertIn('Using asyncio event loop: uvloop.Loop', log)

    @mark.requires_uvloop
    def test_asyncio_enabled_reactor_different_loop(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('asyncio_enabled_reactor_different_loop.py')
        self.assertNotIn('Spider closed (finished)', log)
        self.assertIn('does not match the one specified in the ASYNCIO_EVENT_LOOP setting (uvloop.Loop)', log)

    def test_default_loop_asyncio_deferred_signal(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('asyncio_deferred_signal.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('Using reactor: twisted.internet.asyncioreactor.AsyncioSelectorReactor', log)
        self.assertNotIn('Using asyncio event loop: uvloop.Loop', log)
        self.assertIn('async pipeline opened!', log)

    def test_args_change_settings(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('args_settings.py')
        self.assertIn('Spider closed (finished)', log)
        self.assertIn('The value of FOO is 42', log)

    def test_shutdown_graceful(self):
        if False:
            while True:
                i = 10
        sig = signal.SIGINT if sys.platform != 'win32' else signal.SIGBREAK
        args = self.get_script_args('sleeping.py', '-a', 'sleep=3')
        p = PopenSpawn(args, timeout=5)
        p.expect_exact('Spider opened')
        p.expect_exact('Crawled (200)')
        p.kill(sig)
        p.expect_exact('shutting down gracefully')
        p.expect_exact('Spider closed (shutdown)')
        p.wait()

    @defer.inlineCallbacks
    def test_shutdown_forced(self):
        if False:
            print('Hello World!')
        from twisted.internet import reactor
        sig = signal.SIGINT if sys.platform != 'win32' else signal.SIGBREAK
        args = self.get_script_args('sleeping.py', '-a', 'sleep=10')
        p = PopenSpawn(args, timeout=5)
        p.expect_exact('Spider opened')
        p.expect_exact('Crawled (200)')
        p.kill(sig)
        p.expect_exact('shutting down gracefully')
        d = defer.Deferred()
        reactor.callLater(0.1, d.callback, None)
        yield d
        p.kill(sig)
        p.expect_exact('forcing unclean shutdown')
        p.wait()

class CrawlerRunnerSubprocess(ScriptRunnerMixin, unittest.TestCase):
    script_dir = Path(__file__).parent.resolve() / 'CrawlerRunner'

    def test_response_ip_address(self):
        if False:
            for i in range(10):
                print('nop')
        log = self.run_script('ip_address.py')
        self.assertIn('INFO: Spider closed (finished)', log)
        self.assertIn('INFO: Host: not.a.real.domain', log)
        self.assertIn("INFO: Type: <class 'ipaddress.IPv4Address'>", log)
        self.assertIn('INFO: IP address: 127.0.0.1', log)