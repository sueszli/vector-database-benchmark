import argparse
import os
from pathlib import Path
from twisted.internet import defer
from scrapy.commands import parse
from scrapy.settings import Settings
from scrapy.utils.python import to_unicode
from scrapy.utils.testproc import ProcessTest
from scrapy.utils.testsite import SiteTest
from tests.test_commands import CommandTest

def _textmode(bstr):
    if False:
        return 10
    'Normalize input the same as writing to a file\n    and reading from it in text mode'
    return to_unicode(bstr).replace(os.linesep, '\n')

class ParseCommandTest(ProcessTest, SiteTest, CommandTest):
    command = 'parse'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.spider_name = 'parse_spider'
        (self.proj_mod_path / 'spiders' / 'myspider.py').write_text(f"""\nimport scrapy\nfrom scrapy.linkextractors import LinkExtractor\nfrom scrapy.spiders import CrawlSpider, Rule\nfrom scrapy.utils.test import get_from_asyncio_queue\nimport asyncio\n\n\nclass AsyncDefAsyncioReturnSpider(scrapy.Spider):\n    name = "asyncdef_asyncio_return"\n\n    async def parse(self, response):\n        await asyncio.sleep(0.2)\n        status = await get_from_asyncio_queue(response.status)\n        self.logger.info(f"Got response {{status}}")\n        return [{{'id': 1}}, {{'id': 2}}]\n\nclass AsyncDefAsyncioReturnSingleElementSpider(scrapy.Spider):\n    name = "asyncdef_asyncio_return_single_element"\n\n    async def parse(self, response):\n        await asyncio.sleep(0.1)\n        status = await get_from_asyncio_queue(response.status)\n        self.logger.info(f"Got response {{status}}")\n        return {{'foo': 42}}\n\nclass AsyncDefAsyncioGenLoopSpider(scrapy.Spider):\n    name = "asyncdef_asyncio_gen_loop"\n\n    async def parse(self, response):\n        for i in range(10):\n            await asyncio.sleep(0.1)\n            yield {{'foo': i}}\n        self.logger.info(f"Got response {{response.status}}")\n\nclass AsyncDefAsyncioSpider(scrapy.Spider):\n    name = "asyncdef_asyncio"\n\n    async def parse(self, response):\n        await asyncio.sleep(0.2)\n        status = await get_from_asyncio_queue(response.status)\n        self.logger.debug(f"Got response {{status}}")\n\nclass AsyncDefAsyncioGenExcSpider(scrapy.Spider):\n    name = "asyncdef_asyncio_gen_exc"\n\n    async def parse(self, response):\n        for i in range(10):\n            await asyncio.sleep(0.1)\n            yield {{'foo': i}}\n            if i > 5:\n                raise ValueError("Stopping the processing")\n\nclass MySpider(scrapy.Spider):\n    name = '{self.spider_name}'\n\n    def parse(self, response):\n        if getattr(self, 'test_arg', None):\n            self.logger.debug('It Works!')\n        return [scrapy.Item(), dict(foo='bar')]\n\n    def parse_request_with_meta(self, response):\n        foo = response.meta.get('foo', 'bar')\n\n        if foo == 'bar':\n            self.logger.debug('It Does Not Work :(')\n        else:\n            self.logger.debug('It Works!')\n\n    def parse_request_with_cb_kwargs(self, response, foo=None, key=None):\n        if foo == 'bar' and key == 'value':\n            self.logger.debug('It Works!')\n        else:\n            self.logger.debug('It Does Not Work :(')\n\n    def parse_request_without_meta(self, response):\n        foo = response.meta.get('foo', 'bar')\n\n        if foo == 'bar':\n            self.logger.debug('It Works!')\n        else:\n            self.logger.debug('It Does Not Work :(')\n\nclass MyGoodCrawlSpider(CrawlSpider):\n    name = 'goodcrawl{self.spider_name}'\n\n    rules = (\n        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),\n        Rule(LinkExtractor(allow=r'/text'), follow=True),\n    )\n\n    def parse_item(self, response):\n        return [scrapy.Item(), dict(foo='bar')]\n\n    def parse(self, response):\n        return [scrapy.Item(), dict(nomatch='default')]\n\n\nclass MyBadCrawlSpider(CrawlSpider):\n    '''Spider which doesn't define a parse_item callback while using it in a rule.'''\n    name = 'badcrawl{self.spider_name}'\n\n    rules = (\n        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),\n    )\n\n    def parse(self, response):\n        return [scrapy.Item(), dict(foo='bar')]\n""", encoding='utf-8')
        (self.proj_mod_path / 'pipelines.py').write_text("\nimport logging\n\nclass MyPipeline:\n    component_name = 'my_pipeline'\n\n    def process_item(self, item, spider):\n        logging.info('It Works!')\n        return item\n", encoding='utf-8')
        with (self.proj_mod_path / 'settings.py').open('a', encoding='utf-8') as f:
            f.write(f"\nITEM_PIPELINES = {{'{self.project_name}.pipelines.MyPipeline': 1}}\n")

    @defer.inlineCallbacks
    def test_spider_arguments(self):
        if False:
            print('Hello World!')
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '-a', 'test_arg=1', '-c', 'parse', '--verbose', self.url('/html')]))
        self.assertIn('DEBUG: It Works!', _textmode(stderr))

    @defer.inlineCallbacks
    def test_request_with_meta(self):
        if False:
            while True:
                i = 10
        raw_json_string = '{"foo" : "baz"}'
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '--meta', raw_json_string, '-c', 'parse_request_with_meta', '--verbose', self.url('/html')]))
        self.assertIn('DEBUG: It Works!', _textmode(stderr))
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '-m', raw_json_string, '-c', 'parse_request_with_meta', '--verbose', self.url('/html')]))
        self.assertIn('DEBUG: It Works!', _textmode(stderr))

    @defer.inlineCallbacks
    def test_request_with_cb_kwargs(self):
        if False:
            while True:
                i = 10
        raw_json_string = '{"foo" : "bar", "key": "value"}'
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '--cbkwargs', raw_json_string, '-c', 'parse_request_with_cb_kwargs', '--verbose', self.url('/html')]))
        self.assertIn('DEBUG: It Works!', _textmode(stderr))

    @defer.inlineCallbacks
    def test_request_without_meta(self):
        if False:
            for i in range(10):
                print('nop')
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '-c', 'parse_request_without_meta', '--nolinks', self.url('/html')]))
        self.assertIn('DEBUG: It Works!', _textmode(stderr))

    @defer.inlineCallbacks
    def test_pipelines(self):
        if False:
            while True:
                i = 10
        (_, _, stderr) = (yield self.execute(['--spider', self.spider_name, '--pipelines', '-c', 'parse', '--verbose', self.url('/html')]))
        self.assertIn('INFO: It Works!', _textmode(stderr))

    @defer.inlineCallbacks
    def test_async_def_asyncio_parse_items_list(self):
        if False:
            for i in range(10):
                print('nop')
        (status, out, stderr) = (yield self.execute(['--spider', 'asyncdef_asyncio_return', '-c', 'parse', self.url('/html')]))
        self.assertIn('INFO: Got response 200', _textmode(stderr))
        self.assertIn("{'id': 1}", _textmode(out))
        self.assertIn("{'id': 2}", _textmode(out))

    @defer.inlineCallbacks
    def test_async_def_asyncio_parse_items_single_element(self):
        if False:
            return 10
        (status, out, stderr) = (yield self.execute(['--spider', 'asyncdef_asyncio_return_single_element', '-c', 'parse', self.url('/html')]))
        self.assertIn('INFO: Got response 200', _textmode(stderr))
        self.assertIn("{'foo': 42}", _textmode(out))

    @defer.inlineCallbacks
    def test_async_def_asyncgen_parse_loop(self):
        if False:
            print('Hello World!')
        (status, out, stderr) = (yield self.execute(['--spider', 'asyncdef_asyncio_gen_loop', '-c', 'parse', self.url('/html')]))
        self.assertIn('INFO: Got response 200', _textmode(stderr))
        for i in range(10):
            self.assertIn(f"{{'foo': {i}}}", _textmode(out))

    @defer.inlineCallbacks
    def test_async_def_asyncgen_parse_exc(self):
        if False:
            while True:
                i = 10
        (status, out, stderr) = (yield self.execute(['--spider', 'asyncdef_asyncio_gen_exc', '-c', 'parse', self.url('/html')]))
        self.assertIn('ValueError', _textmode(stderr))
        for i in range(7):
            self.assertIn(f"{{'foo': {i}}}", _textmode(out))

    @defer.inlineCallbacks
    def test_async_def_asyncio_parse(self):
        if False:
            for i in range(10):
                print('nop')
        (_, _, stderr) = (yield self.execute(['--spider', 'asyncdef_asyncio', '-c', 'parse', self.url('/html')]))
        self.assertIn('DEBUG: Got response 200', _textmode(stderr))

    @defer.inlineCallbacks
    def test_parse_items(self):
        if False:
            print('Hello World!')
        (status, out, stderr) = (yield self.execute(['--spider', self.spider_name, '-c', 'parse', self.url('/html')]))
        self.assertIn("[{}, {'foo': 'bar'}]", _textmode(out))

    @defer.inlineCallbacks
    def test_parse_items_no_callback_passed(self):
        if False:
            print('Hello World!')
        (status, out, stderr) = (yield self.execute(['--spider', self.spider_name, self.url('/html')]))
        self.assertIn("[{}, {'foo': 'bar'}]", _textmode(out))

    @defer.inlineCallbacks
    def test_wrong_callback_passed(self):
        if False:
            while True:
                i = 10
        (status, out, stderr) = (yield self.execute(['--spider', self.spider_name, '-c', 'dummy', self.url('/html')]))
        self.assertRegex(_textmode(out), '# Scraped Items  -+\\n\\[\\]')
        self.assertIn('Cannot find callback', _textmode(stderr))

    @defer.inlineCallbacks
    def test_crawlspider_matching_rule_callback_set(self):
        if False:
            for i in range(10):
                print('nop')
        "If a rule matches the URL, use it's defined callback."
        (status, out, stderr) = (yield self.execute(['--spider', 'goodcrawl' + self.spider_name, '-r', self.url('/html')]))
        self.assertIn("[{}, {'foo': 'bar'}]", _textmode(out))

    @defer.inlineCallbacks
    def test_crawlspider_matching_rule_default_callback(self):
        if False:
            return 10
        "If a rule match but it has no callback set, use the 'parse' callback."
        (status, out, stderr) = (yield self.execute(['--spider', 'goodcrawl' + self.spider_name, '-r', self.url('/text')]))
        self.assertIn("[{}, {'nomatch': 'default'}]", _textmode(out))

    @defer.inlineCallbacks
    def test_spider_with_no_rules_attribute(self):
        if False:
            return 10
        'Using -r with a spider with no rule should not produce items.'
        (status, out, stderr) = (yield self.execute(['--spider', self.spider_name, '-r', self.url('/html')]))
        self.assertRegex(_textmode(out), '# Scraped Items  -+\\n\\[\\]')
        self.assertIn('No CrawlSpider rules found', _textmode(stderr))

    @defer.inlineCallbacks
    def test_crawlspider_missing_callback(self):
        if False:
            print('Hello World!')
        (status, out, stderr) = (yield self.execute(['--spider', 'badcrawl' + self.spider_name, '-r', self.url('/html')]))
        self.assertRegex(_textmode(out), '# Scraped Items  -+\\n\\[\\]')

    @defer.inlineCallbacks
    def test_crawlspider_no_matching_rule(self):
        if False:
            for i in range(10):
                print('nop')
        'The requested URL has no matching rule, so no items should be scraped'
        (status, out, stderr) = (yield self.execute(['--spider', 'badcrawl' + self.spider_name, '-r', self.url('/enc-gb18030')]))
        self.assertRegex(_textmode(out), '# Scraped Items  -+\\n\\[\\]')
        self.assertIn('Cannot find a rule that matches', _textmode(stderr))

    @defer.inlineCallbacks
    def test_crawlspider_not_exists_with_not_matched_url(self):
        if False:
            while True:
                i = 10
        (status, out, stderr) = (yield self.execute([self.url('/invalid_url')]))
        self.assertEqual(status, 0)

    @defer.inlineCallbacks
    def test_output_flag(self):
        if False:
            print('Hello World!')
        'Checks if a file was created successfully having\n        correct format containing correct data in it.\n        '
        file_name = 'data.json'
        file_path = Path(self.proj_path, file_name)
        yield self.execute(['--spider', self.spider_name, '-c', 'parse', '-o', file_name, self.url('/html')])
        self.assertTrue(file_path.exists())
        self.assertTrue(file_path.is_file())
        content = '[\n{},\n{"foo": "bar"}\n]'
        self.assertEqual(file_path.read_text(encoding='utf-8'), content)

    def test_parse_add_options(self):
        if False:
            return 10
        command = parse.Command()
        command.settings = Settings()
        parser = argparse.ArgumentParser(prog='scrapy', formatter_class=argparse.HelpFormatter, conflict_handler='resolve', prefix_chars='-')
        command.add_options(parser)
        namespace = parser.parse_args(['--verbose', '--nolinks', '-d', '2', '--spider', self.spider_name])
        self.assertTrue(namespace.nolinks)
        self.assertEqual(namespace.depth, 2)
        self.assertEqual(namespace.spider, self.spider_name)
        self.assertTrue(namespace.verbose)