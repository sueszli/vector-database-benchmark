from tests.test_commands import CommandTest

class CheckCommandTest(CommandTest):
    command = 'check'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.spider_name = 'check_spider'
        self.spider = (self.proj_mod_path / 'spiders' / 'checkspider.py').resolve()

    def _write_contract(self, contracts, parse_def):
        if False:
            return 10
        self.spider.write_text(f'''\nimport scrapy\n\nclass CheckSpider(scrapy.Spider):\n    name = '{self.spider_name}'\n    start_urls = ['http://toscrape.com']\n\n    def parse(self, response, **cb_kwargs):\n        """\n        @url http://toscrape.com\n        {contracts}\n        """\n        {parse_def}\n        ''', encoding='utf-8')

    def _test_contract(self, contracts='', parse_def='pass'):
        if False:
            while True:
                i = 10
        self._write_contract(contracts, parse_def)
        (p, out, err) = self.proc('check')
        self.assertNotIn('F', out)
        self.assertIn('OK', err)
        self.assertEqual(p.returncode, 0)

    def test_check_returns_requests_contract(self):
        if False:
            for i in range(10):
                print('nop')
        contracts = '\n        @returns requests 1\n        '
        parse_def = "\n        yield scrapy.Request(url='http://next-url.com')\n        "
        self._test_contract(contracts, parse_def)

    def test_check_returns_items_contract(self):
        if False:
            while True:
                i = 10
        contracts = '\n        @returns items 1\n        '
        parse_def = "\n        yield {'key1': 'val1', 'key2': 'val2'}\n        "
        self._test_contract(contracts, parse_def)

    def test_check_cb_kwargs_contract(self):
        if False:
            while True:
                i = 10
        contracts = '\n        @cb_kwargs {"arg1": "val1", "arg2": "val2"}\n        '
        parse_def = '\n        if len(cb_kwargs.items()) == 0:\n            raise Exception("Callback args not set")\n        '
        self._test_contract(contracts, parse_def)

    def test_check_scrapes_contract(self):
        if False:
            print('Hello World!')
        contracts = '\n        @scrapes key1 key2\n        '
        parse_def = "\n        yield {'key1': 'val1', 'key2': 'val2'}\n        "
        self._test_contract(contracts, parse_def)

    def test_check_all_default_contracts(self):
        if False:
            while True:
                i = 10
        contracts = '\n        @returns items 1\n        @returns requests 1\n        @scrapes key1 key2\n        @cb_kwargs {"arg1": "val1", "arg2": "val2"}\n        '
        parse_def = '\n        yield {\'key1\': \'val1\', \'key2\': \'val2\'}\n        yield scrapy.Request(url=\'http://next-url.com\')\n        if len(cb_kwargs.items()) == 0:\n            raise Exception("Callback args not set")\n        '
        self._test_contract(contracts, parse_def)

    def test_SCRAPY_CHECK_set(self):
        if False:
            while True:
                i = 10
        parse_def = "\n        import os\n        if not os.environ.get('SCRAPY_CHECK'):\n            raise Exception('SCRAPY_CHECK not set')\n        "
        self._test_contract(parse_def=parse_def)