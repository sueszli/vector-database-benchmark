import unittest
from pocsuite3.api import requests, init_pocsuite

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        init_pocsuite()

    def tearDown(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='significant latency')
    def test_get(self):
        if False:
            return 10
        raw = '\n        GET /get?a=1&b=2 HTTP/1.1\n        Host: httpbin.org\n        Connection: keep-alive\n        Upgrade-Insecure-Requests: 1\n        User-Agent: pocsuite v3.0\n        Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8\n        Accept-Encoding: gzip, deflate\n        Accept-Language: zh-CN,zh;q=0.9,en;q=0.8\n        Cookie: _gauges_unique_hour=1; _gauges_unique_day=1; _gauges_unique_month=1; _gauges_unique_year=1; _gauges_unique=1\n        '
        r = requests.httpraw(raw)
        self.assertTrue(r.json()['args'] == {'a': '1', 'b': '2'})

    @unittest.skip(reason='significant latency')
    def test_post(self):
        if False:
            return 10
        raw = '\n        POST /post HTTP/1.1\n        Host: httpbin.org\n        Connection: keep-alive\n        Upgrade-Insecure-Requests: 1\n        User-Agent: pocsuite v3.0\n        Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8\n        Accept-Encoding: gzip, deflate\n        Accept-Language: zh-CN,zh;q=0.9,en;q=0.8\n        Cookie: _gauges_unique_hour=1; _gauges_unique_day=1; _gauges_unique_month=1; _gauges_unique_year=1; _gauges_unique=1\n\n        a=1&b=2\n        '
        r = requests.httpraw(raw)
        self.assertTrue(r.json()['data'] == 'a=1&b=2')

    @unittest.skip(reason='significant latency')
    def test_json(self):
        if False:
            i = 10
            return i + 15
        raw = '\n        POST /post HTTP/1.1\n        Host: httpbin.org\n        Connection: keep-alive\n        Upgrade-Insecure-Requests: 1\n        User-Agent: pocsuite v3.0\n        Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8\n        Accept-Encoding: gzip, deflate\n        Accept-Language: zh-CN,zh;q=0.9,en;q=0.8\n        Cookie: _gauges_unique_hour=1; _gauges_unique_day=1; _gauges_unique_month=1; _gauges_unique_year=1; _gauges_unique=1\n\n        {"pocsuite":"v3.0"}\n        '
        r = requests.httpraw(raw)
        self.assertTrue(r.json()['json'] == '{"pocsuite":"v3.0"}')