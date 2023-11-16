import unittest
from pocsuite3.api import crawl

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.url = 'http://xxxxx'

    def tearDown(self):
        if False:
            return 10
        pass

    def verify_result(self, urls):
        if False:
            while True:
                i = 10
        links = urls['url']
        self.assertTrue(len(links) > 0)
        url = links.pop()
        url = url.split('?')[0]
        self.assertTrue(url.endswith(('.action', '.do')))

    def test_import_run(self):
        if False:
            while True:
                i = 10
        return self.assertTrue(1)
        urls = crawl(self.url, url_ext=('.action', '.do'))
        self.verify_result(urls)