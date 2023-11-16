from test.picardtestcase import PicardTestCase
from picard.util import parse_amazon_url

class ParseAmazonUrlTest(PicardTestCase):

    def test_1(self):
        if False:
            return 10
        url = 'http://www.amazon.com/dp/020530902X'
        expected = {'asin': '020530902X', 'host': 'amazon.com'}
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_2(self):
        if False:
            return 10
        url = 'http://ec1.amazon.co.jp/gp/product/020530902X'
        expected = {'asin': '020530902X', 'host': 'ec1.amazon.co.jp'}
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_3(self):
        if False:
            print('Hello World!')
        url = 'http://amazon.com/Dark-Side-Moon-Pink-Floyd/dp/B004ZN9RWK/ref=sr_1_1?s=music&ie=UTF8&qid=1372605047&sr=1-1&keywords=pink+floyd+dark+side+of+the+moon'
        expected = {'asin': 'B004ZN9RWK', 'host': 'amazon.com'}
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_4(self):
        if False:
            i = 10
            return i + 15
        url = 'https://www.amazon.co.jp/gp/product/B00005FMYV'
        expected = {'asin': 'B00005FMYV', 'host': 'amazon.co.jp'}
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_incorrect_asin_1(self):
        if False:
            print('Hello World!')
        url = 'http://www.amazon.com/dp/A20530902X'
        expected = None
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_incorrect_asin_2(self):
        if False:
            while True:
                i = 10
        url = 'http://www.amazon.com/dp/020530902x'
        expected = None
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)

    def test_incorrect_url_scheme(self):
        if False:
            i = 10
            return i + 15
        url = 'httpsa://www.amazon.co.jp/gp/product/B00005FMYV'
        expected = None
        r = parse_amazon_url(url)
        self.assertEqual(r, expected)