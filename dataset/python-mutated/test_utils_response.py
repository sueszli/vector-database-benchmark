import unittest
from pathlib import Path
from urllib.parse import urlparse
from scrapy.http import HtmlResponse, Response, TextResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.response import get_base_url, get_meta_refresh, open_in_browser, response_status_message
__doctests__ = ['scrapy.utils.response']

class ResponseUtilsTest(unittest.TestCase):
    dummy_response = TextResponse(url='http://example.org/', body=b'dummy_response')

    def test_open_in_browser(self):
        if False:
            while True:
                i = 10
        url = 'http:///www.example.com/some/page.html'
        body = b'<html> <head> <title>test page</title> </head> <body>test body</body> </html>'

        def browser_open(burl):
            if False:
                return 10
            path = urlparse(burl).path
            if not path or not Path(path).exists():
                path = burl.replace('file://', '')
            bbody = Path(path).read_bytes()
            self.assertIn(b'<base href="' + to_bytes(url) + b'">', bbody)
            return True
        response = HtmlResponse(url, body=body)
        assert open_in_browser(response, _openfunc=browser_open), 'Browser not called'
        resp = Response(url, body=body)
        self.assertRaises(TypeError, open_in_browser, resp, debug=True)

    def test_get_meta_refresh(self):
        if False:
            return 10
        r1 = HtmlResponse('http://www.example.com', body=b'\n        <html>\n        <head><title>Dummy</title><meta http-equiv="refresh" content="5;url=http://example.org/newpage" /></head>\n        <body>blahablsdfsal&amp;</body>\n        </html>')
        r2 = HtmlResponse('http://www.example.com', body=b'\n        <html>\n        <head><title>Dummy</title><noScript>\n        <meta http-equiv="refresh" content="5;url=http://example.org/newpage" /></head>\n        </noSCRIPT>\n        <body>blahablsdfsal&amp;</body>\n        </html>')
        r3 = HtmlResponse('http://www.example.com', body=b'\n    <noscript><meta http-equiv="REFRESH" content="0;url=http://www.example.com/newpage</noscript>\n    <script type="text/javascript">\n    if(!checkCookies()){\n        document.write(\'<meta http-equiv="REFRESH" content="0;url=http://www.example.com/newpage">\');\n    }\n    </script>\n        ')
        self.assertEqual(get_meta_refresh(r1), (5.0, 'http://example.org/newpage'))
        self.assertEqual(get_meta_refresh(r2), (None, None))
        self.assertEqual(get_meta_refresh(r3), (None, None))

    def test_get_base_url(self):
        if False:
            print('Hello World!')
        resp = HtmlResponse('http://www.example.com', body=b'\n        <html>\n        <head><base href="http://www.example.com/img/" target="_blank"></head>\n        <body>blahablsdfsal&amp;</body>\n        </html>')
        self.assertEqual(get_base_url(resp), 'http://www.example.com/img/')
        resp2 = HtmlResponse('http://www.example.com', body=b'\n        <html><body>blahablsdfsal&amp;</body></html>')
        self.assertEqual(get_base_url(resp2), 'http://www.example.com')

    def test_response_status_message(self):
        if False:
            return 10
        self.assertEqual(response_status_message(200), '200 OK')
        self.assertEqual(response_status_message(404), '404 Not Found')
        self.assertEqual(response_status_message(573), '573 Unknown Status')

    def test_inject_base_url(self):
        if False:
            return 10
        url = 'http://www.example.com'

        def check_base_url(burl):
            if False:
                while True:
                    i = 10
            path = urlparse(burl).path
            if not path or not Path(path).exists():
                path = burl.replace('file://', '')
            bbody = Path(path).read_bytes()
            self.assertEqual(bbody.count(b'<base href="' + to_bytes(url) + b'">'), 1)
            return True
        r1 = HtmlResponse(url, body=b'\n        <html>\n            <head><title>Dummy</title></head>\n            <body><p>Hello world.</p></body>\n        </html>')
        r2 = HtmlResponse(url, body=b'\n        <html>\n            <head id="foo"><title>Dummy</title></head>\n            <body>Hello world.</body>\n        </html>')
        r3 = HtmlResponse(url, body=b'\n        <html>\n            <head><title>Dummy</title></head>\n            <body>\n                <header>Hello header</header>\n                <p>Hello world.</p>\n            </body>\n        </html>')
        r4 = HtmlResponse(url, body=b'\n        <html>\n            <!-- <head>Dummy comment</head> -->\n            <head><title>Dummy</title></head>\n            <body><p>Hello world.</p></body>\n        </html>')
        r5 = HtmlResponse(url, body=b'\n        <html>\n            <!--[if IE]>\n            <head><title>IE head</title></head>\n            <![endif]-->\n            <!--[if !IE]>-->\n            <head><title>Standard head</title></head>\n            <!--<![endif]-->\n            <body><p>Hello world.</p></body>\n        </html>')
        assert open_in_browser(r1, _openfunc=check_base_url), 'Inject base url'
        assert open_in_browser(r2, _openfunc=check_base_url), 'Inject base url with argumented head'
        assert open_in_browser(r3, _openfunc=check_base_url), 'Inject unique base url with misleading tag'
        assert open_in_browser(r4, _openfunc=check_base_url), 'Inject unique base url with misleading comment'
        assert open_in_browser(r5, _openfunc=check_base_url), 'Inject unique base url with conditional comment'