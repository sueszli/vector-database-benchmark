import logging
import sys
try:
    from django.urls import reverse
except ImportError:
    from django.core.urlresolvers import reverse
from .base import TestCase
LOGGER = logging.getLogger()
if hasattr(logging, 'NullHandler'):
    LOGGER.addHandler(logging.NullHandler())
if sys.version_info[0] >= 3:

    def resp_text(r):
        if False:
            for i in range(10):
                print('nop')
        return r.content.decode('utf-8')
else:

    def resp_text(r):
        if False:
            i = 10
            return i + 15
        return r.content

class RenderXSSTest(TestCase):

    def test_render_xss(self):
        if False:
            while True:
                i = 10
        url = reverse('render')
        xssStr = '<noscript><p title="</noscript><img src=x onerror=alert() onmouseover=alert()>">'
        response = self.client.get(url, {'target': 'test', 'format': 'raw', 'cacheTimeout': xssStr, 'from': xssStr})
        self.assertXSS(response, status_code=400, msg_prefix='XSS detected: ')

class FindXSSTest(TestCase):

    def test_render_xss(self):
        if False:
            return 10
        url = reverse('metrics_find')
        xssStr = '<noscript><p title="</noscript><img src=x onerror=alert() onmouseover=alert()>">'
        response = self.client.get(url, {'query': 'test', 'local': xssStr, 'from': xssStr, 'tz': xssStr})
        self.assertXSS(response, status_code=400, msg_prefix='XSS detected: ')