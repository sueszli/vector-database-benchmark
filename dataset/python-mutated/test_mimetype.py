from unittest import TestCase
from lib.utils.mimetype import MimeTypeUtils

class TestMimeTypeUtils(TestCase):

    def test_is_json(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(MimeTypeUtils.is_json('{"foo": "bar"}'), 'Failed to detect JSON mimetype')

    def test_is_xml(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(MimeTypeUtils.is_xml('<?xml version="1.0" encoding="UTF-8"?><foo>bar</foo>'), 'Failed to detect XML mimetype')

    def test_is_query_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(MimeTypeUtils.is_query_string('foo=1&bar=&foobar=2'), 'Failed to detect query string')