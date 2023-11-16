from twisted.internet import defer
from twisted.trial import unittest
from scrapy.utils.testproc import ProcessTest
from scrapy.utils.testsite import SiteTest

class FetchTest(ProcessTest, SiteTest, unittest.TestCase):
    command = 'fetch'

    @defer.inlineCallbacks
    def test_output(self):
        if False:
            for i in range(10):
                print('nop')
        (_, out, _) = (yield self.execute([self.url('/text')]))
        self.assertEqual(out.strip(), b'Works')

    @defer.inlineCallbacks
    def test_redirect_default(self):
        if False:
            while True:
                i = 10
        (_, out, _) = (yield self.execute([self.url('/redirect')]))
        self.assertEqual(out.strip(), b'Redirected here')

    @defer.inlineCallbacks
    def test_redirect_disabled(self):
        if False:
            i = 10
            return i + 15
        (_, out, err) = (yield self.execute(['--no-redirect', self.url('/redirect-no-meta-refresh')]))
        err = err.strip()
        self.assertIn(b'downloader/response_status_count/302', err, err)
        self.assertNotIn(b'downloader/response_status_count/200', err, err)

    @defer.inlineCallbacks
    def test_headers(self):
        if False:
            for i in range(10):
                print('nop')
        (_, out, _) = (yield self.execute([self.url('/text'), '--headers']))
        out = out.replace(b'\r', b'')
        assert b'Server: TwistedWeb' in out, out
        assert b'Content-Type: text/plain' in out