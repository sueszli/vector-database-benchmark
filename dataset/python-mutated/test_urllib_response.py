"""Unit tests for code in urllib.response."""
import socket
import tempfile
import urllib.response
import unittest

class TestResponse(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fp = self.sock.makefile('rb')
        self.test_headers = {'Host': 'www.python.org', 'Connection': 'close'}

    def test_with(self):
        if False:
            print('Hello World!')
        addbase = urllib.response.addbase(self.fp)
        self.assertIsInstance(addbase, tempfile._TemporaryFileWrapper)

        def f():
            if False:
                i = 10
                return i + 15
            with addbase as spam:
                pass
        self.assertFalse(self.fp.closed)
        f()
        self.assertTrue(self.fp.closed)
        self.assertRaises(ValueError, f)

    def test_addclosehook(self):
        if False:
            while True:
                i = 10
        closehook_called = False

        def closehook():
            if False:
                i = 10
                return i + 15
            nonlocal closehook_called
            closehook_called = True
        closehook = urllib.response.addclosehook(self.fp, closehook)
        closehook.close()
        self.assertTrue(self.fp.closed)
        self.assertTrue(closehook_called)

    def test_addinfo(self):
        if False:
            for i in range(10):
                print('nop')
        info = urllib.response.addinfo(self.fp, self.test_headers)
        self.assertEqual(info.info(), self.test_headers)
        self.assertEqual(info.headers, self.test_headers)

    def test_addinfourl(self):
        if False:
            return 10
        url = 'http://www.python.org'
        code = 200
        infourl = urllib.response.addinfourl(self.fp, self.test_headers, url, code)
        self.assertEqual(infourl.info(), self.test_headers)
        self.assertEqual(infourl.geturl(), url)
        self.assertEqual(infourl.getcode(), code)
        self.assertEqual(infourl.headers, self.test_headers)
        self.assertEqual(infourl.url, url)
        self.assertEqual(infourl.status, code)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.sock.close()
if __name__ == '__main__':
    unittest.main()