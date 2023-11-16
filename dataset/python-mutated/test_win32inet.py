import unittest
import winerror
from pywin32_testutil import TestSkipped, testmain
from win32inet import *
from win32inetcon import *

class CookieTests(unittest.TestCase):

    def testCookies(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'TestData=Test'
        InternetSetCookie('http://www.python.org', None, data)
        got = InternetGetCookie('http://www.python.org', None)
        bits = (x.strip() for x in got.split(';'))
        self.assertTrue(data in bits)

    def testCookiesEmpty(self):
        if False:
            print('Hello World!')
        try:
            InternetGetCookie('http://site-with-no-cookie.python.org', None)
            self.fail('expected win32 exception')
        except error as exc:
            self.assertEqual(exc.winerror, winerror.ERROR_NO_MORE_ITEMS)

class UrlTests(unittest.TestCase):

    def testSimpleCanonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        ret = InternetCanonicalizeUrl('foo bar')
        self.assertEqual(ret, 'foo%20bar')

    def testLongCanonicalize(self):
        if False:
            print('Hello World!')
        big = 'x' * 2048
        ret = InternetCanonicalizeUrl(big + ' ' + big)
        self.assertEqual(ret, big + '%20' + big)

class TestNetwork(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.hi = InternetOpen('test', INTERNET_OPEN_TYPE_DIRECT, None, None, 0)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.hi.Close()

    def testPythonDotOrg(self):
        if False:
            print('Hello World!')
        hdl = InternetOpenUrl(self.hi, 'http://www.python.org', None, INTERNET_FLAG_EXISTING_CONNECT)
        chunks = []
        while 1:
            chunk = InternetReadFile(hdl, 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b''.join(chunks)
        assert data.find(b'Python') > 0, repr(data)

    def testFtpCommand(self):
        if False:
            i = 10
            return i + 15
        try:
            hcon = InternetConnect(self.hi, 'ftp.gnu.org', INTERNET_INVALID_PORT_NUMBER, None, None, INTERNET_SERVICE_FTP, 0, 0)
            try:
                hftp = FtpCommand(hcon, True, FTP_TRANSFER_TYPE_ASCII, 'NLST', 0)
                try:
                    print('Connected - response info is', InternetGetLastResponseInfo())
                    got = InternetReadFile(hftp, 2048)
                    print('Read', len(got), 'bytes')
                finally:
                    hftp.Close()
            finally:
                hcon.Close()
        except error as e:
            raise TestSkipped(e)
if __name__ == '__main__':
    testmain()