import unittest
from test import support
from test.support import os_helper
from test.support import socket_helper
import contextlib
import socket
import urllib.parse
import urllib.request
import os
import email.message
import time
support.requires('network')

class URLTimeoutTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        socket.setdefaulttimeout(support.INTERNET_TIMEOUT)

    def tearDown(self):
        if False:
            return 10
        socket.setdefaulttimeout(None)

    def testURLread(self):
        if False:
            for i in range(10):
                print('nop')
        self.addCleanup(urllib.request.urlcleanup)
        domain = urllib.parse.urlparse(support.TEST_HTTP_URL).netloc
        with socket_helper.transient_internet(domain):
            f = urllib.request.urlopen(support.TEST_HTTP_URL)
            f.read()

class urlopenNetworkTests(unittest.TestCase):
    """Tests urllib.request.urlopen using the network.

    These tests are not exhaustive.  Assuming that testing using files does a
    good job overall of some of the basic interface features.  There are no
    tests exercising the optional 'data' and 'proxies' arguments.  No tests
    for transparent redirection have been written.

    setUp is not used for always constructing a connection to
    http://www.pythontest.net/ since there a few tests that don't use that address
    and making a connection is expensive enough to warrant minimizing unneeded
    connections.

    """
    url = 'http://www.pythontest.net/'

    def setUp(self):
        if False:
            print('Hello World!')
        self.addCleanup(urllib.request.urlcleanup)

    @contextlib.contextmanager
    def urlopen(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        resource = args[0]
        with socket_helper.transient_internet(resource):
            r = urllib.request.urlopen(*args, **kwargs)
            try:
                yield r
            finally:
                r.close()

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        with self.urlopen(self.url) as open_url:
            for attr in ('read', 'readline', 'readlines', 'fileno', 'close', 'info', 'geturl'):
                self.assertTrue(hasattr(open_url, attr), 'object returned from urlopen lacks the %s attribute' % attr)
            self.assertTrue(open_url.read(), "calling 'read' failed")

    def test_readlines(self):
        if False:
            for i in range(10):
                print('nop')
        with self.urlopen(self.url) as open_url:
            self.assertIsInstance(open_url.readline(), bytes, 'readline did not return a string')
            self.assertIsInstance(open_url.readlines(), list, 'readlines did not return a list')

    def test_info(self):
        if False:
            i = 10
            return i + 15
        with self.urlopen(self.url) as open_url:
            info_obj = open_url.info()
            self.assertIsInstance(info_obj, email.message.Message, "object returned by 'info' is not an instance of email.message.Message")
            self.assertEqual(info_obj.get_content_subtype(), 'html')

    def test_geturl(self):
        if False:
            for i in range(10):
                print('nop')
        with self.urlopen(self.url) as open_url:
            gotten_url = open_url.geturl()
            self.assertEqual(gotten_url, self.url)

    def test_getcode(self):
        if False:
            i = 10
            return i + 15
        URL = self.url + 'XXXinvalidXXX'
        with socket_helper.transient_internet(URL):
            with self.assertWarns(DeprecationWarning):
                open_url = urllib.request.FancyURLopener().open(URL)
            try:
                code = open_url.getcode()
            finally:
                open_url.close()
            self.assertEqual(code, 404)

    def test_bad_address(self):
        if False:
            print('Hello World!')
        bogus_domain = 'sadflkjsasf.i.nvali.d.'
        try:
            socket.gethostbyname(bogus_domain)
        except OSError:
            pass
        else:
            self.skipTest('%r should not resolve for test to work' % bogus_domain)
        failure_explanation = 'opening an invalid URL did not raise OSError; can be caused by a broken DNS server (e.g. returns 404 or hijacks page)'
        with self.assertRaises(OSError, msg=failure_explanation):
            urllib.request.urlopen('http://{}/'.format(bogus_domain))

class urlretrieveNetworkTests(unittest.TestCase):
    """Tests urllib.request.urlretrieve using the network."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.addCleanup(urllib.request.urlcleanup)

    @contextlib.contextmanager
    def urlretrieve(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        resource = args[0]
        with socket_helper.transient_internet(resource):
            (file_location, info) = urllib.request.urlretrieve(*args, **kwargs)
            try:
                yield (file_location, info)
            finally:
                os_helper.unlink(file_location)

    def test_basic(self):
        if False:
            return 10
        with self.urlretrieve(self.logo) as (file_location, info):
            self.assertTrue(os.path.exists(file_location), 'file location returned by urlretrieve is not a valid path')
            with open(file_location, 'rb') as f:
                self.assertTrue(f.read(), 'reading from the file location returned by urlretrieve failed')

    def test_specified_path(self):
        if False:
            return 10
        with self.urlretrieve(self.logo, os_helper.TESTFN) as (file_location, info):
            self.assertEqual(file_location, os_helper.TESTFN)
            self.assertTrue(os.path.exists(file_location))
            with open(file_location, 'rb') as f:
                self.assertTrue(f.read(), 'reading from temporary file failed')

    def test_header(self):
        if False:
            while True:
                i = 10
        with self.urlretrieve(self.logo) as (file_location, info):
            self.assertIsInstance(info, email.message.Message, 'info is not an instance of email.message.Message')
    logo = 'http://www.pythontest.net/'

    def test_data_header(self):
        if False:
            return 10
        with self.urlretrieve(self.logo) as (file_location, fileheaders):
            datevalue = fileheaders.get('Date')
            dateformat = '%a, %d %b %Y %H:%M:%S GMT'
            try:
                time.strptime(datevalue, dateformat)
            except ValueError:
                self.fail('Date value not in %r format' % dateformat)

    def test_reporthook(self):
        if False:
            return 10
        records = []

        def recording_reporthook(blocks, block_size, total_size):
            if False:
                for i in range(10):
                    print('nop')
            records.append((blocks, block_size, total_size))
        with self.urlretrieve(self.logo, reporthook=recording_reporthook) as (file_location, fileheaders):
            expected_size = int(fileheaders['Content-Length'])
        records_repr = repr(records)
        self.assertGreater(len(records), 1, msg='There should always be two calls; the first one before the transfer starts.')
        self.assertEqual(records[0][0], 0)
        self.assertGreater(records[0][1], 0, msg="block size can't be 0 in %s" % records_repr)
        self.assertEqual(records[0][2], expected_size)
        self.assertEqual(records[-1][2], expected_size)
        block_sizes = {block_size for (_, block_size, _) in records}
        self.assertEqual({records[0][1]}, block_sizes, msg='block sizes in %s must be equal' % records_repr)
        self.assertGreaterEqual(records[-1][0] * records[0][1], expected_size, msg='number of blocks * block size must be >= total size in %s' % records_repr)
if __name__ == '__main__':
    unittest.main()