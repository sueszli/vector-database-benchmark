"""Testing of bzrlib.transport.http.ca_bundle module"""
import os
import sys
from bzrlib.tests import TestCaseInTempDir, TestSkipped
from bzrlib.transport.http import ca_bundle

class TestGetCAPath(TestCaseInTempDir):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestGetCAPath, self).setUp()
        self.overrideEnv('CURL_CA_BUNDLE', None)
        self.overrideEnv('PATH', None)

    def _make_file(self, in_dir='.'):
        if False:
            print('Hello World!')
        fname = os.path.join(in_dir, 'curl-ca-bundle.crt')
        f = file(fname, 'w')
        f.write('spam')
        f.close()

    def test_found_nothing(self):
        if False:
            return 10
        self.assertEqual('', ca_bundle.get_ca_path(use_cache=False))

    def test_env_var(self):
        if False:
            for i in range(10):
                print('nop')
        self.overrideEnv('CURL_CA_BUNDLE', 'foo.bar')
        self._make_file()
        self.assertEqual('foo.bar', ca_bundle.get_ca_path(use_cache=False))

    def test_in_path(self):
        if False:
            print('Hello World!')
        if sys.platform != 'win32':
            raise TestSkipped('Searching in PATH implemented only for win32')
        os.mkdir('foo')
        in_dir = os.path.join(os.getcwd(), 'foo')
        self._make_file(in_dir=in_dir)
        self.overrideEnv('PATH', in_dir)
        shouldbe = os.path.join(in_dir, 'curl-ca-bundle.crt')
        self.assertEqual(shouldbe, ca_bundle.get_ca_path(use_cache=False))