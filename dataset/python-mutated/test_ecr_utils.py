from unittest import TestCase
from samcli.lib.package.ecr_utils import is_ecr_url

class TestECRUtils(TestCase):

    def test_valid_ecr_url(self):
        if False:
            return 10
        url = '000000000000.dkr.ecr.eu-west-1.amazonaws.com/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_long_ecr_url(self):
        if False:
            print('Hello World!')
        url = '000000000000.dkr.ecr.eu-west-1.amazonaws.com/a/longer/path/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_long_ecr_url_special_chars(self):
        if False:
            return 10
        url = '000000000000.dkr.ecr.eu-west-1.amazonaws.com/a/weird.er/pa_th/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_localhost_ecr_url(self):
        if False:
            return 10
        url = 'localhost/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_localhost_ecr_url_port(self):
        if False:
            print('Hello World!')
        url = 'localhost:8084/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_127_0_0_1_ecr_url(self):
        if False:
            print('Hello World!')
        url = '127.0.0.1/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_valid_127_0_0_1_ecr_url_port(self):
        if False:
            while True:
                i = 10
        url = '127.0.0.1:12345/my-repo'
        self.assertTrue(is_ecr_url(url))

    def test_ecr_url_only_hostname(self):
        if False:
            print('Hello World!')
        url = '000000000000.dkr.ecr.eu-west-1.amazonaws.com'
        self.assertFalse(is_ecr_url(url))

    def test_ecr_url_only_hostname2(self):
        if False:
            for i in range(10):
                print('nop')
        url = '000000000000.dkr.ecr.eu-west-1.amazonaws.com/'
        self.assertFalse(is_ecr_url(url))

    def test_ecr_url_non_alphanum_starting_char(self):
        if False:
            while True:
                i = 10
        url = '_00000000000.dkr.ecr.eu-west-1.amazonaws.com/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_localhost_ecr_url_only_hostname(self):
        if False:
            while True:
                i = 10
        url = 'localhost'
        self.assertFalse(is_ecr_url(url))

    def test_localhost_ecr_url_long_port_name(self):
        if False:
            print('Hello World!')
        url = 'localhost:123456/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_localhost_ecr_url_bad_port_name(self):
        if False:
            print('Hello World!')
        url = 'localhost:abc/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_localhost_ecr_url_malform(self):
        if False:
            for i in range(10):
                print('nop')
        url = 'localhost:/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_127_0_0_1_ecr_url_only_hostname(self):
        if False:
            print('Hello World!')
        url = '127.0.0.1'
        self.assertFalse(is_ecr_url(url))

    def test_127_0_0_1_ecr_url_long_port_name(self):
        if False:
            while True:
                i = 10
        url = '127.0.0.1:123456/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_127_0_0_1_ecr_url_bad_port_name(self):
        if False:
            print('Hello World!')
        url = '127.0.0.1:abc/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_127_0_0_1_ecr_url_malform(self):
        if False:
            while True:
                i = 10
        url = '127.0.0.1:/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_localhost_ecr_url_wronghostname(self):
        if False:
            while True:
                i = 10
        url = 'notlocalhost:1234/my-repo'
        self.assertFalse(is_ecr_url(url))

    def test_127_0_0_1_ecr_url_wronghostname(self):
        if False:
            print('Hello World!')
        url = '127.0.0.2:1234/my-repo'
        self.assertFalse(is_ecr_url(url))