import unittest
from ssl import match_hostname, CertificateError
import pytest
from docker.transport import ssladapter
try:
    from ssl import OP_NO_SSLv3, OP_NO_SSLv2, OP_NO_TLSv1
except ImportError:
    OP_NO_SSLv2 = 16777216
    OP_NO_SSLv3 = 33554432
    OP_NO_TLSv1 = 67108864

class SSLAdapterTest(unittest.TestCase):

    def test_only_uses_tls(self):
        if False:
            for i in range(10):
                print('nop')
        ssl_context = ssladapter.urllib3.util.ssl_.create_urllib3_context()
        assert ssl_context.options & OP_NO_SSLv3
        assert not bool(OP_NO_SSLv2) or ssl_context.options & OP_NO_SSLv2
        assert not ssl_context.options & OP_NO_TLSv1

class MatchHostnameTest(unittest.TestCase):
    cert = {'issuer': ((('countryName', 'US'),), (('stateOrProvinceName', 'California'),), (('localityName', 'San Francisco'),), (('organizationName', 'Docker Inc'),), (('organizationalUnitName', 'Docker-Python'),), (('commonName', 'localhost'),), (('emailAddress', 'info@docker.com'),)), 'notAfter': 'Mar 25 23:08:23 2030 GMT', 'notBefore': 'Mar 25 23:08:23 2016 GMT', 'serialNumber': 'BD5F894C839C548F', 'subject': ((('countryName', 'US'),), (('stateOrProvinceName', 'California'),), (('localityName', 'San Francisco'),), (('organizationName', 'Docker Inc'),), (('organizationalUnitName', 'Docker-Python'),), (('commonName', 'localhost'),), (('emailAddress', 'info@docker.com'),)), 'subjectAltName': (('DNS', 'localhost'), ('DNS', '*.gensokyo.jp'), ('IP Address', '127.0.0.1')), 'version': 3}

    def test_match_ip_address_success(self):
        if False:
            i = 10
            return i + 15
        assert match_hostname(self.cert, '127.0.0.1') is None

    def test_match_localhost_success(self):
        if False:
            return 10
        assert match_hostname(self.cert, 'localhost') is None

    def test_match_dns_success(self):
        if False:
            print('Hello World!')
        assert match_hostname(self.cert, 'touhou.gensokyo.jp') is None

    def test_match_ip_address_failure(self):
        if False:
            return 10
        with pytest.raises(CertificateError):
            match_hostname(self.cert, '192.168.0.25')

    def test_match_dns_failure(self):
        if False:
            print('Hello World!')
        with pytest.raises(CertificateError):
            match_hostname(self.cert, 'foobar.co.uk')