"""
Tests for the salt-run command
"""
import functools
import random
import string
import tempfile
import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID
from tests.support.case import ShellCase

def _random_name(prefix=''):
    if False:
        print('Hello World!')
    ret = prefix
    for _ in range(8):
        ret += random.choice(string.ascii_lowercase)
    return ret

def with_random_name(func):
    if False:
        while True:
            i = 10
    '\n    generate a randomized name for a container\n    '

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            print('Hello World!')
        name = _random_name(prefix='salt_')
        return func(self, _random_name(prefix='salt-test-'), *args, **kwargs)
    return wrapper

class VenafiTest(ShellCase):
    """
    Test the venafi runner
    """

    @with_random_name
    @pytest.mark.slow_test
    def test_request(self, name):
        if False:
            return 10
        cn = '{}.example.com'.format(name)
        if not isinstance(cn, str):
            cn = cn.decode()
        ret = self.run_run_plus(fun='venafi.request', minion_id=cn, dns_name=cn, key_password='secretPassword', zone='fake')
        cert_output = ret['return'][0]
        assert cert_output is not None, 'venafi_certificate not found in `output_value`'
        cert = x509.load_pem_x509_certificate(cert_output.encode(), default_backend())
        assert isinstance(cert, x509.Certificate)
        assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME) == [x509.NameAttribute(NameOID.COMMON_NAME, cn)]
        pkey_output = ret['return'][1]
        assert pkey_output is not None, 'venafi_private key not found in output_value'
        pkey = serialization.load_pem_private_key(pkey_output.encode(), password=b'secretPassword', backend=default_backend())
        pkey_public_key_pem = pkey.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        cert_public_key_pem = cert.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        assert pkey_public_key_pem == cert_public_key_pem

    @with_random_name
    @pytest.mark.slow_test
    def test_sign(self, name):
        if False:
            while True:
                i = 10
        csr_pem = '-----BEGIN CERTIFICATE REQUEST-----\nMIIFbDCCA1QCAQAwgbQxCzAJBgNVBAYTAlVTMQ0wCwYDVQQIDARVdGFoMRIwEAYD\nVQQHDAlTYWx0IExha2UxFDASBgNVBAoMC1ZlbmFmaSBJbmMuMRQwEgYDVQQLDAtJ\nbnRlZ3JhdGlvbjEnMCUGCSqGSIb3DQEJARYYZW1haWxAdmVuYWZpLmV4YW1wbGUu\nY29tMS0wKwYDVQQDDCR0ZXN0LWNzci0zMjMxMzEzMS52ZW5hZmkuZXhhbXBsZS5j\nb20wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQC4T0bdjq+mF+DABhF+\nXWCwOXXUWbPNWa72VVhxoelbyTS0iIeZEe64AvNGykytFdOuT/F9pdkZa+Io07R1\nZMp6Ak8dp2Wjt4c5rayVZus6ZK+0ZwBRJO7if/cqhEpxy8Wz1RMfVLf2AE1u/xZS\nQSYY0BTRWGmPqrFJrIGbnyQfvmGVPk3cA0RfdrwYJZXtZ2/4QNrbNCoSoSmqTHzt\nNAtZhvT2dPU9U48Prx4b2460x+ck3xA1OdJNXV7n5u53QbxOIcjdGT0lJ62ml70G\n5gvEHmdPcg+t5cw/Sm5cfDSUEDtNEXvD4oJXfP98ty6f1cYsZpcrgxRwk9RfGain\nhvoweXhZP3NWnU5nRdn2nOfExv+xMeQOyB/rYv98zqzK6LvwKhwI5UB1l/n9KTpg\njgaNCP4x/KAsrPecbHK91oiqGSbPn4wtTYOmPkDxSzATN317u7fE20iqvVAUy/O+\n7SCNNKEDPX2NP9LLz0IPK0roQxLiwd2CVyN6kEXuzs/3psptkNRMSlhyeAZdfrOE\nCNOp46Pam9f9HGBqzXxxoIlfzLqHHL584kgFlBm7qmivVrgp6zdLPDa+UayXEl2N\nO17SnGS8nkOTmfg3cez7lzX/LPLO9X/Y1xKYqx5hoGZhh754K8mzDWCVCYThWgou\nyBOYY8uNXiX6ldqzQUHpbxxQgwIDAQABoHIwcAYJKoZIhvcNAQkOMWMwYTBfBgNV\nHREEWDBWgilhbHQxLXRlc3QtY3NyLTMyMzEzMTMxLnZlbmFmaS5leGFtcGxlLmNv\nbYIpYWx0Mi10ZXN0LWNzci0zMjMxMzEzMS52ZW5hZmkuZXhhbXBsZS5jb20wDQYJ\nKoZIhvcNAQELBQADggIBAJd87BIdeh0WWoyQ4IX+ENpNqmm/sLmdfmUB/hj9NpBL\nqbr2UTWaSr1jadoZ+mrDxtm1Z0YJDTTIrEWxkBOW5wQ039lYZNe2tfDXSJZwJn7u\n2keaXtWQ2SdduK1wOPDO9Hra6WnH7aEq5D1AyoghvPsZwTqZkNynt/A1BZW5C/ha\nJ9/mwgWfL4qXBGBOhLwKN5GUo3erUkJIdH0TlMqI906D/c/YAuJ86SRdQtBYci6X\nbJ7C+OnoiV6USn1HtQE6dfOMeS8voJuixpSIvHZ/Aim6kSAN1Za1f6FQAkyqbF+o\noKTJHDS1CPWikCeLdpPUcOCDIbsiISTsMZkEvIkzZ7dKBIlIugauxw3vaEpk47jN\nWq09r639RbSv/Qs8D6uY66m1IpL4zHm4lTAknrjM/BqihPxc8YiN76ssajvQ4SFT\nDHPrDweEVe4KL1ENw8nv4wdkIFKwJTDarV5ZygbETzIhfa2JSBZFTdN+Wmd2Mh5h\nOTu+vuHrJF2TO8g1G48EB/KWGt+yvVUpWAanRMwldnFX80NcUlM7GzNn6IXTeE+j\nBttIbvAAVJPG8rVCP8u3DdOf+vgm5macj9oLoVP8RBYo/z0E3e+H50nXv3uS6JhN\nxlAKgaU6i03jOm5+sww5L2YVMi1eeBN+kx7o94ogpRemC/EUidvl1PUJ6+e7an9V\n-----END CERTIFICATE REQUEST-----\n        '
        with tempfile.NamedTemporaryFile('w+') as f:
            f.write(csr_pem)
            f.flush()
            csr_path = f.name
            cn = 'test-csr-32313131.venafi.example.com'
            if not isinstance(cn, str):
                cn = cn.decode()
            ret = self.run_run_plus(fun='venafi.request', minion_id=cn, csr_path=csr_path, zone='fake')
            cert_output = ret['return'][0]
            assert cert_output is not None, 'venafi_certificate not found in `output_value`'
            cert = x509.load_pem_x509_certificate(cert_output.encode(), default_backend())
            assert isinstance(cert, x509.Certificate)
            assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME) == [x509.NameAttribute(NameOID.COMMON_NAME, cn)]