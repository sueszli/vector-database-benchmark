"""Tests for ocsp.py"""
import contextlib
from datetime import datetime
from datetime import timedelta
import sys
import unittest
from unittest import mock
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.x509 import ocsp as ocsp_lib
import pytest
import pytz
from certbot import errors
from certbot.tests import util as test_util
out = 'Missing = in header key=value\nocsp: Use -help for summary.\n'

class OCSPTestOpenSSL(unittest.TestCase):
    """
    OCSP revocation tests using OpenSSL binary.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        from certbot import ocsp
        with mock.patch('certbot.ocsp.subprocess.run') as mock_run:
            with mock.patch('certbot.util.exe_exists') as mock_exists:
                mock_run.stderr = out
                mock_exists.return_value = True
                self.checker = ocsp.RevocationChecker(enforce_openssl_binary_usage=True)

    @mock.patch('certbot.ocsp.logger.info')
    @mock.patch('certbot.ocsp.subprocess.run')
    @mock.patch('certbot.util.exe_exists')
    def test_init(self, mock_exists, mock_run, mock_log):
        if False:
            i = 10
            return i + 15
        mock_run.return_value.stderr = out
        mock_exists.return_value = True
        from certbot import ocsp
        checker = ocsp.RevocationChecker(enforce_openssl_binary_usage=True)
        assert mock_run.call_count == 1
        assert checker.host_args('x') == ['Host=x']
        mock_run.return_value.stderr = out.partition('\n')[2]
        checker = ocsp.RevocationChecker(enforce_openssl_binary_usage=True)
        assert checker.host_args('x') == ['Host', 'x']
        assert checker.broken is False
        mock_exists.return_value = False
        mock_run.call_count = 0
        checker = ocsp.RevocationChecker(enforce_openssl_binary_usage=True)
        assert mock_run.call_count == 0
        assert mock_log.call_count == 1
        assert checker.broken is True

    @mock.patch('certbot.ocsp._determine_ocsp_server')
    @mock.patch('certbot.ocsp.crypto_util.notAfter')
    @mock.patch('certbot.util.run_script')
    def test_ocsp_revoked(self, mock_run, mock_na, mock_determine):
        if False:
            i = 10
            return i + 15
        now = datetime.now(pytz.UTC)
        cert_obj = mock.MagicMock()
        cert_obj.cert_path = 'x'
        cert_obj.chain_path = 'y'
        mock_na.return_value = now + timedelta(hours=2)
        self.checker.broken = True
        mock_determine.return_value = ('', '')
        assert self.checker.ocsp_revoked(cert_obj) is False
        self.checker.broken = False
        mock_run.return_value = tuple(openssl_happy[1:])
        assert self.checker.ocsp_revoked(cert_obj) is False
        assert mock_run.call_count == 0
        mock_determine.return_value = ('http://x.co', 'x.co')
        assert self.checker.ocsp_revoked(cert_obj) is False
        mock_run.side_effect = errors.SubprocessError('Unable to load certificate launcher')
        assert self.checker.ocsp_revoked(cert_obj) is False
        assert mock_run.call_count == 2
        mock_na.return_value = now
        mock_determine.return_value = ('', '')
        count_before = mock_determine.call_count
        assert self.checker.ocsp_revoked(cert_obj) is False
        assert mock_determine.call_count == count_before

    def test_determine_ocsp_server(self):
        if False:
            for i in range(10):
                print('nop')
        cert_path = test_util.vector_path('ocsp_certificate.pem')
        from certbot import ocsp
        result = ocsp._determine_ocsp_server(cert_path)
        assert ('http://ocsp.test4.buypass.com', 'ocsp.test4.buypass.com') == result

    @mock.patch('certbot.ocsp.logger')
    @mock.patch('certbot.util.run_script')
    def test_translate_ocsp(self, mock_run, mock_log):
        if False:
            while True:
                i = 10
        mock_run.return_value = openssl_confused
        from certbot import ocsp
        assert ocsp._translate_ocsp_query(*openssl_happy) is False
        assert ocsp._translate_ocsp_query(*openssl_confused) is False
        assert mock_log.debug.call_count == 1
        assert mock_log.warning.call_count == 0
        mock_log.debug.call_count = 0
        assert ocsp._translate_ocsp_query(*openssl_unknown) is False
        assert mock_log.debug.call_count == 1
        assert mock_log.warning.call_count == 0
        assert ocsp._translate_ocsp_query(*openssl_expired_ocsp) is False
        assert mock_log.debug.call_count == 2
        assert ocsp._translate_ocsp_query(*openssl_broken) is False
        assert mock_log.warning.call_count == 1
        mock_log.info.call_count = 0
        assert ocsp._translate_ocsp_query(*openssl_revoked) is True
        assert mock_log.info.call_count == 0
        assert ocsp._translate_ocsp_query(*openssl_expired_ocsp_revoked) is True
        assert mock_log.info.call_count == 1

class OSCPTestCryptography(unittest.TestCase):
    """
    OCSP revokation tests using Cryptography >= 2.4.0
    """

    def setUp(self):
        if False:
            print('Hello World!')
        from certbot import ocsp
        self.checker = ocsp.RevocationChecker()
        self.cert_path = test_util.vector_path('ocsp_certificate.pem')
        self.chain_path = test_util.vector_path('ocsp_issuer_certificate.pem')
        self.cert_obj = mock.MagicMock()
        self.cert_obj.cert_path = self.cert_path
        self.cert_obj.chain_path = self.chain_path
        now = datetime.now(pytz.UTC)
        self.mock_notAfter = mock.patch('certbot.ocsp.crypto_util.notAfter', return_value=now + timedelta(hours=2))
        self.mock_notAfter.start()
        self.addCleanup(self.mock_notAfter.stop)

    @mock.patch('certbot.ocsp._determine_ocsp_server')
    @mock.patch('certbot.ocsp._check_ocsp_cryptography')
    def test_ensure_cryptography_toggled(self, mock_check, mock_determine):
        if False:
            return 10
        mock_determine.return_value = ('http://example.com', 'example.com')
        self.checker.ocsp_revoked(self.cert_obj)
        mock_check.assert_called_once_with(self.cert_path, self.chain_path, 'http://example.com', 10)

    def test_revoke(self):
        if False:
            for i in range(10):
                print('nop')
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked

    def test_responder_is_issuer(self):
        if False:
            for i in range(10):
                print('nop')
        issuer = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_issuer_certificate.pem'), default_backend())
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL) as mocks:
            mocks['mock_response'].return_value.responder_name = issuer.subject
            mocks['mock_response'].return_value.responder_key_hash = None
            self.checker.ocsp_revoked(self.cert_obj)
            key_hash = x509.SubjectKeyIdentifier.from_public_key(issuer.public_key()).digest
            mocks['mock_response'].return_value.responder_name = None
            mocks['mock_response'].return_value.responder_key_hash = key_hash
            self.checker.ocsp_revoked(self.cert_obj)
        assert mocks['mock_check'].call_count == 2
        assert mocks['mock_check'].call_args_list[0][0][0].public_numbers() == issuer.public_key().public_numbers()
        assert mocks['mock_check'].call_args_list[1][0][0].public_numbers() == issuer.public_key().public_numbers()

    def test_responder_is_authorized_delegate(self):
        if False:
            i = 10
            return i + 15
        issuer = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_issuer_certificate.pem'), default_backend())
        responder = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_responder_certificate.pem'), default_backend())
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL) as mocks:
            mocks['mock_response'].return_value.responder_name = responder.subject
            mocks['mock_response'].return_value.responder_key_hash = None
            self.checker.ocsp_revoked(self.cert_obj)
            key_hash = x509.SubjectKeyIdentifier.from_public_key(responder.public_key()).digest
            mocks['mock_response'].return_value.responder_name = None
            mocks['mock_response'].return_value.responder_key_hash = key_hash
            self.checker.ocsp_revoked(self.cert_obj)
        assert mocks['mock_check'].call_count == 4
        assert mocks['mock_check'].call_args_list[0][0][0].public_numbers() == issuer.public_key().public_numbers()
        assert mocks['mock_check'].call_args_list[1][0][0].public_numbers() == responder.public_key().public_numbers()
        assert mocks['mock_check'].call_args_list[2][0][0].public_numbers() == issuer.public_key().public_numbers()
        assert mocks['mock_check'].call_args_list[3][0][0].public_numbers() == responder.public_key().public_numbers()

    def test_revoke_resiliency(self):
        if False:
            print('Hello World!')
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.UNKNOWN, ocsp_lib.OCSPResponseStatus.SUCCESSFUL, http_status_code=400):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.UNKNOWN, ocsp_lib.OCSPResponseStatus.UNAUTHORIZED):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.UNKNOWN, ocsp_lib.OCSPResponseStatus.SUCCESSFUL):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL):
            with mock.patch('cryptography.x509.Extensions.get_extension_for_class', side_effect=x509.ExtensionNotFound('Not found', x509.AuthorityInformationAccessOID.OCSP)):
                revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL, check_signature_side_effect=UnsupportedAlgorithm('foo')):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL, check_signature_side_effect=InvalidSignature('foo')):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL, check_signature_side_effect=AssertionError('foo')):
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL) as mocks:
            mocks['mock_response'].return_value.certificates = []
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL) as mocks:
            cert = mocks['mock_response'].return_value.certificates[0]
            mocks['mock_response'].return_value.certificates[0] = mock.Mock(issuer='fake', subject=cert.subject)
            revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False
        with _ocsp_mock(ocsp_lib.OCSPCertStatus.REVOKED, ocsp_lib.OCSPResponseStatus.SUCCESSFUL):
            with mock.patch('certbot.ocsp._determine_ocsp_server') as mock_server:
                mock_server.return_value = ('https://example.com', 'example.com')
                with mock.patch('cryptography.x509.Extensions.get_extension_for_class', side_effect=x509.ExtensionNotFound('Not found', x509.AuthorityInformationAccessOID.OCSP)):
                    revoked = self.checker.ocsp_revoked(self.cert_obj)
        assert revoked is False

@contextlib.contextmanager
def _ocsp_mock(certificate_status, response_status, http_status_code=200, check_signature_side_effect=None):
    if False:
        i = 10
        return i + 15
    with mock.patch('certbot.ocsp.ocsp.load_der_ocsp_response') as mock_response:
        mock_response.return_value = _construct_mock_ocsp_response(certificate_status, response_status)
        with mock.patch('certbot.ocsp.requests.post') as mock_post:
            mock_post.return_value = mock.Mock(status_code=http_status_code)
            with mock.patch('certbot.ocsp.crypto_util.verify_signed_payload') as mock_check:
                if check_signature_side_effect:
                    mock_check.side_effect = check_signature_side_effect
                yield {'mock_response': mock_response, 'mock_post': mock_post, 'mock_check': mock_check}

def _construct_mock_ocsp_response(certificate_status, response_status):
    if False:
        print('Hello World!')
    cert = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_certificate.pem'), default_backend())
    issuer = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_issuer_certificate.pem'), default_backend())
    responder = x509.load_pem_x509_certificate(test_util.load_vector('ocsp_responder_certificate.pem'), default_backend())
    builder = ocsp_lib.OCSPRequestBuilder()
    builder = builder.add_certificate(cert, issuer, hashes.SHA1())
    request = builder.build()
    return mock.Mock(response_status=response_status, certificate_status=certificate_status, serial_number=request.serial_number, issuer_key_hash=request.issuer_key_hash, issuer_name_hash=request.issuer_name_hash, responder_name=responder.subject, certificates=[responder], hash_algorithm=hashes.SHA1(), next_update=datetime.now(pytz.UTC).replace(tzinfo=None) + timedelta(days=1), this_update=datetime.now(pytz.UTC).replace(tzinfo=None) - timedelta(days=1), signature_algorithm_oid=x509.oid.SignatureAlgorithmOID.RSA_WITH_SHA1)
openssl_confused = ('', '\n/etc/letsencrypt/live/example.org/cert.pem: good\n\tThis Update: Dec 17 00:00:00 2016 GMT\n\tNext Update: Dec 24 00:00:00 2016 GMT\n', '\nResponse Verify Failure\n139903674214048:error:27069065:OCSP routines:OCSP_basic_verify:certificate verify error:ocsp_vfy.c:138:Verify error:unable to get local issuer certificate\n')
openssl_happy = ('blah.pem', '\nblah.pem: good\n\tThis Update: Dec 20 18:00:00 2016 GMT\n\tNext Update: Dec 27 18:00:00 2016 GMT\n', 'Response verify OK')
openssl_revoked = ('blah.pem', '\nblah.pem: revoked\n\tThis Update: Dec 20 01:00:00 2016 GMT\n\tNext Update: Dec 27 01:00:00 2016 GMT\n\tRevocation Time: Dec 20 01:46:34 2016 GMT\n', 'Response verify OK')
openssl_unknown = ('blah.pem', '\nblah.pem: unknown\n\tThis Update: Dec 20 18:00:00 2016 GMT\n\tNext Update: Dec 27 18:00:00 2016 GMT\n', 'Response verify OK')
openssl_broken = ('', 'tentacles', 'Response verify OK')
openssl_expired_ocsp = ('blah.pem', '\nblah.pem: WARNING: Status times invalid.\n140659132298912:error:2707307D:OCSP routines:OCSP_check_validity:status expired:ocsp_cl.c:372:\ngood\n\tThis Update: Apr  6 00:00:00 2016 GMT\n\tNext Update: Apr 13 00:00:00 2016 GMT\n', 'Response verify OK')
openssl_expired_ocsp_revoked = ('blah.pem', '\nblah.pem: WARNING: Status times invalid.\n140659132298912:error:2707307D:OCSP routines:OCSP_check_validity:status expired:ocsp_cl.c:372:\nrevoked\n\tThis Update: Apr  6 00:00:00 2016 GMT\n\tNext Update: Apr 13 00:00:00 2016 GMT\n', 'Response verify OK')
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))