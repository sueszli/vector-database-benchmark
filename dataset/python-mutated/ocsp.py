"""Tools for checking certificate revocation."""
from datetime import datetime
from datetime import timedelta
import logging
import re
import subprocess
from subprocess import PIPE
from typing import Optional
from typing import Tuple
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import ocsp
import pytz
import requests
from certbot import crypto_util
from certbot import errors
from certbot import util
from certbot.compat.os import getenv
from certbot.interfaces import RenewableCert
logger = logging.getLogger(__name__)

class RevocationChecker:
    """This class figures out OCSP checking on this system, and performs it."""

    def __init__(self, enforce_openssl_binary_usage: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.broken = False
        self.use_openssl_binary = enforce_openssl_binary_usage
        if self.use_openssl_binary:
            if not util.exe_exists('openssl'):
                logger.info("openssl not installed, can't check revocation")
                self.broken = True
                return
            test_host_format = subprocess.run(['openssl', 'ocsp', '-header', 'var', 'val'], stdout=PIPE, stderr=PIPE, universal_newlines=True, check=False, env=util.env_no_snap_for_external_calls())
            if 'Missing =' in test_host_format.stderr:
                self.host_args = lambda host: ['Host=' + host]
            else:
                self.host_args = lambda host: ['Host', host]

    def ocsp_revoked(self, cert: RenewableCert) -> bool:
        if False:
            print('Hello World!')
        'Get revoked status for a particular cert version.\n\n        .. todo:: Make this a non-blocking call\n\n        :param `.interfaces.RenewableCert` cert: Certificate object\n        :returns: True if revoked; False if valid or the check failed or cert is expired.\n        :rtype: bool\n\n        '
        return self.ocsp_revoked_by_paths(cert.cert_path, cert.chain_path)

    def ocsp_revoked_by_paths(self, cert_path: str, chain_path: str, timeout: int=10) -> bool:
        if False:
            while True:
                i = 10
        'Performs the OCSP revocation check\n\n        :param str cert_path: Certificate filepath\n        :param str chain_path: Certificate chain\n        :param int timeout: Timeout (in seconds) for the OCSP query\n\n        :returns: True if revoked; False if valid or the check failed or cert is expired.\n        :rtype: bool\n\n        '
        if self.broken:
            return False
        now = datetime.now(pytz.UTC)
        if crypto_util.notAfter(cert_path) <= now:
            return False
        (url, host) = _determine_ocsp_server(cert_path)
        if not host or not url:
            return False
        if self.use_openssl_binary:
            return self._check_ocsp_openssl_bin(cert_path, chain_path, host, url, timeout)
        return _check_ocsp_cryptography(cert_path, chain_path, url, timeout)

    def _check_ocsp_openssl_bin(self, cert_path: str, chain_path: str, host: str, url: str, timeout: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        env_http_proxy = getenv('http_proxy')
        env_HTTP_PROXY = getenv('HTTP_PROXY')
        proxy_host = None
        if env_http_proxy is not None or env_HTTP_PROXY is not None:
            proxy_host = env_http_proxy if env_http_proxy is not None else env_HTTP_PROXY
        if proxy_host is None:
            url_opts = ['-url', url]
        else:
            if proxy_host.startswith('http://'):
                proxy_host = proxy_host[len('http://'):]
            url_opts = ['-host', proxy_host, '-path', url]
        cmd = ['openssl', 'ocsp', '-no_nonce', '-issuer', chain_path, '-cert', cert_path, '-CAfile', chain_path, '-verify_other', chain_path, '-trust_other', '-timeout', str(timeout), '-header'] + self.host_args(host) + url_opts
        logger.debug('Querying OCSP for %s', cert_path)
        logger.debug(' '.join(cmd))
        try:
            (output, err) = util.run_script(cmd, log=logger.debug)
        except errors.SubprocessError:
            logger.info('OCSP check failed for %s (are we offline?)', cert_path)
            return False
        return _translate_ocsp_query(cert_path, output, err)

def _determine_ocsp_server(cert_path: str) -> Tuple[Optional[str], Optional[str]]:
    if False:
        while True:
            i = 10
    "Extract the OCSP server host from a certificate.\n\n    :param str cert_path: Path to the cert we're checking OCSP for\n    :rtype tuple:\n    :returns: (OCSP server URL or None, OCSP server host or None)\n\n    "
    with open(cert_path, 'rb') as file_handler:
        cert = x509.load_pem_x509_certificate(file_handler.read(), default_backend())
    try:
        extension = cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
        ocsp_oid = x509.AuthorityInformationAccessOID.OCSP
        descriptions = [description for description in extension.value if description.access_method == ocsp_oid]
        url = descriptions[0].access_location.value
    except (x509.ExtensionNotFound, IndexError):
        logger.info('Cannot extract OCSP URI from %s', cert_path)
        return (None, None)
    url = url.rstrip()
    host = url.partition('://')[2].rstrip('/')
    if host:
        return (url, host)
    logger.info('Cannot process OCSP host from URL (%s) in certificate at %s', url, cert_path)
    return (None, None)

def _check_ocsp_cryptography(cert_path: str, chain_path: str, url: str, timeout: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    with open(chain_path, 'rb') as file_handler:
        issuer = x509.load_pem_x509_certificate(file_handler.read(), default_backend())
    with open(cert_path, 'rb') as file_handler:
        cert = x509.load_pem_x509_certificate(file_handler.read(), default_backend())
    builder = ocsp.OCSPRequestBuilder()
    builder = builder.add_certificate(cert, issuer, hashes.SHA1())
    request = builder.build()
    request_binary = request.public_bytes(serialization.Encoding.DER)
    try:
        response = requests.post(url, data=request_binary, headers={'Content-Type': 'application/ocsp-request'}, timeout=timeout)
    except requests.exceptions.RequestException:
        logger.info('OCSP check failed for %s (are we offline?)', cert_path, exc_info=True)
        return False
    if response.status_code != 200:
        logger.info('OCSP check failed for %s (HTTP status: %d)', cert_path, response.status_code)
        return False
    response_ocsp = ocsp.load_der_ocsp_response(response.content)
    if response_ocsp.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
        logger.warning('Invalid OCSP response status for %s: %s', cert_path, response_ocsp.response_status)
        return False
    try:
        _check_ocsp_response(response_ocsp, request, issuer, cert_path)
    except UnsupportedAlgorithm as e:
        logger.warning(str(e))
    except errors.Error as e:
        logger.warning(str(e))
    except InvalidSignature:
        logger.warning('Invalid signature on OCSP response for %s', cert_path)
    except AssertionError as error:
        logger.warning('Invalid OCSP response for %s: %s.', cert_path, str(error))
    else:
        logger.debug('OCSP certificate status for %s is: %s', cert_path, response_ocsp.certificate_status)
        return response_ocsp.certificate_status == ocsp.OCSPCertStatus.REVOKED
    return False

def _check_ocsp_response(response_ocsp: 'ocsp.OCSPResponse', request_ocsp: 'ocsp.OCSPRequest', issuer_cert: x509.Certificate, cert_path: str) -> None:
    if False:
        return 10
    'Verify that the OCSP is valid for several criteria'
    if response_ocsp.serial_number != request_ocsp.serial_number:
        raise AssertionError('the certificate in response does not correspond to the certificate in request')
    _check_ocsp_response_signature(response_ocsp, issuer_cert, cert_path)
    if not isinstance(response_ocsp.hash_algorithm, type(request_ocsp.hash_algorithm)) or response_ocsp.issuer_key_hash != request_ocsp.issuer_key_hash or response_ocsp.issuer_name_hash != request_ocsp.issuer_name_hash:
        raise AssertionError('the issuer does not correspond to issuer of the certificate.')
    now = datetime.now(pytz.UTC).replace(tzinfo=None)
    if not response_ocsp.this_update:
        raise AssertionError('param thisUpdate is not set.')
    if response_ocsp.this_update > now + timedelta(minutes=5):
        raise AssertionError('param thisUpdate is in the future.')
    if response_ocsp.next_update and response_ocsp.next_update < now - timedelta(minutes=5):
        raise AssertionError('param nextUpdate is in the past.')

def _check_ocsp_response_signature(response_ocsp: 'ocsp.OCSPResponse', issuer_cert: x509.Certificate, cert_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Verify an OCSP response signature against certificate issuer or responder'

    def _key_hash(cert: x509.Certificate) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return x509.SubjectKeyIdentifier.from_public_key(cert.public_key()).digest
    if response_ocsp.responder_name == issuer_cert.subject or response_ocsp.responder_key_hash == _key_hash(issuer_cert):
        logger.debug("OCSP response for certificate %s is signed by the certificate's issuer.", cert_path)
        responder_cert = issuer_cert
    else:
        logger.debug('OCSP response for certificate %s is delegated to an external responder.', cert_path)
        responder_certs = [cert for cert in response_ocsp.certificates if response_ocsp.responder_name == cert.subject or response_ocsp.responder_key_hash == _key_hash(cert)]
        if not responder_certs:
            raise AssertionError('no matching responder certificate could be found')
        responder_cert = responder_certs[0]
        if responder_cert.issuer != issuer_cert.subject:
            raise AssertionError("responder certificate is not signed by the certificate's issuer")
        try:
            extension = responder_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
            delegate_authorized = x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING in extension.value
        except (x509.ExtensionNotFound, IndexError):
            delegate_authorized = False
        if not delegate_authorized:
            raise AssertionError('responder is not authorized by issuer to sign OCSP responses')
        chosen_cert_hash = responder_cert.signature_hash_algorithm
        assert chosen_cert_hash
        crypto_util.verify_signed_payload(issuer_cert.public_key(), responder_cert.signature, responder_cert.tbs_certificate_bytes, chosen_cert_hash)
    chosen_response_hash = response_ocsp.signature_hash_algorithm
    if not chosen_response_hash:
        raise AssertionError('no signature hash algorithm defined')
    crypto_util.verify_signed_payload(responder_cert.public_key(), response_ocsp.signature, response_ocsp.tbs_response_bytes, chosen_response_hash)

def _translate_ocsp_query(cert_path: str, ocsp_output: str, ocsp_errors: str) -> bool:
    if False:
        return 10
    "Parse openssl's weird output to work out what it means."
    states = ('good', 'revoked', 'unknown')
    patterns = ['{0}: (WARNING.*)?{1}'.format(cert_path, s) for s in states]
    (good, revoked, unknown) = (re.search(p, ocsp_output, flags=re.DOTALL) for p in patterns)
    warning = good.group(1) if good else None
    if 'Response verify OK' not in ocsp_errors or (good and warning) or unknown:
        logger.info('Revocation status for %s is unknown', cert_path)
        logger.debug('Uncertain output:\n%s\nstderr:\n%s', ocsp_output, ocsp_errors)
        return False
    elif good and (not warning):
        return False
    elif revoked:
        warning = revoked.group(1)
        if warning:
            logger.info('OCSP revocation warning: %s', warning)
        return True
    else:
        logger.warning('Unable to properly parse OCSP output: %s\nstderr:%s', ocsp_output, ocsp_errors)
        return False