import time
from base64 import b32encode
from urllib.parse import parse_qsl
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hashes import SHA1
from cryptography.hazmat.primitives.twofactor.totp import TOTP
from urllib3.util import parse_url
import warehouse.utils.otp as otp

def test_generate_totp_secret():
    if False:
        i = 10
        return i + 15
    secret = otp.generate_totp_secret()
    assert type(secret) is bytes
    assert len(secret) == 20

def test_generate_totp_provisioning_uri():
    if False:
        print('Hello World!')
    secret = b'F' * 32
    username = 'pony'
    issuer_name = 'pypi.org'
    uri = otp.generate_totp_provisioning_uri(secret, username, issuer_name=issuer_name)
    parsed = parse_url(uri)
    assert parsed.scheme == 'otpauth'
    assert parsed.netloc == 'totp'
    assert parsed.path == f'/{issuer_name}:{username}'
    query = parse_qsl(parsed.query)
    assert ('digits', '6') in query
    assert ('secret', b32encode(secret).decode()) in query
    assert ('algorithm', 'SHA1') in query
    assert ('issuer', issuer_name) in query
    assert ('period', '30') in query

@pytest.mark.parametrize('skew', [0, -20, 20])
def test_verify_totp_success(skew):
    if False:
        print('Hello World!')
    secret = otp.generate_totp_secret()
    totp = TOTP(secret, otp.TOTP_LENGTH, SHA1(), otp.TOTP_INTERVAL, backend=default_backend())
    value = totp.generate(time.time() + skew)
    assert otp.verify_totp(secret, value)

@pytest.mark.parametrize('skew', [-60, 60])
def test_verify_totp_failure(skew):
    if False:
        print('Hello World!')
    secret = otp.generate_totp_secret()
    totp = TOTP(secret, otp.TOTP_LENGTH, SHA1(), otp.TOTP_INTERVAL, backend=default_backend())
    value = totp.generate(time.time() + skew)
    with pytest.raises(otp.OutOfSyncTOTPError):
        otp.verify_totp(secret, value)