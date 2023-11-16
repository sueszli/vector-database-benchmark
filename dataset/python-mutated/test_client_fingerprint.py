import hashlib
from typing import Any
from unittest import mock
import pytest
import aiohttp
ssl: Any = pytest.importorskip('ssl')

def test_fingerprint_sha256() -> None:
    if False:
        for i in range(10):
            print('nop')
    sha256 = hashlib.sha256(b'12345678' * 64).digest()
    fp = aiohttp.Fingerprint(sha256)
    assert fp.fingerprint == sha256

def test_fingerprint_sha1() -> None:
    if False:
        i = 10
        return i + 15
    sha1 = hashlib.sha1(b'12345678' * 64).digest()
    with pytest.raises(ValueError):
        aiohttp.Fingerprint(sha1)

def test_fingerprint_md5() -> None:
    if False:
        i = 10
        return i + 15
    md5 = hashlib.md5(b'12345678' * 64).digest()
    with pytest.raises(ValueError):
        aiohttp.Fingerprint(md5)

def test_fingerprint_check_no_ssl() -> None:
    if False:
        print('Hello World!')
    sha256 = hashlib.sha256(b'12345678' * 64).digest()
    fp = aiohttp.Fingerprint(sha256)
    transport = mock.Mock()
    transport.get_extra_info.return_value = None
    fp.check(transport)