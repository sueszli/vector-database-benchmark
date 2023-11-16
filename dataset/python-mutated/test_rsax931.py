"""
Test the RSA ANSI X9.31 signer and verifier
"""
import ctypes
import ctypes.util
import fnmatch
import glob
import os
import platform
import sys
import pytest
import salt.utils.platform
from salt.utils.rsax931 import RSAX931Signer, RSAX931Verifier, _find_libcrypto, _load_libcrypto
from tests.support.mock import patch

@pytest.fixture
def privkey_data():
    if False:
        for i in range(10):
            print('nop')
    return '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA75GR6ZTv5JOv90Vq8tKhKC7YQnhDIo2hM0HVziTEk5R4UQBW\na0CKytFMbTONY2msEDwX9iA0x7F5Lgj0X8eD4ZMsYqLzqjWMekLC8bjhxc+EuPo9\nDygu3mJ2VgRC7XhlFpmdo5NN8J2E7B/CNB3R4hOcMMZNZdi0xLtFoTfwU61UPfFX\n14mV2laqLbvDEfQLJhUTDeFFV8EN5Z4H1ttLP3sMXJvc3EvM0JiDVj4l1TWFUHHz\neFgCA1Im0lv8i7PFrgW7nyMfK9uDSsUmIp7k6ai4tVzwkTmV5PsriP1ju88Lo3MB\n4/sUmDv/JmlZ9YyzTO3Po8Uz3Aeq9HJWyBWHAQIDAQABAoIBAGOzBzBYZUWRGOgl\nIY8QjTT12dY/ymC05GM6gMobjxuD7FZ5d32HDLu/QrknfS3kKlFPUQGDAbQhbbb0\nzw6VL5NO9mfOPO2W/3FaG1sRgBQcerWonoSSSn8OJwVBHMFLG3a+U1Zh1UvPoiPK\nS734swIM+zFpNYivGPvOm/muF/waFf8tF/47t1cwt/JGXYQnkG/P7z0vp47Irpsb\nYjw7vPe4BnbY6SppSxscW3KoV7GtJLFKIxAXbxsuJMF/rYe3O3w2VKJ1Sug1VDJl\n/GytwAkSUer84WwP2b07Wn4c5pCnmLslMgXCLkENgi1NnJMhYVOnckxGDZk54hqP\n9RbLnkkCgYEA/yKuWEvgdzYRYkqpzB0l9ka7Y00CV4Dha9Of6GjQi9i4VCJ/UFVr\nUlhTo5y0ZzpcDAPcoZf5CFZsD90a/BpQ3YTtdln2MMCL/Kr3QFmetkmDrt+3wYnX\nsKESfsa2nZdOATRpl1antpwyD4RzsAeOPwBiACj4fkq5iZJBSI0bxrMCgYEA8GFi\nqAjgKh81/Uai6KWTOW2kX02LEMVRrnZLQ9VPPLGid4KZDDk1/dEfxjjkcyOxX1Ux\nKlu4W8ZEdZyzPcJrfk7PdopfGOfrhWzkREK9C40H7ou/1jUecq/STPfSOmxh3Y+D\nifMNO6z4sQAHx8VaHaxVsJ7SGR/spr0pkZL+NXsCgYEA84rIgBKWB1W+TGRXJzdf\nyHIGaCjXpm2pQMN3LmP3RrcuZWm0vBt94dHcrR5l+u/zc6iwEDTAjJvqdU4rdyEr\ntfkwr7v6TNlQB3WvpWanIPyVzfVSNFX/ZWSsAgZvxYjr9ixw6vzWBXOeOb/Gqu7b\ncvpLkjmJ0wxDhbXtyXKhZA8CgYBZyvcQb+hUs732M4mtQBSD0kohc5TsGdlOQ1AQ\nMcFcmbpnzDghkclyW8jzwdLMk9uxEeDAwuxWE/UEvhlSi6qdzxC+Zifp5NBc0fVe\n7lMx2mfJGxj5CnSqQLVdHQHB4zSXkAGB6XHbBd0MOUeuvzDPfs2voVQ4IG3FR0oc\n3/znuwKBgQChZGH3McQcxmLA28aUwOVbWssfXKdDCsiJO+PEXXlL0maO3SbnFn+Q\nTyf8oHI5cdP7AbwDSx9bUfRPjg9dKKmATBFr2bn216pjGxK0OjYOCntFTVr0psRB\nCrKg52Qrq71/2l4V2NLQZU40Dr1bN9V+Ftd9L0pvpCAEAWpIbLXGDw==\n-----END RSA PRIVATE KEY-----'

@pytest.fixture
def pubkey_data():
    if False:
        return 10
    return '-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA75GR6ZTv5JOv90Vq8tKh\nKC7YQnhDIo2hM0HVziTEk5R4UQBWa0CKytFMbTONY2msEDwX9iA0x7F5Lgj0X8eD\n4ZMsYqLzqjWMekLC8bjhxc+EuPo9Dygu3mJ2VgRC7XhlFpmdo5NN8J2E7B/CNB3R\n4hOcMMZNZdi0xLtFoTfwU61UPfFX14mV2laqLbvDEfQLJhUTDeFFV8EN5Z4H1ttL\nP3sMXJvc3EvM0JiDVj4l1TWFUHHzeFgCA1Im0lv8i7PFrgW7nyMfK9uDSsUmIp7k\n6ai4tVzwkTmV5PsriP1ju88Lo3MB4/sUmDv/JmlZ9YyzTO3Po8Uz3Aeq9HJWyBWH\nAQIDAQAB\n-----END PUBLIC KEY-----'

@pytest.fixture
def hello_world():
    if False:
        i = 10
        return i + 15
    return b'hello, world'

@pytest.fixture
def hello_world_sig():
    if False:
        return 10
    return b'c\xa0p\xd2\xe4\xd4k\x8a\xa2Y\'_\x00i\x1e<P\xedP\x13\t\x80\xe3GN\x14\xb5|\x07&N t\xea\x0e\xf8\xda\xff\x1eW\x8cgvs\xaa\xea\x0f\n\xe7\xa2\xe3\x88\xfc\t\x876\x01:\xb7L@\xe0\xf4T\xc5\xf1\xaa\xb2\x1d\x7f\xb6\xd3\xa8\xdd(i\x8b\x88\xe4B\x1eH>\x1f\xe2+<|\x85\x11\xe9Y\xd7\xf3\xc2!\xd3U\xcb\x9c<\x93\xcc \xdfd\x81\xd0\r\xbf\x8e\x8dG\xec\x1d\x9e\'\xec\x12\xed\x8b_\xd6\x1d\xec\x8dwZX\x8a$\xb6\x0f\x12\xb7Q\xef}\x85\x0fI9\x02\x81\x15\x08p\xd6\xe0\x0b1\xff_\xf9\xd1\x928Y\x8c"\x9c\xbb\xbf\xcf\x854\xe2G\xf5\xe2\xaa\xb4b3<\x13x3\x87\x08\x9e\xb5\xbc]\xc1\xbfy|\xfa_\x06j;\x17@\t\xb9\t\xbf2\xc3\x00\xe2\xbc\x91w\x14\xa5#\xf5\xf5\xf1\t\x128\xda;j\x82\x81{^\x1c\xcb\xaa6\x9b\x086\x03\x14\x96\xa319Y\x16u\xc9\xb6f\x94\x1b\x97\xff\xc8\xa1\xe3!5#\x06L\x9b\xf4\xee'

def test_signer(privkey_data, pubkey_data, hello_world, hello_world_sig):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        signer = RSAX931Signer('bogus key data')
    with pytest.raises(ValueError):
        signer = RSAX931Signer(pubkey_data)
    signer = RSAX931Signer(privkey_data)
    with pytest.raises(ValueError):
        signer.sign('x' * 255)
    sig = signer.sign(hello_world)
    assert hello_world_sig == sig

def test_verifier(privkey_data, pubkey_data, hello_world, hello_world_sig):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        verifier = RSAX931Verifier('bogus key data')
    with pytest.raises(ValueError):
        verifier = RSAX931Verifier(privkey_data)
    verifier = RSAX931Verifier(pubkey_data)
    with pytest.raises(ValueError):
        verifier.verify('')
    with pytest.raises(ValueError):
        verifier.verify(hello_world_sig + b'junk')
    msg = verifier.verify(hello_world_sig)
    assert hello_world == msg

@pytest.mark.skip_unless_on_windows
def test_find_libcrypto_win32():
    if False:
        return 10
    '\n    Test _find_libcrypto on Windows hosts.\n    '
    lib_path = _find_libcrypto()
    assert 'libcrypto' in lib_path

@pytest.mark.skip_unless_on_smartos
def test_find_libcrypto_smartos():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test _find_libcrypto on a SmartOS host.\n    '
    lib_path = _find_libcrypto()
    assert fnmatch.fnmatch(lib_path, os.path.join(os.path.dirname(sys.executable), 'libcrypto*'))

@pytest.mark.skip_unless_on_sunos
def test_find_libcrypto_sunos():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test _find_libcrypto on a Solaris-like host.\n    '
    lib_path = _find_libcrypto()
    passed = False
    for i in ('/opt/local/lib/libcrypto.so*', '/opt/tools/lib/libcrypto.so*'):
        if fnmatch.fnmatch(lib_path, i):
            passed = True
            break
    assert passed

@pytest.mark.skip_unless_on_aix
def test_find_libcrypto_aix():
    if False:
        while True:
            i = 10
    '\n    Test _find_libcrypto on an IBM AIX host.\n    '
    lib_path = _find_libcrypto()
    if os.path.isdir('/opt/salt/lib'):
        assert fnmatch.fnmatch(lib_path, '/opt/salt/lib/libcrypto.so*')
    else:
        assert fnmatch.fnmatch(lib_path, '/opt/freeware/lib/libcrypto.so*')

def test_find_libcrypto_with_system_before_catalina():
    if False:
        print('Hello World!')
    '\n    Test _find_libcrypto on a pre-Catalina macOS host by simulating not\n    finding any other libcryptos and verifying that it defaults to system.\n    '
    with patch.object(salt.utils.platform, 'is_darwin', lambda : True), patch.object(platform, 'mac_ver', lambda : ('10.14.2', (), '')), patch.object(glob, 'glob', lambda _: []), patch.object(sys, 'platform', 'macosx'):
        lib_path = _find_libcrypto()
        assert lib_path == '/usr/lib/libcrypto.dylib'

def test_find_libcrypto_darwin_catalina():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test _find_libcrypto on a macOS Catalina host where there are no custom\n    libcryptos and defaulting to the versioned system libraries.\n    '
    available = ['/usr/lib/libcrypto.0.9.7.dylib', '/usr/lib/libcrypto.0.9.8.dylib', '/usr/lib/libcrypto.35.dylib', '/usr/lib/libcrypto.41.dylib', '/usr/lib/libcrypto.42.dylib', '/usr/lib/libcrypto.44.dylib', '/usr/lib/libcrypto.dylib']

    def test_glob(pattern):
        if False:
            while True:
                i = 10
        return [lib for lib in available if fnmatch.fnmatch(lib, pattern)]
    with patch.object(salt.utils.platform, 'is_darwin', lambda : True), patch.object(platform, 'mac_ver', lambda : ('10.15.2', (), '')), patch.object(sys, 'platform', 'macosx'), patch.object(glob, 'glob', test_glob):
        lib_path = _find_libcrypto()
    assert '/usr/lib/libcrypto.44.dylib' == lib_path

def test_find_libcrypto_darwin_bigsur_packaged():
    if False:
        return 10
    "\n    Test _find_libcrypto on a Darwin-like macOS host where there isn't a\n    lacation returned by ctypes.util.find_library() and the libcrypto\n    installation comes from a package manager (ports, brew, salt).\n    "
    managed_paths = {'salt': '/opt/salt/lib/libcrypto.dylib', 'brew': '/test/homebrew/prefix/opt/openssl/lib/libcrypto.dylib', 'port': '/opt/local/lib/libcrypto.dylib'}
    saved_getenv = os.getenv

    def mock_getenv(env):
        if False:
            print('Hello World!')

        def test_getenv(var, default=None):
            if False:
                return 10
            return env.get(var, saved_getenv(var, default))
        return test_getenv

    def mock_glob(expected_lib):
        if False:
            for i in range(10):
                print('nop')

        def test_glob(pattern):
            if False:
                while True:
                    i = 10
            if fnmatch.fnmatch(expected_lib, pattern):
                return [expected_lib]
            return []
        return test_glob
    with patch.object(salt.utils.platform, 'is_darwin', lambda : True), patch.object(platform, 'mac_ver', lambda : ('11.2.2', (), '')), patch.object(sys, 'platform', 'macosx'):
        for (package_manager, expected_lib) in managed_paths.items():
            if package_manager == 'brew':
                env = {'HOMEBREW_PREFIX': '/test/homebrew/prefix'}
            else:
                env = {'HOMEBREW_PREFIX': ''}
            with patch.object(os, 'getenv', mock_getenv(env)):
                with patch.object(glob, 'glob', mock_glob(expected_lib)):
                    lib_path = _find_libcrypto()
            assert expected_lib == lib_path
        with patch.object(glob, 'glob', lambda _: []):
            with pytest.raises(OSError):
                lib_path = _find_libcrypto()

def test_find_libcrypto_unsupported():
    if False:
        while True:
            i = 10
    '\n    Ensure that _find_libcrypto works correctly on an unsupported host OS.\n    '
    with patch.object(ctypes.util, 'find_library', lambda a: None), patch.object(glob, 'glob', lambda a: []), patch.object(sys, 'platform', 'unknown'), patch.object(salt.utils.platform, 'is_darwin', lambda : False), pytest.raises(OSError):
        _find_libcrypto()

def test_load_libcrypto():
    if False:
        return 10
    '\n    Test _load_libcrypto generically.\n    '
    lib = _load_libcrypto()
    assert isinstance(lib, ctypes.CDLL)
    assert hasattr(lib, 'OpenSSL_version_num') or hasattr(lib, 'OPENSSL_init_crypto') or hasattr(lib, 'OPENSSL_no_config')

def test_find_libcrypto_darwin_onedir():
    if False:
        return 10
    '\n    Test _find_libcrypto on a macOS\n    libcryptos and defaulting to the versioned system libraries.\n    '
    available = ['/usr/lib/libcrypto.0.9.7.dylib', '/usr/lib/libcrypto.0.9.8.dylib', '/usr/lib/libcrypto.35.dylib', '/usr/lib/libcrypto.41.dylib', '/usr/lib/libcrypto.42.dylib', '/usr/lib/libcrypto.44.dylib', '/test/homebrew/prefix/opt/openssl/lib/libcrypto.dylib', '/opt/local/lib/libcrypto.dylib', 'lib/libcrypto.dylib']

    def test_glob(pattern):
        if False:
            while True:
                i = 10
        return [lib for lib in available if fnmatch.fnmatch(lib, pattern)]
    with patch.object(glob, 'glob', test_glob), patch.object(salt.utils.platform, 'is_darwin', lambda : True), patch.object(platform, 'mac_ver', lambda : ('10.15.2', (), '')), patch.object(sys, 'platform', 'macosx'):
        lib_path = _find_libcrypto()
    assert 'lib/libcrypto.dylib' == lib_path