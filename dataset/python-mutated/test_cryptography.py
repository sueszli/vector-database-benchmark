from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import binary, integers
from pytest_pyodide import run_in_pyodide
from pytest_pyodide.fixture import selenium_context_manager

@run_in_pyodide(packages=['cryptography'])
def test_cryptography(selenium):
    if False:
        while True:
            i = 10
    import base64
    from cryptography.fernet import Fernet, MultiFernet
    f1 = Fernet(base64.urlsafe_b64encode(b'\x00' * 32))
    f2 = Fernet(base64.urlsafe_b64encode(b'\x01' * 32))
    f = MultiFernet([f1, f2])
    assert f1.decrypt(f.encrypt(b'abc')) == b'abc'

@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=binary())
def test_fernet(selenium_module_scope, data):
    if False:
        return 10
    sbytes = list(data)
    with selenium_context_manager(selenium_module_scope) as selenium:
        selenium.load_package('cryptography')
        selenium.run(f'\n            from cryptography.fernet import Fernet\n            data = bytes({sbytes})\n            f = Fernet(Fernet.generate_key())\n            ct = f.encrypt(data)\n            assert f.decrypt(ct) == data\n            ')

@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(block_size=integers(min_value=1, max_value=255), data=binary())
def test_pkcs7(selenium_module_scope, block_size, data):
    if False:
        while True:
            i = 10
    sbytes = list(data)
    with selenium_context_manager(selenium_module_scope) as selenium:
        selenium.load_package('cryptography')
        selenium.run(f'\n            from cryptography.hazmat.primitives.padding import ANSIX923, PKCS7\n            block_size = {block_size}\n            data = bytes({sbytes})\n            # Generate in [1, 31] so we can easily get block_size in bits by\n            # multiplying by 8.\n            p = PKCS7(block_size=block_size * 8)\n            padder = p.padder()\n            unpadder = p.unpadder()\n\n            padded = padder.update(data) + padder.finalize()\n\n            assert unpadder.update(padded) + unpadder.finalize() == data\n            ')

@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(block_size=integers(min_value=1, max_value=255), data=binary())
def test_ansix923(selenium_module_scope, block_size, data):
    if False:
        for i in range(10):
            print('nop')
    sbytes = list(data)
    with selenium_context_manager(selenium_module_scope) as selenium:
        selenium.load_package('cryptography')
        selenium.run(f'\n            from cryptography.hazmat.primitives.padding import ANSIX923, PKCS7\n            block_size = {block_size}\n            data = bytes({sbytes})\n            a = ANSIX923(block_size=block_size * 8)\n            padder = a.padder()\n            unpadder = a.unpadder()\n\n            padded = padder.update(data) + padder.finalize()\n\n            assert unpadder.update(padded) + unpadder.finalize() == data\n            ')