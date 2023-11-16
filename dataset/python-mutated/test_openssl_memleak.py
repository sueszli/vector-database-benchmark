import json
import os
import platform
import subprocess
import sys
import textwrap
import pytest
from cryptography.hazmat.bindings.openssl.binding import Binding
MEMORY_LEAK_SCRIPT = '\nimport sys\n\n\ndef main(argv):\n    import gc\n    import json\n\n    import cffi\n\n    from cryptography.hazmat.bindings._rust import _openssl\n\n    heap = {}\n    start_heap = {}\n    start_heap_realloc_delta = [0]  # 1-item list so callbacks can mutate it\n\n    BACKTRACE_ENABLED = False\n    if BACKTRACE_ENABLED:\n        backtrace_ffi = cffi.FFI()\n        backtrace_ffi.cdef(\'\'\'\n            int backtrace(void **, int);\n            char **backtrace_symbols(void *const *, int);\n        \'\'\')\n        backtrace_lib = backtrace_ffi.dlopen(None)\n\n        def backtrace():\n            buf = backtrace_ffi.new("void*[]", 24)\n            length = backtrace_lib.backtrace(buf, len(buf))\n            return (buf, length)\n\n        def symbolize_backtrace(trace):\n            (buf, length) = trace\n            symbols = backtrace_lib.backtrace_symbols(buf, length)\n            stack = [\n                backtrace_ffi.string(symbols[i]).decode()\n                for i in range(length)\n            ]\n            _openssl.lib.Cryptography_free_wrapper(\n                symbols, backtrace_ffi.NULL, 0\n            )\n            return stack\n    else:\n        def backtrace():\n            return None\n\n        def symbolize_backtrace(trace):\n            return None\n\n    @_openssl.ffi.callback("void *(size_t, const char *, int)")\n    def malloc(size, path, line):\n        ptr = _openssl.lib.Cryptography_malloc_wrapper(size, path, line)\n        heap[ptr] = (size, path, line, backtrace())\n        return ptr\n\n    @_openssl.ffi.callback("void *(void *, size_t, const char *, int)")\n    def realloc(ptr, size, path, line):\n        if ptr != _openssl.ffi.NULL:\n            del heap[ptr]\n        new_ptr = _openssl.lib.Cryptography_realloc_wrapper(\n            ptr, size, path, line\n        )\n        heap[new_ptr] = (size, path, line, backtrace())\n\n        # It is possible that something during the test will cause a\n        # realloc of memory allocated during the startup phase. (This\n        # was observed in conda-forge Windows builds of this package with\n        # provider operation_bits pointers in crypto/provider_core.c.) If\n        # we don\'t pay attention to that, the realloc\'ed pointer will show\n        # up as a leak; but we also don\'t want to allow this kind of realloc\n        # to consume large amounts of additional memory. So we track the\n        # realloc and the change in memory consumption.\n        startup_info = start_heap.pop(ptr, None)\n        if startup_info is not None:\n            start_heap[new_ptr] = heap[new_ptr]\n            start_heap_realloc_delta[0] += size - startup_info[0]\n\n        return new_ptr\n\n    @_openssl.ffi.callback("void(void *, const char *, int)")\n    def free(ptr, path, line):\n        if ptr != _openssl.ffi.NULL:\n            del heap[ptr]\n            _openssl.lib.Cryptography_free_wrapper(ptr, path, line)\n\n    result = _openssl.lib.Cryptography_CRYPTO_set_mem_functions(\n        malloc, realloc, free\n    )\n    assert result == 1\n\n    # Trigger a bunch of initialization stuff.\n    import hashlib\n    from cryptography.hazmat.backends.openssl.backend import backend\n\n    hashlib.sha256()\n\n    start_heap.update(heap)\n\n    try:\n        func(*argv[1:])\n    finally:\n        gc.collect()\n        gc.collect()\n        gc.collect()\n\n        if _openssl.lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:\n            _openssl.lib.OSSL_PROVIDER_unload(backend._binding._legacy_provider)\n            _openssl.lib.OSSL_PROVIDER_unload(backend._binding._default_provider)\n\n        _openssl.lib.OPENSSL_cleanup()\n\n        # Swap back to the original functions so that if OpenSSL tries to free\n        # something from its atexit handle it won\'t be going through a Python\n        # function, which will be deallocated when this function returns\n        result = _openssl.lib.Cryptography_CRYPTO_set_mem_functions(\n            _openssl.ffi.addressof(\n                _openssl.lib, "Cryptography_malloc_wrapper"\n            ),\n            _openssl.ffi.addressof(\n                _openssl.lib, "Cryptography_realloc_wrapper"\n            ),\n            _openssl.ffi.addressof(_openssl.lib, "Cryptography_free_wrapper"),\n        )\n        assert result == 1\n\n    remaining = set(heap) - set(start_heap)\n\n    # The constant here is the number of additional bytes of memory\n    # consumption that are allowed in reallocs of start_heap memory.\n    if remaining or start_heap_realloc_delta[0] > 3072:\n        info = dict(\n            (int(_openssl.ffi.cast("size_t", ptr)), {\n                "size": heap[ptr][0],\n                "path": _openssl.ffi.string(heap[ptr][1]).decode(),\n                "line": heap[ptr][2],\n                "backtrace": symbolize_backtrace(heap[ptr][3]),\n            })\n            for ptr in remaining\n        )\n        info["start_heap_realloc_delta"] = start_heap_realloc_delta[0]\n        sys.stdout.write(json.dumps(info))\n        sys.stdout.flush()\n        sys.exit(255)\n\nmain(sys.argv)\n'

def assert_no_memory_leaks(s, argv=[]):
    if False:
        print('Hello World!')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    env.pop('COV_CORE_CONFIG', None)
    env.pop('COV_CORE_DATAFILE', None)
    env.pop('COV_CORE_SOURCE', None)
    argv = [sys.executable, '-c', f'{s}\n\n{MEMORY_LEAK_SCRIPT}', *argv]
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        proc.wait()
        if proc.returncode == 255:
            out = json.loads(proc.stdout.read().decode())
            raise AssertionError(out)
        elif proc.returncode != 0:
            raise ValueError(proc.stdout.read(), proc.stderr.read())
    finally:
        proc.stdout.close()
        proc.stderr.close()

def skip_if_memtesting_not_supported():
    if False:
        print('Hello World!')
    return pytest.mark.skipif(not Binding().lib.Cryptography_HAS_MEM_FUNCTIONS or platform.python_implementation() == 'PyPy', reason='Requires OpenSSL memory functions (>=1.1.0) and not PyPy')

@pytest.mark.skip_fips(reason='FIPS self-test sets allow_customize = 0')
@skip_if_memtesting_not_supported()
class TestAssertNoMemoryLeaks:

    def test_no_leak_no_malloc(self):
        if False:
            for i in range(10):
                print('nop')
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            pass\n        '))

    def test_no_leak_free(self):
        if False:
            print('Hello World!')
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            from cryptography.hazmat.bindings.openssl.binding import Binding\n            b = Binding()\n            name = b.lib.X509_NAME_new()\n            b.lib.X509_NAME_free(name)\n        '))

    def test_no_leak_gc(self):
        if False:
            i = 10
            return i + 15
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            from cryptography.hazmat.bindings.openssl.binding import Binding\n            b = Binding()\n            name = b.lib.X509_NAME_new()\n            b.ffi.gc(name, b.lib.X509_NAME_free)\n        '))

    def test_leak(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(AssertionError):
            assert_no_memory_leaks(textwrap.dedent('\n            def func():\n                from cryptography.hazmat.bindings.openssl.binding import (\n                    Binding\n                )\n                b = Binding()\n                b.lib.X509_NAME_new()\n            '))

    def test_errors(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='ZeroDivisionError'):
            assert_no_memory_leaks(textwrap.dedent('\n            def func():\n                raise ZeroDivisionError\n            '))

@pytest.mark.skip_fips(reason='FIPS self-test sets allow_customize = 0')
@skip_if_memtesting_not_supported()
class TestOpenSSLMemoryLeaks:

    def test_ec_private_numbers_private_key(self):
        if False:
            i = 10
            return i + 15
        assert_no_memory_leaks(textwrap.dedent("\n        def func():\n            from cryptography.hazmat.backends.openssl import backend\n            from cryptography.hazmat.primitives.asymmetric import ec\n\n            ec.EllipticCurvePrivateNumbers(\n                private_value=int(\n                    '280814107134858470598753916394807521398239633534281633982576099083'\n                    '35787109896602102090002196616273211495718603965098'\n                ),\n                public_numbers=ec.EllipticCurvePublicNumbers(\n                    curve=ec.SECP384R1(),\n                    x=int(\n                        '10036914308591746758780165503819213553101287571902957054148542'\n                        '504671046744460374996612408381962208627004841444205030'\n                    ),\n                    y=int(\n                        '17337335659928075994560513699823544906448896792102247714689323'\n                        '575406618073069185107088229463828921069465902299522926'\n                    )\n                )\n            ).private_key(backend)\n        "))

    def test_ec_derive_private_key(self):
        if False:
            while True:
                i = 10
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            from cryptography.hazmat.backends.openssl import backend\n            from cryptography.hazmat.primitives.asymmetric import ec\n            ec.derive_private_key(1, ec.SECP256R1(), backend)\n        '))

    def test_x25519_pubkey_from_private_key(self):
        if False:
            return 10
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            from cryptography.hazmat.primitives.asymmetric import x25519\n            private_key = x25519.X25519PrivateKey.generate()\n            private_key.public_key()\n        '))

    @pytest.mark.parametrize('path', ['pkcs12/cert-aes256cbc-no-key.p12', 'pkcs12/cert-key-aes256cbc.p12'])
    def test_load_pkcs12_key_and_certificates(self, path):
        if False:
            return 10
        assert_no_memory_leaks(textwrap.dedent('\n        def func(path):\n            from cryptography import x509\n            from cryptography.hazmat.backends.openssl import backend\n            from cryptography.hazmat.primitives.serialization import pkcs12\n            import cryptography_vectors\n\n            with cryptography_vectors.open_vector_file(path, "rb") as f:\n                pkcs12.load_key_and_certificates(\n                    f.read(), b"cryptography", backend\n                )\n        '), [path])

    def test_write_pkcs12_key_and_certificates(self):
        if False:
            print('Hello World!')
        assert_no_memory_leaks(textwrap.dedent('\n        def func():\n            import os\n            from cryptography import x509\n            from cryptography.hazmat.backends.openssl import backend\n            from cryptography.hazmat.primitives import serialization\n            from cryptography.hazmat.primitives.serialization import pkcs12\n            import cryptography_vectors\n\n            path = os.path.join(\'x509\', \'custom\', \'ca\', \'ca.pem\')\n            with cryptography_vectors.open_vector_file(path, "rb") as f:\n                cert = x509.load_pem_x509_certificate(\n                    f.read(), backend\n                )\n            path2 = os.path.join(\'x509\', \'custom\', \'dsa_selfsigned_ca.pem\')\n            with cryptography_vectors.open_vector_file(path2, "rb") as f:\n                cert2 = x509.load_pem_x509_certificate(\n                    f.read(), backend\n                )\n            path3 = os.path.join(\'x509\', \'letsencryptx3.pem\')\n            with cryptography_vectors.open_vector_file(path3, "rb") as f:\n                cert3 = x509.load_pem_x509_certificate(\n                    f.read(), backend\n                )\n            key_path = os.path.join("x509", "custom", "ca", "ca_key.pem")\n            with cryptography_vectors.open_vector_file(key_path, "rb") as f:\n                key = serialization.load_pem_private_key(\n                    f.read(), None, backend\n                )\n            encryption = serialization.NoEncryption()\n            pkcs12.serialize_key_and_certificates(\n                b"name", key, cert, [cert2, cert3], encryption)\n        '))