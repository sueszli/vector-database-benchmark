__all__ = ('apply_ssl_hacks',)
import sys
import socket

def set_default_timeout(timeout=60):
    if False:
        for i in range(10):
            print('nop')
    socket.setdefaulttimeout(timeout)

def change_default_verify_paths():
    if False:
        while True:
            i = 10
    if sys.platform == 'win32':
        return
    import ssl
    setattr(ssl, '_SSL_FILES', ['/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/ca-bundle.pem', '/etc/pki/tls/cacert.pem', '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem'])
    setattr(ssl, '_SSL_PATHS', ['/etc/ssl/certs', '/system/etc/security/cacerts', '/usr/local/share/certs', '/etc/pki/tls/certs', '/etc/openssl/certs', '/etc/opt/csw/ssl/certs'])

    def set_default_verify_paths(self):
        if False:
            return 10
        for path in ssl._SSL_PATHS:
            try:
                self.load_verify_locations(capath=path)
            except:
                pass
        for path in ssl._SSL_FILES:
            try:
                self.load_verify_locations(cafile=path)
            except:
                pass
        del path
    ssl.SSLContext.set_default_verify_paths = set_default_verify_paths

def apply_ssl_hacks():
    if False:
        i = 10
        return i + 15
    set_default_timeout()
    change_default_verify_paths()