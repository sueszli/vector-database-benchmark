import contextlib
import os
import re
import ssl
import typing
_CA_FILE_CANDIDATES = ['/etc/ssl/cert.pem', '/etc/pki/tls/cert.pem', '/etc/ssl/certs/ca-certificates.crt', '/etc/ssl/ca-bundle.pem']
_HASHED_CERT_FILENAME_RE = re.compile('^[0-9a-fA-F]{8}\\.[0-9]$')

@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    if False:
        print('Hello World!')
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile or (defaults.capath and _capath_contains_certs(defaults.capath)):
        ctx.set_default_verify_paths()
    else:
        for cafile in _CA_FILE_CANDIDATES:
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile=cafile)
                break
    yield

def _capath_contains_certs(capath: str) -> bool:
    if False:
        return 10
    'Check whether capath exists and contains certs in the expected format.'
    if not os.path.isdir(capath):
        return False
    for name in os.listdir(capath):
        if _HASHED_CERT_FILENAME_RE.match(name):
            return True
    return False

def _verify_peercerts_impl(ssl_context: ssl.SSLContext, cert_chain: list[bytes], server_hostname: str | None=None) -> None:
    if False:
        while True:
            i = 10
    pass