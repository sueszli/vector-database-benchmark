import sys
from tribler.core.utilities.utilities import is_frozen

def patch_crypto_be_discovery():
    if False:
        return 10
    "\n    Monkey patches cryptography's backend detection.\n    Objective: support pyinstaller freezing.\n    "
    if (sys.platform == 'win32' or sys.platform == 'darwin') and is_frozen():
        from cryptography.hazmat import backends
        try:
            from cryptography.hazmat.backends.commoncrypto.backend import backend as be_cc
        except ImportError:
            be_cc = None
        try:
            from cryptography.hazmat.backends.openssl.backend import backend as be_ossl
        except ImportError:
            be_ossl = None
        backends._available_backends_list = [be for be in (be_cc, be_ossl) if be is not None]