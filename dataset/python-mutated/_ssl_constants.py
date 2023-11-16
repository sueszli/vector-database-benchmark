import ssl
import sys
import typing
_original_SSLContext = ssl.SSLContext
_original_super_SSLContext = super(_original_SSLContext, _original_SSLContext)
_truststore_SSLContext_dunder_class: typing.Optional[type]
_truststore_SSLContext_super_class: type
if sys.implementation.name == 'cpython':
    _truststore_SSLContext_super_class = _original_SSLContext
    _truststore_SSLContext_dunder_class = None
else:
    _truststore_SSLContext_super_class = object
    _truststore_SSLContext_dunder_class = _original_SSLContext

def _set_ssl_context_verify_mode(ssl_context: ssl.SSLContext, verify_mode: ssl.VerifyMode) -> None:
    if False:
        print('Hello World!')
    _original_super_SSLContext.verify_mode.__set__(ssl_context, verify_mode)