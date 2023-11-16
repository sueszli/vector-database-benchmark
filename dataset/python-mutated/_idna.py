"""
Shared interface to IDNA encoding and decoding, using the C{idna} PyPI package
if available, otherwise the stdlib implementation.
"""

def _idnaBytes(text: str) -> bytes:
    if False:
        print('Hello World!')
    "\n    Convert some text typed by a human into some ASCII bytes.\n\n    This is provided to allow us to use the U{partially-broken IDNA\n    implementation in the standard library <http://bugs.python.org/issue17305>}\n    if the more-correct U{idna <https://pypi.python.org/pypi/idna>} package is\n    not available; C{service_identity} is somewhat stricter about this.\n\n    @param text: A domain name, hopefully.\n    @type text: L{unicode}\n\n    @return: The domain name's IDNA representation, encoded as bytes.\n    @rtype: L{bytes}\n    "
    try:
        import idna
    except ImportError:
        return text.encode('idna')
    else:
        return idna.encode(text)

def _idnaText(octets: bytes) -> str:
    if False:
        return 10
    '\n    Convert some IDNA-encoded octets into some human-readable text.\n\n    Currently only used by the tests.\n\n    @param octets: Some bytes representing a hostname.\n    @type octets: L{bytes}\n\n    @return: A human-readable domain name.\n    @rtype: L{unicode}\n    '
    try:
        import idna
    except ImportError:
        return octets.decode('idna')
    else:
        return idna.decode(octets)