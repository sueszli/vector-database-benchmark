"""
Calculations for HTTP Digest authentication.

@see: U{http://www.faqs.org/rfcs/rfc2617.html}
"""
from binascii import hexlify
from hashlib import md5, sha1
algorithms = {b'md5': md5, b'md5-sess': md5, b'sha': sha1}

def calcHA1(pszAlg, pszUserName, pszRealm, pszPassword, pszNonce, pszCNonce, preHA1=None):
    if False:
        while True:
            i = 10
    '\n    Compute H(A1) from RFC 2617.\n\n    @param pszAlg: The name of the algorithm to use to calculate the digest.\n        Currently supported are md5, md5-sess, and sha.\n    @param pszUserName: The username\n    @param pszRealm: The realm\n    @param pszPassword: The password\n    @param pszNonce: The nonce\n    @param pszCNonce: The cnonce\n\n    @param preHA1: If available this is a str containing a previously\n       calculated H(A1) as a hex string.  If this is given then the values for\n       pszUserName, pszRealm, and pszPassword must be L{None} and are ignored.\n    '
    if preHA1 and (pszUserName or pszRealm or pszPassword):
        raise TypeError('preHA1 is incompatible with the pszUserName, pszRealm, and pszPassword arguments')
    if preHA1 is None:
        m = algorithms[pszAlg]()
        m.update(pszUserName)
        m.update(b':')
        m.update(pszRealm)
        m.update(b':')
        m.update(pszPassword)
        HA1 = hexlify(m.digest())
    else:
        HA1 = preHA1
    if pszAlg == b'md5-sess':
        m = algorithms[pszAlg]()
        m.update(HA1)
        m.update(b':')
        m.update(pszNonce)
        m.update(b':')
        m.update(pszCNonce)
        HA1 = hexlify(m.digest())
    return HA1

def calcHA2(algo, pszMethod, pszDigestUri, pszQop, pszHEntity):
    if False:
        for i in range(10):
            print('nop')
    "\n    Compute H(A2) from RFC 2617.\n\n    @param algo: The name of the algorithm to use to calculate the digest.\n        Currently supported are md5, md5-sess, and sha.\n    @param pszMethod: The request method.\n    @param pszDigestUri: The request URI.\n    @param pszQop: The Quality-of-Protection value.\n    @param pszHEntity: The hash of the entity body or L{None} if C{pszQop} is\n        not C{'auth-int'}.\n    @return: The hash of the A2 value for the calculation of the response\n        digest.\n    "
    m = algorithms[algo]()
    m.update(pszMethod)
    m.update(b':')
    m.update(pszDigestUri)
    if pszQop == b'auth-int':
        m.update(b':')
        m.update(pszHEntity)
    return hexlify(m.digest())

def calcResponse(HA1, HA2, algo, pszNonce, pszNonceCount, pszCNonce, pszQop):
    if False:
        print('Hello World!')
    '\n    Compute the digest for the given parameters.\n\n    @param HA1: The H(A1) value, as computed by L{calcHA1}.\n    @param HA2: The H(A2) value, as computed by L{calcHA2}.\n    @param pszNonce: The challenge nonce.\n    @param pszNonceCount: The (client) nonce count value for this response.\n    @param pszCNonce: The client nonce.\n    @param pszQop: The Quality-of-Protection value.\n    '
    m = algorithms[algo]()
    m.update(HA1)
    m.update(b':')
    m.update(pszNonce)
    m.update(b':')
    if pszNonceCount and pszCNonce:
        m.update(pszNonceCount)
        m.update(b':')
        m.update(pszCNonce)
        m.update(b':')
        m.update(pszQop)
        m.update(b':')
    m.update(HA2)
    respHash = hexlify(m.digest())
    return respHash