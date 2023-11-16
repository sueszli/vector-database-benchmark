"""
SSH key exchange handling.
"""
from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error

class _IKexAlgorithm(Interface):
    """
    An L{_IKexAlgorithm} describes a key exchange algorithm.
    """
    preference = Attribute('An L{int} giving the preference of the algorithm when negotiating key exchange. Algorithms with lower precedence values are more preferred.')
    hashProcessor = Attribute('A callable hash algorithm constructor (e.g. C{hashlib.sha256}) suitable for use with this key exchange algorithm.')

class _IFixedGroupKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IFixedGroupKexAlgorithm} describes a key exchange algorithm with a
    fixed prime / generator group.
    """
    prime = Attribute('An L{int} giving the prime number used in Diffie-Hellman key exchange, or L{None} if not applicable.')
    generator = Attribute('An L{int} giving the generator number used in Diffie-Hellman key exchange, or L{None} if not applicable. (This is not related to Python generator functions.)')

class _IEllipticCurveExchangeKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IEllipticCurveExchangeKexAlgorithm} describes a key exchange algorithm
    that uses an elliptic curve exchange between the client and server.
    """

class _IGroupExchangeKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IGroupExchangeKexAlgorithm} describes a key exchange algorithm
    that uses group exchange between the client and server.

    A prime / generator group should be chosen at run time based on the
    requested size. See RFC 4419.
    """

@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _Curve25519SHA256:
    """
    Elliptic Curve Key Exchange using Curve25519 and SHA256. Defined in
    U{https://datatracker.ietf.org/doc/draft-ietf-curdle-ssh-curves/}.
    """
    preference = 1
    hashProcessor = sha256

@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _Curve25519SHA256LibSSH:
    """
    As L{_Curve25519SHA256}, but with a pre-standardized algorithm name.
    """
    preference = 2
    hashProcessor = sha256

@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _ECDH256:
    """
    Elliptic Curve Key Exchange with SHA-256 as HASH. Defined in
    RFC 5656.

    Note that C{ecdh-sha2-nistp256} takes priority over nistp384 or nistp512.
    This is the same priority from OpenSSH.

    C{ecdh-sha2-nistp256} is considered preety good cryptography.
    If you need something better consider using C{curve25519-sha256}.
    """
    preference = 3
    hashProcessor = sha256

@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _ECDH384:
    """
    Elliptic Curve Key Exchange with SHA-384 as HASH. Defined in
    RFC 5656.
    """
    preference = 4
    hashProcessor = sha384

@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _ECDH512:
    """
    Elliptic Curve Key Exchange with SHA-512 as HASH. Defined in
    RFC 5656.
    """
    preference = 5
    hashProcessor = sha512

@implementer(_IGroupExchangeKexAlgorithm)
class _DHGroupExchangeSHA256:
    """
    Diffie-Hellman Group and Key Exchange with SHA-256 as HASH. Defined in
    RFC 4419, 4.2.
    """
    preference = 6
    hashProcessor = sha256

@implementer(_IGroupExchangeKexAlgorithm)
class _DHGroupExchangeSHA1:
    """
    Diffie-Hellman Group and Key Exchange with SHA-1 as HASH. Defined in
    RFC 4419, 4.1.
    """
    preference = 7
    hashProcessor = sha1

@implementer(_IFixedGroupKexAlgorithm)
class _DHGroup14SHA1:
    """
    Diffie-Hellman key exchange with SHA-1 as HASH and Oakley Group 14
    (2048-bit MODP Group). Defined in RFC 4253, 8.2.
    """
    preference = 8
    hashProcessor = sha1
    prime = int('32317006071311007300338913926423828248817941241140239112842009751400741706634354222619689417363569347117901737909704191754605873209195028853758986185622153212175412514901774520270235796078236248884246189477587641105928646099411723245426622522193230540919037680524235519125679715870117001058055877651038861847280257976054903569732561526167081339361799541336476559160368317896729073178384589680639671900977202194168647225871031411336429319536193471636533209717077448227988588565369208645296636077250268955505928362751121174096972998068410554359584866583291642136218231078990999448652468262416972035911852507045361090559')
    generator = 2
_kexAlgorithms = {b'curve25519-sha256': _Curve25519SHA256(), b'curve25519-sha256@libssh.org': _Curve25519SHA256LibSSH(), b'diffie-hellman-group-exchange-sha256': _DHGroupExchangeSHA256(), b'diffie-hellman-group-exchange-sha1': _DHGroupExchangeSHA1(), b'diffie-hellman-group14-sha1': _DHGroup14SHA1(), b'ecdh-sha2-nistp256': _ECDH256(), b'ecdh-sha2-nistp384': _ECDH384(), b'ecdh-sha2-nistp521': _ECDH512()}

def getKex(kexAlgorithm):
    if False:
        return 10
    '\n    Get a description of a named key exchange algorithm.\n\n    @param kexAlgorithm: The key exchange algorithm name.\n    @type kexAlgorithm: L{bytes}\n\n    @return: A description of the key exchange algorithm named by\n        C{kexAlgorithm}.\n    @rtype: L{_IKexAlgorithm}\n\n    @raises ConchError: if the key exchange algorithm is not found.\n    '
    if kexAlgorithm not in _kexAlgorithms:
        raise error.ConchError(f'Unsupported key exchange algorithm: {kexAlgorithm}')
    return _kexAlgorithms[kexAlgorithm]

def isEllipticCurve(kexAlgorithm):
    if False:
        print('Hello World!')
    '\n    Returns C{True} if C{kexAlgorithm} is an elliptic curve.\n\n    @param kexAlgorithm: The key exchange algorithm name.\n    @type kexAlgorithm: C{str}\n\n    @return: C{True} if C{kexAlgorithm} is an elliptic curve,\n        otherwise C{False}.\n    @rtype: C{bool}\n    '
    return _IEllipticCurveExchangeKexAlgorithm.providedBy(getKex(kexAlgorithm))

def isFixedGroup(kexAlgorithm):
    if False:
        print('Hello World!')
    '\n    Returns C{True} if C{kexAlgorithm} has a fixed prime / generator group.\n\n    @param kexAlgorithm: The key exchange algorithm name.\n    @type kexAlgorithm: L{bytes}\n\n    @return: C{True} if C{kexAlgorithm} has a fixed prime / generator group,\n        otherwise C{False}.\n    @rtype: L{bool}\n    '
    return _IFixedGroupKexAlgorithm.providedBy(getKex(kexAlgorithm))

def getHashProcessor(kexAlgorithm):
    if False:
        print('Hello World!')
    '\n    Get the hash algorithm callable to use in key exchange.\n\n    @param kexAlgorithm: The key exchange algorithm name.\n    @type kexAlgorithm: L{bytes}\n\n    @return: A callable hash algorithm constructor (e.g. C{hashlib.sha256}).\n    @rtype: C{callable}\n    '
    kex = getKex(kexAlgorithm)
    return kex.hashProcessor

def getDHGeneratorAndPrime(kexAlgorithm):
    if False:
        i = 10
        return i + 15
    '\n    Get the generator and the prime to use in key exchange.\n\n    @param kexAlgorithm: The key exchange algorithm name.\n    @type kexAlgorithm: L{bytes}\n\n    @return: A L{tuple} containing L{int} generator and L{int} prime.\n    @rtype: L{tuple}\n    '
    kex = getKex(kexAlgorithm)
    return (kex.generator, kex.prime)

def getSupportedKeyExchanges():
    if False:
        return 10
    '\n    Get a list of supported key exchange algorithm names in order of\n    preference.\n\n    @return: A C{list} of supported key exchange algorithm names.\n    @rtype: C{list} of L{bytes}\n    '
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import ec
    from twisted.conch.ssh.keys import _curveTable
    backend = default_backend()
    kexAlgorithms = _kexAlgorithms.copy()
    for keyAlgorithm in list(kexAlgorithms):
        if keyAlgorithm.startswith(b'ecdh'):
            keyAlgorithmDsa = keyAlgorithm.replace(b'ecdh', b'ecdsa')
            supported = backend.elliptic_curve_exchange_algorithm_supported(ec.ECDH(), _curveTable[keyAlgorithmDsa])
        elif keyAlgorithm.startswith(b'curve25519-sha256'):
            supported = backend.x25519_supported()
        else:
            supported = True
        if not supported:
            kexAlgorithms.pop(keyAlgorithm)
    return sorted(kexAlgorithms, key=lambda kexAlgorithm: kexAlgorithms[kexAlgorithm].preference)