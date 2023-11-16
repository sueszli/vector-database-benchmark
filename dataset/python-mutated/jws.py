"""ACME-specific JWS.

The JWS implementation in josepy only implements the base JOSE standard. In
order to support the new header fields defined in ACME, this module defines some
ACME-specific classes that layer on top of josepy.
"""
from typing import Optional
import josepy as jose

class Header(jose.Header):
    """ACME-specific JOSE Header. Implements nonce, kid, and url.
    """
    nonce: Optional[bytes] = jose.field('nonce', omitempty=True, encoder=jose.encode_b64jose)
    kid: Optional[str] = jose.field('kid', omitempty=True)
    url: Optional[str] = jose.field('url', omitempty=True)

    @nonce.decoder
    def nonce(value: str) -> bytes:
        if False:
            while True:
                i = 10
        try:
            return jose.decode_b64jose(value)
        except jose.DeserializationError as error:
            raise jose.DeserializationError('Invalid nonce: {0}'.format(error))

class Signature(jose.Signature):
    """ACME-specific Signature. Uses ACME-specific Header for customer fields."""
    __slots__ = jose.Signature._orig_slots
    header_cls = Header
    header: Header = jose.field('header', omitempty=True, default=header_cls(), decoder=header_cls.from_json)

class JWS(jose.JWS):
    """ACME-specific JWS. Includes none, url, and kid in protected header."""
    signature_cls = Signature
    __slots__ = jose.JWS._orig_slots

    @classmethod
    def sign(cls, payload: bytes, key: jose.JWK, alg: jose.JWASignature, nonce: Optional[bytes], url: Optional[str]=None, kid: Optional[str]=None) -> jose.JWS:
        if False:
            return 10
        include_jwk = kid is None
        return super().sign(payload, key=key, alg=alg, protect=frozenset(['nonce', 'url', 'kid', 'jwk', 'alg']), nonce=nonce, url=url, kid=kid, include_jwk=include_jwk)