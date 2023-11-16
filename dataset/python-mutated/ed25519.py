from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization

class Ed25519PublicKey(metaclass=abc.ABCMeta):

    @classmethod
    def from_public_bytes(cls, data: bytes) -> Ed25519PublicKey:
        if False:
            for i in range(10):
                print('nop')
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.ed25519_supported():
            raise UnsupportedAlgorithm('ed25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)
        return rust_openssl.ed25519.from_public_bytes(data)

    @abc.abstractmethod
    def public_bytes(self, encoding: _serialization.Encoding, format: _serialization.PublicFormat) -> bytes:
        if False:
            while True:
                i = 10
        '\n        The serialized bytes of the public key.\n        '

    @abc.abstractmethod
    def public_bytes_raw(self) -> bytes:
        if False:
            print('Hello World!')
        '\n        The raw bytes of the public key.\n        Equivalent to public_bytes(Raw, Raw).\n        '

    @abc.abstractmethod
    def verify(self, signature: bytes, data: bytes) -> None:
        if False:
            print('Hello World!')
        '\n        Verify the signature.\n        '

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks equality.\n        '
Ed25519PublicKey.register(rust_openssl.ed25519.Ed25519PublicKey)

class Ed25519PrivateKey(metaclass=abc.ABCMeta):

    @classmethod
    def generate(cls) -> Ed25519PrivateKey:
        if False:
            return 10
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.ed25519_supported():
            raise UnsupportedAlgorithm('ed25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)
        return rust_openssl.ed25519.generate_key()

    @classmethod
    def from_private_bytes(cls, data: bytes) -> Ed25519PrivateKey:
        if False:
            return 10
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.ed25519_supported():
            raise UnsupportedAlgorithm('ed25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)
        return rust_openssl.ed25519.from_private_bytes(data)

    @abc.abstractmethod
    def public_key(self) -> Ed25519PublicKey:
        if False:
            while True:
                i = 10
        '\n        The Ed25519PublicKey derived from the private key.\n        '

    @abc.abstractmethod
    def private_bytes(self, encoding: _serialization.Encoding, format: _serialization.PrivateFormat, encryption_algorithm: _serialization.KeySerializationEncryption) -> bytes:
        if False:
            while True:
                i = 10
        '\n        The serialized bytes of the private key.\n        '

    @abc.abstractmethod
    def private_bytes_raw(self) -> bytes:
        if False:
            while True:
                i = 10
        '\n        The raw bytes of the private key.\n        Equivalent to private_bytes(Raw, Raw, NoEncryption()).\n        '

    @abc.abstractmethod
    def sign(self, data: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Signs the data.\n        '
Ed25519PrivateKey.register(rust_openssl.ed25519.Ed25519PrivateKey)