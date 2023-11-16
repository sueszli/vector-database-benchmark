from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization

class X25519PublicKey(metaclass=abc.ABCMeta):

    @classmethod
    def from_public_bytes(cls, data: bytes) -> X25519PublicKey:
        if False:
            print('Hello World!')
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.x25519_supported():
            raise UnsupportedAlgorithm('X25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
        return rust_openssl.x25519.from_public_bytes(data)

    @abc.abstractmethod
    def public_bytes(self, encoding: _serialization.Encoding, format: _serialization.PublicFormat) -> bytes:
        if False:
            print('Hello World!')
        '\n        The serialized bytes of the public key.\n        '

    @abc.abstractmethod
    def public_bytes_raw(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        The raw bytes of the public key.\n        Equivalent to public_bytes(Raw, Raw).\n        '

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks equality.\n        '
X25519PublicKey.register(rust_openssl.x25519.X25519PublicKey)

class X25519PrivateKey(metaclass=abc.ABCMeta):

    @classmethod
    def generate(cls) -> X25519PrivateKey:
        if False:
            while True:
                i = 10
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.x25519_supported():
            raise UnsupportedAlgorithm('X25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
        return rust_openssl.x25519.generate_key()

    @classmethod
    def from_private_bytes(cls, data: bytes) -> X25519PrivateKey:
        if False:
            i = 10
            return i + 15
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.x25519_supported():
            raise UnsupportedAlgorithm('X25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
        return rust_openssl.x25519.from_private_bytes(data)

    @abc.abstractmethod
    def public_key(self) -> X25519PublicKey:
        if False:
            i = 10
            return i + 15
        '\n        Returns the public key assosciated with this private key\n        '

    @abc.abstractmethod
    def private_bytes(self, encoding: _serialization.Encoding, format: _serialization.PrivateFormat, encryption_algorithm: _serialization.KeySerializationEncryption) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        The serialized bytes of the private key.\n        '

    @abc.abstractmethod
    def private_bytes_raw(self) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        The raw bytes of the private key.\n        Equivalent to private_bytes(Raw, Raw, NoEncryption()).\n        '

    @abc.abstractmethod
    def exchange(self, peer_public_key: X25519PublicKey) -> bytes:
        if False:
            while True:
                i = 10
        "\n        Performs a key exchange operation using the provided peer's public key.\n        "
X25519PrivateKey.register(rust_openssl.x25519.X25519PrivateKey)