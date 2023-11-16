from __future__ import annotations
import abc
import datetime
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives.hashes import HashAlgorithm

class LogEntryType(utils.Enum):
    X509_CERTIFICATE = 0
    PRE_CERTIFICATE = 1

class Version(utils.Enum):
    v1 = 0

class SignatureAlgorithm(utils.Enum):
    """
    Signature algorithms that are valid for SCTs.

    These are exactly the same as SignatureAlgorithm in RFC 5246 (TLS 1.2).

    See: <https://datatracker.ietf.org/doc/html/rfc5246#section-7.4.1.4.1>
    """
    ANONYMOUS = 0
    RSA = 1
    DSA = 2
    ECDSA = 3

class SignedCertificateTimestamp(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def version(self) -> Version:
        if False:
            return 10
        '\n        Returns the SCT version.\n        '

    @property
    @abc.abstractmethod
    def log_id(self) -> bytes:
        if False:
            return 10
        '\n        Returns an identifier indicating which log this SCT is for.\n        '

    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime.datetime:
        if False:
            i = 10
            return i + 15
        '\n        Returns the timestamp for this SCT.\n        '

    @property
    @abc.abstractmethod
    def entry_type(self) -> LogEntryType:
        if False:
            while True:
                i = 10
        '\n        Returns whether this is an SCT for a certificate or pre-certificate.\n        '

    @property
    @abc.abstractmethod
    def signature_hash_algorithm(self) -> HashAlgorithm:
        if False:
            i = 10
            return i + 15
        "\n        Returns the hash algorithm used for the SCT's signature.\n        "

    @property
    @abc.abstractmethod
    def signature_algorithm(self) -> SignatureAlgorithm:
        if False:
            i = 10
            return i + 15
        "\n        Returns the signing algorithm used for the SCT's signature.\n        "

    @property
    @abc.abstractmethod
    def signature(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the signature for this SCT.\n        '

    @property
    @abc.abstractmethod
    def extension_bytes(self) -> bytes:
        if False:
            return 10
        '\n        Returns the raw bytes of any extensions for this SCT.\n        '
SignedCertificateTimestamp.register(rust_x509.Sct)