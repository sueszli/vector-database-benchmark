"""
PDF’s standard security handler shall allow access permissions and up to two passwords to be specified for a
document: an owner password and a user password. An application’s decision to encrypt a document shall be
based on whether the user creating the document specifies any passwords or access restrictions.
"""
import hashlib
import typing
import zlib
from borb.io.read.encryption.rc4 import RC4
from borb.io.read.pdf_object import PDFObject
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Boolean
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import HexadecimalString
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.io.read.types import Stream
from borb.io.read.types import String

class StandardSecurityHandler:
    """
    PDF’s standard security handler shall allow access permissions and up to two passwords to be specified for a
    document: an owner password and a user password. An application’s decision to encrypt a document shall be
    based on whether the user creating the document specifies any passwords or access restrictions.
    """

    def __init__(self, encryption_dictionary: Dictionary, owner_password: typing.Optional[str]=None, user_password: typing.Optional[str]=None):
        if False:
            while True:
                i = 10
        self._v = int(encryption_dictionary.get('V', bDecimal(0)))
        self._u: bytes = StandardSecurityHandler._str_to_bytes(StandardSecurityHandler._unescape_pdf_syntax(encryption_dictionary.get('U'))) or b''
        assert len(self._u) == 32
        self._o: bytes = StandardSecurityHandler._str_to_bytes(StandardSecurityHandler._unescape_pdf_syntax(encryption_dictionary.get('O'))) or b''
        assert self._o is not None
        assert len(self._o) == 32
        trailer: typing.Optional[PDFObject] = encryption_dictionary.get_parent()
        assert trailer is not None
        assert isinstance(trailer, Dictionary)
        if 'ID' in trailer:
            self._document_id: bytes = trailer['ID'][0].get_content_bytes()
        assert 'P' in encryption_dictionary
        self._permissions: int = int(encryption_dictionary.get('P'))
        self._key_length: int = int(encryption_dictionary.get('Length', bDecimal(40)))
        assert self._key_length % 8 == 0, 'The length of the encryption key, in bits must be a multiple of 8.'
        self._revision: int = int(encryption_dictionary.get('R', bDecimal(0)))
        self._encrypt_metadata: bool = encryption_dictionary.get('EncryptMetadata', Boolean(True))
        password: typing.Optional[bytes] = None
        if user_password is not None:
            self._authenticate_user_password(bytes(user_password, encoding='charmap'))
            password = bytes(user_password, encoding='charmap')
        if owner_password is not None:
            self._authenticate_owner_password(bytes(owner_password, encoding='charmap'))
            password = bytes(owner_password, encoding='charmap')
        self._encryption_key: bytes = self._compute_encryption_key(password)

    def _authenticate_owner_password(self, owner_password: bytes) -> bool:
        if False:
            print('Hello World!')
        '\n        Algorithm 7: Authenticating the owner password\n        '
        return False

    def _authenticate_user_password(self, user_password: bytes) -> bool:
        if False:
            while True:
                i = 10
        '\n        Algorithm 6: Authenticating the user password\n        '
        previous_u_value: bytes = self._u
        self._compute_encryption_dictionary_u_value(user_password)
        u_value: bytes = self._u
        self._u = previous_u_value
        return self._u == u_value

    def _compute_encryption_dictionary_o_value(self, owner_password: typing.Optional[bytes], user_password: typing.Optional[bytes]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Algorithm 3: Computing the encryption dictionary’s O (owner password) value\n        '
        padded_owner_password: bytes = StandardSecurityHandler._pad_or_truncate(owner_password)
        h = hashlib.md5()
        h.update(padded_owner_password)
        if self._revision >= 3:
            prev_digest: bytes = h.digest()[0:int(self._key_length / 8)]
            for _ in range(0, 50):
                h = hashlib.md5()
                h.update(prev_digest)
                prev_digest = h.digest()[0:int(self._key_length / 8)]
        key: bytes = h.digest()[0:5]
        if self._revision >= 3:
            key = h.digest()[0:int(self._key_length / 8)]
        padded_user_password: bytes = StandardSecurityHandler._pad_or_truncate(user_password)
        rc4: RC4 = RC4()
        owner_key: bytes = rc4.encrypt(key, padded_user_password)
        if self._revision >= 3:
            for i in range(1, 20):
                key2: bytes = bytes([b ^ i for b in key])
                owner_key = RC4().encrypt(key2, owner_key)
        self._o = owner_key
        return owner_key

    def _compute_encryption_dictionary_u_value(self, user_password_string: bytes):
        if False:
            print('Hello World!')
        '\n        Algorithm 4: Computing the encryption dictionary’s U (user password) value (Security handlers of revision 2)\n        Algorithm 5: Computing the encryption dictionary’s U (user password) value (Security handlers of revision 3 or greater)\n        '
        if self._revision == 2:
            key_rev_2: bytes = self._compute_encryption_key(user_password_string)
            self._u = RC4().encrypt(key_rev_2, StandardSecurityHandler._pad_or_truncate(None))
            return self._u
        if self._revision >= 3:
            key_rev_3: bytes = self._compute_encryption_key(user_password_string)
            h = hashlib.md5()
            h.update(StandardSecurityHandler._pad_or_truncate(None))
            h.update(self._document_id)
            digest: bytes = h.digest()
            digest = RC4().encrypt(key_rev_3, digest)
            if self._revision >= 3:
                for i in range(1, 20):
                    key2: bytes = bytes([b ^ i for b in key_rev_3])
                    digest = RC4().encrypt(key2, digest)
            digest += bytes([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
            self._u = digest

    def _compute_encryption_key(self, password: typing.Optional[bytes]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        h = hashlib.md5()
        h.update(StandardSecurityHandler._pad_or_truncate(password))
        h.update(self._o)
        h.update(self._permissions.to_bytes(length=4, byteorder='little', signed=True))
        h.update(self._document_id)
        if self._revision >= 4 and (not self._encrypt_metadata):
            h.update(bytes([255, 255, 255, 255]))
        digest: bytes = h.digest()
        n: int = 0
        if self._revision >= 3:
            n = int(self._key_length / 8)
            for _ in range(0, 50):
                h2 = hashlib.md5()
                h2.update(digest[0:n])
                digest = h2.digest()
        n = 5
        if self._revision >= 3:
            n = int(self._key_length / 8)
        encryption_key: bytes = digest[0:n]
        return encryption_key

    def _decrypt_data(self, object: AnyPDFType) -> AnyPDFType:
        if False:
            print('Hello World!')
        return self._encrypt_data(object)

    def _encrypt_data(self, object: AnyPDFType) -> AnyPDFType:
        if False:
            while True:
                i = 10
        reference: typing.Optional[Reference] = object.get_reference()
        if reference is None:
            parent: typing.Optional[PDFObject] = object.get_parent()
            assert parent is not None
            reference = parent.get_reference()
        assert reference is not None
        assert reference.object_number is not None
        assert reference.generation_number is not None
        object_number: int = reference.object_number
        generation_number: int = reference.generation_number
        encryption_key = self._encryption_key + object_number.to_bytes(3, byteorder='little', signed=False) + generation_number.to_bytes(2, byteorder='little', signed=False)
        n: int = 5
        if self._v > 1:
            n = int(self._key_length / 8)
        h = hashlib.md5()
        h.update(encryption_key)
        n_plus_5: int = min(16, n + 5)
        if isinstance(object, String):
            str_new_content_bytes: bytes = RC4().encrypt(h.digest()[0:n_plus_5], object.get_content_bytes())
        if isinstance(object, HexadecimalString):
            hex_str_new_content_bytes: bytes = RC4().encrypt(h.digest()[0:n_plus_5], object.get_content_bytes())
        if isinstance(object, Stream):
            stream_new_content_bytes: bytes = RC4().encrypt(h.digest()[0:n_plus_5], object['DecodedBytes'])
            object[Name('DecodedBytes')] = stream_new_content_bytes
            object[Name('Bytes')] = zlib.compress(object['DecodedBytes'], 9)
            return object
        return object

    @staticmethod
    def _pad_or_truncate(b: typing.Optional[bytes]) -> bytes:
        if False:
            while True:
                i = 10
        padding: bytes = bytes([40, 191, 78, 94, 78, 117, 138, 65, 100, 0, 78, 86, 255, 250, 1, 8, 46, 46, 0, 182, 208, 104, 62, 128, 47, 12, 169, 254, 100, 83, 105, 122])
        if b is None:
            return padding
        if len(b) > 32:
            return b[0:32]
        if len(b) < 32:
            b2: bytes = b + padding
            return b2[0:32]
        return b

    @staticmethod
    def _str_to_bytes(s: typing.Optional[str]) -> typing.Optional[bytes]:
        if False:
            print('Hello World!')
        if s is None:
            return None
        return bytes(s, encoding='charmap')

    @staticmethod
    def _unescape_pdf_syntax(s: typing.Union[str, String, None]) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if s is None:
            return None
        if isinstance(s, String):
            return str(s.get_content_bytes(), encoding='latin1')
        return str(String(s).get_content_bytes(), encoding='latin1')