import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import CryptAES, CryptBase, CryptIdentity, CryptRC4, aes_cbc_decrypt, aes_cbc_encrypt, aes_ecb_decrypt, aes_ecb_encrypt, rc4_decrypt, rc4_encrypt
from ._utils import b_, logger_warning
from .generic import ArrayObject, ByteStringObject, DictionaryObject, NameObject, NumberObject, PdfObject, StreamObject, TextStringObject, create_string_object

class CryptFilter:

    def __init__(self, stm_crypt: CryptBase, str_crypt: CryptBase, ef_crypt: CryptBase) -> None:
        if False:
            return 10
        self.stm_crypt = stm_crypt
        self.str_crypt = str_crypt
        self.ef_crypt = ef_crypt

    def encrypt_object(self, obj: PdfObject) -> PdfObject:
        if False:
            return 10
        if isinstance(obj, ByteStringObject):
            data = self.str_crypt.encrypt(obj.original_bytes)
            obj = ByteStringObject(data)
        if isinstance(obj, TextStringObject):
            data = self.str_crypt.encrypt(obj.get_encoded_bytes())
            obj = ByteStringObject(data)
        elif isinstance(obj, StreamObject):
            obj2 = StreamObject()
            obj2.update(obj)
            obj2.set_data(self.stm_crypt.encrypt(b_(obj._data)))
            for (key, value) in obj.items():
                obj2[key] = self.encrypt_object(value)
            obj = obj2
        elif isinstance(obj, DictionaryObject):
            obj2 = DictionaryObject()
            for (key, value) in obj.items():
                obj2[key] = self.encrypt_object(value)
            obj = obj2
        elif isinstance(obj, ArrayObject):
            obj = ArrayObject((self.encrypt_object(x) for x in obj))
        return obj

    def decrypt_object(self, obj: PdfObject) -> PdfObject:
        if False:
            print('Hello World!')
        if isinstance(obj, (ByteStringObject, TextStringObject)):
            data = self.str_crypt.decrypt(obj.original_bytes)
            obj = create_string_object(data)
        elif isinstance(obj, StreamObject):
            obj._data = self.stm_crypt.decrypt(b_(obj._data))
            for (key, value) in obj.items():
                obj[key] = self.decrypt_object(value)
        elif isinstance(obj, DictionaryObject):
            for (key, value) in obj.items():
                obj[key] = self.decrypt_object(value)
        elif isinstance(obj, ArrayObject):
            for i in range(len(obj)):
                obj[i] = self.decrypt_object(obj[i])
        return obj
_PADDING = b'(\xbfN^Nu\x8aAd\x00NV\xff\xfa\x01\x08..\x00\xb6\xd0h>\x80/\x0c\xa9\xfedSiz'

def _padding(data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    return (data + _PADDING)[:32]

class AlgV4:

    @staticmethod
    def compute_key(password: bytes, rev: int, key_size: int, o_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        if False:
            return 10
        '\n        Algorithm 2: Computing an encryption key.\n\n        a) Pad or truncate the password string to exactly 32 bytes. If the\n           password string is more than 32 bytes long,\n           use only its first 32 bytes; if it is less than 32 bytes long, pad it\n           by appending the required number of\n           additional bytes from the beginning of the following padding string:\n                < 28 BF 4E 5E 4E 75 8A 41 64 00 4E 56 FF FA 01 08\n                2E 2E 00 B6 D0 68 3E 80 2F 0C A9 FE 64 53 69 7A >\n           That is, if the password string is n bytes long, append\n           the first 32 - n bytes of the padding string to the end\n           of the password string. If the password string is empty\n           (zero-length), meaning there is no user password,\n           substitute the entire padding string in its place.\n\n        b) Initialize the MD5 hash function and pass the result of step (a)\n           as input to this function.\n        c) Pass the value of the encryption dictionary’s O entry to the\n           MD5 hash function. ("Algorithm 3: Computing\n           the encryption dictionary’s O (owner password) value" shows how the\n           O value is computed.)\n        d) Convert the integer value of the P entry to a 32-bit unsigned binary\n           number and pass these bytes to the\n           MD5 hash function, low-order byte first.\n        e) Pass the first element of the file’s file identifier array (the value\n           of the ID entry in the document’s trailer\n           dictionary; see Table 15) to the MD5 hash function.\n        f) (Security handlers of revision 4 or greater) If document metadata is\n           not being encrypted, pass 4 bytes with\n           the value 0xFFFFFFFF to the MD5 hash function.\n        g) Finish the hash.\n        h) (Security handlers of revision 3 or greater) Do the following\n           50 times: Take the output from the previous\n           MD5 hash and pass the first n bytes of the output as input into a new\n           MD5 hash, where n is the number of\n           bytes of the encryption key as defined by the value of the encryption\n           dictionary’s Length entry.\n        i) Set the encryption key to the first n bytes of the output from the\n           final MD5 hash, where n shall always be 5\n           for security handlers of revision 2 but, for security handlers of\n           revision 3 or greater, shall depend on the\n           value of the encryption dictionary’s Length entry.\n\n        Args:\n            password: The encryption secret as a bytes-string\n            rev: The encryption revision (see PDF standard)\n            key_size: The size of the key in bytes\n            o_entry: The owner entry\n            P: A set of flags specifying which operations shall be permitted\n                when the document is opened with user access. If bit 2 is set to 1,\n                all other bits are ignored and all operations are permitted.\n                If bit 2 is set to 0, permission for operations are based on the\n                values of the remaining flags defined in Table 24.\n            id1_entry:\n            metadata_encrypted: A boolean indicating if the metadata is encrypted.\n\n        Returns:\n            The u_hash digest of length key_size\n        '
        a = _padding(password)
        u_hash = hashlib.md5(a)
        u_hash.update(o_entry)
        u_hash.update(struct.pack('<I', P))
        u_hash.update(id1_entry)
        if rev >= 4 and (not metadata_encrypted):
            u_hash.update(b'\xff\xff\xff\xff')
        u_hash_digest = u_hash.digest()
        length = key_size // 8
        if rev >= 3:
            for _ in range(50):
                u_hash_digest = hashlib.md5(u_hash_digest[:length]).digest()
        return u_hash_digest[:length]

    @staticmethod
    def compute_O_value_key(owner_password: bytes, rev: int, key_size: int) -> bytes:
        if False:
            print('Hello World!')
        '\n        Algorithm 3: Computing the encryption dictionary’s O (owner password) value.\n\n        a) Pad or truncate the owner password string as described in step (a)\n           of "Algorithm 2: Computing an encryption key".\n           If there is no owner password, use the user password instead.\n        b) Initialize the MD5 hash function and pass the result of step (a) as\n           input to this function.\n        c) (Security handlers of revision 3 or greater) Do the following 50 times:\n           Take the output from the previous\n           MD5 hash and pass it as input into a new MD5 hash.\n        d) Create an RC4 encryption key using the first n bytes of the output\n           from the final MD5 hash, where n shall\n           always be 5 for security handlers of revision 2 but, for security\n           handlers of revision 3 or greater, shall\n           depend on the value of the encryption dictionary’s Length entry.\n        e) Pad or truncate the user password string as described in step (a) of\n           "Algorithm 2: Computing an encryption key".\n        f) Encrypt the result of step (e), using an RC4 encryption function with\n           the encryption key obtained in step (d).\n        g) (Security handlers of revision 3 or greater) Do the following 19 times:\n           Take the output from the previous\n           invocation of the RC4 function and pass it as input to a new\n           invocation of the function; use an encryption\n           key generated by taking each byte of the encryption key obtained in\n           step (d) and performing an XOR\n           (exclusive or) operation between that byte and the single-byte value\n           of the iteration counter (from 1 to 19).\n        h) Store the output from the final invocation of the RC4 function as\n           the value of the O entry in the encryption dictionary.\n\n        Args:\n            owner_password:\n            rev: The encryption revision (see PDF standard)\n            key_size: The size of the key in bytes\n\n        Returns:\n            The RC4 key\n        '
        a = _padding(owner_password)
        o_hash_digest = hashlib.md5(a).digest()
        if rev >= 3:
            for _ in range(50):
                o_hash_digest = hashlib.md5(o_hash_digest).digest()
        rc4_key = o_hash_digest[:key_size // 8]
        return rc4_key

    @staticmethod
    def compute_O_value(rc4_key: bytes, user_password: bytes, rev: int) -> bytes:
        if False:
            print('Hello World!')
        '\n        See :func:`compute_O_value_key`.\n\n        Args:\n            rc4_key:\n            user_password:\n            rev: The encryption revision (see PDF standard)\n\n        Returns:\n            The RC4 encrypted\n        '
        a = _padding(user_password)
        rc4_enc = rc4_encrypt(rc4_key, a)
        if rev >= 3:
            for i in range(1, 20):
                key = bytes(bytearray((x ^ i for x in rc4_key)))
                rc4_enc = rc4_encrypt(key, rc4_enc)
        return rc4_enc

    @staticmethod
    def compute_U_value(key: bytes, rev: int, id1_entry: bytes) -> bytes:
        if False:
            return 10
        '\n        Algorithm 4: Computing the encryption dictionary’s U (user password) value.\n\n        (Security handlers of revision 2)\n\n        a) Create an encryption key based on the user password string, as\n           described in "Algorithm 2: Computing an encryption key".\n        b) Encrypt the 32-byte padding string shown in step (a) of\n           "Algorithm 2: Computing an encryption key", using an RC4 encryption\n           function with the encryption key from the preceding step.\n        c) Store the result of step (b) as the value of the U entry in the\n           encryption dictionary.\n\n        Args:\n            key:\n            rev: The encryption revision (see PDF standard)\n            id1_entry:\n\n        Returns:\n            The value\n        '
        if rev <= 2:
            value = rc4_encrypt(key, _PADDING)
            return value
        '\n        Algorithm 5: Computing the encryption dictionary’s U (user password) value.\n\n        (Security handlers of revision 3 or greater)\n\n        a) Create an encryption key based on the user password string, as\n           described in "Algorithm 2: Computing an encryption key".\n        b) Initialize the MD5 hash function and pass the 32-byte padding string\n           shown in step (a) of "Algorithm 2:\n           Computing an encryption key" as input to this function.\n        c) Pass the first element of the file’s file identifier array (the value\n           of the ID entry in the document’s trailer\n           dictionary; see Table 15) to the hash function and finish the hash.\n        d) Encrypt the 16-byte result of the hash, using an RC4 encryption\n           function with the encryption key from step (a).\n        e) Do the following 19 times: Take the output from the previous\n           invocation of the RC4 function and pass it as input to a new\n           invocation of the function; use an encryption key generated by\n           taking each byte of the original encryption key obtained in\n           step (a) and performing an XOR (exclusive or) operation between that\n           byte and the single-byte value of the iteration counter (from 1 to 19).\n        f) Append 16 bytes of arbitrary padding to the output from the final\n           invocation of the RC4 function and store the 32-byte result as the\n           value of the U entry in the encryption dictionary.\n        '
        u_hash = hashlib.md5(_PADDING)
        u_hash.update(id1_entry)
        rc4_enc = rc4_encrypt(key, u_hash.digest())
        for i in range(1, 20):
            rc4_key = bytes(bytearray((x ^ i for x in key)))
            rc4_enc = rc4_encrypt(rc4_key, rc4_enc)
        return _padding(rc4_enc)

    @staticmethod
    def verify_user_password(user_password: bytes, rev: int, key_size: int, o_entry: bytes, u_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Algorithm 6: Authenticating the user password.\n\n        a) Perform all but the last step of "Algorithm 4: Computing the\n           encryption dictionary’s U (user password) value (Security handlers of\n           revision 2)" or "Algorithm 5: Computing the encryption dictionary’s U\n           (user password) value (Security handlers of revision 3 or greater)"\n           using the supplied password string.\n        b) If the result of step (a) is equal to the value of the encryption\n           dictionary’s U entry (comparing on the first 16 bytes in the case of\n           security handlers of revision 3 or greater), the password supplied is\n           the correct user password. The key obtained in step (a) (that is, in\n           the first step of "Algorithm 4: Computing the encryption\n           dictionary’s U (user password) value\n           (Security handlers of revision 2)" or\n           "Algorithm 5: Computing the encryption dictionary’s U (user password)\n           value (Security handlers of revision 3 or greater)") shall be used\n           to decrypt the document.\n\n        Args:\n            user_password: The user password as a bytes stream\n            rev: The encryption revision (see PDF standard)\n            key_size: The size of the key in bytes\n            o_entry: The owner entry\n            u_entry: The user entry\n            P: A set of flags specifying which operations shall be permitted\n                when the document is opened with user access. If bit 2 is set to 1,\n                all other bits are ignored and all operations are permitted.\n                If bit 2 is set to 0, permission for operations are based on the\n                values of the remaining flags defined in Table 24.\n            id1_entry:\n            metadata_encrypted: A boolean indicating if the metadata is encrypted.\n\n        Returns:\n            The key\n        '
        key = AlgV4.compute_key(user_password, rev, key_size, o_entry, P, id1_entry, metadata_encrypted)
        u_value = AlgV4.compute_U_value(key, rev, id1_entry)
        if rev >= 3:
            u_value = u_value[:16]
            u_entry = u_entry[:16]
        if u_value != u_entry:
            key = b''
        return key

    @staticmethod
    def verify_owner_password(owner_password: bytes, rev: int, key_size: int, o_entry: bytes, u_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        if False:
            print('Hello World!')
        '\n        Algorithm 7: Authenticating the owner password.\n\n        a) Compute an encryption key from the supplied password string, as\n           described in steps (a) to (d) of\n           "Algorithm 3: Computing the encryption dictionary’s O (owner password)\n           value".\n        b) (Security handlers of revision 2 only) Decrypt the value of the\n           encryption dictionary’s O entry, using an RC4\n           encryption function with the encryption key computed in step (a).\n           (Security handlers of revision 3 or greater) Do the following 20 times:\n           Decrypt the value of the encryption dictionary’s O entry (first iteration)\n           or the output from the previous iteration (all subsequent iterations),\n           using an RC4 encryption function with a different encryption key at\n           each iteration. The key shall be generated by taking the original key\n           (obtained in step (a)) and performing an XOR (exclusive or) operation\n           between each byte of the key and the single-byte value of the\n           iteration counter (from 19 to 0).\n        c) The result of step (b) purports to be the user password.\n           Authenticate this user password using\n           "Algorithm 6: Authenticating the user password".\n           If it is correct, the password supplied is the correct owner password.\n\n        Args:\n            owner_password:\n            rev: The encryption revision (see PDF standard)\n            key_size: The size of the key in bytes\n            o_entry: The owner entry\n            u_entry: The user entry\n            P: A set of flags specifying which operations shall be permitted\n                when the document is opened with user access. If bit 2 is set to 1,\n                all other bits are ignored and all operations are permitted.\n                If bit 2 is set to 0, permission for operations are based on the\n                values of the remaining flags defined in Table 24.\n            id1_entry:\n            metadata_encrypted: A boolean indicating if the metadata is encrypted.\n\n        Returns:\n            bytes\n        '
        rc4_key = AlgV4.compute_O_value_key(owner_password, rev, key_size)
        if rev <= 2:
            user_password = rc4_decrypt(rc4_key, o_entry)
        else:
            user_password = o_entry
            for i in range(19, -1, -1):
                key = bytes(bytearray((x ^ i for x in rc4_key)))
                user_password = rc4_decrypt(key, user_password)
        return AlgV4.verify_user_password(user_password, rev, key_size, o_entry, u_entry, P, id1_entry, metadata_encrypted)

class AlgV5:

    @staticmethod
    def verify_owner_password(R: int, password: bytes, o_value: bytes, oe_value: bytes, u_value: bytes) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Algorithm 3.2a Computing an encryption key.\n\n        To understand the algorithm below, it is necessary to treat the O and U\n        strings in the Encrypt dictionary as made up of three sections.\n        The first 32 bytes are a hash value (explained below). The next 8 bytes\n        are called the Validation Salt. The final 8 bytes are called the Key Salt.\n\n        1. The password string is generated from Unicode input by processing the\n           input string with the SASLprep (IETF RFC 4013) profile of\n           stringprep (IETF RFC 3454), and then converting to a UTF-8\n           representation.\n        2. Truncate the UTF-8 representation to 127 bytes if it is longer than\n           127 bytes.\n        3. Test the password against the owner key by computing the SHA-256 hash\n           of the UTF-8 password concatenated with the 8 bytes of owner\n           Validation Salt, concatenated with the 48-byte U string. If the\n           32-byte result matches the first 32 bytes of the O string, this is\n           the owner password.\n           Compute an intermediate owner key by computing the SHA-256 hash of\n           the UTF-8 password concatenated with the 8 bytes of owner Key Salt,\n           concatenated with the 48-byte U string. The 32-byte result is the\n           key used to decrypt the 32-byte OE string using AES-256 in CBC mode\n           with no padding and an initialization vector of zero.\n           The 32-byte result is the file encryption key.\n        4. Test the password against the user key by computing the SHA-256 hash\n           of the UTF-8 password concatenated with the 8 bytes of user\n           Validation Salt. If the 32 byte result matches the first 32 bytes of\n           the U string, this is the user password.\n           Compute an intermediate user key by computing the SHA-256 hash of the\n           UTF-8 password concatenated with the 8 bytes of user Key Salt.\n           The 32-byte result is the key used to decrypt the 32-byte\n           UE string using AES-256 in CBC mode with no padding and an\n           initialization vector of zero. The 32-byte result is the file\n           encryption key.\n        5. Decrypt the 16-byte Perms string using AES-256 in ECB mode with an\n           initialization vector of zero and the file encryption key as the key.\n           Verify that bytes 9-11 of the result are the characters ‘a’, ‘d’, ‘b’.\n           Bytes 0-3 of the decrypted Perms entry, treated as a little-endian\n           integer, are the user permissions.\n           They should match the value in the P key.\n\n        Args:\n            R: A number specifying which revision of the standard security\n                handler shall be used to interpret this dictionary\n            password: The owner password\n            o_value: A 32-byte string, based on both the owner and user passwords,\n                that shall be used in computing the encryption key and in\n                determining whether a valid owner password was entered\n            oe_value:\n            u_value: A 32-byte string, based on the user password, that shall be\n                used in determining whether to prompt the user for a password and,\n                if so, whether a valid user or owner password was entered.\n\n        Returns:\n            The key\n        '
        password = password[:127]
        if AlgV5.calculate_hash(R, password, o_value[32:40], u_value[:48]) != o_value[:32]:
            return b''
        iv = bytes((0 for _ in range(16)))
        tmp_key = AlgV5.calculate_hash(R, password, o_value[40:48], u_value[:48])
        key = aes_cbc_decrypt(tmp_key, iv, oe_value)
        return key

    @staticmethod
    def verify_user_password(R: int, password: bytes, u_value: bytes, ue_value: bytes) -> bytes:
        if False:
            print('Hello World!')
        '\n        See :func:`verify_owner_password`.\n\n        Args:\n            R: A number specifying which revision of the standard security\n                handler shall be used to interpret this dictionary\n            password: The user password\n            u_value: A 32-byte string, based on the user password, that shall be\n                used in determining whether to prompt the user for a password\n                and, if so, whether a valid user or owner password was entered.\n            ue_value:\n\n        Returns:\n            bytes\n        '
        password = password[:127]
        if AlgV5.calculate_hash(R, password, u_value[32:40], b'') != u_value[:32]:
            return b''
        iv = bytes((0 for _ in range(16)))
        tmp_key = AlgV5.calculate_hash(R, password, u_value[40:48], b'')
        return aes_cbc_decrypt(tmp_key, iv, ue_value)

    @staticmethod
    def calculate_hash(R: int, password: bytes, salt: bytes, udata: bytes) -> bytes:
        if False:
            return 10
        k = hashlib.sha256(password + salt + udata).digest()
        if R < 6:
            return k
        count = 0
        while True:
            count += 1
            k1 = password + k + udata
            e = aes_cbc_encrypt(k[:16], k[16:32], k1 * 64)
            hash_fn = (hashlib.sha256, hashlib.sha384, hashlib.sha512)[sum(e[:16]) % 3]
            k = hash_fn(e).digest()
            if count >= 64 and e[-1] <= count - 32:
                break
        return k[:32]

    @staticmethod
    def verify_perms(key: bytes, perms: bytes, p: int, metadata_encrypted: bool) -> bool:
        if False:
            while True:
                i = 10
        '\n        See :func:`verify_owner_password` and :func:`compute_perms_value`.\n\n        Args:\n            key: The owner password\n            perms:\n            p: A set of flags specifying which operations shall be permitted\n                when the document is opened with user access.\n                If bit 2 is set to 1, all other bits are ignored and all\n                operations are permitted.\n                If bit 2 is set to 0, permission for operations are based on\n                the values of the remaining flags defined in Table 24.\n            metadata_encrypted:\n\n        Returns:\n            A boolean\n        '
        b8 = b'T' if metadata_encrypted else b'F'
        p1 = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb'
        p2 = aes_ecb_decrypt(key, perms)
        return p1 == p2[:12]

    @staticmethod
    def generate_values(R: int, user_password: bytes, owner_password: bytes, key: bytes, p: int, metadata_encrypted: bool) -> Dict[Any, Any]:
        if False:
            return 10
        user_password = user_password[:127]
        owner_password = owner_password[:127]
        (u_value, ue_value) = AlgV5.compute_U_value(R, user_password, key)
        (o_value, oe_value) = AlgV5.compute_O_value(R, owner_password, key, u_value)
        perms = AlgV5.compute_Perms_value(key, p, metadata_encrypted)
        return {'/U': u_value, '/UE': ue_value, '/O': o_value, '/OE': oe_value, '/Perms': perms}

    @staticmethod
    def compute_U_value(R: int, password: bytes, key: bytes) -> Tuple[bytes, bytes]:
        if False:
            return 10
        '\n        Algorithm 3.8 Computing the encryption dictionary’s U (user password)\n        and UE (user encryption key) values.\n\n        1. Generate 16 random bytes of data using a strong random number generator.\n           The first 8 bytes are the User Validation Salt. The second 8 bytes\n           are the User Key Salt. Compute the 32-byte SHA-256 hash of the\n           password concatenated with the User Validation Salt. The 48-byte\n           string consisting of the 32-byte hash followed by the User\n           Validation Salt followed by the User Key Salt is stored as the U key.\n        2. Compute the 32-byte SHA-256 hash of the password concatenated with\n           the User Key Salt. Using this hash as the key, encrypt the file\n           encryption key using AES-256 in CBC mode with no padding and an\n           initialization vector of zero. The resulting 32-byte string is stored\n           as the UE key.\n\n        Args:\n            R:\n            password:\n            key:\n\n        Returns:\n            A tuple (u-value, ue value)\n        '
        random_bytes = secrets.token_bytes(16)
        val_salt = random_bytes[:8]
        key_salt = random_bytes[8:]
        u_value = AlgV5.calculate_hash(R, password, val_salt, b'') + val_salt + key_salt
        tmp_key = AlgV5.calculate_hash(R, password, key_salt, b'')
        iv = bytes((0 for _ in range(16)))
        ue_value = aes_cbc_encrypt(tmp_key, iv, key)
        return (u_value, ue_value)

    @staticmethod
    def compute_O_value(R: int, password: bytes, key: bytes, u_value: bytes) -> Tuple[bytes, bytes]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Algorithm 3.9 Computing the encryption dictionary’s O (owner password)\n        and OE (owner encryption key) values.\n\n        1. Generate 16 random bytes of data using a strong random number\n           generator. The first 8 bytes are the Owner Validation Salt. The\n           second 8 bytes are the Owner Key Salt. Compute the 32-byte SHA-256\n           hash of the password concatenated with the Owner Validation Salt and\n           then concatenated with the 48-byte U string as generated in\n           Algorithm 3.8. The 48-byte string consisting of the 32-byte hash\n           followed by the Owner Validation Salt followed by the Owner Key Salt\n           is stored as the O key.\n        2. Compute the 32-byte SHA-256 hash of the password concatenated with\n           the Owner Key Salt and then concatenated with the 48-byte U string as\n           generated in Algorithm 3.8. Using this hash as the key,\n           encrypt the file encryption key using AES-256 in CBC mode with\n           no padding and an initialization vector of zero.\n           The resulting 32-byte string is stored as the OE key.\n\n        Args:\n            R:\n            password:\n            key:\n            u_value: A 32-byte string, based on the user password, that shall be\n                used in determining whether to prompt the user for a password\n                and, if so, whether a valid user or owner password was entered.\n\n        Returns:\n            A tuple (O value, OE value)\n        '
        random_bytes = secrets.token_bytes(16)
        val_salt = random_bytes[:8]
        key_salt = random_bytes[8:]
        o_value = AlgV5.calculate_hash(R, password, val_salt, u_value) + val_salt + key_salt
        tmp_key = AlgV5.calculate_hash(R, password, key_salt, u_value[:48])
        iv = bytes((0 for _ in range(16)))
        oe_value = aes_cbc_encrypt(tmp_key, iv, key)
        return (o_value, oe_value)

    @staticmethod
    def compute_Perms_value(key: bytes, p: int, metadata_encrypted: bool) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        "\n        Algorithm 3.10 Computing the encryption dictionary’s Perms\n        (permissions) value.\n\n        1. Extend the permissions (contents of the P integer) to 64 bits by\n           setting the upper 32 bits to all 1’s.\n           (This allows for future extension without changing the format.)\n        2. Record the 8 bytes of permission in the bytes 0-7 of the block,\n           low order byte first.\n        3. Set byte 8 to the ASCII value ' T ' or ' F ' according to the\n           EncryptMetadata Boolean.\n        4. Set bytes 9-11 to the ASCII characters ' a ', ' d ', ' b '.\n        5. Set bytes 12-15 to 4 bytes of random data, which will be ignored.\n        6. Encrypt the 16-byte block using AES-256 in ECB mode with an\n           initialization vector of zero, using the file encryption key as the\n           key. The result (16 bytes) is stored as the Perms string, and checked\n           for validity when the file is opened.\n\n        Args:\n            key:\n            p: A set of flags specifying which operations shall be permitted\n                when the document is opened with user access. If bit 2 is set to 1,\n                all other bits are ignored and all operations are permitted.\n                If bit 2 is set to 0, permission for operations are based on the\n                values of the remaining flags defined in Table 24.\n            metadata_encrypted: A boolean indicating if the metadata is encrypted.\n\n        Returns:\n            The perms value\n        "
        b8 = b'T' if metadata_encrypted else b'F'
        rr = secrets.token_bytes(4)
        data = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb' + rr
        perms = aes_ecb_encrypt(key, data)
        return perms

class PasswordType(IntEnum):
    NOT_DECRYPTED = 0
    USER_PASSWORD = 1
    OWNER_PASSWORD = 2

class EncryptAlgorithm(tuple, Enum):
    RC4_40 = (1, 2, 40)
    RC4_128 = (2, 3, 128)
    AES_128 = (4, 4, 128)
    AES_256_R5 = (5, 5, 256)
    AES_256 = (5, 6, 256)

class EncryptionValues:
    O: bytes
    U: bytes
    OE: bytes
    UE: bytes
    Perms: bytes

class Encryption:
    """
    Collects and manages parameters for PDF document encryption and decryption.

    Args:
        V: A code specifying the algorithm to be used in encrypting and
           decrypting the document.
        R: The revision of the standard security handler.
        Length: The length of the encryption key in bits.
        P: A set of flags specifying which operations shall be permitted
           when the document is opened with user access
        entry: The encryption dictionary object.
        EncryptMetadata: Whether to encrypt metadata in the document.
        first_id_entry: The first 16 bytes of the file's original ID.
        StmF: The name of the crypt filter that shall be used by default
              when decrypting streams.
        StrF: The name of the crypt filter that shall be used when decrypting
              all strings in the document.
        EFF: The name of the crypt filter that shall be used when
             encrypting embedded file streams that do not have their own
             crypt filter specifier.
        values: Additional encryption parameters.
    """

    def __init__(self, V: int, R: int, Length: int, P: int, entry: DictionaryObject, EncryptMetadata: bool, first_id_entry: bytes, StmF: str, StrF: str, EFF: str, values: Optional[EncryptionValues]) -> None:
        if False:
            return 10
        self.V = V
        self.R = R
        self.Length = Length
        self.P = (P + 4294967296) % 4294967296
        self.EncryptMetadata = EncryptMetadata
        self.id1_entry = first_id_entry
        self.StmF = StmF
        self.StrF = StrF
        self.EFF = EFF
        self.values: EncryptionValues = values if values else EncryptionValues()
        self._password_type = PasswordType.NOT_DECRYPTED
        self._key: Optional[bytes] = None

    def is_decrypted(self) -> bool:
        if False:
            print('Hello World!')
        return self._password_type != PasswordType.NOT_DECRYPTED

    def encrypt_object(self, obj: PdfObject, idnum: int, generation: int) -> PdfObject:
        if False:
            print('Hello World!')
        if not self._is_encryption_object(obj):
            return obj
        cf = self._make_crypt_filter(idnum, generation)
        return cf.encrypt_object(obj)

    def decrypt_object(self, obj: PdfObject, idnum: int, generation: int) -> PdfObject:
        if False:
            i = 10
            return i + 15
        if not self._is_encryption_object(obj):
            return obj
        cf = self._make_crypt_filter(idnum, generation)
        return cf.decrypt_object(obj)

    @staticmethod
    def _is_encryption_object(obj: PdfObject) -> bool:
        if False:
            print('Hello World!')
        return isinstance(obj, (ByteStringObject, TextStringObject, StreamObject, ArrayObject, DictionaryObject))

    def _make_crypt_filter(self, idnum: int, generation: int) -> CryptFilter:
        if False:
            while True:
                i = 10
        '\n        Algorithm 1: Encryption of data using the RC4 or AES algorithms.\n\n        a) Obtain the object number and generation number from the object\n           identifier of the string or stream to be encrypted\n           (see 7.3.10, "Indirect Objects"). If the string is a direct object,\n           use the identifier of the indirect object containing it.\n        b) For all strings and streams without crypt filter specifier; treating\n           the object number and generation number as binary integers, extend\n           the original n-byte encryption key to n + 5 bytes by appending the\n           low-order 3 bytes of the object number and the low-order 2 bytes of\n           the generation number in that order, low-order byte first.\n           (n is 5 unless the value of V in the encryption dictionary is greater\n           than 1, in which case n is the value of Length divided by 8.)\n           If using the AES algorithm, extend the encryption key an additional\n           4 bytes by adding the value “sAlT”, which corresponds to the\n           hexadecimal values 0x73, 0x41, 0x6C, 0x54. (This addition is done for\n           backward compatibility and is not intended to provide additional\n           security.)\n        c) Initialize the MD5 hash function and pass the result of step (b) as\n           input to this function.\n        d) Use the first (n + 5) bytes, up to a maximum of 16, of the output\n           from the MD5 hash as the key for the RC4 or AES symmetric key\n           algorithms, along with the string or stream data to be encrypted.\n           If using the AES algorithm, the Cipher Block Chaining (CBC) mode,\n           which requires an initialization vector, is used. The block size\n           parameter is set to 16 bytes, and the initialization vector is a\n           16-byte random number that is stored as the first 16 bytes of the\n           encrypted stream or string.\n\n        Algorithm 3.1a Encryption of data using the AES algorithm\n        1. Use the 32-byte file encryption key for the AES-256 symmetric key\n           algorithm, along with the string or stream data to be encrypted.\n           Use the AES algorithm in Cipher Block Chaining (CBC) mode, which\n           requires an initialization vector. The block size parameter is set to\n           16 bytes, and the initialization vector is a 16-byte random number\n           that is stored as the first 16 bytes of the encrypted stream or string.\n           The output is the encrypted data to be stored in the PDF file.\n        '
        pack1 = struct.pack('<i', idnum)[:3]
        pack2 = struct.pack('<i', generation)[:2]
        assert self._key
        key = self._key
        n = 5 if self.V == 1 else self.Length // 8
        key_data = key[:n] + pack1 + pack2
        key_hash = hashlib.md5(key_data)
        rc4_key = key_hash.digest()[:min(n + 5, 16)]
        key_hash.update(b'sAlT')
        aes128_key = key_hash.digest()[:min(n + 5, 16)]
        aes256_key = key
        stm_crypt = self._get_crypt(self.StmF, rc4_key, aes128_key, aes256_key)
        str_crypt = self._get_crypt(self.StrF, rc4_key, aes128_key, aes256_key)
        ef_crypt = self._get_crypt(self.EFF, rc4_key, aes128_key, aes256_key)
        return CryptFilter(stm_crypt, str_crypt, ef_crypt)

    @staticmethod
    def _get_crypt(method: str, rc4_key: bytes, aes128_key: bytes, aes256_key: bytes) -> CryptBase:
        if False:
            return 10
        if method == '/AESV3':
            return CryptAES(aes256_key)
        if method == '/AESV2':
            return CryptAES(aes128_key)
        elif method == '/Identity':
            return CryptIdentity()
        else:
            return CryptRC4(rc4_key)

    @staticmethod
    def _encode_password(password: Union[bytes, str]) -> bytes:
        if False:
            while True:
                i = 10
        if isinstance(password, str):
            try:
                pwd = password.encode('latin-1')
            except Exception:
                pwd = password.encode('utf-8')
        else:
            pwd = password
        return pwd

    def verify(self, password: Union[bytes, str]) -> PasswordType:
        if False:
            return 10
        pwd = self._encode_password(password)
        (key, rc) = self.verify_v4(pwd) if self.V <= 4 else self.verify_v5(pwd)
        if rc != PasswordType.NOT_DECRYPTED:
            self._password_type = rc
            self._key = key
        return rc

    def verify_v4(self, password: bytes) -> Tuple[bytes, PasswordType]:
        if False:
            i = 10
            return i + 15
        key = AlgV4.verify_owner_password(password, self.R, self.Length, self.values.O, self.values.U, self.P, self.id1_entry, self.EncryptMetadata)
        if key:
            return (key, PasswordType.OWNER_PASSWORD)
        key = AlgV4.verify_user_password(password, self.R, self.Length, self.values.O, self.values.U, self.P, self.id1_entry, self.EncryptMetadata)
        if key:
            return (key, PasswordType.USER_PASSWORD)
        return (b'', PasswordType.NOT_DECRYPTED)

    def verify_v5(self, password: bytes) -> Tuple[bytes, PasswordType]:
        if False:
            i = 10
            return i + 15
        key = AlgV5.verify_owner_password(self.R, password, self.values.O, self.values.OE, self.values.U)
        rc = PasswordType.OWNER_PASSWORD
        if not key:
            key = AlgV5.verify_user_password(self.R, password, self.values.U, self.values.UE)
            rc = PasswordType.USER_PASSWORD
        if not key:
            return (b'', PasswordType.NOT_DECRYPTED)
        if not AlgV5.verify_perms(key, self.values.Perms, self.P, self.EncryptMetadata):
            logger_warning("ignore '/Perms' verify failed", __name__)
        return (key, rc)

    def write_entry(self, user_password: str, owner_password: Optional[str]) -> DictionaryObject:
        if False:
            return 10
        user_pwd = self._encode_password(user_password)
        owner_pwd = self._encode_password(owner_password) if owner_password else None
        if owner_pwd is None:
            owner_pwd = user_pwd
        if self.V <= 4:
            self.compute_values_v4(user_pwd, owner_pwd)
        else:
            self._key = secrets.token_bytes(self.Length // 8)
            values = AlgV5.generate_values(self.R, user_pwd, owner_pwd, self._key, self.P, self.EncryptMetadata)
            self.values.O = values['/O']
            self.values.U = values['/U']
            self.values.OE = values['/OE']
            self.values.UE = values['/UE']
            self.values.Perms = values['/Perms']
        dict_obj = DictionaryObject()
        dict_obj[NameObject('/V')] = NumberObject(self.V)
        dict_obj[NameObject('/R')] = NumberObject(self.R)
        dict_obj[NameObject('/Length')] = NumberObject(self.Length)
        dict_obj[NameObject('/P')] = NumberObject(self.P)
        dict_obj[NameObject('/Filter')] = NameObject('/Standard')
        dict_obj[NameObject('/O')] = ByteStringObject(self.values.O)
        dict_obj[NameObject('/U')] = ByteStringObject(self.values.U)
        if self.V >= 4:
            std_cf = DictionaryObject()
            std_cf[NameObject('/AuthEvent')] = NameObject('/DocOpen')
            std_cf[NameObject('/CFM')] = NameObject(self.StmF)
            std_cf[NameObject('/Length')] = NumberObject(self.Length // 8)
            cf = DictionaryObject()
            cf[NameObject('/StdCF')] = std_cf
            dict_obj[NameObject('/CF')] = cf
            dict_obj[NameObject('/StmF')] = NameObject('/StdCF')
            dict_obj[NameObject('/StrF')] = NameObject('/StdCF')
        if self.V >= 5:
            dict_obj[NameObject('/OE')] = ByteStringObject(self.values.OE)
            dict_obj[NameObject('/UE')] = ByteStringObject(self.values.UE)
            dict_obj[NameObject('/Perms')] = ByteStringObject(self.values.Perms)
        return dict_obj

    def compute_values_v4(self, user_password: bytes, owner_password: bytes) -> None:
        if False:
            i = 10
            return i + 15
        rc4_key = AlgV4.compute_O_value_key(owner_password, self.R, self.Length)
        o_value = AlgV4.compute_O_value(rc4_key, user_password, self.R)
        key = AlgV4.compute_key(user_password, self.R, self.Length, o_value, self.P, self.id1_entry, self.EncryptMetadata)
        u_value = AlgV4.compute_U_value(key, self.R, self.id1_entry)
        self._key = key
        self.values.O = o_value
        self.values.U = u_value

    @staticmethod
    def read(encryption_entry: DictionaryObject, first_id_entry: bytes) -> 'Encryption':
        if False:
            return 10
        if encryption_entry.get('/Filter') != '/Standard':
            raise NotImplementedError('only Standard PDF encryption handler is available')
        if '/SubFilter' in encryption_entry:
            raise NotImplementedError('/SubFilter NOT supported')
        stm_filter = '/V2'
        str_filter = '/V2'
        ef_filter = '/V2'
        alg_ver = encryption_entry.get('/V', 0)
        if alg_ver not in (1, 2, 3, 4, 5):
            raise NotImplementedError(f'Encryption V={alg_ver} NOT supported')
        if alg_ver >= 4:
            filters = encryption_entry['/CF']
            stm_filter = encryption_entry.get('/StmF', '/Identity')
            str_filter = encryption_entry.get('/StrF', '/Identity')
            ef_filter = encryption_entry.get('/EFF', stm_filter)
            if stm_filter != '/Identity':
                stm_filter = filters[stm_filter]['/CFM']
            if str_filter != '/Identity':
                str_filter = filters[str_filter]['/CFM']
            if ef_filter != '/Identity':
                ef_filter = filters[ef_filter]['/CFM']
            allowed_methods = ('/Identity', '/V2', '/AESV2', '/AESV3')
            if stm_filter not in allowed_methods:
                raise NotImplementedError(f'StmF Method {stm_filter} NOT supported!')
            if str_filter not in allowed_methods:
                raise NotImplementedError(f'StrF Method {str_filter} NOT supported!')
            if ef_filter not in allowed_methods:
                raise NotImplementedError(f'EFF Method {ef_filter} NOT supported!')
        alg_rev = cast(int, encryption_entry['/R'])
        perm_flags = cast(int, encryption_entry['/P'])
        key_bits = encryption_entry.get('/Length', 40)
        encrypt_metadata = encryption_entry.get('/EncryptMetadata')
        encrypt_metadata = encrypt_metadata.value if encrypt_metadata is not None else True
        values = EncryptionValues()
        values.O = cast(ByteStringObject, encryption_entry['/O']).original_bytes
        values.U = cast(ByteStringObject, encryption_entry['/U']).original_bytes
        values.OE = encryption_entry.get('/OE', ByteStringObject()).original_bytes
        values.UE = encryption_entry.get('/UE', ByteStringObject()).original_bytes
        values.Perms = encryption_entry.get('/Perms', ByteStringObject()).original_bytes
        return Encryption(V=alg_ver, R=alg_rev, Length=key_bits, P=perm_flags, EncryptMetadata=encrypt_metadata, first_id_entry=first_id_entry, values=values, StrF=str_filter, StmF=stm_filter, EFF=ef_filter, entry=encryption_entry)

    @staticmethod
    def make(alg: EncryptAlgorithm, permissions: int, first_id_entry: bytes) -> 'Encryption':
        if False:
            print('Hello World!')
        (alg_ver, alg_rev, key_bits) = alg
        (stm_filter, str_filter, ef_filter) = ('/V2', '/V2', '/V2')
        if alg == EncryptAlgorithm.AES_128:
            (stm_filter, str_filter, ef_filter) = ('/AESV2', '/AESV2', '/AESV2')
        elif alg in (EncryptAlgorithm.AES_256_R5, EncryptAlgorithm.AES_256):
            (stm_filter, str_filter, ef_filter) = ('/AESV3', '/AESV3', '/AESV3')
        return Encryption(V=alg_ver, R=alg_rev, Length=key_bits, P=permissions, EncryptMetadata=True, first_id_entry=first_id_entry, values=None, StrF=str_filter, StmF=stm_filter, EFF=ef_filter, entry=DictionaryObject())