import base64
import os
from tempfile import _TemporaryFileWrapper
from typing import Optional, Union
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import CTR

class SecureTemporaryFile(_TemporaryFileWrapper):
    """Temporary file that provides on-the-fly encryption.

    Buffering large submissions in memory as they come in requires too
    much memory for too long a period. By writing the file to disk as it
    comes in using a stream cipher, we are able to minimize memory usage
    as submissions come in, while minimizing the chances of plaintext
    recovery through forensic disk analysis. They key used to encrypt
    each secure temporary file is also ephemeral, and is only stored in
    memory only for as long as needed.

    Adapted from Globaleaks' GLSecureTemporaryFile:
    https://github.com/globaleaks/GlobaLeaks/blob/master/backend/globaleaks/security.py#L35

    WARNING: you can't use this like a normal file object. It supports
    being appended to however many times you wish (although content may not be
    overwritten), and then it's contents may be read only once (although it may
    be done in chunks) and only after it's been written to.
    """
    AES_key_size = 256
    AES_block_size = 128

    def __init__(self, store_dir: str) -> None:
        if False:
            while True:
                i = 10
        'Generates an AES key and an initialization vector, and opens\n        a file in the `store_dir` directory with a\n        pseudorandomly-generated filename.\n\n        Args:\n            store_dir (str): the directory to create the secure\n                temporary file under.\n\n        Returns: self\n        '
        self.last_action = 'init'
        self.create_key()
        data = base64.urlsafe_b64encode(os.urandom(32))
        self.tmp_file_id = data.decode('utf-8').strip('=')
        self.filepath = os.path.join(store_dir, f'{self.tmp_file_id}.aes')
        self.file = open(self.filepath, 'w+b')
        super().__init__(self.file, self.filepath)

    def create_key(self) -> None:
        if False:
            i = 10
            return i + 15
        'Generates a unique, pseudorandom AES key, stored ephemerally in\n        memory as an instance attribute. Its destruction is ensured by the\n        automatic nightly reboots of the SecureDrop application server combined\n        with the freed memory-overwriting PAX_MEMORY_SANITIZE feature of the\n        grsecurity-patched kernel it uses (for further details consult\n        https://github.com/freedomofpress/securedrop/pull/477#issuecomment-168445450).\n        '
        self.key = os.urandom(self.AES_key_size // 8)
        self.iv = os.urandom(self.AES_block_size // 8)
        self.initialize_cipher()

    def initialize_cipher(self) -> None:
        if False:
            i = 10
            return i + 15
        'Creates the cipher-related objects needed for AES-CTR\n        encryption and decryption.\n        '
        self.cipher = Cipher(AES(self.key), CTR(self.iv), default_backend())
        self.encryptor = self.cipher.encryptor()
        self.decryptor = self.cipher.decryptor()

    def write(self, data: Union[bytes, str]) -> int:
        if False:
            i = 10
            return i + 15
        'Write `data` to the secure temporary file. This method may be\n        called any number of times following instance initialization,\n        but after calling :meth:`read`, you cannot write to the file\n        again.\n        '
        if self.last_action == 'read':
            raise AssertionError('You cannot write after reading!')
        self.last_action = 'write'
        if isinstance(data, str):
            data_as_bytes = data.encode('utf-8')
        else:
            data_as_bytes = data
        return self.file.write(self.encryptor.update(data_as_bytes))

    def read(self, count: Optional[int]=None) -> bytes:
        if False:
            print('Hello World!')
        "Read `data` from the secure temporary file. This method may\n        be called any number of times following instance initialization\n        and once :meth:`write has been called at least once, but not\n        before.\n\n        Before the first read operation, `seek(0, 0)` is called. So\n        while you can call this method any number of times, the full\n        contents of the file can only be read once. Additional calls to\n        read will return an empty str, which is desired behavior in that\n        it matches :class:`file` and because other modules depend on\n        this behavior to let them know they've reached the end of the\n        file.\n\n        Args:\n            count (int): the number of bytes to try to read from the\n                file from the current position.\n        "
        if self.last_action == 'init':
            raise AssertionError('You must write before reading!')
        if self.last_action == 'write':
            self.seek(0, 0)
            self.last_action = 'read'
        if count:
            return self.decryptor.update(self.file.read(count))
        else:
            return self.decryptor.update(self.file.read())

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'The __del__ method in tempfile._TemporaryFileWrapper (which\n        SecureTemporaryFile class inherits from) calls close() when the\n        temporary file is deleted.\n        '
        try:
            self.decryptor.finalize()
        except AlreadyFinalized:
            pass
        super().close()