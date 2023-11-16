import hashlib
import io
import json
import os
from hmac import compare_digest
from typing import Callable
from ..helpers import IntegrityError
from ..logger import create_logger
from ..checksums import StreamingXXH64
logger = create_logger()

class FileLikeWrapper:

    def __init__(self, fd):
        if False:
            i = 10
            return i + 15
        self.fd = fd

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.fd.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.fd.__exit__(exc_type, exc_val, exc_tb)

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fd.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            print('Hello World!')
        return self.fd.seek(offset, whence)

    def write(self, data):
        if False:
            i = 10
            return i + 15
        return self.fd.write(data)

    def read(self, n=None):
        if False:
            while True:
                i = 10
        return self.fd.read(n)

    def flush(self):
        if False:
            i = 10
            return i + 15
        self.fd.flush()

    def fileno(self):
        if False:
            print('Hello World!')
        return self.fd.fileno()

class FileHashingWrapper(FileLikeWrapper):
    """
    Wrapper for file-like objects that computes a hash on-the-fly while reading/writing.

    WARNING: Seeks should only be used to query the size of the file, not
    to skip data, because skipped data isn't read and not hashed into the digest.

    Similarly skipping while writing to create sparse files is also not supported.

    Data has to be read/written in a symmetric fashion, otherwise different
    digests will be generated.

    Note: When used as a context manager read/write operations outside the enclosed scope
    are illegal.
    """
    ALGORITHM: str = None
    FACTORY: Callable = None

    def __init__(self, backing_fd, write):
        if False:
            return 10
        super().__init__(backing_fd)
        self.writing = write
        self.hash = self.FACTORY()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        if exc_type is None:
            self.hash_length()
        super().__exit__(exc_type, exc_val, exc_tb)

    def write(self, data):
        if False:
            while True:
                i = 10
        '\n        Write *data* to backing file and update internal state.\n        '
        n = super().write(data)
        self.hash.update(data)
        return n

    def read(self, n=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read *data* from backing file (*n* has the usual meaning) and update internal state.\n        '
        data = super().read(n)
        self.hash.update(data)
        return data

    def hexdigest(self):
        if False:
            while True:
                i = 10
        '\n        Return current digest bytes as hex-string.\n\n        Note: this can be called multiple times.\n        '
        return self.hash.hexdigest()

    def update(self, data: bytes):
        if False:
            for i in range(10):
                print('nop')
        self.hash.update(data)

    def hash_length(self, seek_to_end=False):
        if False:
            return 10
        if seek_to_end:
            self.seek(0, io.SEEK_END)
        self.hash.update(str(self.tell()).encode())

class SHA512FileHashingWrapper(FileHashingWrapper):
    ALGORITHM = 'SHA512'
    FACTORY = hashlib.sha512

class XXH64FileHashingWrapper(FileHashingWrapper):
    ALGORITHM = 'XXH64'
    FACTORY = StreamingXXH64
SUPPORTED_ALGORITHMS = {SHA512FileHashingWrapper.ALGORITHM: SHA512FileHashingWrapper, XXH64FileHashingWrapper.ALGORITHM: XXH64FileHashingWrapper}

class FileIntegrityError(IntegrityError):
    """File failed integrity check: {}"""

class IntegrityCheckedFile(FileLikeWrapper):

    def __init__(self, path, write, filename=None, override_fd=None, integrity_data=None):
        if False:
            while True:
                i = 10
        self.path = path
        self.writing = write
        mode = 'wb' if write else 'rb'
        self.file_fd = override_fd or open(path, mode)
        self.file_opened = override_fd is None
        self.digests = {}
        hash_cls = XXH64FileHashingWrapper
        if not write:
            algorithm_and_digests = self.load_integrity_data(path, integrity_data)
            if algorithm_and_digests:
                (algorithm, self.digests) = algorithm_and_digests
                hash_cls = SUPPORTED_ALGORITHMS[algorithm]
        self.hasher = hash_cls(backing_fd=self.file_fd, write=write)
        super().__init__(self.hasher)
        self.hash_filename(filename)

    def load_integrity_data(self, path, integrity_data):
        if False:
            while True:
                i = 10
        if integrity_data is not None:
            return self.parse_integrity_data(path, integrity_data)

    def hash_filename(self, filename=None):
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.basename(filename or self.path)
        self.hasher.update(('%10d' % len(filename)).encode())
        self.hasher.update(filename.encode())

    @classmethod
    def parse_integrity_data(cls, path: str, data: str):
        if False:
            return 10
        try:
            integrity_data = json.loads(data)
            algorithm = integrity_data['algorithm']
            if algorithm not in SUPPORTED_ALGORITHMS:
                logger.warning('Cannot verify integrity of %s: Unknown algorithm %r', path, algorithm)
                return
            digests = integrity_data['digests']
            digests['final']
            return (algorithm, digests)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning('Could not parse integrity data for %s: %s', path, e)
            raise FileIntegrityError(path)

    def hash_part(self, partname, is_final=False):
        if False:
            while True:
                i = 10
        if not self.writing and (not self.digests):
            return
        self.hasher.update(('%10d' % len(partname)).encode())
        self.hasher.update(partname.encode())
        self.hasher.hash_length(seek_to_end=is_final)
        digest = self.hasher.hexdigest()
        if self.writing:
            self.digests[partname] = digest
        elif self.digests and (not compare_digest(self.digests.get(partname, ''), digest)):
            raise FileIntegrityError(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        exception = exc_type is not None
        try:
            if not exception:
                self.hash_part('final', is_final=True)
            self.hasher.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self.file_opened:
                self.file_fd.close()
        if exception:
            return
        if self.writing:
            self.store_integrity_data(json.dumps({'algorithm': self.hasher.ALGORITHM, 'digests': self.digests}))
        elif self.digests:
            logger.debug('Verified integrity of %s', self.path)

    def store_integrity_data(self, data: str):
        if False:
            return 10
        self.integrity_data = data

class DetachedIntegrityCheckedFile(IntegrityCheckedFile):

    def __init__(self, path, write, filename=None, override_fd=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(path, write, filename, override_fd)
        filename = filename or os.path.basename(path)
        output_dir = os.path.dirname(path)
        self.output_integrity_file = self.integrity_file_path(os.path.join(output_dir, filename))

    def load_integrity_data(self, path, integrity_data):
        if False:
            while True:
                i = 10
        assert not integrity_data, 'Cannot pass explicit integrity_data to DetachedIntegrityCheckedFile'
        return self.read_integrity_file(self.path)

    @staticmethod
    def integrity_file_path(path):
        if False:
            while True:
                i = 10
        return path + '.integrity'

    @classmethod
    def read_integrity_file(cls, path):
        if False:
            while True:
                i = 10
        try:
            with open(cls.integrity_file_path(path)) as fd:
                return cls.parse_integrity_data(path, fd.read())
        except FileNotFoundError:
            logger.info('No integrity file found for %s', path)
        except OSError as e:
            logger.warning('Could not read integrity file for %s: %s', path, e)
            raise FileIntegrityError(path)

    def store_integrity_data(self, data: str):
        if False:
            return 10
        with open(self.output_integrity_file, 'w') as fd:
            fd.write(data)