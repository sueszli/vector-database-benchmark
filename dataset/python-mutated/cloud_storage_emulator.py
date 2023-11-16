"""An emulator that mocks the core.platform.storage API."""
from __future__ import annotations
import mimetypes
from core import feconf
import redis
from typing import Dict, List, Mapping, Optional, Union
REDIS_CLIENT = redis.StrictRedis(host=feconf.REDISHOST, port=feconf.REDISPORT, db=feconf.STORAGE_EMULATOR_REDIS_DB_INDEX, decode_responses=False)

class EmulatorBlob:
    """Object for storing the file data."""

    def __init__(self, name: str, data: Union[bytes, str], content_type: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize blob.\n\n        Args:\n            name: str. The name of the blob.\n            data: str|bytes. The data of the blob. If the data are string,\n                they are encoded to bytes. Note that data is always retrieved\n                from Cloud Storage as bytes.\n            content_type: str|None. The content type of the blob. It should\n                be in the MIME format.\n\n        Raises:\n            Exception. Content type contains unknown MIME type.\n        '
        self._name = name
        self._raw_bytes = data.encode('utf-8') if isinstance(data, str) else data
        if content_type is None:
            (guessed_content_type, _) = mimetypes.guess_type(name)
            self._content_type = guessed_content_type if guessed_content_type else 'application/octet-stream'
        elif content_type == 'audio/mp3':
            self._content_type = content_type
        elif content_type == 'image/webp':
            self._content_type = content_type
        else:
            if mimetypes.guess_extension(content_type) is None:
                raise Exception('Content type contains unknown MIME type.')
            self._content_type = content_type

    @classmethod
    def create_copy(cls, original_blob: EmulatorBlob, new_name: str) -> EmulatorBlob:
        if False:
            for i in range(10):
                print('nop')
        'Create new instance of EmulatorBlob with the same values.\n\n        Args:\n            original_blob: EmulatorBlob. Original blob to copy.\n            new_name: str. New name of the blob.\n\n        Returns:\n            EmulatorBlob. New instance with the same values as original_blob.\n        '
        return cls(new_name, original_blob.download_as_bytes(), original_blob.content_type)

    def to_dict(self) -> Mapping[bytes, bytes]:
        if False:
            i = 10
            return i + 15
        'Transform the EmulatorBlob into dictionary that can be saved\n        into Redis.\n\n        Returns:\n            dict(bytes, bytes). Dictionary containing all values of\n            EmulatorBlob.\n        '
        blob_dict = {b'name': self._name.encode('utf-8'), b'raw_bytes': self._raw_bytes, b'content_type': self._content_type.encode('utf-8')}
        return blob_dict

    @classmethod
    def from_dict(cls, blob_dict: Dict[bytes, bytes]) -> EmulatorBlob:
        if False:
            print('Hello World!')
        'Transform dictionary from Redis into EmulatorBlob.\n\n        Args:\n            blob_dict: dict(bytes, bytes). Dictionary containing all values\n                of EmulatorBlob.\n\n        Returns:\n            EmulatorBlob. EmulatorBlob created from the dictionary.\n        '
        return cls(blob_dict[b'name'].decode('utf-8'), blob_dict[b'raw_bytes'], blob_dict[b'content_type'].decode('utf-8'))

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        "Get the filepath of the blob. This is called 'name' since this mimics\n        the corresponding property in the Google Cloud Storage API.\n\n        Returns:\n            str. The filepath of the blob.\n        "
        return self._name

    @property
    def content_type(self) -> str:
        if False:
            return 10
        'Get the content type of the blob.\n\n        Returns:\n            str. The content type of the blob.\n        '
        return self._content_type

    def download_as_bytes(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Get the raw bytes of the blob.\n\n        Returns:\n            bytes. The raw bytes of the blob.\n        '
        return self._raw_bytes

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(self.name)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'EmulatorBlob(name=%s, content_type=%s)' % (self.name, self.content_type)

class CloudStorageEmulator:
    """Emulator for the storage client."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        'Initialize the CloudStorageEmulator class..'
        self.namespace = ''

    def _get_redis_key(self, filepath: str) -> str:
        if False:
            print('Hello World!')
        "Construct and return the Redis key for the given filepath. The key\n        is the filepath prepended with namespace and ':'.\n\n        Args:\n            filepath: str. Path to do the file we want to get key for.\n\n        Returns:\n            str. Filepath prepended by the current namespace.\n        "
        return '%s:%s' % (self.namespace, filepath)

    def get_blob(self, filepath: str) -> Optional[EmulatorBlob]:
        if False:
            for i in range(10):
                print('nop')
        'Get the blob located at the given filepath.\n\n        Args:\n            filepath: str. Filepath to the blob.\n\n        Returns:\n            EmulatorBlob. The blob.\n        '
        blob_dict = REDIS_CLIENT.hgetall(self._get_redis_key(filepath))
        return EmulatorBlob.from_dict(blob_dict) if blob_dict else None

    def upload_blob(self, filepath: str, blob: EmulatorBlob) -> None:
        if False:
            print('Hello World!')
        'Upload the given blob to the filepath.\n\n        Args:\n            filepath: str. Filepath to upload the blob to.\n            blob: EmulatorBlob. The blob to upload.\n        '
        REDIS_CLIENT.hset(self._get_redis_key(filepath), mapping=blob.to_dict())

    def delete_blob(self, filepath: str) -> None:
        if False:
            while True:
                i = 10
        'Delete the blob at the given filepath.\n\n        Args:\n            filepath: str. Filepath of the blob.\n        '
        REDIS_CLIENT.delete(self._get_redis_key(filepath))

    def copy_blob(self, blob: EmulatorBlob, filepath: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Copy existing blob to new filepath.\n\n        Args:\n            blob: EmulatorBlob. The blob to copy.\n            filepath: str. The filepath to copy the blob to.\n        '
        REDIS_CLIENT.hset(self._get_redis_key(filepath), mapping=EmulatorBlob.create_copy(blob, filepath).to_dict())

    def list_blobs(self, prefix: str) -> List[EmulatorBlob]:
        if False:
            i = 10
            return i + 15
        'Get blobs whose filepaths start with the given prefix.\n\n        Args:\n            prefix: str. The prefix to match.\n\n        Returns:\n            list(EmulatorBlob). The list of blobs whose filepaths start with\n            the given prefix.\n        '
        matching_filepaths = REDIS_CLIENT.scan_iter(match='%s*' % self._get_redis_key(prefix))
        pipeline = REDIS_CLIENT.pipeline()
        for filepath in matching_filepaths:
            pipeline.hgetall(filepath)
        blob_dicts = pipeline.execute()
        return [EmulatorBlob.from_dict(blob_dict) for blob_dict in blob_dicts]

    def reset(self) -> None:
        if False:
            return 10
        'Reset the emulator and remove all blobs.'
        for key in REDIS_CLIENT.scan_iter(match='%s*' % self._get_redis_key('')):
            REDIS_CLIENT.delete(key)