import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
ParsedURL = namedtuple('ParsedURL', 'base_url, offset, size')
logger = logging.getLogger(__name__)

def create_url_with_offset(*, url: str, offset: int, size: int) -> str:
    if False:
        print('Hello World!')
    'Methods to create a URL with offset.\n\n    When ray spills objects, it fuses multiple objects\n    into one file to optimize the performance. That says, each object\n    needs to keep tracking of its own special url to store metadata.\n\n    This method creates an url_with_offset, which is used internally\n    by Ray.\n\n    Created url_with_offset can be passed to the self._get_base_url method\n    to parse the filename used to store files.\n\n    Example) file://path/to/file?offset=""&size=""\n\n    Args:\n        url: url to the object stored in the external storage.\n        offset: Offset from the beginning of the file to\n            the first bytes of this object.\n        size: Size of the object that is stored in the url.\n            It is used to calculate the last offset.\n\n    Returns:\n        url_with_offset stored internally to find\n        objects from external storage.\n    '
    return f'{url}?offset={offset}&size={size}'

def parse_url_with_offset(url_with_offset: str) -> Tuple[str, int, int]:
    if False:
        while True:
            i = 10
    'Parse url_with_offset to retrieve information.\n\n    base_url is the url where the object ref\n    is stored in the external storage.\n\n    Args:\n        url_with_offset: url created by create_url_with_offset.\n\n    Returns:\n        named tuple of base_url, offset, and size.\n    '
    parsed_result = urllib.parse.urlparse(url_with_offset)
    query_dict = urllib.parse.parse_qs(parsed_result.query)
    base_url = parsed_result.geturl().split('?')[0]
    if 'offset' not in query_dict or 'size' not in query_dict:
        raise ValueError(f'Failed to parse URL: {url_with_offset}')
    offset = int(query_dict['offset'][0])
    size = int(query_dict['size'][0])
    return ParsedURL(base_url=base_url, offset=offset, size=size)

class ExternalStorage(metaclass=abc.ABCMeta):
    """The base class for external storage.

    This class provides some useful functions for zero-copy object
    put/get from plasma store. Also it specifies the interface for
    object spilling.

    When inheriting this class, please make sure to implement validation
    logic inside __init__ method. When ray instance starts, it will
    instantiating external storage to validate the config.

    Raises:
        ValueError: when given configuration for
            the external storage is invalid.
    """
    HEADER_LENGTH = 24

    def _get_objects_from_store(self, object_refs):
        if False:
            return 10
        worker = ray._private.worker.global_worker
        ray_object_pairs = worker.core_worker.get_if_local(object_refs)
        return ray_object_pairs

    def _put_object_to_store(self, metadata, data_size, file_like, object_ref, owner_address):
        if False:
            for i in range(10):
                print('nop')
        worker = ray._private.worker.global_worker
        worker.core_worker.put_file_like_object(metadata, data_size, file_like, object_ref, owner_address)

    def _write_multiple_objects(self, f: IO, object_refs: List[ObjectRef], owner_addresses: List[str], url: str) -> List[str]:
        if False:
            return 10
        'Fuse all given objects into a given file handle.\n\n        Args:\n            f: File handle to fusion all given object refs.\n            object_refs: Object references to fusion to a single file.\n            owner_addresses: Owner addresses for the provided objects.\n            url: url where the object ref is stored\n                in the external storage.\n\n        Return:\n            List of urls_with_offset of fused objects.\n            The order of returned keys are equivalent to the one\n            with given object_refs.\n        '
        keys = []
        offset = 0
        ray_object_pairs = self._get_objects_from_store(object_refs)
        for (ref, (buf, metadata), owner_address) in zip(object_refs, ray_object_pairs, owner_addresses):
            address_len = len(owner_address)
            metadata_len = len(metadata)
            if buf is None and len(metadata) == 0:
                error = f'Object {ref.hex()} does not exist.'
                raise ValueError(error)
            buf_len = 0 if buf is None else len(buf)
            payload = address_len.to_bytes(8, byteorder='little') + metadata_len.to_bytes(8, byteorder='little') + buf_len.to_bytes(8, byteorder='little') + owner_address + metadata + (memoryview(buf) if buf_len else b'')
            payload_len = len(payload)
            assert self.HEADER_LENGTH + address_len + metadata_len + buf_len == payload_len
            written_bytes = f.write(payload)
            assert written_bytes == payload_len
            url_with_offset = create_url_with_offset(url=url, offset=offset, size=written_bytes)
            keys.append(url_with_offset.encode())
            offset += written_bytes
        f.flush()
        return keys

    def _size_check(self, address_len, metadata_len, buffer_len, obtained_data_size):
        if False:
            for i in range(10):
                print('nop')
        'Check whether or not the obtained_data_size is as expected.\n\n        Args:\n             metadata_len: Actual metadata length of the object.\n             buffer_len: Actual buffer length of the object.\n             obtained_data_size: Data size specified in the\n                url_with_offset.\n\n        Raises:\n            ValueError if obtained_data_size is different from\n            address_len + metadata_len + buffer_len +\n            24 (first 8 bytes to store length).\n        '
        data_size_in_bytes = address_len + metadata_len + buffer_len + self.HEADER_LENGTH
        if data_size_in_bytes != obtained_data_size:
            raise ValueError(f'Obtained data has a size of {data_size_in_bytes}, although it is supposed to have the size of {obtained_data_size}.')

    @abc.abstractmethod
    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            return 10
        'Spill objects to the external storage. Objects are specified\n        by their object refs.\n\n        Args:\n            object_refs: The list of the refs of the objects to be spilled.\n            owner_addresses: Owner addresses for the provided objects.\n        Returns:\n            A list of internal URLs with object offset.\n        '

    @abc.abstractmethod
    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]) -> int:
        if False:
            i = 10
            return i + 15
        'Restore objects from the external storage.\n\n        Args:\n            object_refs: List of object IDs (note that it is not ref).\n            url_with_offset_list: List of url_with_offset.\n\n        Returns:\n            The total number of bytes restored.\n        '

    @abc.abstractmethod
    def delete_spilled_objects(self, urls: List[str]):
        if False:
            return 10
        'Delete objects that are spilled to the external storage.\n\n        Args:\n            urls: URLs that store spilled object files.\n\n        NOTE: This function should not fail if some of the urls\n        do not exist.\n        '

    @abc.abstractmethod
    def destroy_external_storage(self):
        if False:
            i = 10
            return i + 15
        'Destroy external storage when a head node is down.\n\n        NOTE: This is currently working when the cluster is\n        started by ray.init\n        '

class NullStorage(ExternalStorage):
    """The class that represents an uninitialized external storage."""

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('External storage is not initialized')

    def restore_spilled_objects(self, object_refs, url_with_offset_list):
        if False:
            return 10
        raise NotImplementedError('External storage is not initialized')

    def delete_spilled_objects(self, urls: List[str]):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('External storage is not initialized')

    def destroy_external_storage(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('External storage is not initialized')

class FileSystemStorage(ExternalStorage):
    """The class for filesystem-like external storage.

    Raises:
        ValueError: Raises directory path to
            spill objects doesn't exist.
    """

    def __init__(self, directory_path, buffer_size=None):
        if False:
            while True:
                i = 10
        self._spill_dir_name = DEFAULT_OBJECT_PREFIX
        self._directory_paths = []
        self._current_directory_index = 0
        self._buffer_size = -1
        assert directory_path is not None, 'directory_path should be provided to use object spilling.'
        if isinstance(directory_path, str):
            directory_path = [directory_path]
        assert isinstance(directory_path, list), 'Directory_path must be either a single string or a list of strings'
        if buffer_size is not None:
            assert isinstance(buffer_size, int), 'buffer_size must be an integer.'
            self._buffer_size = buffer_size
        for path in directory_path:
            full_dir_path = os.path.join(path, self._spill_dir_name)
            os.makedirs(full_dir_path, exist_ok=True)
            if not os.path.exists(full_dir_path):
                raise ValueError(f'The given directory path to store objects, {full_dir_path}, could not be created.')
            self._directory_paths.append(full_dir_path)
        assert len(self._directory_paths) == len(directory_path)
        self._current_directory_index = random.randrange(0, len(self._directory_paths))

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        if len(object_refs) == 0:
            return []
        self._current_directory_index = (self._current_directory_index + 1) % len(self._directory_paths)
        directory_path = self._directory_paths[self._current_directory_index]
        filename = _get_unique_spill_filename(object_refs)
        url = f'{os.path.join(directory_path, filename)}'
        with open(url, 'wb', buffering=self._buffer_size) as f:
            return self._write_multiple_objects(f, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        if False:
            return 10
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with open(base_url, 'rb') as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                total += buf_len
                owner_address = f.read(address_len)
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        if False:
            while True:
                i = 10
        for url in urls:
            path = parse_url_with_offset(url.decode()).base_url
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def destroy_external_storage(self):
        if False:
            for i in range(10):
                print('nop')
        for directory_path in self._directory_paths:
            self._destroy_external_storage(directory_path)

    def _destroy_external_storage(self, directory_path):
        if False:
            print('Hello World!')
        while os.path.isdir(directory_path):
            try:
                shutil.rmtree(directory_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception('Error cleaning up spill files. You might still have remaining spilled objects inside `ray_spilled_objects` directory.')
                break

class ExternalStorageRayStorageImpl(ExternalStorage):
    """Implements the external storage interface using the ray storage API."""

    def __init__(self, session_name: str, buffer_size=1024 * 1024, _force_storage_for_testing: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        from ray._private import storage
        if _force_storage_for_testing:
            storage._reset()
            storage._init_storage(_force_storage_for_testing, True)
        (self._fs, storage_prefix) = storage._get_filesystem_internal()
        self._buffer_size = buffer_size
        self._prefix = os.path.join(storage_prefix, 'spilled_objects', session_name)
        self._fs.create_dir(self._prefix)

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            while True:
                i = 10
        if len(object_refs) == 0:
            return []
        filename = _get_unique_spill_filename(object_refs)
        url = f'{os.path.join(self._prefix, filename)}'
        with self._fs.open_output_stream(url, buffer_size=self._buffer_size) as f:
            return self._write_multiple_objects(f, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        if False:
            for i in range(10):
                print('nop')
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with self._fs.open_input_file(base_url) as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                total += buf_len
                owner_address = f.read(address_len)
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        if False:
            while True:
                i = 10
        for url in urls:
            path = parse_url_with_offset(url.decode()).base_url
            try:
                self._fs.delete_file(path)
            except FileNotFoundError:
                pass

    def destroy_external_storage(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._fs.delete_dir(self._prefix)
        except Exception:
            logger.exception('Error cleaning up spill files. You might still have remaining spilled objects inside `{}`.'.format(self._prefix))

class ExternalStorageSmartOpenImpl(ExternalStorage):
    """The external storage class implemented by smart_open.
    (https://github.com/RaRe-Technologies/smart_open)

    Smart open supports multiple backend with the same APIs.

    To use this implementation, you should pre-create the given uri.
    For example, if your uri is a local file path, you should pre-create
    the directory.

    Args:
        uri: Storage URI used for smart open.
        prefix: Prefix of objects that are stored.
        override_transport_params: Overriding the default value of
            transport_params for smart-open library.

    Raises:
        ModuleNotFoundError: If it fails to setup.
            For example, if smart open library
            is not downloaded, this will fail.
    """

    def __init__(self, uri: str or list, prefix: str=DEFAULT_OBJECT_PREFIX, override_transport_params: dict=None, buffer_size=1024 * 1024):
        if False:
            i = 10
            return i + 15
        try:
            from smart_open import open
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f'Smart open is chosen to be a object spilling external storage, but smart_open and boto3 is not downloaded. Original error: {e}')
        assert uri is not None, 'uri should be provided to use object spilling.'
        if isinstance(uri, str):
            uri = [uri]
        assert isinstance(uri, list), 'uri must be a single string or list of strings.'
        assert isinstance(buffer_size, int), 'buffer_size must be an integer.'
        uri_is_s3 = [u.startswith('s3://') for u in uri]
        self.is_for_s3 = all(uri_is_s3)
        if not self.is_for_s3:
            assert not any(uri_is_s3), "all uri's must be s3 or none can be s3."
            self._uris = uri
        else:
            self._uris = [u.strip('/') for u in uri]
        assert len(self._uris) == len(uri)
        self._current_uri_index = random.randrange(0, len(self._uris))
        self.prefix = prefix
        self.override_transport_params = override_transport_params or {}
        if self.is_for_s3:
            import boto3
            self.s3 = boto3.resource(service_name='s3')
            self.transport_params = {'defer_seek': True, 'resource': self.s3, 'buffer_size': buffer_size}
        else:
            self.transport_params = {}
        self.transport_params.update(self.override_transport_params)

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            i = 10
            return i + 15
        if len(object_refs) == 0:
            return []
        from smart_open import open
        self._current_uri_index = (self._current_uri_index + 1) % len(self._uris)
        uri = self._uris[self._current_uri_index]
        key = f'{self.prefix}-{_get_unique_spill_filename(object_refs)}'
        url = f'{uri}/{key}'
        with open(url, mode='wb', transport_params=self.transport_params) as file_like:
            return self._write_multiple_objects(file_like, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        if False:
            for i in range(10):
                print('nop')
        from smart_open import open
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with open(base_url, 'rb', transport_params=self.transport_params) as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                owner_address = f.read(address_len)
                total += buf_len
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        if False:
            i = 10
            return i + 15
        pass

    def destroy_external_storage(self):
        if False:
            print('Hello World!')
        pass
_external_storage = NullStorage()

class UnstableFileStorage(FileSystemStorage):
    """This class is for testing with writing failure."""

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self._failure_rate = 0.1
        self._partial_failure_ratio = 0.2

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            while True:
                i = 10
        r = random.random() < self._failure_rate
        failed = r < self._failure_rate
        partial_failed = r < self._partial_failure_ratio
        if failed:
            raise IOError('Spilling object failed')
        elif partial_failed:
            i = random.choice(range(len(object_refs)))
            return super().spill_objects(object_refs[:i], owner_addresses)
        else:
            return super().spill_objects(object_refs, owner_addresses)

class SlowFileStorage(FileSystemStorage):
    """This class is for testing slow object spilling."""

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self._min_delay = 1
        self._max_delay = 2

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if False:
            while True:
                i = 10
        delay = random.random() * (self._max_delay - self._min_delay) + self._min_delay
        time.sleep(delay)
        return super().spill_objects(object_refs, owner_addresses)

def setup_external_storage(config, session_name):
    if False:
        for i in range(10):
            print('nop')
    'Setup the external storage according to the config.'
    global _external_storage
    if config:
        storage_type = config['type']
        if storage_type == 'filesystem':
            _external_storage = FileSystemStorage(**config['params'])
        elif storage_type == 'ray_storage':
            _external_storage = ExternalStorageRayStorageImpl(session_name, **config['params'])
        elif storage_type == 'smart_open':
            _external_storage = ExternalStorageSmartOpenImpl(**config['params'])
        elif storage_type == 'mock_distributed_fs':
            _external_storage = FileSystemStorage(**config['params'])
        elif storage_type == 'unstable_fs':
            _external_storage = UnstableFileStorage(**config['params'])
        elif storage_type == 'slow_fs':
            _external_storage = SlowFileStorage(**config['params'])
        else:
            raise ValueError(f'Unknown external storage type: {storage_type}')
    else:
        _external_storage = NullStorage()
    return _external_storage

def reset_external_storage():
    if False:
        i = 10
        return i + 15
    global _external_storage
    _external_storage = NullStorage()

def spill_objects(object_refs, owner_addresses):
    if False:
        for i in range(10):
            print('nop')
    'Spill objects to the external storage. Objects are specified\n    by their object refs.\n\n    Args:\n        object_refs: The list of the refs of the objects to be spilled.\n        owner_addresses: The owner addresses of the provided object refs.\n    Returns:\n        A list of keys corresponding to the input object refs.\n    '
    return _external_storage.spill_objects(object_refs, owner_addresses)

def restore_spilled_objects(object_refs: List[ObjectRef], url_with_offset_list: List[str]):
    if False:
        i = 10
        return i + 15
    'Restore objects from the external storage.\n\n    Args:\n        object_refs: List of object IDs (note that it is not ref).\n        url_with_offset_list: List of url_with_offset.\n    '
    return _external_storage.restore_spilled_objects(object_refs, url_with_offset_list)

def delete_spilled_objects(urls: List[str]):
    if False:
        for i in range(10):
            print('nop')
    'Delete objects that are spilled to the external storage.\n\n    Args:\n        urls: URLs that store spilled object files.\n    '
    _external_storage.delete_spilled_objects(urls)

def _get_unique_spill_filename(object_refs: List[ObjectRef]):
    if False:
        i = 10
        return i + 15
    'Generate a unqiue spill file name.\n\n    Args:\n        object_refs: objects to be spilled in this file.\n    '
    return f'{uuid.uuid4().hex}-multi-{len(object_refs)}'