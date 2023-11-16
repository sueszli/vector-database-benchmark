import abc
import logging
import os
import shutil
from typing import TYPE_CHECKING, Callable, Optional
from synapse.config._base import Config
from synapse.logging.context import defer_to_thread, run_in_background
from synapse.logging.opentracing import start_active_span, trace_with_opname
from synapse.util.async_helpers import maybe_awaitable
from ._base import FileInfo, Responder
from .media_storage import FileResponder
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from synapse.server import HomeServer

class StorageProvider(metaclass=abc.ABCMeta):
    """A storage provider is a service that can store uploaded media and
    retrieve them.
    """

    @abc.abstractmethod
    async def store_file(self, path: str, file_info: FileInfo) -> None:
        """Store the file described by file_info. The actual contents can be
        retrieved by reading the file in file_info.upload_path.

        Args:
            path: Relative path of file in local cache
            file_info: The metadata of the file.
        """

    @abc.abstractmethod
    async def fetch(self, path: str, file_info: FileInfo) -> Optional[Responder]:
        """Attempt to fetch the file described by file_info and stream it
        into writer.

        Args:
            path: Relative path of file in local cache
            file_info: The metadata of the file.

        Returns:
            Returns a Responder if the provider has the file, otherwise returns None.
        """

class StorageProviderWrapper(StorageProvider):
    """Wraps a storage provider and provides various config options

    Args:
        backend: The storage provider to wrap.
        store_local: Whether to store new local files or not.
        store_synchronous: Whether to wait for file to be successfully
            uploaded, or todo the upload in the background.
        store_remote: Whether remote media should be uploaded
    """

    def __init__(self, backend: StorageProvider, store_local: bool, store_synchronous: bool, store_remote: bool):
        if False:
            return 10
        self.backend = backend
        self.store_local = store_local
        self.store_synchronous = store_synchronous
        self.store_remote = store_remote

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'StorageProviderWrapper[%s]' % (self.backend,)

    @trace_with_opname('StorageProviderWrapper.store_file')
    async def store_file(self, path: str, file_info: FileInfo) -> None:
        if not file_info.server_name and (not self.store_local):
            return None
        if file_info.server_name and (not self.store_remote):
            return None
        if file_info.url_cache:
            return None
        if self.store_synchronous:
            await maybe_awaitable(self.backend.store_file(path, file_info))
        else:

            async def store() -> None:
                try:
                    return await maybe_awaitable(self.backend.store_file(path, file_info))
                except Exception:
                    logger.exception('Error storing file')
            run_in_background(store)

    @trace_with_opname('StorageProviderWrapper.fetch')
    async def fetch(self, path: str, file_info: FileInfo) -> Optional[Responder]:
        if file_info.url_cache:
            return None
        return await maybe_awaitable(self.backend.fetch(path, file_info))

class FileStorageProviderBackend(StorageProvider):
    """A storage provider that stores files in a directory on a filesystem.

    Args:
        hs
        config: The config returned by `parse_config`.
    """

    def __init__(self, hs: 'HomeServer', config: str):
        if False:
            for i in range(10):
                print('nop')
        self.hs = hs
        self.cache_directory = hs.config.media.media_store_path
        self.base_directory = config

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'FileStorageProviderBackend[%s]' % (self.base_directory,)

    @trace_with_opname('FileStorageProviderBackend.store_file')
    async def store_file(self, path: str, file_info: FileInfo) -> None:
        """See StorageProvider.store_file"""
        primary_fname = os.path.join(self.cache_directory, path)
        backup_fname = os.path.join(self.base_directory, path)
        dirname = os.path.dirname(backup_fname)
        os.makedirs(dirname, exist_ok=True)
        shutil_copyfile: Callable[[str, str], str] = shutil.copyfile
        with start_active_span('shutil_copyfile'):
            await defer_to_thread(self.hs.get_reactor(), shutil_copyfile, primary_fname, backup_fname)

    @trace_with_opname('FileStorageProviderBackend.fetch')
    async def fetch(self, path: str, file_info: FileInfo) -> Optional[Responder]:
        """See StorageProvider.fetch"""
        backup_fname = os.path.join(self.base_directory, path)
        if os.path.isfile(backup_fname):
            return FileResponder(open(backup_fname, 'rb'))
        return None

    @staticmethod
    def parse_config(config: dict) -> str:
        if False:
            return 10
        "Called on startup to parse config supplied. This should parse\n        the config and raise if there is a problem.\n\n        The returned value is passed into the constructor.\n\n        In this case we only care about a single param, the directory, so let's\n        just pull that out.\n        "
        return Config.ensure_directory(config['directory'])