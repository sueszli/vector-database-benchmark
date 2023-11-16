"""MediaFileStorage implementation that stores files in memory."""
import contextlib
import hashlib
import mimetypes
import os.path
from typing import Dict, List, NamedTuple, Optional, Union
from typing_extensions import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage, MediaFileStorageError
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.util import HASHLIB_KWARGS
LOGGER = get_logger(__name__)
PREFERRED_MIMETYPE_EXTENSION_MAP: Final = {'audio/wav': '.wav'}

def _calculate_file_id(data: bytes, mimetype: str, filename: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Hash data, mimetype, and an optional filename to generate a stable file ID.\n\n    Parameters\n    ----------\n    data\n        Content of in-memory file in bytes. Other types will throw TypeError.\n    mimetype\n        Any string. Will be converted to bytes and used to compute a hash.\n    filename\n        Any string. Will be converted to bytes and used to compute a hash.\n    '
    filehash = hashlib.new('sha224', **HASHLIB_KWARGS)
    filehash.update(data)
    filehash.update(bytes(mimetype.encode()))
    if filename is not None:
        filehash.update(bytes(filename.encode()))
    return filehash.hexdigest()

def get_extension_for_mimetype(mimetype: str) -> str:
    if False:
        while True:
            i = 10
    if mimetype in PREFERRED_MIMETYPE_EXTENSION_MAP:
        return PREFERRED_MIMETYPE_EXTENSION_MAP[mimetype]
    extension = mimetypes.guess_extension(mimetype, strict=False)
    if extension is None:
        return ''
    return extension

class MemoryFile(NamedTuple):
    """A MediaFile stored in memory."""
    content: bytes
    mimetype: str
    kind: MediaFileKind
    filename: Optional[str]

    @property
    def content_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.content)

class MemoryMediaFileStorage(MediaFileStorage, CacheStatsProvider):

    def __init__(self, media_endpoint: str):
        if False:
            while True:
                i = 10
        'Create a new MemoryMediaFileStorage instance\n\n        Parameters\n        ----------\n        media_endpoint\n            The name of the local endpoint that media is served from.\n            This endpoint should start with a forward-slash (e.g. "/media").\n        '
        self._files_by_id: Dict[str, MemoryFile] = {}
        self._media_endpoint = media_endpoint

    def load_and_get_id(self, path_or_data: Union[str, bytes], mimetype: str, kind: MediaFileKind, filename: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Add a file to the manager and return its ID.'
        file_data: bytes
        if isinstance(path_or_data, str):
            file_data = self._read_file(path_or_data)
        else:
            file_data = path_or_data
        file_id = _calculate_file_id(file_data, mimetype, filename)
        if file_id not in self._files_by_id:
            LOGGER.debug('Adding media file %s', file_id)
            media_file = MemoryFile(content=file_data, mimetype=mimetype, kind=kind, filename=filename)
            self._files_by_id[file_id] = media_file
        return file_id

    def get_file(self, filename: str) -> MemoryFile:
        if False:
            print('Hello World!')
        'Return the MemoryFile with the given filename. Filenames are of the\n        form "file_id.extension". (Note that this is *not* the optional\n        user-specified filename for download files.)\n\n        Raises a MediaFileStorageError if no such file exists.\n        '
        file_id = os.path.splitext(filename)[0]
        try:
            return self._files_by_id[file_id]
        except KeyError as e:
            raise MediaFileStorageError(f"Bad filename '{filename}'. (No media file with id '{file_id}')") from e

    def get_url(self, file_id: str) -> str:
        if False:
            print('Hello World!')
        'Get a URL for a given media file. Raise a MediaFileStorageError if\n        no such file exists.\n        '
        media_file = self.get_file(file_id)
        extension = get_extension_for_mimetype(media_file.mimetype)
        return f'{self._media_endpoint}/{file_id}{extension}'

    def delete_file(self, file_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Delete the file with the given ID.'
        with contextlib.suppress(KeyError):
            del self._files_by_id[file_id]

    def _read_file(self, filename: str) -> bytes:
        if False:
            return 10
        "Read a file into memory. Raise MediaFileStorageError if we can't."
        try:
            with open(filename, 'rb') as f:
                return f.read()
        except Exception as ex:
            raise MediaFileStorageError(f"Error opening '{filename}'") from ex

    def get_stats(self) -> List[CacheStat]:
        if False:
            for i in range(10):
                print('nop')
        files_by_id = self._files_by_id.copy()
        stats: List[CacheStat] = []
        for (file_id, file) in files_by_id.items():
            stats.append(CacheStat(category_name='st_memory_media_file_storage', cache_name='', byte_length=len(file.content)))
        return stats