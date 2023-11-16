from typing import Optional
from urllib.parse import quote
import tornado.web
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage, get_extension_for_mimetype
from streamlit.web.server import allow_cross_origin_requests
_LOGGER = get_logger(__name__)

class MediaFileHandler(tornado.web.StaticFileHandler):
    _storage: MemoryMediaFileStorage

    @classmethod
    def initialize_storage(cls, storage: MemoryMediaFileStorage) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the MemoryMediaFileStorage object used by instances of this\n        handler. Must be called on server startup.\n        '
        cls._storage = storage

    def set_default_headers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if allow_cross_origin_requests():
            self.set_header('Access-Control-Allow-Origin', '*')

    def set_extra_headers(self, path: str) -> None:
        if False:
            print('Hello World!')
        'Add Content-Disposition header for downloadable files.\n\n        Set header value to "attachment" indicating that file should be saved\n        locally instead of displaying inline in browser.\n\n        We also set filename to specify the filename for downloaded files.\n        Used for serving downloadable files, like files stored via the\n        `st.download_button` widget.\n        '
        media_file = self._storage.get_file(path)
        if media_file and media_file.kind == MediaFileKind.DOWNLOADABLE:
            filename = media_file.filename
            if not filename:
                filename = f'streamlit_download{get_extension_for_mimetype(media_file.mimetype)}'
            try:
                filename.encode('latin1')
                file_expr = 'filename="{}"'.format(filename)
            except UnicodeEncodeError:
                file_expr = "filename*=utf-8''{}".format(quote(filename))
            self.set_header('Content-Disposition', f'attachment; {file_expr}')

    def validate_absolute_path(self, root: str, absolute_path: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        try:
            self._storage.get_file(absolute_path)
        except MediaFileStorageError:
            _LOGGER.error('MediaFileHandler: Missing file %s', absolute_path)
            raise tornado.web.HTTPError(404, 'not found')
        return absolute_path

    def get_content_size(self) -> int:
        if False:
            return 10
        abspath = self.absolute_path
        if abspath is None:
            return 0
        media_file = self._storage.get_file(abspath)
        return media_file.content_size

    def get_modified_time(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        return None

    @classmethod
    def get_absolute_path(cls, root: str, path: str) -> str:
        if False:
            return 10
        return path

    @classmethod
    def get_content(cls, abspath: str, start: Optional[int]=None, end: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        _LOGGER.debug('MediaFileHandler: GET %s', abspath)
        try:
            media_file = cls._storage.get_file(abspath)
        except Exception:
            _LOGGER.error('MediaFileHandler: Missing file %s', abspath)
            return None
        _LOGGER.debug('MediaFileHandler: Sending %s file %s', media_file.mimetype, abspath)
        if start is None and end is None:
            return media_file.content
        if start is None:
            start = 0
        if end is None:
            end = len(media_file.content)
        return media_file.content[start:end]