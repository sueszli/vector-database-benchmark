from unittest import mock
from unittest.mock import MagicMock
import tornado.testing
import tornado.web
from parameterized import parameterized
from typing_extensions import Final
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.web.server.media_file_handler import MediaFileHandler
MOCK_ENDPOINT: Final = '/mock/media'

class MediaFileHandlerTest(tornado.testing.AsyncHTTPTestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        storage = MemoryMediaFileStorage(MOCK_ENDPOINT)
        self.media_file_manager = MediaFileManager(storage)
        MediaFileHandler.initialize_storage(storage)

    def get_app(self) -> tornado.web.Application:
        if False:
            for i in range(10):
                print('nop')
        return tornado.web.Application([(f'{MOCK_ENDPOINT}/(.*)', MediaFileHandler, {'path': ''})])

    @mock.patch('streamlit.runtime.media_file_manager._get_session_id', MagicMock(return_value='mock_session_id'))
    def test_media_file(self) -> None:
        if False:
            return 10
        'Requests for media files in MediaFileManager should succeed.'
        url = self.media_file_manager.add(b'mock_data', 'video/mp4', 'mock_coords')
        rsp = self.fetch(url, method='GET')
        self.assertEqual(200, rsp.code)
        self.assertEqual(b'mock_data', rsp.body)
        self.assertEqual('video/mp4', rsp.headers['Content-Type'])
        self.assertEqual(str(len(b'mock_data')), rsp.headers['Content-Length'])

    @parameterized.expand([('MockVideo.mp4', 'video/mp4', 'attachment; filename="MockVideo.mp4"'), (b'\xe6\xbc\xa2\xe5\xad\x97.mp3'.decode(), 'video/mp4', "attachment; filename*=utf-8''%E6%BC%A2%E5%AD%97.mp3"), (None, 'text/plain', 'attachment; filename="streamlit_download.txt"'), (None, 'video/mp4', 'attachment; filename="streamlit_download.mp4"'), (None, 'application/octet-stream', 'attachment; filename="streamlit_download.bin"')])
    @mock.patch('streamlit.runtime.media_file_manager._get_session_id', MagicMock(return_value='mock_session_id'))
    def test_downloadable_file(self, file_name, mimetype, content_disposition_header) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Downloadable files get an additional 'Content-Disposition' header\n        that includes their user-specified filename or\n        generic filename if filename is not specified.\n        "
        url = self.media_file_manager.add(b'mock_data', mimetype, 'mock_coords', file_name=file_name, is_for_static_download=True)
        rsp = self.fetch(url, method='GET')
        self.assertEqual(200, rsp.code)
        self.assertEqual(b'mock_data', rsp.body)
        self.assertEqual(mimetype, rsp.headers['Content-Type'])
        self.assertEqual(str(len(b'mock_data')), rsp.headers['Content-Length'])
        self.assertEqual(content_disposition_header, rsp.headers['Content-Disposition'])

    def test_invalid_file(self) -> None:
        if False:
            while True:
                i = 10
        'Requests for invalid files fail with 404.'
        url = f'{MOCK_ENDPOINT}/invalid_media_file.mp4'
        rsp = self.fetch(url, method='GET')
        self.assertEqual(404, rsp.code)