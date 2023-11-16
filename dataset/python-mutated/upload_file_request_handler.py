from typing import Any, Callable, Dict, List
import tornado.httputil
import tornado.web
from streamlit import config
from streamlit.logger import get_logger
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileManager, UploadedFileRec
from streamlit.web.server import routes, server_util
LOGGER = get_logger(__name__)

class UploadFileRequestHandler(tornado.web.RequestHandler):
    """Implements the POST /upload_file endpoint."""

    def initialize(self, file_mgr: MemoryUploadedFileManager, is_active_session: Callable[[str], bool]):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        file_mgr : UploadedFileManager\n            The server's singleton UploadedFileManager. All file uploads\n            go here.\n        is_active_session:\n            A function that returns true if a session_id belongs to an active\n            session.\n        "
        self._file_mgr = file_mgr
        self._is_active_session = is_active_session

    def set_default_headers(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_header('Access-Control-Allow-Methods', 'PUT, OPTIONS, DELETE')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')
        if config.get_option('server.enableXsrfProtection'):
            self.set_header('Access-Control-Allow-Origin', server_util.get_url(config.get_option('browser.serverAddress')))
            self.set_header('Access-Control-Allow-Headers', 'X-Xsrftoken, Content-Type')
            self.set_header('Vary', 'Origin')
            self.set_header('Access-Control-Allow-Credentials', 'true')
        elif routes.allow_cross_origin_requests():
            self.set_header('Access-Control-Allow-Origin', '*')

    def options(self, **kwargs):
        if False:
            while True:
                i = 10
        '/OPTIONS handler for preflight CORS checks.\n\n        When a browser is making a CORS request, it may sometimes first\n        send an OPTIONS request, to check whether the server understands the\n        CORS protocol. This is optional, and doesn\'t happen for every request\n        or in every browser. If an OPTIONS request does get sent, and is not\n        then handled by the server, the browser will fail the underlying\n        request.\n\n        The proper way to handle this is to send a 204 response ("no content")\n        with the CORS headers attached. (These headers are automatically added\n        to every outgoing response, including OPTIONS responses,\n        via set_default_headers().)\n\n        See https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request\n        '
        self.set_status(204)
        self.finish()

    def put(self, **kwargs):
        if False:
            print('Hello World!')
        'Receive an uploaded file and add it to our UploadedFileManager.'
        args: Dict[str, List[bytes]] = {}
        files: Dict[str, List[Any]] = {}
        session_id = self.path_kwargs['session_id']
        file_id = self.path_kwargs['file_id']
        tornado.httputil.parse_body_arguments(content_type=self.request.headers['Content-Type'], body=self.request.body, arguments=args, files=files)
        try:
            if not self._is_active_session(session_id):
                raise Exception(f'Invalid session_id')
        except Exception as e:
            self.send_error(400, reason=str(e))
            return
        uploaded_files: List[UploadedFileRec] = []
        for (_, flist) in files.items():
            for file in flist:
                uploaded_files.append(UploadedFileRec(file_id=file_id, name=file['filename'], type=file['content_type'], data=file['body']))
        if len(uploaded_files) != 1:
            self.send_error(400, reason=f'Expected 1 file, but got {len(uploaded_files)}')
            return
        self._file_mgr.add_file(session_id=session_id, file=uploaded_files[0])
        self.set_status(204)

    def delete(self, **kwargs):
        if False:
            print('Hello World!')
        'Delete file request handler.'
        session_id = self.path_kwargs['session_id']
        file_id = self.path_kwargs['file_id']
        self._file_mgr.remove_file(session_id=session_id, file_id=file_id)
        self.set_status(204)