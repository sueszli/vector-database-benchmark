import os
import tempfile
from unittest.mock import MagicMock, patch
import tornado.httpserver
import tornado.testing
import tornado.web
import tornado.websocket
from streamlit.web.server.app_static_file_handler import MAX_APP_STATIC_FILE_SIZE, AppStaticFileHandler

class AppStaticFileHandlerTest(tornado.testing.AsyncHTTPTestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self._tmpdir = tempfile.TemporaryDirectory(dir=os.getcwd())
        self._tmpfile = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, delete=False)
        self._tmp_js_file = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, suffix='script.js', delete=False)
        self._tmp_png_image_file = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, suffix='image.png', delete=False)
        self._tmp_webp_image_file = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, suffix='image.webp', delete=False)
        self._symlink_outside_directory = 'symlink_outside'
        self._symlink_inside_directory = 'symlink_inside'
        os.symlink('/', os.path.join(self._tmpdir.name, self._symlink_outside_directory))
        os.symlink(self._tmpfile.name, os.path.join(self._tmpdir.name, self._symlink_inside_directory))
        self._filename = os.path.basename(self._tmpfile.name)
        self._js_filename = os.path.basename(self._tmp_js_file.name)
        self._png_image_filename = os.path.basename(self._tmp_png_image_file.name)
        self._webp_image_filename = os.path.basename(self._tmp_webp_image_file.name)
        super().setUp()

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        self._tmpdir.cleanup()

    def get_app(self):
        if False:
            return 10
        return tornado.web.Application([('/app/static/(.*)', AppStaticFileHandler, {'path': '%s/' % self._tmpdir.name})])

    def test_static_files_200(self):
        if False:
            i = 10
            return i + 15
        'Files with extensions NOT listed in app_static_file_handler.py\n        `SAFE_APP_STATIC_FILE_EXTENSIONS` should have the `Content-Type` header value\n        equals to `text-plain`.\n        '
        responses = [self.fetch(f'/app/static/{self._filename}'), self.fetch(f'/app/static/{self._js_filename}'), self.fetch(f'/app/static/{self._symlink_inside_directory}')]
        for r in responses:
            assert r.headers['Content-Type'] == 'text/plain'
            assert r.headers['X-Content-Type-Options'] == 'nosniff'
            assert r.code == 200

    def test_static_png_image_200(self):
        if False:
            print('Hello World!')
        'Files with extensions listed in app_static_file_handler.py\n        `SAFE_APP_STATIC_FILE_EXTENSIONS` (e.g. png) should have the\n        `Content-Type` header based on their extension.\n        '
        response = self.fetch(f'/app/static/{self._png_image_filename}')
        assert response.code == 200
        assert response.headers['Content-Type'] == 'image/png'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'

    def test_static_webp_image_200(self):
        if False:
            print('Hello World!')
        'Files with extensions listed in app_static_file_handler.py\n        `SAFE_APP_STATIC_FILE_EXTENSIONS` (e.g. webp) should have the\n        `Content-Type` header based on their extension.\n        '
        response = self.fetch(f'/app/static/{self._webp_image_filename}')
        assert response.code == 200
        assert response.headers['Content-Type'] == 'image/webp'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'

    @patch('os.path.getsize', MagicMock(return_value=MAX_APP_STATIC_FILE_SIZE + 1))
    def test_big_file_404(self):
        if False:
            while True:
                i = 10
        'Files with size greater than MAX_APP_STATIC_FILE_SIZE should return 404.'
        response = self.fetch(f'/app/static/{self._png_image_filename}')
        assert response.code == 404
        self.assertEqual(b'<html><title>404: File is too large</title><body>404: File is too large</body></html>', response.body)

    def test_staticfiles_404(self):
        if False:
            return 10
        'Non-existent files, files outside static directory and symlinks pointing to\n        files outside static directory and directories should return 404.\n        '
        responses = [self.fetch('/app/static'), self.fetch('/app/static/'), self.fetch('/app/static/../test_file_outside_directory.py'), self.fetch(f'/app/static/{self._symlink_outside_directory}'), self.fetch('/app/static/nonexistent.jpg')]
        for r in responses:
            assert r.code == 404
            assert r.body == b'<html><title>404: Not Found</title><body>404: Not Found</body></html>'