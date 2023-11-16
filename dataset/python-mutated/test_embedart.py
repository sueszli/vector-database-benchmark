import os.path
import shutil
import tempfile
import unittest
from test import _common
from test.helper import TestHelper
from test.plugins.test_art import FetchImageHelper
from test.test_art_resize import DummyIMBackend
from unittest.mock import MagicMock, patch
from mediafile import MediaFile
from beets import art, config, logging, ui
from beets.util import bytestring_path, displayable_path, syspath
from beets.util.artresizer import ArtResizer

def require_artresizer_compare(test):
    if False:
        i = 10
        return i + 15

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        if not ArtResizer.shared.can_compare:
            raise unittest.SkipTest('compare not available')
        else:
            return test(*args, **kwargs)
    wrapper.__name__ = test.__name__
    return wrapper

class EmbedartCliTest(TestHelper, FetchImageHelper):
    small_artpath = os.path.join(_common.RSRC, b'image-2x3.jpg')
    abbey_artpath = os.path.join(_common.RSRC, b'abbey.jpg')
    abbey_similarpath = os.path.join(_common.RSRC, b'abbey-similar.jpg')
    abbey_differentpath = os.path.join(_common.RSRC, b'abbey-different.jpg')

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.io.install()
        self.setup_beets()
        self.load_plugins('embedart')

    def _setup_data(self, artpath=None):
        if False:
            while True:
                i = 10
        if not artpath:
            artpath = self.small_artpath
        with open(syspath(artpath), 'rb') as f:
            self.image_data = f.read()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.unload_plugins()
        self.teardown_beets()

    def test_embed_art_from_file_with_yes_input(self):
        if False:
            while True:
                i = 10
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.io.addinput('y')
        self.run_command('embedart', '-f', self.small_artpath)
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data)

    def test_embed_art_from_file_with_no_input(self):
        if False:
            while True:
                i = 10
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.io.addinput('n')
        self.run_command('embedart', '-f', self.small_artpath)
        mediafile = MediaFile(syspath(item.path))
        self.assertFalse(mediafile.images)

    def test_embed_art_from_file(self):
        if False:
            print('Hello World!')
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.run_command('embedart', '-y', '-f', self.small_artpath)
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data)

    def test_embed_art_from_album(self):
        if False:
            i = 10
            return i + 15
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        album.artpath = self.small_artpath
        album.store()
        self.run_command('embedart', '-y')
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data)

    def test_embed_art_remove_art_file(self):
        if False:
            for i in range(10):
                print('nop')
        self._setup_data()
        album = self.add_album_fixture()
        logging.getLogger('beets.embedart').setLevel(logging.DEBUG)
        (handle, tmp_path) = tempfile.mkstemp()
        tmp_path = bytestring_path(tmp_path)
        os.write(handle, self.image_data)
        os.close(handle)
        album.artpath = tmp_path
        album.store()
        config['embedart']['remove_art_file'] = True
        self.run_command('embedart', '-y')
        if os.path.isfile(syspath(tmp_path)):
            os.remove(syspath(tmp_path))
            self.fail('Artwork file {} was not deleted'.format(displayable_path(tmp_path)))

    def test_art_file_missing(self):
        if False:
            return 10
        self.add_album_fixture()
        logging.getLogger('beets.embedart').setLevel(logging.DEBUG)
        with self.assertRaises(ui.UserError):
            self.run_command('embedart', '-y', '-f', '/doesnotexist')

    def test_embed_non_image_file(self):
        if False:
            i = 10
            return i + 15
        album = self.add_album_fixture()
        logging.getLogger('beets.embedart').setLevel(logging.DEBUG)
        (handle, tmp_path) = tempfile.mkstemp()
        tmp_path = bytestring_path(tmp_path)
        os.write(handle, b'I am not an image.')
        os.close(handle)
        try:
            self.run_command('embedart', '-y', '-f', tmp_path)
        finally:
            os.remove(syspath(tmp_path))
        mediafile = MediaFile(syspath(album.items()[0].path))
        self.assertFalse(mediafile.images)

    @require_artresizer_compare
    def test_reject_different_art(self):
        if False:
            while True:
                i = 10
        self._setup_data(self.abbey_artpath)
        album = self.add_album_fixture()
        item = album.items()[0]
        self.run_command('embedart', '-y', '-f', self.abbey_artpath)
        config['embedart']['compare_threshold'] = 20
        self.run_command('embedart', '-y', '-f', self.abbey_differentpath)
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data, 'Image written is not {}'.format(displayable_path(self.abbey_artpath)))

    @require_artresizer_compare
    def test_accept_similar_art(self):
        if False:
            i = 10
            return i + 15
        self._setup_data(self.abbey_similarpath)
        album = self.add_album_fixture()
        item = album.items()[0]
        self.run_command('embedart', '-y', '-f', self.abbey_artpath)
        config['embedart']['compare_threshold'] = 20
        self.run_command('embedart', '-y', '-f', self.abbey_similarpath)
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data, 'Image written is not {}'.format(displayable_path(self.abbey_similarpath)))

    def test_non_ascii_album_path(self):
        if False:
            i = 10
            return i + 15
        resource_path = os.path.join(_common.RSRC, b'image.mp3')
        album = self.add_album_fixture()
        trackpath = album.items()[0].path
        albumpath = album.path
        shutil.copy(syspath(resource_path), syspath(trackpath))
        self.run_command('extractart', '-n', 'extracted')
        self.assertExists(os.path.join(albumpath, b'extracted.png'))

    def test_extracted_extension(self):
        if False:
            i = 10
            return i + 15
        resource_path = os.path.join(_common.RSRC, b'image-jpeg.mp3')
        album = self.add_album_fixture()
        trackpath = album.items()[0].path
        albumpath = album.path
        shutil.copy(syspath(resource_path), syspath(trackpath))
        self.run_command('extractart', '-n', 'extracted')
        self.assertExists(os.path.join(albumpath, b'extracted.jpg'))

    def test_clear_art_with_yes_input(self):
        if False:
            for i in range(10):
                print('nop')
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.io.addinput('y')
        self.run_command('embedart', '-f', self.small_artpath)
        self.io.addinput('y')
        self.run_command('clearart')
        mediafile = MediaFile(syspath(item.path))
        self.assertFalse(mediafile.images)

    def test_clear_art_with_no_input(self):
        if False:
            i = 10
            return i + 15
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.io.addinput('y')
        self.run_command('embedart', '-f', self.small_artpath)
        self.io.addinput('n')
        self.run_command('clearart')
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.image_data)

    def test_embed_art_from_url_with_yes_input(self):
        if False:
            return 10
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.mock_response('http://example.com/test.jpg', 'image/jpeg')
        self.io.addinput('y')
        self.run_command('embedart', '-u', 'http://example.com/test.jpg')
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.IMAGEHEADER.get('image/jpeg').ljust(32, b'\x00'))

    def test_embed_art_from_url_png(self):
        if False:
            while True:
                i = 10
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.mock_response('http://example.com/test.png', 'image/png')
        self.run_command('embedart', '-y', '-u', 'http://example.com/test.png')
        mediafile = MediaFile(syspath(item.path))
        self.assertEqual(mediafile.images[0].data, self.IMAGEHEADER.get('image/png').ljust(32, b'\x00'))

    def test_embed_art_from_url_not_image(self):
        if False:
            return 10
        self._setup_data()
        album = self.add_album_fixture()
        item = album.items()[0]
        self.mock_response('http://example.com/test.html', 'text/html')
        self.run_command('embedart', '-y', '-u', 'http://example.com/test.html')
        mediafile = MediaFile(syspath(item.path))
        self.assertFalse(mediafile.images)

class DummyArtResizer(ArtResizer):
    """An `ArtResizer` which pretends that ImageMagick is available, and has
    a sufficiently recent version to support image comparison.
    """

    def __init__(self):
        if False:
            return 10
        self.local_method = DummyIMBackend()

@patch('beets.util.artresizer.subprocess')
@patch('beets.art.extract')
class ArtSimilarityTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.item = _common.item()
        self.log = logging.getLogger('beets.embedart')
        self.artresizer = DummyArtResizer()

    def _similarity(self, threshold):
        if False:
            while True:
                i = 10
        return art.check_art_similarity(self.log, self.item, b'path', threshold, artresizer=self.artresizer)

    def _popen(self, status=0, stdout='', stderr=''):
        if False:
            for i in range(10):
                print('nop')
        'Create a mock `Popen` object.'
        popen = MagicMock(returncode=status)
        popen.communicate.return_value = (stdout, stderr)
        return popen

    def _mock_popens(self, mock_extract, mock_subprocess, compare_status=0, compare_stdout='', compare_stderr='', convert_status=0):
        if False:
            for i in range(10):
                print('nop')
        mock_extract.return_value = b'extracted_path'
        mock_subprocess.Popen.side_effect = [self._popen(convert_status), self._popen(compare_status, compare_stdout, compare_stderr)]

    def test_compare_success_similar(self, mock_extract, mock_subprocess):
        if False:
            while True:
                i = 10
        self._mock_popens(mock_extract, mock_subprocess, 0, '10', 'err')
        self.assertTrue(self._similarity(20))

    def test_compare_success_different(self, mock_extract, mock_subprocess):
        if False:
            i = 10
            return i + 15
        self._mock_popens(mock_extract, mock_subprocess, 0, '10', 'err')
        self.assertFalse(self._similarity(5))

    def test_compare_status1_similar(self, mock_extract, mock_subprocess):
        if False:
            while True:
                i = 10
        self._mock_popens(mock_extract, mock_subprocess, 1, 'out', '10')
        self.assertTrue(self._similarity(20))

    def test_compare_status1_different(self, mock_extract, mock_subprocess):
        if False:
            return 10
        self._mock_popens(mock_extract, mock_subprocess, 1, 'out', '10')
        self.assertFalse(self._similarity(5))

    def test_compare_failed(self, mock_extract, mock_subprocess):
        if False:
            return 10
        self._mock_popens(mock_extract, mock_subprocess, 2, 'out', '10')
        self.assertIsNone(self._similarity(20))

    def test_compare_parsing_error(self, mock_extract, mock_subprocess):
        if False:
            i = 10
            return i + 15
        self._mock_popens(mock_extract, mock_subprocess, 0, 'foo', 'bar')
        self.assertIsNone(self._similarity(20))

    def test_compare_parsing_error_and_failure(self, mock_extract, mock_subprocess):
        if False:
            print('Hello World!')
        self._mock_popens(mock_extract, mock_subprocess, 1, 'foo', 'bar')
        self.assertIsNone(self._similarity(20))

    def test_convert_failure(self, mock_extract, mock_subprocess):
        if False:
            for i in range(10):
                print('nop')
        self._mock_popens(mock_extract, mock_subprocess, convert_status=1)
        self.assertIsNone(self._similarity(20))

def suite():
    if False:
        return 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')