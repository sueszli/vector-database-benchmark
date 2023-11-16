from threading import Event
from unittest import TestCase, mock
from mycroft.util.download import download, _running_downloads, _get_download_tmp
from mycroft.util.file_utils import get_temp_path
TEST_URL = 'http://example.com/mycroft-test.tar.gz'
TEST_DEST = get_temp_path('file.tar.gz')

@mock.patch('mycroft.util.download.subprocess')
@mock.patch('mycroft.util.download.os')
class TestDownload(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Remove any cached instance.'
        for key in list(_running_downloads.keys()):
            _running_downloads.pop(key)

    def test_download_basic(self, mock_os, mock_subprocess):
        if False:
            i = 10
            return i + 15
        'Test the basic download call.'
        mock_subprocess.call.return_value = 0
        downloader = download(url=TEST_URL, dest=TEST_DEST)
        downloader.join()
        mock_subprocess.call.assert_called_once_with(['wget', '-c', TEST_URL, '-O', TEST_DEST + '.part', '--tries=20', '--read-timeout=5'])
        self.assertTrue(downloader.done)

    def test_download_with_header(self, mock_os, mock_subprocess):
        if False:
            for i in range(10):
                print('nop')
        'Test download with specific header.'
        mock_subprocess.call.return_value = 0
        test_hdr = 'TEST_HEADER'
        downloader = download(url=TEST_URL, dest=TEST_DEST, header=test_hdr)
        downloader.join()
        self.assertTrue(downloader.done)
        mock_subprocess.call.assert_called_once_with(['wget', '-c', TEST_URL, '-O', TEST_DEST + '.part', '--tries=20', '--read-timeout=5', '--header=' + test_hdr])

    def test_download_callback(self, mock_os, mock_subprocess):
        if False:
            i = 10
            return i + 15
        'Check that callback function is called with correct destination.'
        mock_subprocess.call.return_value = 0
        action_called_with = None

        def action(dest):
            if False:
                return 10
            nonlocal action_called_with
            action_called_with = dest
        downloader = download(url=TEST_URL, dest=TEST_DEST, complete_action=action)
        downloader.join()
        self.assertTrue(downloader.done)
        self.assertEqual(action_called_with, TEST_DEST)

    def test_download_cache(self, mock_os, mock_subprocess):
        if False:
            while True:
                i = 10
        'Make sure that a cached download is used if exists.'
        transfer_done = Event()

        def wget_call(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal transfer_done
            transfer_done.wait()
            return 0
        downloader = download(url=TEST_URL, dest=TEST_DEST)
        downloader2 = download(url=TEST_URL, dest=TEST_DEST)
        self.assertTrue(downloader is downloader2)
        transfer_done.set()
        downloader.join()

@mock.patch('mycroft.util.download.glob')
class TestGetTemp(TestCase):

    def test_no_existing(self, mock_glob):
        if False:
            i = 10
            return i + 15
        mock_glob.return_value = []
        dest = get_temp_path('test')
        self.assertEqual(_get_download_tmp(dest), dest + '.part')

    def test_existing(self, mock_glob):
        if False:
            print('Hello World!')
        mock_glob.return_value = [get_temp_path('test.part')]
        dest = get_temp_path('test')
        self.assertEqual(_get_download_tmp(dest), dest + '.part.1')

    def test_multiple_existing(self, mock_glob):
        if False:
            return 10
        mock_glob.return_value = [get_temp_path('test.part'), get_temp_path('test.part.1'), get_temp_path('test.part.2')]
        dest = get_temp_path('test')
        self.assertEqual(_get_download_tmp(dest), dest + '.part.3')