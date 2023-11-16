import unittest
from lbry.schema import mime_types

class TestMimeTypes(unittest.TestCase):

    def test_mp4_video(self):
        if False:
            print('Hello World!')
        self.assertEqual('video/mp4', mime_types.guess_media_type('test.mp4')[0])
        self.assertEqual('video/mp4', mime_types.guess_media_type('test.MP4')[0])

    def test_x_ext_(self):
        if False:
            return 10
        self.assertEqual('application/x-ext-lbry', mime_types.guess_media_type('test.lbry')[0])
        self.assertEqual('application/x-ext-lbry', mime_types.guess_media_type('test.LBRY')[0])

    def test_octet_stream(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('application/octet-stream', mime_types.guess_media_type('test.')[0])
        self.assertEqual('application/octet-stream', mime_types.guess_media_type('test')[0])