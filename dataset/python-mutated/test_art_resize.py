"""Tests for image resizing based on filesize."""
import os
import unittest
from test import _common
from test.helper import TestHelper
from unittest.mock import patch
from beets.util import command_output, syspath
from beets.util.artresizer import IMBackend, PILBackend

class DummyIMBackend(IMBackend):
    """An `IMBackend` which pretends that ImageMagick is available.

    The version is sufficiently recent to support image comparison.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Init a dummy backend class for mocked ImageMagick tests.'
        self.version = (7, 0, 0)
        self.legacy = False
        self.convert_cmd = ['magick']
        self.identify_cmd = ['magick', 'identify']
        self.compare_cmd = ['magick', 'compare']

class DummyPILBackend(PILBackend):
    """An `PILBackend` which pretends that PIL is available."""

    def __init__(self):
        if False:
            return 10
        'Init a dummy backend class for mocked PIL tests.'
        pass

class ArtResizerFileSizeTest(_common.TestCase, TestHelper):
    """Unittest test case for Art Resizer to a specific filesize."""
    IMG_225x225 = os.path.join(_common.RSRC, b'abbey.jpg')
    IMG_225x225_SIZE = os.stat(syspath(IMG_225x225)).st_size

    def setUp(self):
        if False:
            print('Hello World!')
        'Called before each test, setting up beets.'
        self.setup_beets()

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Called after each test, unloading all plugins.'
        self.teardown_beets()

    def _test_img_resize(self, backend):
        if False:
            print('Hello World!')
        'Test resizing based on file size, given a resize_func.'
        im_95_qual = backend.resize(225, self.IMG_225x225, quality=95, max_filesize=0)
        self.assertExists(im_95_qual)
        im_a = backend.resize(225, self.IMG_225x225, quality=95, max_filesize=0.9 * os.stat(syspath(im_95_qual)).st_size)
        self.assertExists(im_a)
        self.assertLess(os.stat(syspath(im_a)).st_size, os.stat(syspath(im_95_qual)).st_size)
        im_75_qual = backend.resize(225, self.IMG_225x225, quality=75, max_filesize=0)
        self.assertExists(im_75_qual)
        im_b = backend.resize(225, self.IMG_225x225, quality=95, max_filesize=0.9 * os.stat(syspath(im_75_qual)).st_size)
        self.assertExists(im_b)
        self.assertLess(os.stat(syspath(im_b)).st_size, os.stat(syspath(im_75_qual)).st_size)

    @unittest.skipUnless(PILBackend.available(), 'PIL not available')
    def test_pil_file_resize(self):
        if False:
            print('Hello World!')
        'Test PIL resize function is lowering file size.'
        self._test_img_resize(PILBackend())

    @unittest.skipUnless(IMBackend.available(), 'ImageMagick not available')
    def test_im_file_resize(self):
        if False:
            for i in range(10):
                print('nop')
        'Test IM resize function is lowering file size.'
        self._test_img_resize(IMBackend())

    @unittest.skipUnless(PILBackend.available(), 'PIL not available')
    def test_pil_file_deinterlace(self):
        if False:
            print('Hello World!')
        'Test PIL deinterlace function.\n\n        Check if the `PILBackend.deinterlace()` function returns images\n        that are non-progressive\n        '
        path = PILBackend().deinterlace(self.IMG_225x225)
        from PIL import Image
        with Image.open(path) as img:
            self.assertFalse('progression' in img.info)

    @unittest.skipUnless(IMBackend.available(), 'ImageMagick not available')
    def test_im_file_deinterlace(self):
        if False:
            return 10
        'Test ImageMagick deinterlace function.\n\n        Check if the `IMBackend.deinterlace()` function returns images\n        that are non-progressive.\n        '
        im = IMBackend()
        path = im.deinterlace(self.IMG_225x225)
        cmd = im.identify_cmd + ['-format', '%[interlace]', syspath(path, prefix=False)]
        out = command_output(cmd).stdout
        self.assertTrue(out == b'None')

    @patch('beets.util.artresizer.util')
    def test_write_metadata_im(self, mock_util):
        if False:
            return 10
        'Test writing image metadata.'
        metadata = {'a': 'A', 'b': 'B'}
        im = DummyIMBackend()
        im.write_metadata('foo', metadata)
        try:
            command = im.convert_cmd + 'foo -set a A -set b B foo'.split()
            mock_util.command_output.assert_called_once_with(command)
        except AssertionError:
            command = im.convert_cmd + 'foo -set b B -set a A foo'.split()
            mock_util.command_output.assert_called_once_with(command)

def suite():
    if False:
        print('Hello World!')
    'Run this suite of tests.'
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')