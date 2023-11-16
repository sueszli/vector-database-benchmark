import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestPixelsToRowCol(unittest.TestCase):
    """
    Test the Worksheet _pixels_to_xxx() methods.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def width_to_pixels(self, width):
        if False:
            i = 10
            return i + 15
        max_digit_width = 7
        padding = 5
        if width < 1:
            pixels = int(width * (max_digit_width + padding) + 0.5)
        else:
            pixels = int(width * max_digit_width + 0.5) + padding
        return pixels

    def height_to_pixels(self, height):
        if False:
            while True:
                i = 10
        return int(4.0 / 3.0 * height)

    def test_pixels_to_width(self):
        if False:
            return 10
        'Test the _pixels_to_width() function'
        for pixels in range(1791):
            exp = pixels
            got = self.width_to_pixels(self.worksheet._pixels_to_width(pixels))
            self.assertEqual(got, exp)

    def test_pixels_to_height(self):
        if False:
            while True:
                i = 10
        'Test the _pixels_to_height() function'
        for pixels in range(546):
            exp = pixels
            got = self.height_to_pixels(self.worksheet._pixels_to_height(pixels))
            self.assertEqual(got, exp)