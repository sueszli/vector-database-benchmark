import unittest
from r2.lib.providers.image_resizing.unsplashit import UnsplashitImageResizingProvider

class TestLocalResizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.provider = UnsplashitImageResizingProvider()

    def test_no_resize(self):
        if False:
            while True:
                i = 10
        image = dict(url='http://s3.amazonaws.com/a.jpg', width=200, height=800)
        url = self.provider.resize_image(image)
        self.assertEqual(url, 'https://unsplash.it/200/400')

    def test_resize(self):
        if False:
            i = 10
            return i + 15
        image = dict(url='http://s3.amazonaws.com/a.jpg', width=1200, height=800)
        for width in (108, 216, 320, 640, 960, 1080):
            url = self.provider.resize_image(image, width)
            self.assertEqual(url, 'https://unsplash.it/%d/%d' % (width, width * 2))