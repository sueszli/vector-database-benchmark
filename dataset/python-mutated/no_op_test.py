import unittest
from r2.lib.providers.image_resizing.no_op import NoOpImageResizingProvider

class TestLocalResizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.provider = NoOpImageResizingProvider()

    def test_no_resize(self):
        if False:
            for i in range(10):
                print('nop')
        image = dict(url='http://s3.amazonaws.com/a.jpg', width=1200, height=800)
        url = self.provider.resize_image(image)
        self.assertEqual(url, 'http://s3.amazonaws.com/a.jpg')

    def test_resize(self):
        if False:
            return 10
        image = dict(url='http://s3.amazonaws.com/a.jpg', width=1200, height=800)
        for width in (108, 216, 320, 640, 960, 1080):
            url = self.provider.resize_image(image, width)
            self.assertEqual(url, 'http://s3.amazonaws.com/a.jpg')