from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
from kivy.tests.common import GraphicUnitTest, ensure_web_server

class AsyncImageTestCase(GraphicUnitTest):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        from kivy import kivy_examples_dir
        ensure_web_server(kivy_examples_dir)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        from kivy.config import Config
        self.maxfps = Config.getint('graphics', 'maxfps')
        assert self.maxfps > 0
        super(AsyncImageTestCase, self).setUp()

    def zip_frames(self, path):
        if False:
            print('Hello World!')
        with ZipFile(path) as zipf:
            return len(zipf.namelist())

    def wait_for_event_or_timeout(self, event):
        if False:
            return 10
        timeout = 30 * self.maxfps
        while timeout and (not event.is_set()):
            self.advance_frames(1)
            timeout -= 1

    def load_zipimage(self, source, frames):
        if False:
            print('Hello World!')
        from kivy.uix.image import AsyncImage
        event = Event()
        image = AsyncImage(anim_delay=0.0333333333333333)
        image.bind(on_load=lambda *args, **kwargs: event.set())
        image.source = source
        self.wait_for_event_or_timeout(event)
        self.render(image)
        proxyimg = image._coreimage
        self.assertTrue(proxyimg.anim_available)
        self.assertEqual(len(proxyimg.image.textures), frames)
        return image

    def test_remote_zipsequence(self):
        if False:
            for i in range(10):
                print('nop')
        zip_cube = 'http://localhost:8000/widgets/sequenced_images/data/images/cube.zip'
        (tempf, headers) = urlretrieve(zip_cube)
        zip_pngs = self.zip_frames(tempf)
        remove(tempf)
        image = self.load_zipimage(zip_cube, zip_pngs)
        self.assertTrue(self.check_sequence_frames(image._coreimage, int(image._coreimage.anim_delay * self.maxfps + 3)))

    def test_local_zipsequence(self):
        if False:
            i = 10
            return i + 15
        from kivy import kivy_examples_dir
        zip_cube = join(kivy_examples_dir, 'widgets', 'sequenced_images', 'data', 'images', 'cube.zip')
        zip_pngs = self.zip_frames(zip_cube)
        image = self.load_zipimage(zip_cube, zip_pngs)
        self.assertTrue(self.check_sequence_frames(image._coreimage, int(image._coreimage.anim_delay * self.maxfps + 3)))

    def check_sequence_frames(self, img, frames, slides=5):
        if False:
            i = 10
            return i + 15
        old = None
        while slides:
            self.assertNotEqual(img.anim_index, old)
            old = img.anim_index
            self.advance_frames(frames)
            slides -= 1
        return True

    def test_reload_asyncimage(self):
        if False:
            return 10
        from kivy.resources import resource_find
        from kivy.uix.image import AsyncImage
        temp_dir = mkdtemp()
        event = Event()
        image = AsyncImage()
        image.bind(on_load=lambda *args, **kwargs: event.set())
        fn = resource_find('data/logo/kivy-icon-16.png')
        source = join(temp_dir, 'source.png')
        copyfile(fn, source)
        event.clear()
        image.source = source
        self.wait_for_event_or_timeout(event)
        self.render(image, framecount=2)
        self.assertEqual(image.texture_size, [16, 16])
        remove(source)
        fn = resource_find('data/logo/kivy-icon-32.png')
        copyfile(fn, source)
        event.clear()
        image.reload()
        self.wait_for_event_or_timeout(event)
        self.render(image, framecount=2)
        self.assertEqual(image.texture_size, [32, 32])
        remove(source)
        rmtree(temp_dir)
if __name__ == '__main__':
    import unittest
    unittest.main()