import os.path
from picard import config
from picard.coverart.image import CoverArtImage, TagCoverArtImage
import picard.formats
from picard.metadata import Metadata
from .common import CommonTests, load_metadata, save_and_load_metadata, skipUnlessTestfile

def file_save_image(filename, image):
    if False:
        for i in range(10):
            print('nop')
    f = picard.formats.open_(filename)
    metadata = Metadata(images=[image])
    f._save(filename, metadata)

def load_coverart_file(filename):
    if False:
        return 10
    with open(os.path.join('test', 'data', filename), 'rb') as f:
        return f.read()

class DummyUnsupportedCoverArt(CoverArtImage):

    def __init__(self, data=b'', mimetype='image/unknown'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.mimetype = mimetype
        self.width = 100
        self.height = 100
        self.extension = '.cvr'
        self.set_data(data)

    def set_data(self, data):
        if False:
            while True:
                i = 10
        self._data = data
        self.datalength = len(data)

    @property
    def data(self):
        if False:
            print('Hello World!')
        return self._data

class CommonCoverArtTests:

    class CoverArtTestCase(CommonTests.BaseFileTestCase):
        supports_types = True

        def setUp(self):
            if False:
                print('Hello World!')
            super().setUp()
            self.set_config_values({'clear_existing_tags': False, 'preserve_images': False})
            self.jpegdata = load_coverart_file('mb.jpg')
            self.pngdata = load_coverart_file('mb.png')

        @skipUnlessTestfile
        def test_cover_art(self):
            if False:
                for i in range(10):
                    print('nop')
            source_types = ['front', 'booklet']
            payload = b'a' * 1024 * 128
            tests = [CoverArtImage(data=self.jpegdata + payload, types=source_types), CoverArtImage(data=self.pngdata + payload, types=source_types)]
            for test in tests:
                file_save_image(self.filename, test)
                loaded_metadata = load_metadata(self.filename)
                image = loaded_metadata.images[0]
                self.assertEqual(test.mimetype, image.mimetype)
                self.assertEqual(test, image)

        def test_cover_art_with_types(self):
            if False:
                i = 10
                return i + 15
            expected = set('abcdefg'[:]) if self.supports_types else set('a')
            loaded_metadata = save_and_load_metadata(self.filename, self._cover_metadata())
            found = {chr(img.data[-1]) for img in loaded_metadata.images}
            self.assertEqual(expected, found)

        @skipUnlessTestfile
        def test_cover_art_types_only_one_front(self):
            if False:
                return 10
            config.setting['embed_only_one_front_image'] = True
            loaded_metadata = save_and_load_metadata(self.filename, self._cover_metadata())
            self.assertEqual(1, len(loaded_metadata.images))
            self.assertEqual(ord('a'), loaded_metadata.images[0].data[-1])

        @skipUnlessTestfile
        def test_unsupported_image_format(self):
            if False:
                for i in range(10):
                    print('nop')
            metadata = Metadata()
            metadata.images.append(DummyUnsupportedCoverArt(b'unsupported', 'image/unknown'))
            metadata.images.append(DummyUnsupportedCoverArt(b'unsupported', 'image/png'))
            loaded_metadata = save_and_load_metadata(self.filename, metadata)
            self.assertEqual(0, len(loaded_metadata.images))

        @skipUnlessTestfile
        def test_cover_art_clear_tags(self):
            if False:
                return 10
            image = CoverArtImage(data=self.pngdata, types=['front'])
            file_save_image(self.filename, image)
            metadata = load_metadata(self.filename)
            self.assertEqual(image, metadata.images[0])
            config.setting['clear_existing_tags'] = True
            config.setting['preserve_images'] = True
            metadata = save_and_load_metadata(self.filename, Metadata())
            self.assertEqual(image, metadata.images[0])
            config.setting['preserve_images'] = False
            metadata = save_and_load_metadata(self.filename, Metadata())
            self.assertEqual(0, len(metadata.images))

        @skipUnlessTestfile
        def test_cover_art_clear_tags_preserve_images_no_existing_images(self):
            if False:
                return 10
            config.setting['clear_existing_tags'] = True
            config.setting['preserve_images'] = True
            image = CoverArtImage(data=self.pngdata, types=['front'])
            file_save_image(self.filename, image)
            metadata = load_metadata(self.filename)
            self.assertEqual(image, metadata.images[0])

        def _cover_metadata(self):
            if False:
                for i in range(10):
                    print('nop')
            imgdata = self.jpegdata
            metadata = Metadata()
            metadata.images.append(TagCoverArtImage(file='a', tag='a', data=imgdata + b'a', support_types=True, types=['booklet', 'front']))
            metadata.images.append(TagCoverArtImage(file='b', tag='b', data=imgdata + b'b', support_types=True, types=['back']))
            metadata.images.append(TagCoverArtImage(file='c', tag='c', data=imgdata + b'c', support_types=True, types=['front']))
            metadata.images.append(TagCoverArtImage(file='d', tag='d', data=imgdata + b'd'))
            metadata.images.append(TagCoverArtImage(file='e', tag='e', data=imgdata + b'e', is_front=False))
            metadata.images.append(TagCoverArtImage(file='f', tag='f', data=imgdata + b'f', types=['front']))
            metadata.images.append(TagCoverArtImage(file='g', tag='g', data=imgdata + b'g', types=['back'], is_front=True))
            return metadata