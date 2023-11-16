from collections import Counter
import os.path
from tempfile import TemporaryDirectory
import unittest
from test.picardtestcase import PicardTestCase, create_fake_png
from picard.const import DEFAULT_COVER_IMAGE_FILENAME
from picard.const.sys import IS_WIN
from picard.coverart.image import CoverArtImage, LocalFileCoverArtImage, TagCoverArtImage
from picard.coverart.utils import Id3ImageType, types_from_id3
from picard.metadata import Metadata
from picard.util import encode_filename
from picard.util.filenaming import WinPathTooLong

def create_image(extra_data, types=None, support_types=False, support_multi_types=False, comment=None, id3_type=None):
    if False:
        return 10
    return CoverArtImage(data=create_fake_png(extra_data), types=types, comment=comment, support_types=support_types, support_multi_types=support_multi_types, id3_type=id3_type)

class TagCoverArtImageTest(PicardTestCase):

    def test_repr_str_1(self):
        if False:
            i = 10
            return i + 15
        image_type = Id3ImageType.COVER_FRONT
        image = TagCoverArtImage(file='testfilename', tag='tag', types=types_from_id3(image_type), comment='description', support_types=True, data=None, id3_type=image_type, is_front=True)
        expected = "TagCoverArtImage('testfilename', tag='tag', types=['front'], support_types=True, support_multi_types=False, is_front=True, comment='description')"
        self.assertEqual(expected, repr(image))
        expected = "TagCoverArtImage from 'testfilename' of type front and comment 'description'"
        self.assertEqual(expected, str(image))

class CoverArtImageTest(PicardTestCase):

    def test_repr_str_1(self):
        if False:
            while True:
                i = 10
        image = CoverArtImage(url='url', types=['booklet', 'front'], comment='comment', support_types=True, support_multi_types=True)
        expected = "CoverArtImage(url='url', types=['booklet', 'front'], support_types=True, support_multi_types=True, comment='comment')"
        self.assertEqual(expected, repr(image))
        expected = "CoverArtImage from url of type booklet,front and comment 'comment'"
        self.assertEqual(expected, str(image))

    def test_repr_str_2(self):
        if False:
            for i in range(10):
                print('nop')
        image = CoverArtImage()
        expected = 'CoverArtImage(support_types=False, support_multi_types=False)'
        self.assertEqual(expected, repr(image))
        expected = 'CoverArtImage'
        self.assertEqual(expected, str(image))

    def test_is_front_image_no_types(self):
        if False:
            return 10
        image = create_image(b'a')
        self.assertTrue(image.is_front_image())
        self.assertEqual(Id3ImageType.COVER_FRONT, image.id3_type)
        image.can_be_saved_to_metadata = False
        self.assertFalse(image.is_front_image())

    def test_is_front_image_types_supported(self):
        if False:
            return 10
        image = create_image(b'a', types=['booklet', 'front'], support_types=True)
        self.assertTrue(image.is_front_image())
        image.is_front = False
        self.assertFalse(image.is_front_image())
        image = create_image(b'a', support_types=True)
        self.assertFalse(image.is_front_image())

    def test_is_front_image_no_types_supported(self):
        if False:
            for i in range(10):
                print('nop')
        image = create_image(b'a', types=['back'], support_types=False)
        self.assertTrue(image.is_front_image())
        self.assertEqual(Id3ImageType.COVER_FRONT, image.id3_type)

    def test_maintype(self):
        if False:
            while True:
                i = 10
        self.assertEqual('front', create_image(b'a').maintype)
        self.assertEqual('front', create_image(b'a', support_types=True).maintype)
        self.assertEqual('front', create_image(b'a', types=['back', 'front'], support_types=True).maintype)
        self.assertEqual('back', create_image(b'a', types=['back', 'medium'], support_types=True).maintype)
        self.assertEqual('front', create_image(b'a', types=['back', 'medium'], support_types=False).maintype)

    def test_id3_type_derived(self):
        if False:
            print('Hello World!')
        self.assertEqual(Id3ImageType.COVER_FRONT, create_image(b'a').id3_type)
        self.assertEqual(Id3ImageType.COVER_FRONT, create_image(b'a', support_types=True).id3_type)
        self.assertEqual(Id3ImageType.COVER_FRONT, create_image(b'a', types=['back', 'front'], support_types=True).id3_type)
        self.assertEqual(Id3ImageType.COVER_BACK, create_image(b'a', types=['back', 'medium'], support_types=True).id3_type)
        self.assertEqual(Id3ImageType.COVER_FRONT, create_image(b'a', types=['back', 'medium'], support_types=False).id3_type)
        self.assertEqual(Id3ImageType.MEDIA, create_image(b'a', types=['medium'], support_types=True).id3_type)
        self.assertEqual(Id3ImageType.LEAFLET_PAGE, create_image(b'a', types=['booklet'], support_types=True).id3_type)
        self.assertEqual(Id3ImageType.OTHER, create_image(b'a', types=['spine'], support_types=True).id3_type)
        self.assertEqual(Id3ImageType.OTHER, create_image(b'a', types=['sticker'], support_types=True).id3_type)

    def test_id3_type_explicit(self):
        if False:
            while True:
                i = 10
        image = create_image(b'a', types=['back'], support_types=True)
        for id3_type in Id3ImageType:
            image.id3_type = id3_type
            self.assertEqual(id3_type, image.id3_type)
        image.id3_type = None
        self.assertEqual(Id3ImageType.COVER_BACK, image.id3_type)

    def test_id3_type_value_error(self):
        if False:
            i = 10
            return i + 15
        image = create_image(b'a')
        for invalid_value in ('foo', 200, -1):
            with self.assertRaises(ValueError):
                image.id3_type = invalid_value

    def test_init_invalid_id3_type(self):
        if False:
            for i in range(10):
                print('nop')
        image = CoverArtImage(id3_type=255)
        self.assertEqual(image.id3_type, Id3ImageType.OTHER)

    def test_compare_without_type(self):
        if False:
            for i in range(10):
                print('nop')
        image1 = create_image(b'a', types=['front'])
        image2 = create_image(b'a', types=['back'])
        image3 = create_image(b'a', types=['back'], support_types=True)
        image4 = create_image(b'b', types=['front'])
        self.assertEqual(image1, image2)
        self.assertEqual(image1, image3)
        self.assertNotEqual(image1, image4)

    def test_compare_with_primary_type(self):
        if False:
            while True:
                i = 10
        image1 = create_image(b'a', types=['front'], support_types=True)
        image2 = create_image(b'a', types=['front', 'booklet'], support_types=True, support_multi_types=True)
        image3 = create_image(b'a', types=['back'], support_types=True)
        image4 = create_image(b'b', types=['front'], support_types=True)
        image5 = create_image(b'a', types=[], support_types=True)
        image6 = create_image(b'a', types=[], support_types=True)
        self.assertEqual(image1, image2)
        self.assertNotEqual(image1, image3)
        self.assertNotEqual(image1, image4)
        self.assertNotEqual(image3, image5)
        self.assertEqual(image5, image6)

    def test_compare_with_multiple_types(self):
        if False:
            while True:
                i = 10
        image1 = create_image(b'a', types=['front'], support_types=True, support_multi_types=True)
        image2 = create_image(b'a', types=['front', 'booklet'], support_types=True, support_multi_types=True)
        image3 = create_image(b'a', types=['front', 'booklet'], support_types=True, support_multi_types=True)
        image4 = create_image(b'b', types=['front', 'booklet'], support_types=True, support_multi_types=True)
        self.assertNotEqual(image1, image2)
        self.assertEqual(image2, image3)
        self.assertNotEqual(image2, image4)

    def test_set_data(self):
        if False:
            i = 10
            return i + 15
        imgdata = create_fake_png(b'a')
        imgdata2 = create_fake_png(b'xxx')
        coverartimage = CoverArtImage(data=imgdata2)
        tmp_file = coverartimage.tempfile_filename
        filesize = os.path.getsize(tmp_file)
        self.assertEqual(filesize, len(imgdata2))
        self.assertEqual(coverartimage.data, imgdata2)
        coverartimage.set_data(imgdata)
        tmp_file = coverartimage.tempfile_filename
        filesize = os.path.getsize(tmp_file)
        self.assertEqual(filesize, len(imgdata))
        self.assertEqual(coverartimage.data, imgdata)

    def test_save(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_config_values({'image_type_as_filename': True, 'windows_compatibility': True, 'win_compat_replacements': {}, 'windows_long_paths': False, 'replace_spaces_with_underscores': False, 'replace_dir_separator': '_', 'enabled_plugins': [], 'ascii_filenames': False, 'save_images_overwrite': False})
        metadata = Metadata()
        counters = Counter()
        with TemporaryDirectory() as d:
            image1 = create_image(b'a', types=['back'], support_types=True)
            expected_filename = os.path.join(d, 'back.png')
            counter_filename = encode_filename(os.path.join(d, 'back'))
            image1.save(d, metadata, counters)
            self.assertTrue(os.path.exists(expected_filename))
            self.assertEqual(len(image1.data), os.path.getsize(expected_filename))
            self.assertEqual(1, counters[counter_filename])
            image2 = create_image(b'bb', types=['back'], support_types=True)
            image2.save(d, metadata, counters)
            expected_filename_2 = os.path.join(d, 'back (1).png')
            self.assertTrue(os.path.exists(expected_filename_2))
            self.assertEqual(len(image2.data), os.path.getsize(expected_filename_2))
            self.assertEqual(2, counters[counter_filename])

class CoverArtImageMakeFilenameTest(PicardTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.image = create_image(b'a', types=['back'], support_types=True)
        self.metadata = Metadata()
        self.set_config_values({'windows_compatibility': False, 'win_compat_replacements': {}, 'enabled_plugins': [], 'ascii_filenames': False, 'replace_spaces_with_underscores': False, 'replace_dir_separator': '_'})

    def compare_paths(self, path1, path2):
        if False:
            print('Hello World!')
        self.assertEqual(encode_filename(os.path.normpath(path1)), encode_filename(os.path.normpath(path2)))

    def test_make_image_filename(self):
        if False:
            for i in range(10):
                print('nop')
        filename = self.image._make_image_filename('AlbumArt', '/music/albumart', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths('/music/albumart/AlbumArt', filename)

    def test_make_image_filename_default(self):
        if False:
            i = 10
            return i + 15
        filename = self.image._make_image_filename('$noop()', '/music/albumart', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths(os.path.join('/music/albumart/', DEFAULT_COVER_IMAGE_FILENAME), filename)

    def test_make_image_filename_relative_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.metadata['album'] = 'TheAlbum'
        filename = self.image._make_image_filename('../covers/%album%', '/music/album', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths('/music/covers/TheAlbum', filename)

    def test_make_image_filename_absolute_path(self):
        if False:
            i = 10
            return i + 15
        filename = self.image._make_image_filename('/foo/bar/AlbumArt', '/music/albumart', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths('/foo/bar/AlbumArt', filename)

    @unittest.skipUnless(IS_WIN, 'windows test')
    def test_make_image_filename_absolute_path_no_common_base(self):
        if False:
            print('Hello World!')
        filename = self.image._make_image_filename('D:/foo/AlbumArt', 'C:/music', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths('D:\\foo\\AlbumArt', filename)

    def test_make_image_filename_script(self):
        if False:
            i = 10
            return i + 15
        cover_script = '%album%-$if($eq(%coverart_maintype%,front),cover,%coverart_maintype%)'
        self.metadata['album'] = 'TheAlbum'
        filename = self.image._make_image_filename(cover_script, '/music/', self.metadata, win_compat=False, win_shorten_path=False)
        self.compare_paths('/music/TheAlbum-back', filename)

    def test_make_image_filename_save_path(self):
        if False:
            print('Hello World!')
        self.set_config_values({'windows_compatibility': True})
        filename = self.image._make_image_filename('.co:ver', '/music/albumart', self.metadata, win_compat=True, win_shorten_path=False)
        self.compare_paths('/music/albumart/_co_ver', filename)

    def test_make_image_filename_win_shorten_path(self):
        if False:
            print('Hello World!')
        requested_path = '/' + 300 * 'a' + '/cover'
        expected_path = '/' + 226 * 'a' + '/cover'
        filename = self.image._make_image_filename(requested_path, '/music/albumart', self.metadata, win_compat=False, win_shorten_path=True)
        self.compare_paths(expected_path, filename)

    def test_make_image_filename_win_shorten_path_too_long_base_path(self):
        if False:
            print('Hello World!')
        base_path = '/' + 244 * 'a'
        with self.assertRaises(WinPathTooLong):
            self.image._make_image_filename('cover', base_path, self.metadata, win_compat=False, win_shorten_path=True)

class LocalFileCoverArtImageTest(PicardTestCase):

    def test_set_file_url(self):
        if False:
            while True:
                i = 10
        path = '/some/path/image.jpeg'
        image = LocalFileCoverArtImage(path)
        self.assertEqual(image.url.toString(), 'file://' + path)

    def test_support_types(self):
        if False:
            while True:
                i = 10
        path = '/some/path/image.jpeg'
        image = LocalFileCoverArtImage(path)
        self.assertFalse(image.support_types)
        self.assertFalse(image.support_multi_types)
        image = LocalFileCoverArtImage(path, support_types=True)
        self.assertTrue(image.support_types)
        self.assertFalse(image.support_multi_types)
        image = LocalFileCoverArtImage(path, support_multi_types=True)
        self.assertFalse(image.support_types)
        self.assertTrue(image.support_multi_types)

    @unittest.skipUnless(IS_WIN, 'windows test')
    def test_windows_path(self):
        if False:
            while True:
                i = 10
        path = 'C:\\Music\\somefile.mp3'
        image = LocalFileCoverArtImage(path)
        self.assertEqual(image.url.toLocalFile(), 'C:/Music/somefile.mp3')