import base64
import os
from unittest.mock import patch
from mutagen.flac import Padding, Picture, SeekPoint, SeekTable, VCFLACDict
from test.picardtestcase import PicardTestCase, create_fake_png
from picard import config
from picard.coverart.image import CoverArtImage
from picard.formats import vorbis
from picard.formats.util import open_ as open_format
from picard.metadata import Metadata
from .common import TAGS, CommonTests, load_metadata, load_raw, save_and_load_metadata, save_metadata, save_raw, skipUnlessTestfile
from .coverart import CommonCoverArtTests, file_save_image, load_coverart_file
VALID_KEYS = [' valid Key}', '{ $ome tag}']
INVALID_KEYS = ['', 'invalid=key', 'invalid\x19key', 'invalid~key']
PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC'

class CommonVorbisTests:

    class VorbisTestCase(CommonTests.TagFormatsTestCase):

        def test_invalid_rating(self):
            if False:
                return 10
            filename = os.path.join('test', 'data', 'test-invalid-rating.ogg')
            metadata = load_metadata(filename)
            self.assertEqual(metadata['~rating'], 'THERATING')

        def test_supports_tags(self):
            if False:
                return 10
            supports_tag = self.format.supports_tag
            for key in VALID_KEYS + list(TAGS.keys()):
                self.assertTrue(supports_tag(key), '%r should be supported' % key)
            for key in INVALID_KEYS:
                self.assertFalse(supports_tag(key), '%r should be unsupported' % key)

        @skipUnlessTestfile
        def test_r128_replaygain_tags(self):
            if False:
                return 10
            tags = {'r128_album_gain': '-2857', 'r128_track_gain': '-2857'}
            self._test_unsupported_tags(tags)

        @skipUnlessTestfile
        def test_invalid_metadata_block_picture_nobase64(self):
            if False:
                print('Hello World!')
            metadata = {'metadata_block_picture': 'notbase64'}
            save_raw(self.filename, metadata)
            loaded_metadata = load_metadata(self.filename)
            self.assertEqual(0, len(loaded_metadata.images))

        @skipUnlessTestfile
        def test_invalid_metadata_block_picture_noflacpicture(self):
            if False:
                for i in range(10):
                    print('nop')
            metadata = {'metadata_block_picture': base64.b64encode(b'notaflacpictureblock').decode('ascii')}
            save_raw(self.filename, metadata)
            loaded_metadata = load_metadata(self.filename)
            self.assertEqual(0, len(loaded_metadata.images))

        @skipUnlessTestfile
        def test_legacy_coverart(self):
            if False:
                print('Hello World!')
            save_raw(self.filename, {'coverart': PNG_BASE64})
            loaded_metadata = load_metadata(self.filename)
            self.assertEqual(1, len(loaded_metadata.images))
            first_image = loaded_metadata.images[0]
            self.assertEqual('image/png', first_image.mimetype)
            self.assertEqual(69, first_image.datalength)

        @skipUnlessTestfile
        def test_clear_tags_preserve_legacy_coverart(self):
            if False:
                print('Hello World!')
            save_raw(self.filename, {'coverart': PNG_BASE64})
            config.setting['clear_existing_tags'] = True
            config.setting['preserve_images'] = True
            metadata = save_and_load_metadata(self.filename, Metadata())
            self.assertEqual(1, len(metadata.images))
            config.setting['preserve_images'] = False
            metadata = save_and_load_metadata(self.filename, Metadata())
            self.assertEqual(0, len(metadata.images))

        @skipUnlessTestfile
        def test_invalid_legacy_coverart_nobase64(self):
            if False:
                for i in range(10):
                    print('nop')
            metadata = {'coverart': 'notbase64'}
            save_raw(self.filename, metadata)
            loaded_metadata = load_metadata(self.filename)
            self.assertEqual(0, len(loaded_metadata.images))

        @skipUnlessTestfile
        def test_invalid_legacy_coverart_noimage(self):
            if False:
                print('Hello World!')
            metadata = {'coverart': base64.b64encode(b'invalidimagedata').decode('ascii')}
            save_raw(self.filename, metadata)
            loaded_metadata = load_metadata(self.filename)
            self.assertEqual(0, len(loaded_metadata.images))

        def test_supports_extended_tags(self):
            if False:
                print('Hello World!')
            performer_tag = 'performer:accordéon clavier « boutons »'
            self.assertTrue(self.format.supports_tag(performer_tag))
            self.assertTrue(self.format.supports_tag('lyrics:foó'))
            self.assertTrue(self.format.supports_tag('comment:foó'))

        @skipUnlessTestfile
        def test_delete_totaldiscs_totaltracks(self):
            if False:
                i = 10
                return i + 15
            save_raw(self.filename, {'disctotal': '3', 'tracktotal': '2'})
            metadata = Metadata()
            del metadata['totaldiscs']
            del metadata['totaltracks']
            save_metadata(self.filename, metadata)
            loaded_metadata = load_raw(self.filename)
            self.assertNotIn('disctotal', loaded_metadata)
            self.assertNotIn('totaldiscs', loaded_metadata)
            self.assertNotIn('tracktotal', loaded_metadata)
            self.assertNotIn('totaltracks', loaded_metadata)

        @skipUnlessTestfile
        def test_delete_invalid_tagname(self):
            if False:
                return 10
            for invalid_tag in INVALID_KEYS:
                metadata = Metadata()
                del metadata[invalid_tag]
                save_metadata(self.filename, metadata)

        @skipUnlessTestfile
        def test_load_strip_trailing_null_char(self):
            if False:
                return 10
            save_raw(self.filename, {'date': '2023-04-18\x00', 'title': 'foo\x00'})
            metadata = load_metadata(self.filename)
            self.assertEqual('2023-04-18', metadata['date'])
            self.assertEqual('foo', metadata['title'])

class FLACTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test.flac'
    supports_ratings = True
    expected_info = {'length': 82, '~channels': '2', '~sample_rate': '44100', '~format': 'FLAC'}
    unexpected_info = ['~video']

    @skipUnlessTestfile
    def test_preserve_waveformatextensible_channel_mask(self):
        if False:
            for i in range(10):
                print('nop')
        config.setting['clear_existing_tags'] = True
        original_metadata = load_metadata(self.filename)
        self.assertEqual(original_metadata['~waveformatextensible_channel_mask'], '0x3')
        new_metadata = save_and_load_metadata(self.filename, original_metadata)
        self.assertEqual(new_metadata['~waveformatextensible_channel_mask'], '0x3')

    @skipUnlessTestfile
    def test_clear_tags_preserve_legacy_coverart(self):
        if False:
            while True:
                i = 10
        pic = Picture()
        pic.data = load_coverart_file('mb.png')
        save_raw(self.filename, {'coverart': PNG_BASE64, 'metadata_block_picture': base64.b64encode(pic.write()).decode('ascii')})
        config.setting['clear_existing_tags'] = True
        config.setting['preserve_images'] = True
        metadata = save_and_load_metadata(self.filename, Metadata())
        self.assertEqual(0, len(metadata.images))

    @skipUnlessTestfile
    def test_sort_pics_after_tags(self):
        if False:
            for i in range(10):
                print('nop')
        pic = Picture()
        pic.data = load_coverart_file('mb.png')
        f = load_raw(self.filename)
        f.metadata_blocks.insert(1, pic)
        f.save()
        metadata = Metadata()
        save_metadata(self.filename, metadata)
        f = load_raw(self.filename)
        tagindex = f.metadata_blocks.index(f.tags)
        haspics = False
        for b in f.metadata_blocks:
            if b.code == Picture.code:
                haspics = True
                self.assertGreater(f.metadata_blocks.index(b), tagindex)
        self.assertTrue(haspics, 'Picture block expected, none found')

    @patch.object(vorbis, 'flac_remove_empty_seektable')
    def test_setting_fix_missing_seekpoints_flac(self, mock_flac_remove_empty_seektable):
        if False:
            print('Hello World!')
        save_metadata(self.filename, Metadata())
        mock_flac_remove_empty_seektable.assert_not_called()
        self.set_config_values({'fix_missing_seekpoints_flac': True})
        save_metadata(self.filename, Metadata())
        mock_flac_remove_empty_seektable.assert_called_once()

    @skipUnlessTestfile
    def test_flac_remove_empty_seektable_remove_empty(self):
        if False:
            return 10
        f = load_raw(self.filename)
        seektable = SeekTable(None)
        f.seektable = seektable
        f.metadata_blocks.append(seektable)
        vorbis.flac_remove_empty_seektable(f)
        self.assertIsNone(f.seektable)
        self.assertNotIn(seektable, f.metadata_blocks)

    @skipUnlessTestfile
    def test_flac_remove_empty_seektable_keep_existing(self):
        if False:
            while True:
                i = 10
        f = load_raw(self.filename)
        seektable = SeekTable(None)
        seekpoint = SeekPoint(0, 0, 0)
        seektable.seekpoints.append(seekpoint)
        f.seektable = seektable
        f.metadata_blocks.append(seektable)
        vorbis.flac_remove_empty_seektable(f)
        self.assertEqual(seektable, f.seektable)
        self.assertIn(seektable, f.metadata_blocks)
        self.assertEqual([seekpoint], f.seektable.seekpoints)

class OggVorbisTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test.ogg'
    supports_ratings = True
    expected_info = {'length': 82, '~channels': '2', '~sample_rate': '44100'}

class OggSpxTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test.spx'
    supports_ratings = True
    expected_info = {'length': 89, '~channels': '2', '~bitrate': '29.6'}
    unexpected_info = ['~video']

class OggOpusTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test.opus'
    supports_ratings = True
    expected_info = {'length': 82, '~channels': '2'}
    unexpected_info = ['~video']

    @skipUnlessTestfile
    def test_r128_replaygain_tags(self):
        if False:
            print('Hello World!')
        tags = {'r128_album_gain': '-2857', 'r128_track_gain': '-2857'}
        self._test_supported_tags(tags)

class OggTheoraTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test.ogv'
    supports_ratings = True
    expected_info = {'length': 520, '~bitrate': '200.0', '~video': '1'}

class OggFlacTest(CommonVorbisTests.VorbisTestCase):
    testfile = 'test-oggflac.oga'
    supports_ratings = True
    expected_info = {'length': 82, '~channels': '2'}
    unexpected_info = ['~video']

class VorbisUtilTest(PicardTestCase):

    def test_sanitize_key(self):
        if False:
            for i in range(10):
                print('nop')
        sanitized = vorbis.sanitize_key(' \x1f=}~')
        self.assertEqual(sanitized, ' }')

    def test_is_valid_key(self):
        if False:
            print('Hello World!')
        for key in VALID_KEYS:
            self.assertTrue(vorbis.is_valid_key(key), '%r is valid' % key)
        for key in INVALID_KEYS:
            self.assertFalse(vorbis.is_valid_key(key), '%r is invalid' % key)

    def test_flac_sort_pics_after_tags(self):
        if False:
            return 10
        pic1 = Picture()
        pic2 = Picture()
        pic3 = Picture()
        tags = VCFLACDict()
        pad = Padding()
        blocks = []
        vorbis.flac_sort_pics_after_tags(blocks)
        self.assertEqual([], blocks)
        blocks = [tags]
        vorbis.flac_sort_pics_after_tags(blocks)
        self.assertEqual([tags], blocks)
        blocks = [tags, pad, pic1]
        vorbis.flac_sort_pics_after_tags(blocks)
        self.assertEqual([tags, pad, pic1], blocks)
        blocks = [pic1, pic2, tags, pad, pic3]
        vorbis.flac_sort_pics_after_tags(blocks)
        self.assertEqual([tags, pic1, pic2, pad, pic3], blocks)
        blocks = [pic1, pic2, pad, pic3]
        vorbis.flac_sort_pics_after_tags(blocks)
        self.assertEqual([pic1, pic2, pad, pic3], blocks)

class FlacCoverArtTest(CommonCoverArtTests.CoverArtTestCase):
    testfile = 'test.flac'

    def test_set_picture_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [CoverArtImage(data=self.jpegdata), CoverArtImage(data=self.pngdata)]
        for test in tests:
            file_save_image(self.filename, test)
            raw_metadata = load_raw(self.filename)
            pic = raw_metadata.pictures[0]
            self.assertNotEqual(pic.width, 0)
            self.assertEqual(pic.width, test.width)
            self.assertNotEqual(pic.height, 0)
            self.assertEqual(pic.height, test.height)

    def test_save_large_pics(self):
        if False:
            return 10
        data = create_fake_png(b'a' * 1024 * 1024 * 16)
        image = CoverArtImage(data=data)
        file_save_image(self.filename, image)
        raw_metadata = load_raw(self.filename)
        self.assertEqual(0, len(raw_metadata.pictures))

class OggAudioVideoFileTest(PicardTestCase):

    def test_ogg_audio(self):
        if False:
            while True:
                i = 10
        self._test_file_is_type(open_format, self._copy_file_tmp('test-oggflac.oga', '.oga'), vorbis.OggFLACFile)
        self._test_file_is_type(open_format, self._copy_file_tmp('test.spx', '.oga'), vorbis.OggSpeexFile)
        self._test_file_is_type(open_format, self._copy_file_tmp('test.ogg', '.oga'), vorbis.OggVorbisFile)

    def test_ogg_opus(self):
        if False:
            while True:
                i = 10
        self._test_file_is_type(open_format, self._copy_file_tmp('test.opus', '.oga'), vorbis.OggOpusFile)
        self._test_file_is_type(open_format, self._copy_file_tmp('test.opus', '.ogg'), vorbis.OggOpusFile)

    def test_ogg_video(self):
        if False:
            i = 10
            return i + 15
        self._test_file_is_type(open_format, self._copy_file_tmp('test.ogv', '.ogv'), vorbis.OggTheoraFile)

    def _test_file_is_type(self, factory, filename, expected_type):
        if False:
            return 10
        f = factory(filename)
        self.assertIsInstance(f, expected_type)

    def _copy_file_tmp(self, filename, ext):
        if False:
            while True:
                i = 10
        path = os.path.join('test', 'data', filename)
        return self.copy_file_tmp(path, ext)

class OggCoverArtTest(CommonCoverArtTests.CoverArtTestCase):
    testfile = 'test.ogg'