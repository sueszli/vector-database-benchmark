import os
import re
import warnings
from io import BytesIO
import pytest
from PIL import ExifTags, Image, ImageFile, ImageOps, JpegImagePlugin, UnidentifiedImageError, features
from .helper import assert_image, assert_image_equal, assert_image_equal_tofile, assert_image_similar, assert_image_similar_tofile, cjpeg_available, djpeg_available, hopper, is_win32, mark_if_feature_version, skip_unless_feature
try:
    from defusedxml import ElementTree
except ImportError:
    ElementTree = None
TEST_FILE = 'Tests/images/hopper.jpg'

@skip_unless_feature('jpg')
class TestFileJpeg:

    def roundtrip(self, im, **options):
        if False:
            print('Hello World!')
        out = BytesIO()
        im.save(out, 'JPEG', **options)
        test_bytes = out.tell()
        out.seek(0)
        im = Image.open(out)
        im.bytes = test_bytes
        return im

    def gen_random_image(self, size, mode='RGB'):
        if False:
            while True:
                i = 10
        'Generates a very hard to compress file\n        :param size: tuple\n        :param mode: optional image mode\n\n        '
        return Image.frombytes(mode, size, os.urandom(size[0] * size[1] * len(mode)))

    def test_sanity(self):
        if False:
            i = 10
            return i + 15
        assert re.search('\\d+\\.\\d+$', features.version_codec('jpg'))
        with Image.open(TEST_FILE) as im:
            im.load()
            assert im.mode == 'RGB'
            assert im.size == (128, 128)
            assert im.format == 'JPEG'
            assert im.get_format_mimetype() == 'image/jpeg'

    @pytest.mark.parametrize('size', ((1, 0), (0, 1), (0, 0)))
    def test_zero(self, size, tmp_path):
        if False:
            while True:
                i = 10
        f = str(tmp_path / 'temp.jpg')
        im = Image.new('RGB', size)
        with pytest.raises(ValueError):
            im.save(f)

    def test_app(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open(TEST_FILE) as im:
            assert im.applist[0] == ('APP0', b'JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00')
            assert im.applist[1] == ('COM', b'File written by Adobe Photoshop\xa8 4.0\x00')
            assert len(im.applist) == 2
            assert im.info['comment'] == b'File written by Adobe Photoshop\xa8 4.0\x00'
            assert im.app['COM'] == im.info['comment']

    def test_comment_write(self):
        if False:
            return 10
        with Image.open(TEST_FILE) as im:
            assert im.info['comment'] == b'File written by Adobe Photoshop\xa8 4.0\x00'
            out = BytesIO()
            im.save(out, format='JPEG')
            with Image.open(out) as reloaded:
                assert im.info['comment'] == reloaded.info['comment']
            for comment in ('', b'', None):
                out = BytesIO()
                im.save(out, format='JPEG', comment=comment)
                with Image.open(out) as reloaded:
                    assert 'comment' not in reloaded.info
            for comment in ('Test comment text', b'Text comment text'):
                out = BytesIO()
                im.save(out, format='JPEG', comment=comment)
                with Image.open(out) as reloaded:
                    if not isinstance(comment, bytes):
                        comment = comment.encode()
                    assert reloaded.info['comment'] == comment

    def test_cmyk(self):
        if False:
            print('Hello World!')
        f = 'Tests/images/pil_sample_cmyk.jpg'
        with Image.open(f) as im:
            (c, m, y, k) = (x / 255.0 for x in im.getpixel((0, 0)))
            assert c == 0.0
            assert m > 0.8
            assert y > 0.8
            assert k == 0.0
            (c, m, y, k) = (x / 255.0 for x in im.getpixel((im.size[0] - 1, im.size[1] - 1)))
            assert k > 0.9
            im = self.roundtrip(im)
            (c, m, y, k) = (x / 255.0 for x in im.getpixel((0, 0)))
            assert c == 0.0
            assert m > 0.8
            assert y > 0.8
            assert k == 0.0
            (c, m, y, k) = (x / 255.0 for x in im.getpixel((im.size[0] - 1, im.size[1] - 1)))
            assert k > 0.9

    @pytest.mark.parametrize('test_image_path', [TEST_FILE, 'Tests/images/pil_sample_cmyk.jpg'])
    def test_dpi(self, test_image_path):
        if False:
            i = 10
            return i + 15

        def test(xdpi, ydpi=None):
            if False:
                while True:
                    i = 10
            with Image.open(test_image_path) as im:
                im = self.roundtrip(im, dpi=(xdpi, ydpi or xdpi))
            return im.info.get('dpi')
        assert test(72) == (72, 72)
        assert test(300) == (300, 300)
        assert test(100, 200) == (100, 200)
        assert test(0) is None

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_icc(self, tmp_path):
        if False:
            i = 10
            return i + 15
        with Image.open('Tests/images/rgb.jpg') as im1:
            icc_profile = im1.info['icc_profile']
            assert len(icc_profile) == 3144
            f = str(tmp_path / 'temp.jpg')
            im1.save(f, icc_profile=icc_profile)
        with Image.open(f) as im2:
            assert im2.info.get('icc_profile') == icc_profile
            im1 = self.roundtrip(hopper())
            im2 = self.roundtrip(hopper(), icc_profile=icc_profile)
            assert_image_equal(im1, im2)
            assert not im1.info.get('icc_profile')
            assert im2.info.get('icc_profile')

    @pytest.mark.parametrize('n', (0, 1, 3, 4, 5, 65533 - 14, 65533 - 14 + 1, ImageFile.MAXBLOCK, ImageFile.MAXBLOCK + 1, ImageFile.MAXBLOCK * 4 + 3))
    def test_icc_big(self, n):
        if False:
            return 10
        icc_profile = (b'Test' * int(n / 4 + 1))[:n]
        assert len(icc_profile) == n
        im1 = self.roundtrip(hopper(), icc_profile=icc_profile)
        assert im1.info.get('icc_profile') == (icc_profile or None)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_large_icc_meta(self, tmp_path):
        if False:
            return 10
        with Image.open('Tests/images/icc_profile_big.jpg') as im:
            f = str(tmp_path / 'temp.jpg')
            icc_profile = im.info['icc_profile']
            im.save(f, progressive=True, quality=95, icc_profile=icc_profile, optimize=True)
        with Image.open('Tests/images/flower2.jpg') as im:
            f = str(tmp_path / 'temp2.jpg')
            im.save(f, progressive=True, quality=94, icc_profile=b' ' * 53955)
        with Image.open('Tests/images/flower2.jpg') as im:
            f = str(tmp_path / 'temp3.jpg')
            im.save(f, progressive=True, quality=94, exif=b' ' * 43668)

    def test_optimize(self):
        if False:
            return 10
        im1 = self.roundtrip(hopper())
        im2 = self.roundtrip(hopper(), optimize=0)
        im3 = self.roundtrip(hopper(), optimize=1)
        assert_image_equal(im1, im2)
        assert_image_equal(im1, im3)
        assert im1.bytes >= im2.bytes
        assert im1.bytes >= im3.bytes

    def test_optimize_large_buffer(self, tmp_path):
        if False:
            while True:
                i = 10
        f = str(tmp_path / 'temp.jpg')
        im = Image.new('RGB', (4096, 4096), 16724787)
        im.save(f, format='JPEG', optimize=True)

    def test_progressive(self):
        if False:
            return 10
        im1 = self.roundtrip(hopper())
        im2 = self.roundtrip(hopper(), progressive=False)
        im3 = self.roundtrip(hopper(), progressive=True)
        assert not im1.info.get('progressive')
        assert not im2.info.get('progressive')
        assert im3.info.get('progressive')
        assert_image_equal(im1, im3)
        assert im1.bytes >= im3.bytes

    def test_progressive_large_buffer(self, tmp_path):
        if False:
            return 10
        f = str(tmp_path / 'temp.jpg')
        im = Image.new('RGB', (4096, 4096), 16724787)
        im.save(f, format='JPEG', progressive=True)

    def test_progressive_large_buffer_highest_quality(self, tmp_path):
        if False:
            while True:
                i = 10
        f = str(tmp_path / 'temp.jpg')
        im = self.gen_random_image((255, 255))
        im.save(f, format='JPEG', progressive=True, quality=100)

    def test_progressive_cmyk_buffer(self):
        if False:
            while True:
                i = 10
        f = BytesIO()
        im = self.gen_random_image((256, 256), 'CMYK')
        im.save(f, format='JPEG', progressive=True, quality=94)

    def test_large_exif(self, tmp_path):
        if False:
            return 10
        f = str(tmp_path / 'temp.jpg')
        im = hopper()
        im.save(f, 'JPEG', quality=90, exif=b'1' * 65533)
        with pytest.raises(ValueError):
            im.save(f, 'JPEG', quality=90, exif=b'1' * 65534)

    def test_exif_typeerror(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/exif_typeerror.jpg') as im:
            im._getexif()

    def test_exif_gps(self, tmp_path):
        if False:
            while True:
                i = 10
        expected_exif_gps = {0: b'\x00\x00\x00\x01', 2: 4294967295, 5: b'\x01', 30: 65535, 29: '1999:99:99 99:99:99'}
        gps_index = 34853
        with Image.open('Tests/images/exif_gps.jpg') as im:
            exif = im._getexif()
            assert exif[gps_index] == expected_exif_gps
        f = str(tmp_path / 'temp.jpg')
        exif = Image.Exif()
        exif[gps_index] = expected_exif_gps
        hopper().save(f, exif=exif)
        with Image.open(f) as reloaded:
            exif = reloaded._getexif()
            assert exif[gps_index] == expected_exif_gps

    def test_empty_exif_gps(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/empty_gps_ifd.jpg') as im:
            exif = im.getexif()
            del exif[34665]
            assert exif[274] == Image.Transpose.TRANSVERSE
            assert exif.get_ifd(34853) == {}
            transposed = ImageOps.exif_transpose(im)
        exif = transposed.getexif()
        assert exif.get_ifd(34853) == {}
        assert 274 not in exif

    def test_exif_equality(self):
        if False:
            i = 10
            return i + 15
        exifs = []
        for i in range(2):
            with Image.open('Tests/images/exif-200dpcm.jpg') as im:
                exifs.append(im._getexif())
        assert exifs[0] == exifs[1]

    def test_exif_rollback(self):
        if False:
            for i in range(10):
                print('nop')
        expected_exif = {34867: 4294967295, 258: (24, 24, 24), 36867: '2099:09:29 10:10:10', 34853: {0: b'\x00\x00\x00\x01', 2: 4294967295, 5: b'\x01', 30: 65535, 29: '1999:99:99 99:99:99'}, 296: 65535, 34665: 185, 41994: 65535, 514: 4294967295, 271: 'Make', 272: 'XXX-XXX', 305: 'PIL', 42034: (1, 1, 1, 1), 42035: 'LensMake', 34856: b'\xaa\xaa\xaa\xaa\xaa\xaa', 282: 4294967295, 33434: 4294967295}
        with Image.open('Tests/images/exif_gps.jpg') as im:
            exif = im._getexif()
        for (tag, value) in expected_exif.items():
            assert value == exif[tag]

    def test_exif_gps_typeerror(self):
        if False:
            return 10
        with Image.open('Tests/images/exif_gps_typeerror.jpg') as im:
            im._getexif()

    def test_progressive_compat(self):
        if False:
            return 10
        im1 = self.roundtrip(hopper())
        assert not im1.info.get('progressive')
        assert not im1.info.get('progression')
        im2 = self.roundtrip(hopper(), progressive=0)
        im3 = self.roundtrip(hopper(), progression=0)
        assert not im2.info.get('progressive')
        assert not im2.info.get('progression')
        assert not im3.info.get('progressive')
        assert not im3.info.get('progression')
        im2 = self.roundtrip(hopper(), progressive=1)
        im3 = self.roundtrip(hopper(), progression=1)
        assert_image_equal(im1, im2)
        assert_image_equal(im1, im3)
        assert im2.info.get('progressive')
        assert im2.info.get('progression')
        assert im3.info.get('progressive')
        assert im3.info.get('progression')

    def test_quality(self):
        if False:
            while True:
                i = 10
        im1 = self.roundtrip(hopper())
        im2 = self.roundtrip(hopper(), quality=50)
        assert_image(im1, im2.mode, im2.size)
        assert im1.bytes >= im2.bytes
        im3 = self.roundtrip(hopper(), quality=0)
        assert_image(im1, im3.mode, im3.size)
        assert im2.bytes > im3.bytes

    def test_smooth(self):
        if False:
            print('Hello World!')
        im1 = self.roundtrip(hopper())
        im2 = self.roundtrip(hopper(), smooth=100)
        assert_image(im1, im2.mode, im2.size)

    def test_subsampling(self):
        if False:
            print('Hello World!')

        def getsampling(im):
            if False:
                print('Hello World!')
            layer = im.layer
            return layer[0][1:3] + layer[1][1:3] + layer[2][1:3]
        im = self.roundtrip(hopper(), subsampling=-1)
        assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling=0)
        assert getsampling(im) == (1, 1, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling=1)
        assert getsampling(im) == (2, 1, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling=2)
        assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling=3)
        assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling='4:4:4')
        assert getsampling(im) == (1, 1, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling='4:2:2')
        assert getsampling(im) == (2, 1, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling='4:2:0')
        assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        im = self.roundtrip(hopper(), subsampling='4:1:1')
        assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        with pytest.raises(TypeError):
            self.roundtrip(hopper(), subsampling='1:1:1')

    def test_exif(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/pil_sample_rgb.jpg') as im:
            info = im._getexif()
            assert info[305] == 'Adobe Photoshop CS Macintosh'

    def test_get_child_images(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/flower.jpg') as im:
            ims = im.get_child_images()
        assert len(ims) == 1
        assert_image_similar_tofile(ims[0], 'Tests/images/flower_thumbnail.png', 2.1)

    def test_mp(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/pil_sample_rgb.jpg') as im:
            assert im._getmp() is None

    def test_quality_keep(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/hopper.jpg') as im:
            f = str(tmp_path / 'temp.jpg')
            im.save(f, quality='keep')
        with Image.open('Tests/images/hopper_gray.jpg') as im:
            f = str(tmp_path / 'temp.jpg')
            im.save(f, quality='keep')
        with Image.open('Tests/images/pil_sample_cmyk.jpg') as im:
            f = str(tmp_path / 'temp.jpg')
            im.save(f, quality='keep')

    def test_junk_jpeg_header(self):
        if False:
            return 10
        filename = 'Tests/images/junk_jpeg_header.jpg'
        with Image.open(filename):
            pass

    def test_ff00_jpeg_header(self):
        if False:
            return 10
        filename = 'Tests/images/jpeg_ff00_header.jpg'
        with Image.open(filename):
            pass

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_truncated_jpeg_should_read_all_the_data(self):
        if False:
            for i in range(10):
                print('nop')
        filename = 'Tests/images/truncated_jpeg.jpg'
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(filename) as im:
            im.load()
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            assert im.getbbox() is not None

    def test_truncated_jpeg_throws_oserror(self):
        if False:
            print('Hello World!')
        filename = 'Tests/images/truncated_jpeg.jpg'
        with Image.open(filename) as im:
            with pytest.raises(OSError):
                im.load()
            with pytest.raises(OSError):
                im.load()

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_qtables(self, tmp_path):
        if False:
            i = 10
            return i + 15

        def _n_qtables_helper(n, test_file):
            if False:
                return 10
            with Image.open(test_file) as im:
                f = str(tmp_path / 'temp.jpg')
                im.save(f, qtables=[[n] * 64] * n)
            with Image.open(f) as im:
                assert len(im.quantization) == n
                reloaded = self.roundtrip(im, qtables='keep')
                assert im.quantization == reloaded.quantization
                assert max(reloaded.quantization[0]) <= 255
        with Image.open('Tests/images/hopper.jpg') as im:
            qtables = im.quantization
            reloaded = self.roundtrip(im, qtables=qtables, subsampling=0)
            assert im.quantization == reloaded.quantization
            assert_image_similar(im, self.roundtrip(im, qtables='web_low'), 30)
            assert_image_similar(im, self.roundtrip(im, qtables='web_high'), 30)
            assert_image_similar(im, self.roundtrip(im, qtables='keep'), 30)
            bounds_qtable = [int(s) for s in ('255 1 ' * 32).split(None)]
            im2 = self.roundtrip(im, qtables=[bounds_qtable])
            assert im2.quantization == {0: bounds_qtable}
            standard_l_qtable = [int(s) for s in '\n                16  11  10  16  24  40  51  61\n                12  12  14  19  26  58  60  55\n                14  13  16  24  40  57  69  56\n                14  17  22  29  51  87  80  62\n                18  22  37  56  68 109 103  77\n                24  35  55  64  81 104 113  92\n                49  64  78  87 103 121 120 101\n                72  92  95  98 112 100 103  99\n                '.split(None)]
            standard_chrominance_qtable = [int(s) for s in '\n                17  18  24  47  99  99  99  99\n                18  21  26  66  99  99  99  99\n                24  26  56  99  99  99  99  99\n                47  66  99  99  99  99  99  99\n                99  99  99  99  99  99  99  99\n                99  99  99  99  99  99  99  99\n                99  99  99  99  99  99  99  99\n                99  99  99  99  99  99  99  99\n                '.split(None)]
            assert_image_similar(im, self.roundtrip(im, qtables=[standard_l_qtable, standard_chrominance_qtable]), 30)
            assert_image_similar(im, self.roundtrip(im, qtables=(standard_l_qtable, standard_chrominance_qtable)), 30)
            assert_image_similar(im, self.roundtrip(im, qtables={0: standard_l_qtable, 1: standard_chrominance_qtable}), 30)
            _n_qtables_helper(1, 'Tests/images/hopper_gray.jpg')
            _n_qtables_helper(1, 'Tests/images/pil_sample_rgb.jpg')
            _n_qtables_helper(2, 'Tests/images/pil_sample_rgb.jpg')
            _n_qtables_helper(3, 'Tests/images/pil_sample_rgb.jpg')
            _n_qtables_helper(1, 'Tests/images/pil_sample_cmyk.jpg')
            _n_qtables_helper(2, 'Tests/images/pil_sample_cmyk.jpg')
            _n_qtables_helper(3, 'Tests/images/pil_sample_cmyk.jpg')
            _n_qtables_helper(4, 'Tests/images/pil_sample_cmyk.jpg')
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables='a')
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[])
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[1, 2, 3, 4, 5])
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[1])
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[[1, 2, 3, 4]])

    def test_load_16bit_qtables(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/hopper_16bit_qtables.jpg') as im:
            assert len(im.quantization) == 2
            assert len(im.quantization[0]) == 64
            assert max(im.quantization[0]) > 255

    def test_save_multiple_16bit_qtables(self):
        if False:
            return 10
        with Image.open('Tests/images/hopper_16bit_qtables.jpg') as im:
            im2 = self.roundtrip(im, qtables='keep')
            assert im.quantization == im2.quantization

    def test_save_single_16bit_qtable(self):
        if False:
            return 10
        with Image.open('Tests/images/hopper_16bit_qtables.jpg') as im:
            im2 = self.roundtrip(im, qtables={0: im.quantization[0]})
            assert len(im2.quantization) == 1
            assert im2.quantization[0] == im.quantization[0]

    def test_save_low_quality_baseline_qtables(self):
        if False:
            while True:
                i = 10
        with Image.open(TEST_FILE) as im:
            im2 = self.roundtrip(im, quality=10)
            assert len(im2.quantization) == 2
            assert max(im2.quantization[0]) <= 255
            assert max(im2.quantization[1]) <= 255

    @pytest.mark.parametrize('blocks, rows, markers', ((0, 0, 0), (1, 0, 15), (3, 0, 5), (8, 0, 1), (0, 1, 3), (0, 2, 1)))
    def test_restart_markers(self, blocks, rows, markers):
        if False:
            while True:
                i = 10
        im = Image.new('RGB', (32, 32))
        out = BytesIO()
        im.save(out, format='JPEG', restart_marker_blocks=blocks, restart_marker_rows=rows, subsampling=0)
        assert len(re.findall(b'\xff[\xd0-\xd7]', out.getvalue())) == markers

    @pytest.mark.skipif(not djpeg_available(), reason='djpeg not available')
    def test_load_djpeg(self):
        if False:
            return 10
        with Image.open(TEST_FILE) as img:
            img.load_djpeg()
            assert_image_similar_tofile(img, TEST_FILE, 5)

    @pytest.mark.skipif(not cjpeg_available(), reason='cjpeg not available')
    def test_save_cjpeg(self, tmp_path):
        if False:
            print('Hello World!')
        with Image.open(TEST_FILE) as img:
            tempfile = str(tmp_path / 'temp.jpg')
            JpegImagePlugin._save_cjpeg(img, 0, tempfile)
            assert_image_similar_tofile(img, tempfile, 17)

    def test_no_duplicate_0x1001_tag(self):
        if False:
            while True:
                i = 10
        tag_ids = {v: k for (k, v) in ExifTags.TAGS.items()}
        assert tag_ids['RelatedImageWidth'] == 4097
        assert tag_ids['RelatedImageLength'] == 4098

    def test_MAXBLOCK_scaling(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        im = self.gen_random_image((512, 512))
        f = str(tmp_path / 'temp.jpeg')
        im.save(f, quality=100, optimize=True)
        with Image.open(f) as reloaded:
            reloaded.save(f, quality='keep')
            reloaded.save(f, quality='keep', progressive=True)
            reloaded.save(f, quality='keep', optimize=True)

    def test_bad_mpo_header(self):
        if False:
            while True:
                i = 10
        'Treat unknown MPO as JPEG'
        fn = 'Tests/images/sugarshack_bad_mpo_header.jpg'
        with pytest.warns(UserWarning, Image.open, fn) as im:
            assert im.format == 'JPEG'

    @pytest.mark.parametrize('mode', ('1', 'L', 'RGB', 'RGBX', 'CMYK', 'YCbCr'))
    def test_save_correct_modes(self, mode):
        if False:
            i = 10
            return i + 15
        out = BytesIO()
        img = Image.new(mode, (20, 20))
        img.save(out, 'JPEG')

    @pytest.mark.parametrize('mode', ('LA', 'La', 'RGBA', 'RGBa', 'P'))
    def test_save_wrong_modes(self, mode):
        if False:
            while True:
                i = 10
        out = BytesIO()
        img = Image.new(mode, (20, 20))
        with pytest.raises(OSError):
            img.save(out, 'JPEG')

    def test_save_tiff_with_dpi(self, tmp_path):
        if False:
            i = 10
            return i + 15
        outfile = str(tmp_path / 'temp.tif')
        with Image.open('Tests/images/hopper.tif') as im:
            im.save(outfile, 'JPEG', dpi=im.info['dpi'])
            with Image.open(outfile) as reloaded:
                reloaded.load()
                assert im.info['dpi'] == reloaded.info['dpi']

    def test_save_dpi_rounding(self, tmp_path):
        if False:
            while True:
                i = 10
        outfile = str(tmp_path / 'temp.jpg')
        with Image.open('Tests/images/hopper.jpg') as im:
            im.save(outfile, dpi=(72.2, 72.2))
            with Image.open(outfile) as reloaded:
                assert reloaded.info['dpi'] == (72, 72)
            im.save(outfile, dpi=(72.8, 72.8))
        with Image.open(outfile) as reloaded:
            assert reloaded.info['dpi'] == (73, 73)

    def test_dpi_tuple_from_exif(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/photoshop-200dpi.jpg') as im:
            assert im.info.get('dpi') == (200, 200)

    def test_dpi_int_from_exif(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/exif-72dpi-int.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_dpi_from_dpcm_exif(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/exif-200dpcm.jpg') as im:
            assert im.info.get('dpi') == (508, 508)

    def test_dpi_exif_zero_division(self):
        if False:
            i = 10
            return i + 15
        with Image.open('Tests/images/exif-dpi-zerodivision.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_dpi_exif_string(self):
        if False:
            return 10
        with Image.open('Tests/images/broken_exif_dpi.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_dpi_exif_truncated(self):
        if False:
            return 10
        with Image.open('Tests/images/truncated_exif_dpi.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_no_dpi_in_exif(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/no-dpi-in-exif.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_invalid_exif(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/invalid-exif.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_exif_x_resolution(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/flower.jpg') as im:
            exif = im.getexif()
            assert exif[282] == 180
            out = str(tmp_path / 'out.jpg')
            with warnings.catch_warnings():
                im.save(out, exif=exif)
        with Image.open(out) as reloaded:
            assert reloaded.getexif()[282] == 180

    def test_invalid_exif_x_resolution(self):
        if False:
            return 10
        with Image.open('Tests/images/invalid-exif-without-x-resolution.jpg') as im:
            assert im.info.get('dpi') == (72, 72)

    def test_ifd_offset_exif(self):
        if False:
            return 10
        with Image.open('Tests/images/exif-ifd-offset.jpg') as im:
            assert im._getexif()[306] == '2017:03:13 23:03:09'

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_photoshop(self):
        if False:
            return 10
        with Image.open('Tests/images/photoshop-200dpi.jpg') as im:
            assert im.info['photoshop'][1005] == {'XResolution': 200.0, 'DisplayedUnitsX': 1, 'YResolution': 200.0, 'DisplayedUnitsY': 1}
            assert_image_equal_tofile(im, 'Tests/images/photoshop-200dpi-broken.jpg')
        with Image.open('Tests/images/app13.jpg') as im:
            assert 'photoshop' not in im.info

    def test_photoshop_malformed_and_multiple(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/app13-multiple.jpg') as im:
            assert 'photoshop' in im.info
            assert 24 == len(im.info['photoshop'])
            apps_13_lengths = [len(v) for (k, v) in im.applist if k == 'APP13']
            assert [65504, 24] == apps_13_lengths

    def test_adobe_transform(self):
        if False:
            i = 10
            return i + 15
        with Image.open('Tests/images/pil_sample_rgb.jpg') as im:
            assert im.info['adobe_transform'] == 1
        with Image.open('Tests/images/pil_sample_cmyk.jpg') as im:
            assert im.info['adobe_transform'] == 2
        with Image.open('Tests/images/truncated_app14.jpg') as im:
            assert 'adobe' in im.info
            assert 'adobe_transform' not in im.info

    def test_icc_after_SOF(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/icc-after-SOF.jpg') as im:
            assert im.info['icc_profile'] == b'profile'

    def test_jpeg_magic_number(self):
        if False:
            return 10
        size = 4097
        buffer = BytesIO(b'\xff' * size)
        buffer.max_pos = 0
        orig_read = buffer.read

        def read(n=-1):
            if False:
                print('Hello World!')
            res = orig_read(n)
            buffer.max_pos = max(buffer.max_pos, buffer.tell())
            return res
        buffer.read = read
        with pytest.raises(UnidentifiedImageError):
            with Image.open(buffer):
                pass
        assert 0 < buffer.max_pos < size

    def test_getxmp(self):
        if False:
            i = 10
            return i + 15
        with Image.open('Tests/images/xmp_test.jpg') as im:
            if ElementTree is None:
                with pytest.warns(UserWarning, match='XMP data cannot be read without defusedxml dependency'):
                    assert im.getxmp() == {}
            else:
                xmp = im.getxmp()
                description = xmp['xmpmeta']['RDF']['Description']
                assert description['DerivedFrom'] == {'documentID': '8367D410E636EA95B7DE7EBA1C43A412', 'originalDocumentID': '8367D410E636EA95B7DE7EBA1C43A412'}
                assert description['Look']['Description']['Group']['Alt']['li'] == {'lang': 'x-default', 'text': 'Profiles'}
                assert description['ToneCurve']['Seq']['li'] == ['0, 0', '255, 255']
                assert description['Version'] == '10.4'
        if ElementTree is not None:
            with Image.open('Tests/images/hopper.jpg') as im:
                assert im.getxmp() == {}

    def test_getxmp_no_prefix(self):
        if False:
            return 10
        with Image.open('Tests/images/xmp_no_prefix.jpg') as im:
            if ElementTree is None:
                with pytest.warns(UserWarning, match='XMP data cannot be read without defusedxml dependency'):
                    assert im.getxmp() == {}
            else:
                assert im.getxmp() == {'xmpmeta': {'key': 'value'}}

    def test_getxmp_padded(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/xmp_padded.jpg') as im:
            if ElementTree is None:
                with pytest.warns(UserWarning, match='XMP data cannot be read without defusedxml dependency'):
                    assert im.getxmp() == {}
            else:
                assert im.getxmp() == {'xmpmeta': None}

    @pytest.mark.timeout(timeout=1)
    def test_eof(self):
        if False:
            return 10

        class InfiniteMockPyDecoder(ImageFile.PyDecoder):

            def decode(self, buffer):
                if False:
                    while True:
                        i = 10
                return (0, 0)
        decoder = InfiniteMockPyDecoder(None)

        def closure(mode, *args):
            if False:
                return 10
            decoder.__init__(mode, *args)
            return decoder
        Image.register_decoder('INFINITE', closure)
        with Image.open(TEST_FILE) as im:
            im.tile = [('INFINITE', (0, 0, 128, 128), 0, ('RGB', 0, 1))]
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im.load()
            ImageFile.LOAD_TRUNCATED_IMAGES = False

    def test_separate_tables(self):
        if False:
            while True:
                i = 10
        im = hopper()
        data = []
        for streamtype in range(3):
            out = BytesIO()
            im.save(out, format='JPEG', streamtype=streamtype)
            data.append(out.getvalue())
        for marker in (b'\xff\xd8', b'\xff\xd9'):
            assert marker in data[1] and marker in data[2]
        for marker in (b'\xff\xc4', b'\xff\xdb'):
            assert marker in data[1] and marker not in data[2]
        for marker in (b'\xff\xc0', b'\xff\xda', b'\xff\xe0'):
            assert marker not in data[1] and marker in data[2]
        with Image.open(BytesIO(data[0])) as interchange_im:
            with Image.open(BytesIO(data[1] + data[2])) as combined_im:
                assert_image_equal(interchange_im, combined_im)

    def test_repr_jpeg(self):
        if False:
            return 10
        im = hopper()
        with Image.open(BytesIO(im._repr_jpeg_())) as repr_jpeg:
            assert repr_jpeg.format == 'JPEG'
            assert_image_similar(im, repr_jpeg, 17)

    def test_repr_jpeg_error_returns_none(self):
        if False:
            i = 10
            return i + 15
        im = hopper('F')
        assert im._repr_jpeg_() is None

@pytest.mark.skipif(not is_win32(), reason='Windows only')
@skip_unless_feature('jpg')
class TestFileCloseW32:

    def test_fd_leak(self, tmp_path):
        if False:
            i = 10
            return i + 15
        tmpfile = str(tmp_path / 'temp.jpg')
        with Image.open('Tests/images/hopper.jpg') as im:
            im.save(tmpfile)
        im = Image.open(tmpfile)
        fp = im.fp
        assert not fp.closed
        with pytest.raises(OSError):
            os.remove(tmpfile)
        im.load()
        assert fp.closed
        os.remove(tmpfile)