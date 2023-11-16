import base64
import io
import itertools
import os
import re
import sys
from collections import namedtuple
import pytest
from PIL import Image, ImageFilter, ImageOps, TiffImagePlugin, TiffTags, features
from PIL.TiffImagePlugin import SAMPLEFORMAT, STRIPOFFSETS, SUBIFD
from .helper import assert_image_equal, assert_image_equal_tofile, assert_image_similar, assert_image_similar_tofile, hopper, mark_if_feature_version, skip_unless_feature

@skip_unless_feature('libtiff')
class LibTiffTestCase:

    def _assert_noerr(self, tmp_path, im):
        if False:
            for i in range(10):
                print('nop')
        'Helper tests that assert basic sanity about the g4 tiff reading'
        assert im.mode == '1'
        im.load()
        im.getdata()
        try:
            assert im._compression == 'group4'
        except AttributeError:
            print('No _compression')
            print(dir(im))
        out = str(tmp_path / 'temp.png')
        im.save(out)
        out_bytes = io.BytesIO()
        im.save(out_bytes, format='tiff', compression='group4')

class TestFileLibTiff(LibTiffTestCase):

    def test_version(self):
        if False:
            i = 10
            return i + 15
        assert re.search('\\d+\\.\\d+\\.\\d+$', features.version_codec('libtiff'))

    def test_g4_tiff(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'Test the ordinary file path load path'
        test_file = 'Tests/images/hopper_g4_500.tif'
        with Image.open(test_file) as im:
            assert im.size == (500, 500)
            self._assert_noerr(tmp_path, im)

    def test_g4_large(self, tmp_path):
        if False:
            while True:
                i = 10
        test_file = 'Tests/images/pport_g4.tif'
        with Image.open(test_file) as im:
            self._assert_noerr(tmp_path, im)

    def test_g4_tiff_file(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'Testing the string load path'
        test_file = 'Tests/images/hopper_g4_500.tif'
        with open(test_file, 'rb') as f:
            with Image.open(f) as im:
                assert im.size == (500, 500)
                self._assert_noerr(tmp_path, im)

    def test_g4_tiff_bytesio(self, tmp_path):
        if False:
            print('Hello World!')
        'Testing the stringio loading code path'
        test_file = 'Tests/images/hopper_g4_500.tif'
        s = io.BytesIO()
        with open(test_file, 'rb') as f:
            s.write(f.read())
            s.seek(0)
        with Image.open(s) as im:
            assert im.size == (500, 500)
            self._assert_noerr(tmp_path, im)

    def test_g4_non_disk_file_object(self, tmp_path):
        if False:
            print('Hello World!')
        'Testing loading from non-disk non-BytesIO file object'
        test_file = 'Tests/images/hopper_g4_500.tif'
        s = io.BytesIO()
        with open(test_file, 'rb') as f:
            s.write(f.read())
            s.seek(0)
        r = io.BufferedReader(s)
        with Image.open(r) as im:
            assert im.size == (500, 500)
            self._assert_noerr(tmp_path, im)

    def test_g4_eq_png(self):
        if False:
            i = 10
            return i + 15
        "Checking that we're actually getting the data that we expect"
        with Image.open('Tests/images/hopper_bw_500.png') as png:
            assert_image_equal_tofile(png, 'Tests/images/hopper_g4_500.tif')

    def test_g4_fillorder_eq_png(self):
        if False:
            for i in range(10):
                print('nop')
        "Checking that we're actually getting the data that we expect"
        with Image.open('Tests/images/g4-fillorder-test.tif') as g4:
            assert_image_equal_tofile(g4, 'Tests/images/g4-fillorder-test.png')

    def test_g4_write(self, tmp_path):
        if False:
            print('Hello World!')
        'Checking to see that the saved image is the same as what we wrote'
        test_file = 'Tests/images/hopper_g4_500.tif'
        with Image.open(test_file) as orig:
            out = str(tmp_path / 'temp.tif')
            rot = orig.transpose(Image.Transpose.ROTATE_90)
            assert rot.size == (500, 500)
            rot.save(out)
            with Image.open(out) as reread:
                assert reread.size == (500, 500)
                self._assert_noerr(tmp_path, reread)
                assert_image_equal(reread, rot)
                assert reread.info['compression'] == 'group4'
                assert reread.info['compression'] == orig.info['compression']
                assert orig.tobytes() != reread.tobytes()

    def test_adobe_deflate_tiff(self):
        if False:
            print('Hello World!')
        test_file = 'Tests/images/tiff_adobe_deflate.tif'
        with Image.open(test_file) as im:
            assert im.mode == 'RGB'
            assert im.size == (278, 374)
            assert im.tile[0][:3] == ('libtiff', (0, 0, 278, 374), 0)
            im.load()
            assert_image_equal_tofile(im, 'Tests/images/tiff_adobe_deflate.png')

    @pytest.mark.parametrize('legacy_api', (False, True))
    def test_write_metadata(self, legacy_api, tmp_path):
        if False:
            print('Hello World!')
        'Test metadata writing through libtiff'
        f = str(tmp_path / 'temp.tiff')
        with Image.open('Tests/images/hopper_g4.tif') as img:
            img.save(f, tiffinfo=img.tag)
            if legacy_api:
                original = img.tag.named()
            else:
                original = img.tag_v2.named()
        ignored = ['StripByteCounts', 'RowsPerStrip', 'PageNumber', 'PhotometricInterpretation']
        with Image.open(f) as loaded:
            if legacy_api:
                reloaded = loaded.tag.named()
            else:
                reloaded = loaded.tag_v2.named()
        for (tag, value) in itertools.chain(reloaded.items(), original.items()):
            if tag not in ignored:
                val = original[tag]
                if tag.endswith('Resolution'):
                    if legacy_api:
                        assert val[0][0] / val[0][1] == 4294967295 / 113653537, f"{tag} didn't roundtrip"
                    else:
                        assert val == 37.79000115940079, f"{tag} didn't roundtrip"
                else:
                    assert val == value, f"{tag} didn't roundtrip"
        requested_fields = ['StripByteCounts', 'RowsPerStrip', 'StripOffsets']
        for field in requested_fields:
            assert field in reloaded, f'{field} not in metadata'

    @pytest.mark.valgrind_known_error(reason='Known invalid metadata')
    def test_additional_metadata(self, tmp_path):
        if False:
            print('Hello World!')
        core_items = {tag: info for (tag, info) in ((s, TiffTags.lookup(s)) for s in TiffTags.LIBTIFF_CORE) if info.type is not None}
        with Image.open('Tests/images/hopper_g4.tif') as im:
            for tag in im.tag_v2:
                try:
                    del core_items[tag]
                except KeyError:
                    pass
            del core_items[320]
            values = {2: 'test', 3: 1, 4: 2 ** 20, 5: TiffImagePlugin.IFDRational(100, 1), 12: 1.05}
            new_ifd = TiffImagePlugin.ImageFileDirectory_v2()
            for (tag, info) in core_items.items():
                if info.length == 1:
                    new_ifd[tag] = values[info.type]
                if info.length == 0:
                    new_ifd[tag] = tuple((values[info.type] for _ in range(3)))
                else:
                    new_ifd[tag] = tuple((values[info.type] for _ in range(info.length)))
            del new_ifd[338]
            out = str(tmp_path / 'temp.tif')
            TiffImagePlugin.WRITE_LIBTIFF = True
            im.save(out, tiffinfo=new_ifd)
        TiffImagePlugin.WRITE_LIBTIFF = False

    def test_custom_metadata(self, tmp_path):
        if False:
            return 10
        tc = namedtuple('test_case', 'value,type,supported_by_default')
        custom = {37000 + k: v for (k, v) in enumerate([tc(4, TiffTags.SHORT, True), tc(123456789, TiffTags.LONG, True), tc(-4, TiffTags.SIGNED_BYTE, False), tc(-4, TiffTags.SIGNED_SHORT, False), tc(-123456789, TiffTags.SIGNED_LONG, False), tc(TiffImagePlugin.IFDRational(4, 7), TiffTags.RATIONAL, True), tc(4.25, TiffTags.FLOAT, True), tc(4.25, TiffTags.DOUBLE, True), tc('custom tag value', TiffTags.ASCII, True), tc(b'custom tag value', TiffTags.BYTE, True), tc((4, 5, 6), TiffTags.SHORT, True), tc((123456789, 9, 34, 234, 219387, 92432323), TiffTags.LONG, True), tc((-4, 9, 10), TiffTags.SIGNED_BYTE, False), tc((-4, 5, 6), TiffTags.SIGNED_SHORT, False), tc((-123456789, 9, 34, 234, 219387, -92432323), TiffTags.SIGNED_LONG, False), tc((4.25, 5.25), TiffTags.FLOAT, True), tc((4.25, 5.25), TiffTags.DOUBLE, True), tc(bytes([4]), TiffTags.BYTE, True), tc(bytes((4, 9, 10)), TiffTags.BYTE, True)])}
        libtiffs = [False]
        if Image.core.libtiff_support_custom_tags:
            libtiffs.append(True)
        for libtiff in libtiffs:
            TiffImagePlugin.WRITE_LIBTIFF = libtiff

            def check_tags(tiffinfo):
                if False:
                    i = 10
                    return i + 15
                im = hopper()
                out = str(tmp_path / 'temp.tif')
                im.save(out, tiffinfo=tiffinfo)
                with Image.open(out) as reloaded:
                    for (tag, value) in tiffinfo.items():
                        reloaded_value = reloaded.tag_v2[tag]
                        if isinstance(reloaded_value, TiffImagePlugin.IFDRational) and libtiff:
                            assert round(abs(float(reloaded_value) - float(value)), 7) == 0
                            continue
                        assert reloaded_value == value
            ifd = TiffImagePlugin.ImageFileDirectory_v2()
            for (tag, tagdata) in custom.items():
                ifd[tag] = tagdata.value
                ifd.tagtype[tag] = tagdata.type
            check_tags(ifd)
            check_tags({tag: tagdata.value for (tag, tagdata) in custom.items() if tagdata.supported_by_default})
        TiffImagePlugin.WRITE_LIBTIFF = False

    def test_subifd(self, tmp_path):
        if False:
            print('Hello World!')
        outfile = str(tmp_path / 'temp.tif')
        with Image.open('Tests/images/g4_orientation_6.tif') as im:
            im.tag_v2[SUBIFD] = 10000
            im.save(outfile)

    def test_xmlpacket_tag(self, tmp_path):
        if False:
            print('Hello World!')
        TiffImagePlugin.WRITE_LIBTIFF = True
        out = str(tmp_path / 'temp.tif')
        hopper().save(out, tiffinfo={700: b'xmlpacket tag'})
        TiffImagePlugin.WRITE_LIBTIFF = False
        with Image.open(out) as reloaded:
            if 700 in reloaded.tag_v2:
                assert reloaded.tag_v2[700] == b'xmlpacket tag'

    def test_int_dpi(self, tmp_path):
        if False:
            print('Hello World!')
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        TiffImagePlugin.WRITE_LIBTIFF = True
        im.save(out, dpi=(72, 72))
        TiffImagePlugin.WRITE_LIBTIFF = False
        with Image.open(out) as reloaded:
            assert reloaded.info['dpi'] == (72.0, 72.0)

    def test_g3_compression(self, tmp_path):
        if False:
            return 10
        with Image.open('Tests/images/hopper_g4_500.tif') as i:
            out = str(tmp_path / 'temp.tif')
            i.save(out, compression='group3')
            with Image.open(out) as reread:
                assert reread.info['compression'] == 'group3'
                assert_image_equal(reread, i)

    def test_little_endian(self, tmp_path):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/16bit.deflate.tif') as im:
            assert im.getpixel((0, 0)) == 480
            assert im.mode == 'I;16'
            b = im.tobytes()
            assert b[0] == ord(b'\xe0')
            assert b[1] == ord(b'\x01')
            out = str(tmp_path / 'temp.tif')
            im.save(out)
        with Image.open(out) as reread:
            assert reread.info['compression'] == im.info['compression']
            assert reread.getpixel((0, 0)) == 480

    def test_big_endian(self, tmp_path):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/16bit.MM.deflate.tif') as im:
            assert im.getpixel((0, 0)) == 480
            assert im.mode == 'I;16B'
            b = im.tobytes()
            assert b[0] == ord(b'\x01')
            assert b[1] == ord(b'\xe0')
            out = str(tmp_path / 'temp.tif')
            im.save(out)
            with Image.open(out) as reread:
                assert reread.info['compression'] == im.info['compression']
                assert reread.getpixel((0, 0)) == 480

    def test_g4_string_info(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'Tests String data in info directory'
        test_file = 'Tests/images/hopper_g4_500.tif'
        with Image.open(test_file) as orig:
            out = str(tmp_path / 'temp.tif')
            orig.tag[269] = 'temp.tif'
            orig.save(out)
        with Image.open(out) as reread:
            assert 'temp.tif' == reread.tag_v2[269]
            assert 'temp.tif' == reread.tag[269][0]

    def test_12bit_rawmode(self):
        if False:
            print('Hello World!')
        'Are we generating the same interpretation\n        of the image as Imagemagick is?'
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/12bit.cropped.tif') as im:
            im.load()
            TiffImagePlugin.READ_LIBTIFF = False
            assert_image_equal_tofile(im, 'Tests/images/12in16bit.tif')

    def test_blur(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        out = str(tmp_path / 'temp.tif')
        with Image.open('Tests/images/pport_g4.tif') as im:
            im = im.convert('L')
        im = im.filter(ImageFilter.GaussianBlur(4))
        im.save(out, compression='tiff_adobe_deflate')
        assert_image_equal_tofile(im, out)

    def test_compressions(self, tmp_path):
        if False:
            i = 10
            return i + 15
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        im.save(out)
        size_raw = os.path.getsize(out)
        for compression in ('packbits', 'tiff_lzw'):
            im.save(out, compression=compression)
            size_compressed = os.path.getsize(out)
            assert_image_equal_tofile(im, out)
        im.save(out, compression='jpeg')
        size_jpeg = os.path.getsize(out)
        with Image.open(out) as im2:
            assert_image_similar(im, im2, 30)
        im.save(out, compression='jpeg', quality=30)
        size_jpeg_30 = os.path.getsize(out)
        assert_image_similar_tofile(im2, out, 30)
        assert size_raw > size_compressed
        assert size_compressed > size_jpeg
        assert size_jpeg > size_jpeg_30

    def test_tiff_jpeg_compression(self, tmp_path):
        if False:
            print('Hello World!')
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        im.save(out, compression='tiff_jpeg')
        with Image.open(out) as reloaded:
            assert reloaded.info['compression'] == 'jpeg'

    def test_tiff_deflate_compression(self, tmp_path):
        if False:
            while True:
                i = 10
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        im.save(out, compression='tiff_deflate')
        with Image.open(out) as reloaded:
            assert reloaded.info['compression'] == 'tiff_adobe_deflate'

    def test_quality(self, tmp_path):
        if False:
            while True:
                i = 10
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        with pytest.raises(ValueError):
            im.save(out, compression='tiff_lzw', quality=50)
        with pytest.raises(ValueError):
            im.save(out, compression='jpeg', quality=-1)
        with pytest.raises(ValueError):
            im.save(out, compression='jpeg', quality=101)
        with pytest.raises(ValueError):
            im.save(out, compression='jpeg', quality='good')
        im.save(out, compression='jpeg', quality=0)
        im.save(out, compression='jpeg', quality=100)

    def test_cmyk_save(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        im = hopper('CMYK')
        out = str(tmp_path / 'temp.tif')
        im.save(out, compression='tiff_adobe_deflate')
        assert_image_equal_tofile(im, out)

    @pytest.mark.parametrize('im', (hopper('P'), Image.new('P', (1, 1), '#000')))
    def test_palette_save(self, im, tmp_path):
        if False:
            return 10
        out = str(tmp_path / 'temp.tif')
        TiffImagePlugin.WRITE_LIBTIFF = True
        im.save(out)
        TiffImagePlugin.WRITE_LIBTIFF = False
        with Image.open(out) as reloaded:
            assert len(reloaded.tag_v2[320]) == 768

    @pytest.mark.parametrize('compression', ('tiff_ccitt', 'group3', 'group4'))
    def test_bw_compression_w_rgb(self, compression, tmp_path):
        if False:
            return 10
        im = hopper('RGB')
        out = str(tmp_path / 'temp.tif')
        with pytest.raises(OSError):
            im.save(out, compression=compression)

    def test_fp_leak(self):
        if False:
            print('Hello World!')
        im = Image.open('Tests/images/hopper_g4_500.tif')
        fn = im.fp.fileno()
        os.fstat(fn)
        im.load()
        with pytest.raises(OSError):
            os.fstat(fn)
        im = None
        with pytest.raises(OSError):
            os.fstat(fn)
        with pytest.raises(OSError):
            os.close(fn)

    def test_multipage(self):
        if False:
            while True:
                i = 10
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/multipage.tiff') as im:
            im.seek(0)
            assert im.size == (10, 10)
            assert im.convert('RGB').getpixel((0, 0)) == (0, 128, 0)
            assert im.tag.next
            im.seek(1)
            assert im.size == (10, 10)
            assert im.convert('RGB').getpixel((0, 0)) == (255, 0, 0)
            assert im.tag.next
            im.seek(2)
            assert not im.tag.next
            assert im.size == (20, 20)
            assert im.convert('RGB').getpixel((0, 0)) == (0, 0, 255)
        TiffImagePlugin.READ_LIBTIFF = False

    def test_multipage_nframes(self):
        if False:
            while True:
                i = 10
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/multipage.tiff') as im:
            frames = im.n_frames
            assert frames == 3
            for _ in range(frames):
                im.seek(0)
                im.load()
        TiffImagePlugin.READ_LIBTIFF = False

    def test_multipage_seek_backwards(self):
        if False:
            return 10
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/multipage.tiff') as im:
            im.seek(1)
            im.load()
            im.seek(0)
            assert im.convert('RGB').getpixel((0, 0)) == (0, 128, 0)
        TiffImagePlugin.READ_LIBTIFF = False

    def test__next(self):
        if False:
            print('Hello World!')
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/hopper.tif') as im:
            assert not im.tag.next
            im.load()
            assert not im.tag.next

    def test_4bit(self):
        if False:
            while True:
                i = 10
        test_file = 'Tests/images/hopper_gray_4bpp.tif'
        original = hopper('L')
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open(test_file) as im:
            TiffImagePlugin.READ_LIBTIFF = False
            assert im.size == (128, 128)
            assert im.mode == 'L'
            assert_image_similar(im, original, 7.3)

    def test_gray_semibyte_per_pixel(self):
        if False:
            print('Hello World!')
        test_files = ((24.8, ('Tests/images/tiff_gray_2_4_bpp/hopper2.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper2I.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper2R.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper2IR.tif')), (7.3, ('Tests/images/tiff_gray_2_4_bpp/hopper4.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper4I.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper4R.tif', 'Tests/images/tiff_gray_2_4_bpp/hopper4IR.tif')))
        original = hopper('L')
        for (epsilon, group) in test_files:
            with Image.open(group[0]) as im:
                assert im.size == (128, 128)
                assert im.mode == 'L'
                assert_image_similar(im, original, epsilon)
            for file in group[1:]:
                with Image.open(file) as im2:
                    assert im2.size == (128, 128)
                    assert im2.mode == 'L'
                    assert_image_equal(im, im2)

    def test_save_bytesio(self):
        if False:
            i = 10
            return i + 15
        TiffImagePlugin.WRITE_LIBTIFF = True
        TiffImagePlugin.READ_LIBTIFF = True
        pilim = hopper()

        def save_bytesio(compression=None):
            if False:
                i = 10
                return i + 15
            buffer_io = io.BytesIO()
            pilim.save(buffer_io, format='tiff', compression=compression)
            buffer_io.seek(0)
            assert_image_similar_tofile(pilim, buffer_io, 0)
        save_bytesio()
        save_bytesio('raw')
        save_bytesio('packbits')
        save_bytesio('tiff_lzw')
        TiffImagePlugin.WRITE_LIBTIFF = False
        TiffImagePlugin.READ_LIBTIFF = False

    def test_save_ycbcr(self, tmp_path):
        if False:
            while True:
                i = 10
        im = hopper('YCbCr')
        outfile = str(tmp_path / 'temp.tif')
        im.save(outfile, compression='jpeg')
        with Image.open(outfile) as reloaded:
            assert reloaded.tag_v2[530] == (1, 1)
            assert reloaded.tag_v2[532] == (0, 255, 128, 255, 128, 255)

    def test_exif_ifd(self, tmp_path):
        if False:
            while True:
                i = 10
        outfile = str(tmp_path / 'temp.tif')
        with Image.open('Tests/images/tiff_adobe_deflate.tif') as im:
            assert im.tag_v2[34665] == 125456
            im.save(outfile)
        with Image.open(outfile) as reloaded:
            if Image.core.libtiff_support_custom_tags:
                assert reloaded.tag_v2[34665] == 125456

    def test_crashing_metadata(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/rdf.tif') as im:
            out = str(tmp_path / 'temp.tif')
            TiffImagePlugin.WRITE_LIBTIFF = True
            im.save(out, format='TIFF')
        TiffImagePlugin.WRITE_LIBTIFF = False

    def test_page_number_x_0(self, tmp_path):
        if False:
            print('Hello World!')
        outfile = str(tmp_path / 'temp.tif')
        infile = 'Tests/images/total-pages-zero.tif'
        with Image.open(infile) as im:
            im.save(outfile)

    def test_fd_duplication(self, tmp_path):
        if False:
            print('Hello World!')
        tmpfile = str(tmp_path / 'temp.tif')
        with open(tmpfile, 'wb') as f:
            with open('Tests/images/g4-multi.tiff', 'rb') as src:
                f.write(src.read())
        im = Image.open(tmpfile)
        im.n_frames
        im.close()
        os.remove(tmpfile)

    def test_read_icc(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/hopper.iccprofile.tif') as img:
            icc = img.info.get('icc_profile')
            assert icc is not None
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/hopper.iccprofile.tif') as img:
            icc_libtiff = img.info.get('icc_profile')
            assert icc_libtiff is not None
        TiffImagePlugin.READ_LIBTIFF = False
        assert icc == icc_libtiff

    def test_write_icc(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')

        def check_write(libtiff):
            if False:
                while True:
                    i = 10
            TiffImagePlugin.WRITE_LIBTIFF = libtiff
            with Image.open('Tests/images/hopper.iccprofile.tif') as img:
                icc_profile = img.info['icc_profile']
                out = str(tmp_path / 'temp.tif')
                img.save(out, icc_profile=icc_profile)
            with Image.open(out) as reloaded:
                assert icc_profile == reloaded.info['icc_profile']
        libtiffs = []
        if Image.core.libtiff_support_custom_tags:
            libtiffs.append(True)
        libtiffs.append(False)
        for libtiff in libtiffs:
            check_write(libtiff)

    def test_multipage_compression(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/compression.tif') as im:
            im.seek(0)
            assert im._compression == 'tiff_ccitt'
            assert im.size == (10, 10)
            im.seek(1)
            assert im._compression == 'packbits'
            assert im.size == (10, 10)
            im.load()
            im.seek(0)
            assert im._compression == 'tiff_ccitt'
            assert im.size == (10, 10)
            im.load()

    def test_save_tiff_with_jpegtables(self, tmp_path):
        if False:
            return 10
        outfile = str(tmp_path / 'temp.tif')
        infile = 'Tests/images/hopper_jpg.tif'
        with Image.open(infile) as im:
            im.save(outfile)

    def test_16bit_RGB_tiff(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/tiff_16bit_RGB.tiff') as im:
            assert im.mode == 'RGB'
            assert im.size == (100, 40)
            assert im.tile, [('libtiff', (0, 0, 100, 40), 0, ('RGB;16N', 'tiff_adobe_deflate', False, 8))]
            im.load()
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGB_target.png')

    def test_16bit_RGBa_tiff(self):
        if False:
            return 10
        with Image.open('Tests/images/tiff_16bit_RGBa.tiff') as im:
            assert im.mode == 'RGBA'
            assert im.size == (100, 40)
            assert im.tile, [('libtiff', (0, 0, 100, 40), 0, ('RGBa;16N', 'tiff_lzw', False, 38236))]
            im.load()
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGBa_target.png')

    @skip_unless_feature('jpg')
    def test_gimp_tiff(self):
        if False:
            print('Hello World!')
        filename = 'Tests/images/pil168.tif'
        with Image.open(filename) as im:
            assert im.mode == 'RGB'
            assert im.size == (256, 256)
            assert im.tile == [('libtiff', (0, 0, 256, 256), 0, ('RGB', 'jpeg', False, 5122))]
            im.load()
            assert_image_equal_tofile(im, 'Tests/images/pil168.png')

    def test_sampleformat(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/copyleft.tiff') as im:
            assert im.mode == 'RGB'
            assert_image_equal_tofile(im, 'Tests/images/copyleft.png', mode='RGB')

    def test_sampleformat_write(self, tmp_path):
        if False:
            while True:
                i = 10
        im = Image.new('F', (1, 1))
        out = str(tmp_path / 'temp.tif')
        TiffImagePlugin.WRITE_LIBTIFF = True
        im.save(out)
        TiffImagePlugin.WRITE_LIBTIFF = False
        with Image.open(out) as reloaded:
            assert reloaded.mode == 'F'
            assert reloaded.getexif()[SAMPLEFORMAT] == 3

    def test_lzma(self, capfd):
        if False:
            return 10
        try:
            with Image.open('Tests/images/hopper_lzma.tif') as im:
                assert im.mode == 'RGB'
                assert im.size == (128, 128)
                assert im.format == 'TIFF'
                im2 = hopper()
                assert_image_similar(im, im2, 5)
        except OSError:
            captured = capfd.readouterr()
            if 'LZMA compression support is not configured' in captured.err:
                pytest.skip('LZMA compression support is not configured')
            sys.stdout.write(captured.out)
            sys.stderr.write(captured.err)
            raise

    def test_webp(self, capfd):
        if False:
            for i in range(10):
                print('nop')
        try:
            with Image.open('Tests/images/hopper_webp.tif') as im:
                assert im.mode == 'RGB'
                assert im.size == (128, 128)
                assert im.format == 'TIFF'
                assert_image_similar_tofile(im, 'Tests/images/hopper_webp.png', 1)
        except OSError:
            captured = capfd.readouterr()
            if 'WEBP compression support is not configured' in captured.err:
                pytest.skip('WEBP compression support is not configured')
            if 'Compression scheme 50001 strip decoding is not implemented' in captured.err:
                pytest.skip('Compression scheme 50001 strip decoding is not implemented')
            sys.stdout.write(captured.out)
            sys.stderr.write(captured.err)
            raise

    def test_lzw(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/hopper_lzw.tif') as im:
            assert im.mode == 'RGB'
            assert im.size == (128, 128)
            assert im.format == 'TIFF'
            im2 = hopper()
            assert_image_similar(im, im2, 5)

    def test_strip_cmyk_jpeg(self):
        if False:
            print('Hello World!')
        infile = 'Tests/images/tiff_strip_cmyk_jpeg.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/pil_sample_cmyk.jpg', 0.5)

    def test_strip_cmyk_16l_jpeg(self):
        if False:
            i = 10
            return i + 15
        infile = 'Tests/images/tiff_strip_cmyk_16l_jpeg.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/pil_sample_cmyk.jpg', 0.5)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_strip_ycbcr_jpeg_2x2_sampling(self):
        if False:
            return 10
        infile = 'Tests/images/tiff_strip_ycbcr_jpeg_2x2_sampling.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/flower.jpg', 1.2)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_strip_ycbcr_jpeg_1x1_sampling(self):
        if False:
            return 10
        infile = 'Tests/images/tiff_strip_ycbcr_jpeg_1x1_sampling.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/flower2.jpg', 0.01)

    def test_tiled_cmyk_jpeg(self):
        if False:
            return 10
        infile = 'Tests/images/tiff_tiled_cmyk_jpeg.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/pil_sample_cmyk.jpg', 0.5)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_tiled_ycbcr_jpeg_1x1_sampling(self):
        if False:
            while True:
                i = 10
        infile = 'Tests/images/tiff_tiled_ycbcr_jpeg_1x1_sampling.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/flower2.jpg', 0.01)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_tiled_ycbcr_jpeg_2x2_sampling(self):
        if False:
            while True:
                i = 10
        infile = 'Tests/images/tiff_tiled_ycbcr_jpeg_2x2_sampling.tif'
        with Image.open(infile) as im:
            assert_image_similar_tofile(im, 'Tests/images/flower.jpg', 1.5)

    def test_strip_planar_rgb(self):
        if False:
            return 10
        infile = 'Tests/images/tiff_strip_planar_lzw.tiff'
        with Image.open(infile) as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_adobe_deflate.png')

    def test_tiled_planar_rgb(self):
        if False:
            while True:
                i = 10
        infile = 'Tests/images/tiff_tiled_planar_lzw.tiff'
        with Image.open(infile) as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_adobe_deflate.png')

    def test_tiled_planar_16bit_RGB(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/tiff_tiled_planar_16bit_RGB.tiff') as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGB_target.png')

    def test_strip_planar_16bit_RGB(self):
        if False:
            return 10
        with Image.open('Tests/images/tiff_strip_planar_16bit_RGB.tiff') as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGB_target.png')

    def test_tiled_planar_16bit_RGBa(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/tiff_tiled_planar_16bit_RGBa.tiff') as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGBa_target.png')

    def test_strip_planar_16bit_RGBa(self):
        if False:
            i = 10
            return i + 15
        with Image.open('Tests/images/tiff_strip_planar_16bit_RGBa.tiff') as im:
            assert_image_equal_tofile(im, 'Tests/images/tiff_16bit_RGBa_target.png')

    @pytest.mark.parametrize('compression', (None, 'jpeg'))
    def test_block_tile_tags(self, compression, tmp_path):
        if False:
            print('Hello World!')
        im = hopper()
        out = str(tmp_path / 'temp.tif')
        tags = {TiffImagePlugin.TILEWIDTH: 256, TiffImagePlugin.TILELENGTH: 256, TiffImagePlugin.TILEOFFSETS: 256, TiffImagePlugin.TILEBYTECOUNTS: 256}
        im.save(out, exif=tags, compression=compression)
        with Image.open(out) as reloaded:
            for tag in tags:
                assert tag not in reloaded.getexif()

    def test_old_style_jpeg(self):
        if False:
            return 10
        with Image.open('Tests/images/old-style-jpeg-compression.tif') as im:
            assert_image_equal_tofile(im, 'Tests/images/old-style-jpeg-compression.png')

    def test_open_missing_samplesperpixel(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/old-style-jpeg-compression-no-samplesperpixel.tif') as im:
            assert_image_equal_tofile(im, 'Tests/images/old-style-jpeg-compression.png')

    @pytest.mark.parametrize('file_name, mode, size, tile', [('tiff_wrong_bits_per_sample.tiff', 'RGBA', (52, 53), [('raw', (0, 0, 52, 53), 160, ('RGBA', 0, 1))]), ('tiff_wrong_bits_per_sample_2.tiff', 'RGB', (16, 16), [('raw', (0, 0, 16, 16), 8, ('RGB', 0, 1))]), ('tiff_wrong_bits_per_sample_3.tiff', 'RGBA', (512, 256), [('libtiff', (0, 0, 512, 256), 0, ('RGBA', 'tiff_lzw', False, 48782))])])
    def test_wrong_bits_per_sample(self, file_name, mode, size, tile):
        if False:
            return 10
        with Image.open('Tests/images/' + file_name) as im:
            assert im.mode == mode
            assert im.size == size
            assert im.tile == tile
            im.load()

    def test_no_rows_per_strip(self):
        if False:
            while True:
                i = 10
        infile = 'Tests/images/no_rows_per_strip.tif'
        with Image.open(infile) as im:
            im.load()
        assert im.size == (950, 975)

    def test_orientation(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/g4_orientation_1.tif') as base_im:
            for i in range(2, 9):
                with Image.open('Tests/images/g4_orientation_' + str(i) + '.tif') as im:
                    assert 274 in im.tag_v2
                    im.load()
                    assert 274 not in im.tag_v2
                    assert_image_similar(base_im, im, 0.7)

    def test_exif_transpose(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/g4_orientation_1.tif') as base_im:
            for i in range(2, 9):
                with Image.open('Tests/images/g4_orientation_' + str(i) + '.tif') as im:
                    im = ImageOps.exif_transpose(im)
                    assert_image_similar(base_im, im, 0.7)

    @pytest.mark.valgrind_known_error(reason='Backtrace in Python Core')
    def test_sampleformat_not_corrupted(self):
        if False:
            while True:
                i = 10
        tiff = io.BytesIO(base64.b64decode(b'SUkqAAgAAAAPAP4ABAABAAAAAAAAAAABBAABAAAAAQAAAAEBBAABAAAAAQAAAAIBAwADAAAAwgAAAAMBAwABAAAACAAAAAYBAwABAAAAAgAAABEBBAABAAAA4AAAABUBAwABAAAAAwAAABYBBAABAAAAAQAAABcBBAABAAAACwAAABoBBQABAAAAyAAAABsBBQABAAAA0AAAABwBAwABAAAAAQAAACgBAwABAAAAAQAAAFMBAwADAAAA2AAAAAAAAAAIAAgACAABAAAAAQAAAAEAAAABAAAAAQABAAEAAAB4nGNgYAAAAAMAAQ=='))
        out = io.BytesIO()
        with Image.open(tiff) as im:
            im.save(out, format='tiff')
        out.seek(0)
        with Image.open(out) as im:
            im.load()

    def test_realloc_overflow(self):
        if False:
            i = 10
            return i + 15
        TiffImagePlugin.READ_LIBTIFF = True
        with Image.open('Tests/images/tiff_overflow_rows_per_strip.tif') as im:
            with pytest.raises(OSError) as e:
                im.load()
            assert str(e.value) == '-9'
        TiffImagePlugin.READ_LIBTIFF = False

    @pytest.mark.parametrize('compression', ('tiff_adobe_deflate', 'jpeg'))
    def test_save_multistrip(self, compression, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        im = hopper('RGB').resize((256, 256))
        out = str(tmp_path / 'temp.tif')
        im.save(out, compression=compression)
        with Image.open(out) as im:
            assert len(im.tag_v2[STRIPOFFSETS]) > 1

    @pytest.mark.parametrize('argument', (True, False))
    def test_save_single_strip(self, argument, tmp_path):
        if False:
            print('Hello World!')
        im = hopper('RGB').resize((256, 256))
        out = str(tmp_path / 'temp.tif')
        if not argument:
            TiffImagePlugin.STRIP_SIZE = 2 ** 18
        try:
            arguments = {'compression': 'tiff_adobe_deflate'}
            if argument:
                arguments['strip_size'] = 2 ** 18
            im.save(out, **arguments)
            with Image.open(out) as im:
                assert len(im.tag_v2[STRIPOFFSETS]) == 1
        finally:
            TiffImagePlugin.STRIP_SIZE = 65536

    @pytest.mark.parametrize('compression', ('tiff_adobe_deflate', None))
    def test_save_zero(self, compression, tmp_path):
        if False:
            print('Hello World!')
        im = Image.new('RGB', (0, 0))
        out = str(tmp_path / 'temp.tif')
        with pytest.raises(SystemError):
            im.save(out, compression=compression)

    def test_save_many_compressed(self, tmp_path):
        if False:
            print('Hello World!')
        im = hopper()
        out = str(tmp_path / 'temp.tif')
        for _ in range(10000):
            im.save(out, compression='jpeg')

    @pytest.mark.parametrize('path, sizes', (('Tests/images/hopper.tif', ()), ('Tests/images/child_ifd.tiff', (16, 8)), ('Tests/images/child_ifd_jpeg.tiff', (20,))))
    def test_get_child_images(self, path, sizes):
        if False:
            for i in range(10):
                print('nop')
        with Image.open(path) as im:
            ims = im.get_child_images()
        assert len(ims) == len(sizes)
        for (i, im) in enumerate(ims):
            w = sizes[i]
            expected = Image.new('RGB', (w, w), '#f00')
            assert_image_similar(im, expected, 1)