from contextlib import contextmanager
import pytest
from PIL import Image, ImageDraw
from .helper import assert_image_equal, assert_image_similar, hopper, mark_if_feature_version

class TestImagingResampleVulnerability:

    def test_overflow(self):
        if False:
            while True:
                i = 10
        im = hopper('L')
        size_too_large = 4294967304 // 4
        size_normal = 1000
        for (xsize, ysize) in ((size_too_large, size_normal), (size_normal, size_too_large)):
            with pytest.raises(MemoryError):
                im.im.resize((xsize, ysize), Image.Resampling.BILINEAR)

    def test_invalid_size(self):
        if False:
            while True:
                i = 10
        im = hopper()
        im.resize((100, 100))
        with pytest.raises(ValueError):
            im.resize((-100, 100))
        with pytest.raises(ValueError):
            im.resize((100, -100))

    def test_modify_after_resizing(self):
        if False:
            while True:
                i = 10
        im = hopper('RGB')
        copy = im.resize(im.size)
        copy.paste('black', (0, 0, im.width // 2, im.height // 2))
        assert im.tobytes() != copy.tobytes()

class TestImagingCoreResampleAccuracy:

    def make_case(self, mode, size, color):
        if False:
            for i in range(10):
                print('nop')
        'Makes a sample image with two dark and two bright squares.\n        For example:\n        e0 e0 1f 1f\n        e0 e0 1f 1f\n        1f 1f e0 e0\n        1f 1f e0 e0\n        '
        case = Image.new('L', size, 255 - color)
        rectangle = ImageDraw.Draw(case).rectangle
        rectangle((0, 0, size[0] // 2 - 1, size[1] // 2 - 1), color)
        rectangle((size[0] // 2, size[1] // 2, size[0], size[1]), color)
        return Image.merge(mode, [case] * len(mode))

    def make_sample(self, data, size):
        if False:
            i = 10
            return i + 15
        'Restores a sample image from given data string which contains\n        hex-encoded pixels from the top left fourth of a sample.\n        '
        data = data.replace(' ', '')
        sample = Image.new('L', size)
        s_px = sample.load()
        (w, h) = (size[0] // 2, size[1] // 2)
        for y in range(h):
            for x in range(w):
                val = int(data[(y * w + x) * 2:(y * w + x + 1) * 2], 16)
                s_px[x, y] = val
                s_px[size[0] - x - 1, size[1] - y - 1] = val
                s_px[x, size[1] - y - 1] = 255 - val
                s_px[size[0] - x - 1, y] = 255 - val
        return sample

    def check_case(self, case, sample):
        if False:
            return 10
        s_px = sample.load()
        c_px = case.load()
        for y in range(case.size[1]):
            for x in range(case.size[0]):
                if c_px[x, y] != s_px[x, y]:
                    message = f'\nHave: \n{self.serialize_image(case)}\n\nExpected: \n{self.serialize_image(sample)}'
                    assert s_px[x, y] == c_px[x, y], message

    def serialize_image(self, image):
        if False:
            i = 10
            return i + 15
        s_px = image.load()
        return '\n'.join((' '.join((f'{s_px[x, y]:02x}' for x in range(image.size[0]))) for y in range(image.size[1])))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_reduce_box(self, mode):
        if False:
            i = 10
            return i + 15
        case = self.make_case(mode, (8, 8), 225)
        case = case.resize((4, 4), Image.Resampling.BOX)
        data = 'e1 e1e1 e1'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_reduce_bilinear(self, mode):
        if False:
            return 10
        case = self.make_case(mode, (8, 8), 225)
        case = case.resize((4, 4), Image.Resampling.BILINEAR)
        data = 'e1 c9c9 b7'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_reduce_hamming(self, mode):
        if False:
            for i in range(10):
                print('nop')
        case = self.make_case(mode, (8, 8), 225)
        case = case.resize((4, 4), Image.Resampling.HAMMING)
        data = 'e1 dada d3'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_reduce_bicubic(self, mode):
        if False:
            print('Hello World!')
        case = self.make_case(mode, (12, 12), 225)
        case = case.resize((6, 6), Image.Resampling.BICUBIC)
        data = 'e1 e3 d4e3 e5 d6d4 d6 c9'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (6, 6)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_reduce_lanczos(self, mode):
        if False:
            i = 10
            return i + 15
        case = self.make_case(mode, (16, 16), 225)
        case = case.resize((8, 8), Image.Resampling.LANCZOS)
        data = 'e1 e0 e4 d7e0 df e3 d6e4 e3 e7 dad7 d6 d9 ce'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (8, 8)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_enlarge_box(self, mode):
        if False:
            return 10
        case = self.make_case(mode, (2, 2), 225)
        case = case.resize((4, 4), Image.Resampling.BOX)
        data = 'e1 e1e1 e1'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_enlarge_bilinear(self, mode):
        if False:
            return 10
        case = self.make_case(mode, (2, 2), 225)
        case = case.resize((4, 4), Image.Resampling.BILINEAR)
        data = 'e1 b0b0 98'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_enlarge_hamming(self, mode):
        if False:
            return 10
        case = self.make_case(mode, (2, 2), 225)
        case = case.resize((4, 4), Image.Resampling.HAMMING)
        data = 'e1 d2d2 c5'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (4, 4)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_enlarge_bicubic(self, mode):
        if False:
            print('Hello World!')
        case = self.make_case(mode, (4, 4), 225)
        case = case.resize((8, 8), Image.Resampling.BICUBIC)
        data = 'e1 e5 ee b9e5 e9 f3 bcee f3 fd c1b9 bc c1 a2'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (8, 8)))

    @pytest.mark.parametrize('mode', ('RGBX', 'RGB', 'La', 'L'))
    def test_enlarge_lanczos(self, mode):
        if False:
            while True:
                i = 10
        case = self.make_case(mode, (6, 6), 225)
        case = case.resize((12, 12), Image.Resampling.LANCZOS)
        data = 'e1 e0 db ed f5 b8e0 df da ec f3 b7db db d6 e7 ee b5ed ec e6 fb ff bff5 f4 ee ff ff c4b8 b7 b4 bf c4 a0'
        for channel in case.split():
            self.check_case(channel, self.make_sample(data, (12, 12)))

    def test_box_filter_correct_range(self):
        if False:
            print('Hello World!')
        im = Image.new('RGB', (8, 8), '#1688ff').resize((100, 100), Image.Resampling.BOX)
        ref = Image.new('RGB', (100, 100), '#1688ff')
        assert_image_equal(im, ref)

class TestCoreResampleConsistency:

    def make_case(self, mode, fill):
        if False:
            while True:
                i = 10
        im = Image.new(mode, (512, 9), fill)
        return (im.resize((9, 512), Image.Resampling.LANCZOS), im.load()[0, 0])

    def run_case(self, case):
        if False:
            while True:
                i = 10
        (channel, color) = case
        px = channel.load()
        for x in range(channel.size[0]):
            for y in range(channel.size[1]):
                if px[x, y] != color:
                    message = f'{px[x, y]} != {color} for pixel {(x, y)}'
                    assert px[x, y] == color, message

    def test_8u(self):
        if False:
            return 10
        (im, color) = self.make_case('RGB', (0, 64, 255))
        (r, g, b) = im.split()
        self.run_case((r, color[0]))
        self.run_case((g, color[1]))
        self.run_case((b, color[2]))
        self.run_case(self.make_case('L', 12))

    def test_32i(self):
        if False:
            i = 10
            return i + 15
        self.run_case(self.make_case('I', 12))
        self.run_case(self.make_case('I', 2147483647))
        self.run_case(self.make_case('I', -12))
        self.run_case(self.make_case('I', -1 << 31))

    def test_32f(self):
        if False:
            while True:
                i = 10
        self.run_case(self.make_case('F', 1))
        self.run_case(self.make_case('F', 3.40282306074e+38))
        self.run_case(self.make_case('F', 1.175494e-38))
        self.run_case(self.make_case('F', 1.192093e-07))

class TestCoreResampleAlphaCorrect:

    def make_levels_case(self, mode):
        if False:
            while True:
                i = 10
        i = Image.new(mode, (256, 16))
        px = i.load()
        for y in range(i.size[1]):
            for x in range(i.size[0]):
                pix = [x] * len(mode)
                pix[-1] = 255 - y * 16
                px[x, y] = tuple(pix)
        return i

    def run_levels_case(self, i):
        if False:
            return 10
        px = i.load()
        for y in range(i.size[1]):
            used_colors = {px[x, y][0] for x in range(i.size[0])}
            assert 256 == len(used_colors), f'All colors should be present in resized image. Only {len(used_colors)} on {y} line.'

    @pytest.mark.xfail(reason="Current implementation isn't precise enough")
    def test_levels_rgba(self):
        if False:
            while True:
                i = 10
        case = self.make_levels_case('RGBA')
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BOX))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BILINEAR))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.HAMMING))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BICUBIC))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.LANCZOS))

    @pytest.mark.xfail(reason="Current implementation isn't precise enough")
    def test_levels_la(self):
        if False:
            while True:
                i = 10
        case = self.make_levels_case('LA')
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BOX))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BILINEAR))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.HAMMING))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.BICUBIC))
        self.run_levels_case(case.resize((512, 32), Image.Resampling.LANCZOS))

    def make_dirty_case(self, mode, clean_pixel, dirty_pixel):
        if False:
            return 10
        i = Image.new(mode, (64, 64), dirty_pixel)
        px = i.load()
        xdiv4 = i.size[0] // 4
        ydiv4 = i.size[1] // 4
        for y in range(ydiv4 * 2):
            for x in range(xdiv4 * 2):
                px[x + xdiv4, y + ydiv4] = clean_pixel
        return i

    def run_dirty_case(self, i, clean_pixel):
        if False:
            i = 10
            return i + 15
        px = i.load()
        for y in range(i.size[1]):
            for x in range(i.size[0]):
                if px[x, y][-1] != 0 and px[x, y][:-1] != clean_pixel:
                    message = f'pixel at ({x}, {y}) is different:\n{px[x, y]}\n{clean_pixel}'
                    assert px[x, y][:3] == clean_pixel, message

    def test_dirty_pixels_rgba(self):
        if False:
            print('Hello World!')
        case = self.make_dirty_case('RGBA', (255, 255, 0, 128), (0, 0, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BOX), (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BILINEAR), (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.HAMMING), (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BICUBIC), (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.LANCZOS), (255, 255, 0))

    def test_dirty_pixels_la(self):
        if False:
            return 10
        case = self.make_dirty_case('LA', (255, 128), (0, 0))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BOX), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BILINEAR), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.HAMMING), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.BICUBIC), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.Resampling.LANCZOS), (255,))

class TestCoreResamplePasses:

    @contextmanager
    def count(self, diff):
        if False:
            for i in range(10):
                print('nop')
        count = Image.core.get_stats()['new_count']
        yield
        assert Image.core.get_stats()['new_count'] - count == diff

    def test_horizontal(self):
        if False:
            return 10
        im = hopper('L')
        with self.count(1):
            im.resize((im.size[0] - 10, im.size[1]), Image.Resampling.BILINEAR)

    def test_vertical(self):
        if False:
            i = 10
            return i + 15
        im = hopper('L')
        with self.count(1):
            im.resize((im.size[0], im.size[1] - 10), Image.Resampling.BILINEAR)

    def test_both(self):
        if False:
            i = 10
            return i + 15
        im = hopper('L')
        with self.count(2):
            im.resize((im.size[0] - 10, im.size[1] - 10), Image.Resampling.BILINEAR)

    def test_box_horizontal(self):
        if False:
            i = 10
            return i + 15
        im = hopper('L')
        box = (20, 0, im.size[0] - 20, im.size[1])
        with self.count(1):
            with_box = im.resize(im.size, Image.Resampling.BILINEAR, box)
        with self.count(2):
            cropped = im.crop(box).resize(im.size, Image.Resampling.BILINEAR)
        assert_image_similar(with_box, cropped, 0.1)

    def test_box_vertical(self):
        if False:
            print('Hello World!')
        im = hopper('L')
        box = (0, 20, im.size[0], im.size[1] - 20)
        with self.count(1):
            with_box = im.resize(im.size, Image.Resampling.BILINEAR, box)
        with self.count(2):
            cropped = im.crop(box).resize(im.size, Image.Resampling.BILINEAR)
        assert_image_similar(with_box, cropped, 0.1)

class TestCoreResampleCoefficients:

    def test_reduce(self):
        if False:
            return 10
        test_color = 254
        for size in range(400000, 400010, 2):
            i = Image.new('L', (size, 1), 0)
            draw = ImageDraw.Draw(i)
            draw.rectangle((0, 0, i.size[0] // 2 - 1, 0), test_color)
            px = i.resize((5, i.size[1]), Image.Resampling.BICUBIC).load()
            if px[2, 0] != test_color // 2:
                assert test_color // 2 == px[2, 0]

    def test_nonzero_coefficients(self):
        if False:
            while True:
                i = 10
        im = Image.new('RGBA', (1280, 1280), (32, 64, 96, 255))
        histogram = im.resize((256, 256), Image.Resampling.BICUBIC).histogram()
        assert histogram[256 * 0 + 32] == 65536
        assert histogram[256 * 1 + 64] == 65536
        assert histogram[256 * 2 + 96] == 65536
        assert histogram[256 * 3 + 255] == 65536

class TestCoreResampleBox:

    @pytest.mark.parametrize('resample', (Image.Resampling.NEAREST, Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.HAMMING, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS))
    def test_wrong_arguments(self, resample):
        if False:
            i = 10
            return i + 15
        im = hopper()
        im.resize((32, 32), resample, (0, 0, im.width, im.height))
        im.resize((32, 32), resample, (20, 20, im.width, im.height))
        im.resize((32, 32), resample, (20, 20, 20, 100))
        im.resize((32, 32), resample, (20, 20, 100, 20))
        with pytest.raises(TypeError, match='must be sequence of length 4'):
            im.resize((32, 32), resample, (im.width, im.height))
        with pytest.raises(ValueError, match="can't be negative"):
            im.resize((32, 32), resample, (-20, 20, 100, 100))
        with pytest.raises(ValueError, match="can't be negative"):
            im.resize((32, 32), resample, (20, -20, 100, 100))
        with pytest.raises(ValueError, match="can't be empty"):
            im.resize((32, 32), resample, (20.1, 20, 20, 100))
        with pytest.raises(ValueError, match="can't be empty"):
            im.resize((32, 32), resample, (20, 20.1, 100, 20))
        with pytest.raises(ValueError, match="can't be empty"):
            im.resize((32, 32), resample, (20.1, 20.1, 20, 20))
        with pytest.raises(ValueError, match="can't exceed"):
            im.resize((32, 32), resample, (0, 0, im.width + 1, im.height))
        with pytest.raises(ValueError, match="can't exceed"):
            im.resize((32, 32), resample, (0, 0, im.width, im.height + 1))

    def resize_tiled(self, im, dst_size, xtiles, ytiles):
        if False:
            for i in range(10):
                print('nop')

        def split_range(size, tiles):
            if False:
                i = 10
                return i + 15
            scale = size / tiles
            for i in range(tiles):
                yield (int(round(scale * i)), int(round(scale * (i + 1))))
        tiled = Image.new(im.mode, dst_size)
        scale = (im.size[0] / tiled.size[0], im.size[1] / tiled.size[1])
        for (y0, y1) in split_range(dst_size[1], ytiles):
            for (x0, x1) in split_range(dst_size[0], xtiles):
                box = (x0 * scale[0], y0 * scale[1], x1 * scale[0], y1 * scale[1])
                tile = im.resize((x1 - x0, y1 - y0), Image.Resampling.BICUBIC, box)
                tiled.paste(tile, (x0, y0))
        return tiled

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_tiles(self):
        if False:
            print('Hello World!')
        with Image.open('Tests/images/flower.jpg') as im:
            assert im.size == (480, 360)
            dst_size = (251, 188)
            reference = im.resize(dst_size, Image.Resampling.BICUBIC)
            for tiles in [(1, 1), (3, 3), (9, 7), (100, 100)]:
                tiled = self.resize_tiled(im, dst_size, *tiles)
                assert_image_similar(reference, tiled, 0.01)

    @mark_if_feature_version(pytest.mark.valgrind_known_error, 'libjpeg_turbo', '2.0', reason='Known Failing')
    def test_subsample(self):
        if False:
            for i in range(10):
                print('nop')
        with Image.open('Tests/images/flower.jpg') as im:
            assert im.size == (480, 360)
            dst_size = (48, 36)
            reference = im.crop((0, 0, 473, 353)).resize(dst_size, Image.Resampling.BICUBIC)
            supersampled = im.resize((60, 45), Image.Resampling.BOX)
        with_box = supersampled.resize(dst_size, Image.Resampling.BICUBIC, (0, 0, 59.125, 44.125))
        without_box = supersampled.resize(dst_size, Image.Resampling.BICUBIC)
        assert_image_similar(reference, with_box, 6)
        with pytest.raises(AssertionError, match='difference 29\\.'):
            assert_image_similar(reference, without_box, 5)

    @pytest.mark.parametrize('mode', ('RGB', 'L', 'RGBA', 'LA', 'I', ''))
    @pytest.mark.parametrize('resample', (Image.Resampling.NEAREST, Image.Resampling.BILINEAR))
    def test_formats(self, mode, resample):
        if False:
            i = 10
            return i + 15
        im = hopper(mode)
        box = (20, 20, im.size[0] - 20, im.size[1] - 20)
        with_box = im.resize((32, 32), resample, box)
        cropped = im.crop(box).resize((32, 32), resample)
        assert_image_similar(cropped, with_box, 0.4)

    def test_passthrough(self):
        if False:
            return 10
        im = hopper()
        for (size, box) in [((40, 50), (0, 0, 40, 50)), ((40, 50), (0, 10, 40, 60)), ((40, 50), (10, 0, 50, 50)), ((40, 50), (10, 20, 50, 70))]:
            res = im.resize(size, Image.Resampling.LANCZOS, box)
            assert res.size == size
            assert_image_equal(res, im.crop(box), f'>>> {size} {box}')

    def test_no_passthrough(self):
        if False:
            return 10
        im = hopper()
        for (size, box) in [((40, 50), (0.4, 0.4, 40.4, 50.4)), ((40, 50), (0.4, 10.4, 40.4, 60.4)), ((40, 50), (10.4, 0.4, 50.4, 50.4)), ((40, 50), (10.4, 20.4, 50.4, 70.4))]:
            res = im.resize(size, Image.Resampling.LANCZOS, box)
            assert res.size == size
            with pytest.raises(AssertionError, match='difference \\d'):
                assert_image_similar(res, im.crop(box), 20, f'>>> {size} {box}')

    @pytest.mark.parametrize('flt', (Image.Resampling.NEAREST, Image.Resampling.BICUBIC))
    def test_skip_horizontal(self, flt):
        if False:
            i = 10
            return i + 15
        im = hopper()
        for (size, box) in [((40, 50), (0, 0, 40, 90)), ((40, 50), (0, 20, 40, 90)), ((40, 50), (10, 0, 50, 90)), ((40, 50), (10, 20, 50, 90))]:
            res = im.resize(size, flt, box)
            assert res.size == size
            assert_image_similar(res, im.crop(box).resize(size, flt), 0.4, f'>>> {size} {box} {flt}')

    @pytest.mark.parametrize('flt', (Image.Resampling.NEAREST, Image.Resampling.BICUBIC))
    def test_skip_vertical(self, flt):
        if False:
            while True:
                i = 10
        im = hopper()
        for (size, box) in [((40, 50), (0, 0, 90, 50)), ((40, 50), (20, 0, 90, 50)), ((40, 50), (0, 10, 90, 60)), ((40, 50), (20, 10, 90, 60))]:
            res = im.resize(size, flt, box)
            assert res.size == size
            assert_image_similar(res, im.crop(box).resize(size, flt), 0.4, f'>>> {size} {box} {flt}')