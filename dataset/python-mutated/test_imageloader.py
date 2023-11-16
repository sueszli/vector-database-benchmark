import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
DEBUG = False
ASSETDIR = 'image-testsuite'
LOADERS = {x.__name__: x for x in ImageLoader.loaders}
if 'ImageLoaderPygame' not in LOADERS:
    try:
        from kivy.core.image.img_pygame import ImageLoaderPygame
        LOADERS['ImageLoaderPygame'] = ImageLoaderPygame
    except:
        pass
v0_PIXELS = {'w': [255, 255, 255], 'x': [0, 0, 0], 'r': [255, 0, 0], 'g': [0, 255, 0], 'b': [0, 0, 255], 'y': [255, 255, 0], 'c': [0, 255, 255], 'p': [255, 0, 255], '0': [0, 0, 0], '1': [17, 17, 17], '2': [34, 34, 34], '3': [51, 51, 51], '4': [68, 68, 68], '5': [85, 85, 85], '6': [102, 102, 102], '7': [119, 119, 119], '8': [136, 136, 136], '9': [153, 153, 153], 'A': [170, 170, 170], 'B': [187, 187, 187], 'C': [204, 204, 204], 'D': [221, 221, 221], 'E': [238, 238, 238], 'F': [255, 255, 255]}
v0_FILE_RE = re.compile('^v0_(\\d+)x(\\d+)_([wxrgbycptA-F0-9]+)_([0-9a-fA-F]{2})_([a-zA-Z0-9-]+)_([a-zA-Z0-9-]+)_([a-zA-Z0-9-]+)\\.([a-z]+)$')

def asset(*fn):
    if False:
        for i in range(10):
            print('nop')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *fn))

def has_alpha(fmt):
    if False:
        for i in range(10):
            print('nop')
    return fmt in ('rgba', 'bgra', 'argb', 'abgr')

def bytes_per_pixel(fmt):
    if False:
        while True:
            i = 10
    if fmt in ('rgb', 'bgr'):
        return 3
    if fmt in ('rgba', 'bgra', 'argb', 'abgr'):
        return 4
    raise Exception('bytes_per_pixel: unknown format {}'.format(fmt))

def get_pixel_alpha(pix, fmt):
    if False:
        while True:
            i = 10
    if fmt in ('rgba', 'bgra'):
        return pix[3]
    elif fmt in ('abgr', 'argb'):
        return pix[0]
    return 255

def rgba_to(pix_in, target_fmt, w, h, pitch=None):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(pix_in, (bytes, bytearray)):
        pix_in = bytearray(pix_in)
    assert w > 0 and h > 0, 'Must specify w and h'
    assert len(pix_in) == w * h * 4, 'Invalid rgba data {}'.format(pix_in)
    assert target_fmt in ('rgba', 'bgra', 'argb', 'abgr', 'rgb', 'bgr')
    if target_fmt == 'rgba':
        return pix_in
    pixels = [pix_in[i:i + 4] for i in range(0, len(pix_in), 4)]
    if target_fmt == 'bgra':
        return b''.join([bytes(p[:3][::-1] + p[3:]) for p in pixels])
    elif target_fmt == 'abgr':
        return b''.join([bytes(p[3:] + p[:3][::-1]) for p in pixels])
    elif target_fmt == 'argb':
        return b''.join([bytes(p[3:] + p[:3]) for p in pixels])
    if pitch is None:
        pitch = 3 * w + 3 & ~3
    elif pitch == 0:
        pitch = 3 * w
    out = b''
    padding = b'\x00' * (pitch - w * 3)
    for row in [pix_in[i:i + w * 4] for i in range(0, len(pix_in), w * 4)]:
        pixelrow = [row[i:i + 4] for i in range(0, len(row), 4)]
        if target_fmt == 'rgb':
            out += b''.join([bytes(p[:3]) for p in pixelrow])
        elif target_fmt == 'bgr':
            out += b''.join([bytes(p[:3][::-1]) for p in pixelrow])
        out += padding
    return out

def match_prediction(pixels, fmt, fd, pitch):
    if False:
        i = 10
        return i + 15
    assert len(fd['alpha']) == 2
    assert len(fd['pattern']) > 0
    bpp = bytes_per_pixel(fmt)
    rowlen = fd['w'] * bpp
    if pitch is None:
        pitch = rowlen + 3 & ~3
    elif pitch == 0:
        pitch = fd['w'] * bpp
    pitchalign = pitch - rowlen
    errors = []
    fail = errors.append
    if len(pixels) != pitch * fd['h']:
        fail('Pitch error: pitch {} * {} height != {} pixelbytes'.format(pitch, fd['h'], len(pixels)))
    ptr = 0
    pixnum = 0
    for char in fd['pattern']:
        pix = list(bytearray(pixels[ptr:ptr + bpp]))
        if len(pix) != bpp:
            fail('Want {} bytes per pixel, got {}: {}'.format(bpp, len(pix), pix))
            break
        if char == 't':
            if get_pixel_alpha(pix, fmt) != 0:
                fail("pixel {} nonzero 't' pixel alpha {:02X}: {}".format(pixnum, get_pixel_alpha(pix, fmt), pix))
        else:
            srcpix = v0_PIXELS[char] + list(bytearray.fromhex(fd['alpha']))
            predict = rgba_to(srcpix, fmt, 1, 1, pitch=0)
            predict = list(bytearray(predict))
            if not predict or not pix or predict != pix:
                fail('pixel {} {} format mismatch: want {} ({}) -- got {}'.format(pixnum, fmt, predict, char, pix))
        if pitchalign and (pixnum + 1) % fd['w'] == 0:
            check = list(bytearray(pixels[ptr + bpp:ptr + bpp + pitchalign]))
            if check != [0] * pitchalign:
                fail('Want {} 0x00 pitch align pixnum={}, pos={} got: {}'.format(pitchalign, pixnum, ptr + bpp, check))
            ptr += pitchalign
        ptr += bpp
        pixnum += 1
    if ptr != len(pixels):
        fail('Excess data: pixnum={} ptr={} bytes={}, bpp={} pitchalign={}'.format(pixnum, ptr, len(pixels), bpp, pitchalign))
    return (len(errors) == 0, errors)

class _TestContext(object):

    def __init__(self, loadercls):
        if False:
            i = 10
            return i + 15
        self.loadercls = loadercls
        self._fd = None
        self._fn = None
        self._ok = 0
        self._skip = 0
        self._fail = 0
        self._stats = defaultdict(dict)

    @property
    def stats(self):
        if False:
            while True:
                i = 10
        return self._stats

    @property
    def results(self):
        if False:
            print('Hello World!')
        return (self._ok, self._skip, self._fail, self._stats)

    def start(self, fn, fd):
        if False:
            return 10
        assert not self._fn, 'unexpected ctx.start(), already started'
        assert isinstance(fd, dict)
        self._fn = fn
        self._fd = fd

    def end(self, fn=None):
        if False:
            while True:
                i = 10
        assert not fn or self._fn == fn, 'unexpected ctx.end(), fn mismatch'
        self._fn = None
        self._fd = None

    def ok(self, info):
        if False:
            while True:
                i = 10
        assert self._fn, 'unexpected ctx.ok(), fn=None'
        self._ok += 1
        self.dbg('PASS', info)
        self._incstat('ok')
        self.end(self._fn)

    def skip(self, info):
        if False:
            return 10
        assert self._fn, 'unexpected ctx.skip(), fn=None'
        self._skip += 1
        self.dbg('SKIP', info)
        self._incstat('skip')
        self.end(self._fn)

    def fail(self, info):
        if False:
            i = 10
            return i + 15
        assert self._fn, 'unexpected ctx.fail(), fn=None'
        self._fail += 1
        self.dbg('FAIL', info)
        self._incstat('fail')
        self.end(self._fn)

    def dbg(self, msgtype, info):
        if False:
            while True:
                i = 10
        assert self._fn, 'unexpected ctx.dbg(), fn=None'
        if DEBUG:
            print('{} {} {}: {}'.format(self.loadercls.__name__, msgtype, self._fn, info))

    def _incstat(self, s):
        if False:
            return 10
        assert self._fd, 'unexpected ctx._incstat(), fd=None'
        fd = self._fd

        def IS(key):
            if False:
                for i in range(10):
                    print('nop')
            self._stats.setdefault(s, defaultdict(int))[key] += 1
        IS('total')
        IS('extension:{}'.format(fd['ext']))
        IS('encoder:{}'.format(fd['encoder']))
        IS('fmtinfo:{}'.format(fd['fmtinfo']))
        IS('testname:{}'.format(fd['testname']))
        IS('testname+ext:{}+{}'.format(fd['testname'], fd['ext']))
        IS('encoder+ext:{}+{}'.format(fd['encoder'], fd['ext']))
        IS('encoder+testname:{}+{}'.format(fd['encoder'], fd['testname']))
        IS('fmtinfo+ext:{}+{}'.format(fd['fmtinfo'], fd['ext']))

@unittest.skipIf(not os.path.isdir(asset(ASSETDIR)), "Need 'make image-testsuite' to run test")
class ImageLoaderTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._context = None
        self._prepare_images()

    def tearDown(self):
        if False:
            print('Hello World!')
        if not DEBUG or not self._context:
            return
        ctx = self._context
        il = ctx.loadercls.__name__
        stats = ctx.stats
        keys = set([k for x in stats.values() for k in x.keys()])
        sg = stats.get
        for k in sorted(keys):
            (ok, skip, fail) = (sg('ok', {}), sg('skip', {}), sg('fail', {}))
            print('REPORT {} {}: ok={}, skip={}, fail={}'.format(il, k, ok.get(k, 0), skip.get(k, 0), fail.get(k, 0)))

    def _prepare_images(self):
        if False:
            print('Hello World!')
        if hasattr(self, '_image_files'):
            return
        self._image_files = {}
        for filename in os.listdir(asset(ASSETDIR)):
            matches = v0_FILE_RE.match(filename)
            if not matches:
                continue
            (w, h, pat, alpha, fmtinfo, tst, encoder, ext) = matches.groups()
            self._image_files[filename] = {'filename': filename, 'w': int(w), 'h': int(h), 'pattern': pat, 'alpha': alpha, 'fmtinfo': fmtinfo, 'testname': tst, 'encoder': encoder, 'ext': ext, 'require_alpha': 'BINARY' in tst or 'ALPHA' in tst}

    def _test_imageloader(self, loadercls, extensions=None):
        if False:
            i = 10
            return i + 15
        if not loadercls:
            return
        if not extensions:
            extensions = loadercls.extensions()
        ctx = _TestContext(loadercls)
        self._context = ctx
        for filename in sorted(self._image_files.keys()):
            filedata = self._image_files[filename]
            if filedata['ext'] not in extensions:
                continue
            try:
                ctx.start(filename, filedata)
                result = loadercls(asset(ASSETDIR, filename), keep_data=True)
                if not result:
                    raise Exception('invalid result')
            except:
                ctx.skip('Error loading file, result=None')
                continue
            self._test_image(filedata, ctx, loadercls, result)
            ctx.end()
        (ok, skip, fail, stats) = ctx.results
        if fail:
            self.fail('{}: {} passed, {} skipped, {} failed'.format(loadercls.__name__, ok, skip, fail))
        return ctx

    def _test_image(self, fd, ctx, loadercls, imgdata):
        if False:
            while True:
                i = 10
        (w, h, pixels, pitch) = imgdata._data[0].get_mipmap(0)
        fmt = imgdata._data[0].fmt
        if not isinstance(pixels, bytes):
            pixels = bytearray(pixels)

        def debug():
            if False:
                for i in range(10):
                    print('nop')
            if not DEBUG:
                return
            print('    format: {}x{} {}'.format(w, h, fmt))
            print('     pitch: got {}, want {}'.format(pitch, want_pitch))
            print('      want: {} in {}'.format(fd['pattern'], fmt))
            print('       got: {}'.format(bytearray(pixels)))
        want_pitch = pitch == 0 and bytes_per_pixel(fmt) * w or pitch
        if pitch == 0 and bytes_per_pixel(fmt) * w * h != len(pixels):
            ctx.dbg('PITCH', 'pitch=0, expected fmt={} to be unaligned @ ({}bpp) = {} bytes, got {}'.format(fmt, bytes_per_pixel(fmt), bytes_per_pixel(fmt) * w * h, len(pixels)))
        elif pitch and want_pitch != pitch:
            ctx.dbg('PITCH', 'fmt={}, pitch={}, expected {}'.format(fmt, pitch, want_pitch))
        (success, msgs) = match_prediction(pixels, fmt, fd, pitch)
        if not success:
            if not msgs:
                ctx.fail('Unknown error')
            elif len(msgs) == 1:
                ctx.fail(msgs[0])
            else:
                for m in msgs:
                    ctx.dbg('PREDICT', m)
                ctx.fail('{} errors, see debug output: {}'.format(len(msgs), msgs[-1]))
            debug()
        elif fd['require_alpha'] and (not has_alpha(fmt)):
            ctx.fail('Missing expected alpha channel')
            debug()
        elif fd['w'] != w:
            ctx.fail('Width mismatch, want {} got {}'.format(fd['w'], w))
            debug()
        elif fd['h'] != h:
            ctx.fail('Height mismatch, want {} got {}'.format(fd['h'], h))
            debug()
        elif w != 1 and h != 1:
            ctx.fail('v0 test protocol mandates w=1 or h=1')
            debug()
        else:
            ctx.ok('Passed test as {}x{} {}'.format(w, h, fmt))
        sys.stdout.flush()

    def test_ImageLoaderSDL2(self):
        if False:
            for i in range(10):
                print('nop')
        loadercls = LOADERS.get('ImageLoaderSDL2')
        if loadercls:
            exts = list(loadercls.extensions()) + ['gif']
            ctx = self._test_imageloader(loadercls, exts)

    def test_ImageLoaderPIL(self):
        if False:
            i = 10
            return i + 15
        loadercls = LOADERS.get('ImageLoaderPIL')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderPygame(self):
        if False:
            i = 10
            return i + 15
        loadercls = LOADERS.get('ImageLoaderPygame')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderFFPy(self):
        if False:
            i = 10
            return i + 15
        loadercls = LOADERS.get('ImageLoaderFFPy')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderGIF(self):
        if False:
            print('Hello World!')
        loadercls = LOADERS.get('ImageLoaderGIF')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderDDS(self):
        if False:
            print('Hello World!')
        loadercls = LOADERS.get('ImageLoaderDDS')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderTex(self):
        if False:
            print('Hello World!')
        loadercls = LOADERS.get('ImageLoaderTex')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderImageIO(self):
        if False:
            for i in range(10):
                print('nop')
        loadercls = LOADERS.get('ImageLoaderImageIO')
        ctx = self._test_imageloader(loadercls)

    def test_missing_tests(self):
        if False:
            i = 10
            return i + 15
        for loader in ImageLoader.loaders:
            key = 'test_{}'.format(loader.__name__)
            msg = 'Missing ImageLoader test case: {}'.format(key)
            self.assertTrue(hasattr(self, key), msg)
            self.assertTrue(callable(getattr(self, key)), msg)

class ConverterTestCase(unittest.TestCase):

    def test_internal_converter_2x1(self):
        if False:
            i = 10
            return i + 15
        correct = {'rgba': b'\x01\x02\x03\xa1\x04\x05\x06\xa2', 'abgr': b'\xa1\x03\x02\x01\xa2\x06\x05\x04', 'bgra': b'\x03\x02\x01\xa1\x06\x05\x04\xa2', 'argb': b'\xa1\x01\x02\x03\xa2\x04\x05\x06', 'rgb': b'\x01\x02\x03\x04\x05\x06', 'bgr': b'\x03\x02\x01\x06\x05\x04', 'rgb_align4': b'\x01\x02\x03\x04\x05\x06\x00\x00', 'bgr_align4': b'\x03\x02\x01\x06\x05\x04\x00\x00'}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'rgba', 2, 1, 0), src('rgba'))
        self.assertEqual(rgba_to(rgba, 'abgr', 2, 1, 0), src('abgr'))
        self.assertEqual(rgba_to(rgba, 'bgra', 2, 1, 0), src('bgra'))
        self.assertEqual(rgba_to(rgba, 'argb', 2, 1, 0), src('argb'))
        self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, 0), src('rgb'))
        self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, 0), src('bgr'))
        self.assertEqual(rgba_to(rgba, 'rgb', 2, 1, None), src('rgb_align4'))
        self.assertEqual(rgba_to(rgba, 'bgr', 2, 1, None), src('bgr_align4'))

    def test_internal_converter_3x1(self):
        if False:
            print('Hello World!')
        pad6 = b'\x00' * 6
        correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'abgr': b'\xff\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07', 'bgra': b'\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07\xff', 'argb': b'\xff\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t', 'rgb_align2': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00' + pad6, 'bgr_align8': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00' + pad6}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'bgra', 3, 1, 0), src('bgra'))
        self.assertEqual(rgba_to(rgba, 'argb', 3, 1, 0), src('argb'))
        self.assertEqual(rgba_to(rgba, 'abgr', 3, 1, 0), src('abgr'))
        self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 10), src('rgb_align2'))
        self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 10), src('bgr_align2'))
        self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 16), src('rgb_align8'))
        self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 16), src('bgr_align8'))

    def test_internal_converter_1x3(self):
        if False:
            for i in range(10):
                print('nop')
        pad5 = b'\x00' * 5
        correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'rgb_raw': b'\x01\x02\x03\x04\x05\x06\x07\x08\t', 'bgr_raw': b'\x03\x02\x01\x06\x05\x04\t\x08\x07', 'rgb_align2': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align4': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align4': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03' + pad5 + b'\x04\x05\x06' + pad5 + b'\x07\x08\t' + pad5, 'bgr_align8': b'\x03\x02\x01' + pad5 + b'\x06\x05\x04' + pad5 + b'\t\x08\x07' + pad5}
        src = correct.get
        rgba = src('rgba')
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 4), src('rgb_align2'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 4), src('bgr_align2'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, None), src('rgb_align4'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, None), src('bgr_align4'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 0), src('rgb_raw'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 0), src('bgr_raw'))
        self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 8), src('rgb_align8'))
        self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 8), src('bgr_align8'))
if __name__ == '__main__':
    accept_filter = ['ImageLoader{}'.format(x) for x in sys.argv[1:]]
    if accept_filter:
        LOADERS = {x: LOADERS[x] for x in accept_filter}
    DEBUG = True
    unittest.main(argv=sys.argv[:1])