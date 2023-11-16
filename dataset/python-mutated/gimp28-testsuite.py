import os
import re
import random
from gimpfu import *
TESTSUITE_CONFIG = {'OPAQUE': {'alpha': [255], 'patterns': ('wxrgbcyp', None, None), RGB_IMAGE: ['xcf', 'png', 'tga', 'tiff', 'ppm', 'sgi', 'pcx', 'fits', 'ras'], INDEXED_IMAGE: ['xcf', 'png', 'tga', 'tiff', 'ppm', 'gif', 'ras']}, 'GRAY-OPAQUE': {'alpha': [255], 'patterns': ('0123456789ABCDEF', None, None), GRAY_IMAGE: ['xcf', 'png', 'tga', 'tiff', 'pgm', 'sgi', 'fits', 'ras'], INDEXED_IMAGE: ['xcf', 'png', 'tga', 'tiff', 'pgm', 'fits', 'ras']}, 'BINARY': {'alpha': [255], 'patterns': ('twrgbcyp', 't', None), RGBA_IMAGE: ['xcf', 'png', 'tga', 'ico', 'sgi'], INDEXEDA_IMAGE: ['xcf', 'png', 'tga', 'gif']}, 'GRAY-BINARY': {'alpha': [255], 'patterns': ('t123456789ABCDEF', 't', None), GRAYA_IMAGE: ['xcf', 'tga', 'png', 'sgi'], INDEXEDA_IMAGE: ['xcf', 'tga', 'png']}, 'ALPHA': {'alpha': [127, 240], 'patterns': ('twxrgbcyp', None, None), RGBA_IMAGE: ['xcf', 'png', 'tga', 'sgi']}, 'GRAY-ALPHA': {'alpha': [127, 240], 'patterns': ('t0123456789ABCDEF', None, None), GRAYA_IMAGE: ['xcf', 'png', 'tga', 'sgi']}}
v0_PIXELS = {'w': [255, 255, 255], 'x': [0, 0, 0], 'r': [255, 0, 0], 'g': [0, 255, 0], 'b': [0, 0, 255], 'y': [255, 255, 0], 'c': [0, 255, 255], 'p': [255, 0, 255], '0': [0, 0, 0], '1': [17, 17, 17], '2': [34, 34, 34], '3': [51, 51, 51], '4': [68, 68, 68], '5': [85, 85, 85], '6': [102, 102, 102], '7': [119, 119, 119], '8': [136, 136, 136], '9': [153, 153, 153], 'A': [170, 170, 170], 'B': [187, 187, 187], 'C': [204, 204, 204], 'D': [221, 221, 221], 'E': [238, 238, 238], 'F': [255, 255, 255]}

def v0_pattern_pixel(char, alpha, fmt):
    if False:
        return 10
    if fmt == 'rgba':
        if char == 't':
            return [0, 0, 0, 0]
        return v0_PIXELS[char] + [alpha]
    if fmt == 'rgb':
        if char == 't':
            return [0, 0, 0]
        return v0_PIXELS[char]
    if fmt == 'gray':
        assert char in '0123456789ABCDEF'
        return [v0_PIXELS[char][0]]
    if fmt == 'graya':
        assert char in 't0123456789ABCDEF'
        if char == 't':
            return [0, 0]
        return [v0_PIXELS[char][0]] + [alpha]
    raise Exception('v0_pattern_pixel: unknown format {}'.format(fmt))

def v0_filename(w, h, pat, alpha, fmtinfo, testname, ext):
    if False:
        return 10
    return 'v0_{}x{}_{}_{:02X}_{}_{}_gimp.{}'.format(w, h, pat, alpha, fmtinfo, testname, ext)

def save_image(dirname, img, lyr, w, h, pat, alpha, v0_fmtinfo, testname, ext):
    if False:
        i = 10
        return i + 15

    def filename(fmtinfo_in=None):
        if False:
            return 10
        fmtinfo = fmtinfo_in and v0_fmtinfo + '-' + fmtinfo_in or v0_fmtinfo
        return v0_filename(w, h, pat, alpha, fmtinfo, testname, ext)

    def savepath(fn):
        if False:
            i = 10
            return i + 15
        return os.path.join(dirname, fn)
    if ext in ('ppm', 'pgm', 'pbm', 'pnm', 'pam'):
        fn = filename('ASCII')
        pdb.file_pnm_save(img, lyr, savepath(fn), fn, 0)
        fn = filename('RAW')
        pdb.file_pnm_save(img, lyr, savepath(fn), fn, 1)
    elif ext == 'tga':
        fn = filename('RAW')
        pdb.file_tga_save(img, lyr, savepath(fn), fn, 0, 0)
        fn = filename('RLE')
        pdb.file_tga_save(img, lyr, savepath(fn), fn, 1, 0)
    elif ext == 'gif':
        fn = filename('I0')
        pdb.file_gif_save(img, lyr, savepath(fn), fn, 0, 0, 0, 0)
        fn = filename('I1')
        pdb.file_gif_save(img, lyr, savepath(fn), fn, 1, 0, 0, 0)
    elif ext == 'png':
        bits = [0, 1]
        for (i, b, g) in [(i, b, g) for i in bits for b in bits for g in bits]:
            fn = filename('I{}B{}G{}'.format(i, b, g))
            pdb.file_png_save(img, lyr, savepath(fn), fn, i, 9, b, g, 1, 1, 1)
    elif ext == 'sgi':
        fn = filename('RAW')
        pdb.file_sgi_save(img, lyr, savepath(fn), fn, 0)
        fn = filename('RLE')
        pdb.file_sgi_save(img, lyr, savepath(fn), fn, 1)
        fn = filename('ARLE')
        pdb.file_sgi_save(img, lyr, savepath(fn), fn, 2)
    elif ext == 'tiff':
        fn = filename('RAW')
        pdb.file_tiff_save(img, lyr, savepath(fn), fn, 0)
        fn = filename('LZW')
        pdb.file_tiff_save(img, lyr, savepath(fn), fn, 1)
        fn = filename('PACKBITS')
        pdb.file_tiff_save(img, lyr, savepath(fn), fn, 2)
        fn = filename('DEFLATE')
        pdb.file_tiff_save(img, lyr, savepath(fn), fn, 3)
    elif ext == 'ras':
        fn = filename('RAW')
        pdb.file_sunras_save(img, lyr, savepath(fn), fn, 0)
        fn = filename('RLE')
        pdb.file_sunras_save(img, lyr, savepath(fn), fn, 1)
    else:
        fn = filename()
        pdb.gimp_file_save(img, lyr, savepath(fn), fn)

def draw_pattern(lyr, pat, alpha, direction, pixelgetter):
    if False:
        for i in range(10):
            print('nop')
    assert 0 <= alpha <= 255
    assert re.match('[twxrgbycp0-9A-F]+$', pat)
    assert direction in ('x', 'y', 'width', 'height')
    dirx = direction in ('x', 'width')
    for i in range(0, len(pat)):
        pixel = pixelgetter(pat[i], alpha)
        if dirx:
            pdb.gimp_drawable_set_pixel(lyr, i, 0, len(pixel), pixel)
        else:
            pdb.gimp_drawable_set_pixel(lyr, 0, i, len(pixel), pixel)

def make_images(testname, pattern, alpha, layertype_in, extensions, dirname):
    if False:
        return 10
    assert testname.upper() == testname
    assert len(pattern) > 0
    assert len(extensions) > 0
    assert isinstance(extensions, (list, tuple))
    assert re.match('[wxtrgbcypA-F0-9]+$', pattern)
    test_alpha = 'ALPHA' in testname or 'BINARY' in testname
    grayscale = 'GRAY' in testname
    (imgtype, v0_fmtinfo) = {GRAY_IMAGE: (GRAY, 'BPP1G'), GRAYA_IMAGE: (GRAY, 'BPP2GA'), RGB_IMAGE: (RGB, 'BPP3'), RGBA_IMAGE: (RGB, 'BPP4'), INDEXED_IMAGE: (grayscale and GRAY or RGB, 'IX'), INDEXEDA_IMAGE: (grayscale and GRAY or RGB, 'IXA')}[layertype_in]
    PP = v0_pattern_pixel
    pixelgetter = {GRAY_IMAGE: lambda c, a: PP(c, a, 'gray'), GRAYA_IMAGE: lambda c, a: PP(c, a, 'graya'), RGB_IMAGE: lambda c, a: PP(c, a, 'rgb'), RGBA_IMAGE: lambda c, a: PP(c, a, 'rgba'), INDEXED_IMAGE: lambda c, a: PP(c, a, grayscale and 'gray' or 'rgb'), INDEXEDA_IMAGE: lambda c, a: PP(c, a, grayscale and 'graya' or 'rgba')}[layertype_in]
    layertype = {INDEXED_IMAGE: grayscale and GRAY_IMAGE or RGB_IMAGE, INDEXEDA_IMAGE: grayscale and GRAYA_IMAGE or RGBA_IMAGE}.get(layertype_in, layertype_in)
    for direction in 'xy':
        (w, h) = direction == 'x' and (len(pattern), 1) or (1, len(pattern))
        img = pdb.gimp_image_new(w, h, imgtype)
        lyr = pdb.gimp_layer_new(img, w, h, layertype, 'P', 100, NORMAL_MODE)
        if test_alpha:
            pdb.gimp_layer_add_alpha(lyr)
            pdb.gimp_drawable_fill(lyr, TRANSPARENT_FILL)
        pdb.gimp_image_add_layer(img, lyr, 0)
        draw_pattern(lyr, pattern, alpha, direction, pixelgetter)
        if layertype_in in (INDEXED_IMAGE, INDEXEDA_IMAGE):
            colors = len(set(pattern)) + (test_alpha and 1 or 0)
            pdb.gimp_convert_indexed(img, 0, 0, colors, 0, 0, 'ignored')
        for ext in extensions:
            save_image(dirname, img, lyr, w, h, pattern, alpha, v0_fmtinfo, testname, ext)

def makepatterns(allow, include=None, exclude=None):
    if False:
        for i in range(10):
            print('nop')
    src = set()
    src.update([x for x in allow])
    src.update([allow[:i] for i in range(1, len(allow) + 1)])
    for i in range(len(allow)):
        (pick1, pick2) = (random.choice(allow), random.choice(allow))
        src.update([pick1 + pick2])
    for i in range(3, 11) + range(14, 18) + range(31, 34):
        src.update([''.join([random.choice(allow) for k in range(i)])])
    out = []
    for srcpat in src:
        if exclude and exclude in srcpat:
            continue
        if include and include not in srcpat:
            out.append(include + srcpat[1:])
            continue
        out.append(srcpat)
    return list(set(out))

def plugin_main(dirname, do_opaque, do_binary, do_alpha):
    if False:
        i = 10
        return i + 15
    if not dirname:
        pdb.gimp_message('No output directory selected, aborting')
        return
    if not os.path.isdir(dirname) or not os.access(dirname, os.W_OK):
        pdb.gimp_message('Invalid / non-writeable output directory, aborting')
        return
    tests = []
    tests.extend({0: ['OPAQUE', 'GRAY-OPAQUE'], 2: ['OPAQUE'], 3: ['GRAY-OPAQUE']}.get(do_opaque, []))
    tests.extend({0: ['BINARY', 'GRAY-BINARY'], 2: ['BINARY'], 3: ['GRAY-BINARY']}.get(do_binary, []))
    tests.extend({0: ['ALPHA', 'GRAY-ALPHA'], 2: ['ALPHA'], 3: ['GRAY-ALPHA']}.get(do_alpha, []))
    suite_cfg = dict(TESTSUITE_CONFIG)
    for (testname, cfg) in suite_cfg.items():
        if testname not in tests:
            continue
        (pchars, inc, exc) = cfg.pop('patterns')
        if not pchars:
            continue
        patterns = makepatterns(pchars, inc, exc)
        for alpha in cfg.pop('alpha', [255]):
            for (layertype, exts) in cfg.items():
                if not exts:
                    continue
                for p in patterns:
                    make_images(testname, p, alpha, layertype, exts, dirname)
register(proc_name='kivy_image_testsuite', help='Creates image test suite for Kivy ImageLoader', blurb='Creates image test suite for Kivy ImageLoader. Warning: This will create thousands of images', author='For kivy.org, Terje Skjaeveland', copyright='Copyright 2017 kivy.org (MIT license)', date='2017', imagetypes='', params=[(PF_DIRNAME, 'outputdir', 'Output directory:', 0), (PF_OPTION, 'opaque', 'OPAQUE tests?', 0, ['All', 'None', 'OPAQUE', 'GRAY-OPAQUE']), (PF_OPTION, 'binary', 'BINARY tests?', 0, ['All', 'None', 'BINARY', 'GRAY-BINARY']), (PF_OPTION, 'alpha', 'ALPHA tests?', 0, ['All', 'None', 'ALPHA', 'GRAY-ALPHA'])], results=[], function=plugin_main, menu='<Image>/Tools/_Kivy image testsuite...', label='Generate images...')
main()