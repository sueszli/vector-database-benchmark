import sys
from past.builtins import cmp
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy

def _generate_segments(im_orig, scale, sigma, min_size):
    if False:
        while True:
            i = 10
    '\n        segment smallest regions by the algorithm of Felzenswalb and\n        Huttenlocher\n    '
    im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)
    im_orig = numpy.append(im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig

def _sim_colour(r1, r2):
    if False:
        for i in range(10):
            print('nop')
    '\n        calculate the sum of histogram intersection of colour\n    '
    return sum([min(a, b) for (a, b) in zip(r1['hist_c'], r2['hist_c'])])

def _sim_texture(r1, r2):
    if False:
        i = 10
        return i + 15
    '\n        calculate the sum of histogram intersection of texture\n    '
    return sum([min(a, b) for (a, b) in zip(r1['hist_t'], r2['hist_t'])])

def _sim_size(r1, r2, imsize):
    if False:
        print('Hello World!')
    '\n        calculate the size similarity over the image\n    '
    return 1.0 - (r1['size'] + r2['size']) / imsize

def _sim_fill(r1, r2, imsize):
    if False:
        print('Hello World!')
    '\n        calculate the fill similarity over the image\n    '
    bbsize = (max(r1['max_x'], r2['max_x']) - min(r1['min_x'], r2['min_x'])) * (max(r1['max_y'], r2['max_y']) - min(r1['min_y'], r2['min_y']))
    return 1.0 - (bbsize - r1['size'] - r2['size']) / imsize

def _calc_sim(r1, r2, imsize):
    if False:
        while True:
            i = 10
    return _sim_colour(r1, r2) + _sim_texture(r1, r2) + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize)

def _calc_colour_hist(img):
    if False:
        return 10
    '\n        calculate colour histogram for each region\n\n        the size of output histogram will be BINS * COLOUR_CHANNELS(3)\n\n        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]\n\n        extract HSV\n    '
    BINS = 25
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        hist = numpy.concatenate([hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])
    hist = hist / len(img)
    return hist

def _calc_texture_gradient(img):
    if False:
        i = 10
        return i + 15
    '\n        calculate texture gradient for entire image\n\n        The original SelectiveSearch algorithm proposed Gaussian derivative\n        for 8 orientations, but we use LBP instead.\n\n        output will be [height(*)][width(*)]\n    '
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)
    return ret

def _calc_texture_hist(img):
    if False:
        print('Hello World!')
    '\n        calculate texture histogram for each region\n\n        calculate the histogram of gradient for each colours\n        the size of output histogram will be\n            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)\n    '
    BINS = 10
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        fd = img[:, colour_channel]
        hist = numpy.concatenate([hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])
    hist = hist / len(img)
    return hist

def _extract_regions(img):
    if False:
        return 10
    R = {}
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    for (y, i) in enumerate(img):
        for (x, (r, g, b, l)) in enumerate(i):
            if l not in R:
                R[l] = {'min_x': 65535, 'min_y': 65535, 'max_x': 0, 'max_y': 0, 'labels': [l]}
            if R[l]['min_x'] > x:
                R[l]['min_x'] = x
            if R[l]['min_y'] > y:
                R[l]['min_y'] = y
            if R[l]['max_x'] < x:
                R[l]['max_x'] = x
            if R[l]['max_y'] < y:
                R[l]['max_y'] = y
    tex_grad = _calc_texture_gradient(img)
    for (k, v) in R.items():
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]['size'] = len(masked_pixels / 4)
        R[k]['hist_c'] = _calc_colour_hist(masked_pixels)
        R[k]['hist_t'] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
    return R

def _extract_neighbours(regions):
    if False:
        return 10

    def intersect(a, b):
        if False:
            i = 10
            return i + 15
        if a['min_x'] < b['min_x'] < a['max_x'] and a['min_y'] < b['min_y'] < a['max_y'] or (a['min_x'] < b['max_x'] < a['max_x'] and a['min_y'] < b['max_y'] < a['max_y']) or (a['min_x'] < b['min_x'] < a['max_x'] and a['min_y'] < b['max_y'] < a['max_y']) or (a['min_x'] < b['max_x'] < a['max_x'] and a['min_y'] < b['min_y'] < a['max_y']):
            return True
        return False
    R = list(regions.items())
    neighbours = []
    for (cur, a) in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours

def _merge_regions(r1, r2):
    if False:
        while True:
            i = 10
    new_size = r1['size'] + r2['size']
    rt = {'min_x': min(r1['min_x'], r2['min_x']), 'min_y': min(r1['min_y'], r2['min_y']), 'max_x': max(r1['max_x'], r2['max_x']), 'max_y': max(r1['max_y'], r2['max_y']), 'size': new_size, 'hist_c': (r1['hist_c'] * r1['size'] + r2['hist_c'] * r2['size']) / new_size, 'hist_t': (r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new_size, 'labels': r1['labels'] + r2['labels']}
    return rt

def mycmp(x, y):
    if False:
        for i in range(10):
            print('nop')
    return cmp(x[1], y[1])

def cmp_to_key(mycmp):
    if False:
        i = 10
        return i + 15
    'Convert a cmp= function into a key= function'

    class K(object):

        def __init__(self, obj, *args):
            if False:
                while True:
                    i = 10
            self.obj = obj

        def __lt__(self, other):
            if False:
                return 10
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            if False:
                while True:
                    i = 10
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            if False:
                i = 10
                return i + 15
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            if False:
                i = 10
                return i + 15
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return mycmp(self.obj, other.obj) != 0
    return K

def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    if False:
        return 10
    "Selective Search\n\n    Parameters\n    ----------\n        im_orig : ndarray\n            Input image\n        scale : int\n            Free parameter. Higher means larger clusters in felzenszwalb segmentation.\n        sigma : float\n            Width of Gaussian kernel for felzenszwalb segmentation.\n        min_size : int\n            Minimum component size for felzenszwalb segmentation.\n    Returns\n    -------\n        img : ndarray\n            image with region label\n            region label is stored in the 4th value of each pixel [r,g,b,(region)]\n        regions : array of dict\n            [\n                {\n                    'rect': (left, top, right, bottom),\n                    'labels': [...]\n                },\n                ...\n            ]\n    "
    assert im_orig.shape[2] == 3, '3ch image is expected'
    img = _generate_segments(im_orig, scale, sigma, min_size)
    if img is None:
        return (None, {})
    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)
    neighbours = list(_extract_neighbours(R))
    S = {}
    for ((ai, ar), (bi, br)) in neighbours:
        S[ai, bi] = _calc_sim(ar, br, imsize)
    while S != {}:
        if sys.version_info[0] < 3:
            (i, j) = sorted(S.items(), cmp=mycmp)[-1][0]
        else:
            (i, j) = sorted(S.items(), key=cmp_to_key(mycmp))[-1][0]
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])
        key_to_delete = []
        for (k, v) in S.items():
            if i in k or j in k:
                key_to_delete.append(k)
        for k in key_to_delete:
            del S[k]
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[t, n] = _calc_sim(R[t], R[n], imsize)
    regions = []
    for (k, r) in R.items():
        regions.append({'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']), 'size': r['size'], 'labels': r['labels']})
    return (img, regions)