"""Miscellaneous functions"""
import numpy as np

def _fast_cross_3d(x, y):
    if False:
        print('Hello World!')
    'Compute cross product between list of 3D vectors\n\n    Much faster than np.cross() when the number of cross products\n    becomes large (>500). This is because np.cross() methods become\n    less memory efficient at this stage.\n\n    Parameters\n    ----------\n    x : array\n        Input array 1.\n    y : array\n        Input array 2.\n\n    Returns\n    -------\n    z : array\n        Cross product of x and y.\n\n    Notes\n    -----\n    x and y must both be 2D row vectors. One must have length 1, or both\n    lengths must match.\n    '
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == 3
    assert y.shape[1] == 3
    assert (x.shape[0] == 1 or y.shape[0] == 1) or x.shape[0] == y.shape[0]
    if max([x.shape[0], y.shape[0]]) >= 500:
        return np.c_[x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1], x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2], x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]]
    else:
        return np.cross(x, y)

def _calculate_normals(rr, tris):
    if False:
        for i in range(10):
            print('nop')
    'Efficiently compute vertex normals for triangulated surface'
    rr = rr.astype(np.float64)
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = _fast_cross_3d(r2 - r1, r3 - r1)
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    size[size == 0] = 1.0
    tri_nn /= size[:, np.newaxis]
    npts = len(rr)
    nn = np.zeros((npts, 3))
    for verts in tris.T:
        for idx in range(3):
            nn[:, idx] += np.bincount(verts.astype(np.int32), tri_nn[:, idx], minlength=npts)
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0
    nn /= size[:, np.newaxis]
    return nn

def resize(image, shape, kind='linear'):
    if False:
        print('Hello World!')
    'Resize an image\n\n    Parameters\n    ----------\n    image : ndarray\n        Array of shape (N, M, ...).\n    shape : tuple\n        2-element shape.\n    kind : str\n        Interpolation, either "linear" or "nearest".\n\n    Returns\n    -------\n    scaled_image : ndarray\n        New image, will have dtype np.float64.\n    '
    image = np.array(image, float)
    shape = np.array(shape, int)
    if shape.ndim != 1 or shape.size != 2:
        raise ValueError('shape must have two elements')
    if image.ndim < 2:
        raise ValueError('image must have two dimensions')
    if not isinstance(kind, str) or kind not in ('nearest', 'linear'):
        raise ValueError('mode must be "nearest" or "linear"')
    r = np.linspace(0, image.shape[0] - 1, shape[0])
    c = np.linspace(0, image.shape[1] - 1, shape[1])
    if kind == 'linear':
        r_0 = np.floor(r).astype(int)
        c_0 = np.floor(c).astype(int)
        r_1 = r_0 + 1
        c_1 = c_0 + 1
        top = (r_1 - r)[:, np.newaxis]
        bot = (r - r_0)[:, np.newaxis]
        lef = (c - c_0)[np.newaxis, :]
        rig = (c_1 - c)[np.newaxis, :]
        c_1 = np.minimum(c_1, image.shape[1] - 1)
        r_1 = np.minimum(r_1, image.shape[0] - 1)
        for arr in (top, bot, lef, rig):
            arr.shape = arr.shape + (1,) * (image.ndim - 2)
        out = top * rig * image[r_0][:, c_0, ...]
        out += bot * rig * image[r_1][:, c_0, ...]
        out += top * lef * image[r_0][:, c_1, ...]
        out += bot * lef * image[r_1][:, c_1, ...]
    else:
        r = np.round(r).astype(int)
        c = np.round(c).astype(int)
        out = image[r][:, c, ...]
    return out