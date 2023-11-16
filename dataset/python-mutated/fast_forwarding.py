from deeplake.constants import ENCODING_DTYPE
import numpy as np
import deeplake
import warnings

def version_compare(v1, v2):
    if False:
        i = 10
        return i + 15
    'Returns -1 if v1 is older than v2, 0 if v1 == v2, and +1 if v1 > v2.'
    arr1 = v1.split('.')
    arr2 = v2.split('.')
    n = len(arr1)
    m = len(arr2)
    arr1 = [int(i) for i in arr1]
    arr2 = [int(i) for i in arr2]
    if n > m:
        for i in range(m, n):
            arr2.append(0)
    elif m > n:
        for i in range(n, m):
            arr1.append(0)
    for i in range(len(arr1)):
        if arr1[i] > arr2[i]:
            return 1
        elif arr2[i] > arr1[i]:
            return -1
    return 0

def _check_version(v):
    if False:
        print('Hello World!')
    'Returns True if no fast forwarding is required (False if otherwise).'
    comparison = version_compare(v, deeplake.__version__)
    if comparison > 0:
        warnings.warn(f"Loading a dataset that was created or updated with a newer version of deeplake. This could lead to corruption or unexpected errors! Dataset version: {v}, current deeplake version: {deeplake.__version__}. It's recommended that you update to a version of deeplake >= {v}.")
    return comparison >= 0

def ffw(func):
    if False:
        return 10
    'Decorator for fast forwarding functions. Will handle checking for valid versions and automatically updating the\n    fast forwarded version to the latest. Also adds an extra parameter to the decoarted function (version).\n    '

    def decor(inp, **kwargs):
        if False:
            return 10
        v = inp.version
        if not _check_version(v):
            out = func(inp, v, **kwargs)
            inp.version = deeplake.__version__
            return out
    return decor

@ffw
def ffw_chunk_id_encoder(chunk_id_encoder, version):
    if False:
        i = 10
        return i + 15
    pass

@ffw
def ffw_dataset_meta(dataset_meta, version):
    if False:
        return 10
    pass

@ffw
def ffw_tensor_meta(tensor_meta, version):
    if False:
        return 10
    versions = ('2.0.2', '2.0.3', '2.0.4', '2.0.5')
    if version in versions and len(tensor_meta.min_shape) == 0:
        tensor_meta.min_shape = [1]
        tensor_meta.max_shape = [1]
        tensor_meta.is_dirty = True
    if not hasattr(tensor_meta, 'chunk_compression'):
        tensor_meta.chunk_compression = None
    if not hasattr(tensor_meta, 'hidden'):
        tensor_meta.hidden = False
    if not hasattr(tensor_meta, 'links'):
        tensor_meta.links = {}
    if not hasattr(tensor_meta, 'is_link'):
        tensor_meta.is_link = False
    if not hasattr(tensor_meta, 'is_sequence'):
        tensor_meta.is_sequence = False
    if not hasattr(tensor_meta, 'typestr'):
        tensor_meta.typestr = None
    required_meta_keys = tensor_meta._required_meta_keys
    tensor_meta._required_meta_keys = tuple(set(required_meta_keys + ('chunk_compression', 'hidden', 'links', 'is_link', 'is_sequence', 'typestr')))
    if version_compare(version, '3.0.15') < 0:
        links = tensor_meta.links
        for k in links:
            l = links[k]
            if 'append' in l:
                l['extend'] = l['append'].replace('append', 'extend')
                del l['append']
            if 'update' in l:
                l['update'] = l['update'].replace('append', 'update')

@ffw
def ffw_chunk(chunk, version):
    if False:
        return 10
    if version in ('2.0.2', '2.0.3', '2.0.4', '2.0.5'):
        shapes = chunk.shapes_encoder
        if shapes.dimensionality == 0:
            a = shapes.array
            if len(a) > 1:
                raise ValueError(f'Cannot fast forward an invalid shapes encoder. The length of the encoding was expected to be == 1, but got {len(a)}.')
            shapes._encoded = np.array([[1, a[0][0]]], dtype=ENCODING_DTYPE)