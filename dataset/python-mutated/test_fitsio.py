"""
This test file uses the https://github.com/esheldon/fitsio package to verify
our compression and decompression routines against the implementation in
cfitsio.

*Note*: The fitsio library is GPL licensed, therefore it could be interpreted
 that so is this test file. Given that this test file isn't imported anywhere
 else in the code this shouldn't cause us any issues. Please bear this in mind
 when editing this file.
"""
import os
import numpy as np
import pytest
from astropy.io import fits
from .conftest import _expand, fitsio_param_to_astropy_param
if 'ASTROPY_ALWAYS_TEST_FITSIO' in os.environ:
    import fitsio
else:
    fitsio = pytest.importorskip('fitsio')

@pytest.fixture(scope='module', params=_expand([((10,),), ((5,), (1,), (3,))], [((12, 12),), ((1, 12), (4, 5), (6, 6), None)], [((15, 15),), ((1, 15), (5, 1), (5, 5))], [((15, 15, 15),), ((5, 5, 1), (5, 7, 1), (1, 5, 4), (1, 1, 15), (15, 1, 5))], [((4, 4, 5), (5, 5, 5)), ((5, 5, 1), None)]), ids=lambda x: f'shape: {x[0]} tile_dims: {x[1]}')
def array_shapes_tile_dims(request, compression_type):
    if False:
        for i in range(10):
            print('nop')
    (shape, tile_dims) = request.param
    if compression_type == 'HCOMPRESS_1':
        if len(shape) < 2 or np.count_nonzero(np.array(tile_dims) != 1) != 2 or tile_dims[0] == 1 or (tile_dims[1] == 1) or (np.count_nonzero(np.array(shape[:2]) % tile_dims[:2]) != 0):
            pytest.xfail('HCOMPRESS requires 2D tiles, from the first twodimensions, and an integer number of tiles along the first twoaxes.')
    return (shape, tile_dims)

@pytest.fixture(scope='module')
def tile_dims(array_shapes_tile_dims):
    if False:
        return 10
    return array_shapes_tile_dims[1]

@pytest.fixture(scope='module')
def data_shape(array_shapes_tile_dims):
    if False:
        print('Hello World!')
    return array_shapes_tile_dims[0]

@pytest.fixture(scope='module')
def base_original_data(data_shape, dtype, numpy_rng, compression_type):
    if False:
        print('Hello World!')
    random = numpy_rng.uniform(high=255, size=data_shape)
    random.ravel()[0] = 0.0
    if compression_type.startswith('HCOMPRESS') and 'i2' in dtype or 'u1' in dtype:
        random = np.arange(np.prod(data_shape)).reshape(data_shape)
    return random.astype(dtype)

@pytest.fixture(scope='module')
def fitsio_compressed_file_path(tmp_path_factory, comp_param_dtype, base_original_data, data_shape, tile_dims):
    if False:
        while True:
            i = 10
    (compression_type, param, dtype) = comp_param_dtype
    if base_original_data.ndim > 2 and 'u1' in dtype and (compression_type == 'HCOMPRESS_1'):
        pytest.xfail("fitsio won't write these")
    if compression_type == 'PLIO_1' and 'f' in dtype:
        pytest.xfail('fitsio fails to write these')
    if compression_type == 'NOCOMPRESS':
        pytest.xfail('fitsio does not support NOCOMPRESS')
    if compression_type == 'HCOMPRESS_1' and 'f' in dtype and (param.get('qmethod', None) == 2):
        pytest.xfail('fitsio writes these files with very large/incorrect zzero values')
    tmp_path = tmp_path_factory.mktemp('fitsio')
    original_data = base_original_data.astype(dtype)
    filename = tmp_path / f'{compression_type}_{dtype}.fits'
    fits = fitsio.FITS(filename, 'rw')
    fits.write(original_data, compress=compression_type, tile_dims=tile_dims, **param)
    return filename

@pytest.fixture(scope='module')
def astropy_compressed_file_path(comp_param_dtype, tmp_path_factory, base_original_data, data_shape, tile_dims):
    if False:
        i = 10
        return i + 15
    (compression_type, param, dtype) = comp_param_dtype
    original_data = base_original_data.astype(dtype)
    tmp_path = tmp_path_factory.mktemp('astropy')
    filename = tmp_path / f'{compression_type}_{dtype}.fits'
    param = fitsio_param_to_astropy_param(param)
    hdu = fits.CompImageHDU(data=original_data, compression_type=compression_type, tile_shape=None if tile_dims is None else tile_dims[::-1], **param)
    hdu.writeto(filename)
    return filename

def test_decompress(fitsio_compressed_file_path, comp_param_dtype):
    if False:
        for i in range(10):
            print('nop')
    (compression_type, param, dtype) = comp_param_dtype
    with fits.open(fitsio_compressed_file_path) as hdul:
        data = hdul[1].data
        assert hdul[1]._header['ZCMPTYPE'].replace('ONE', '1') == compression_type
        assert hdul[1].data.dtype.kind == np.dtype(dtype).kind
        assert hdul[1].data.dtype.itemsize == np.dtype(dtype).itemsize
    fts = fitsio.FITS(fitsio_compressed_file_path)
    data2 = fts[1].read()
    np.testing.assert_allclose(data, data2)
    if param.get('qmethod', None) == 2:
        assert data.ravel()[0] == 0.0

def test_compress(astropy_compressed_file_path, compression_type, dtype):
    if False:
        print('Hello World!')
    if compression_type == 'NOCOMPRESS':
        pytest.xfail('fitsio does not support NOCOMPRESS')
    fts = fitsio.FITS(astropy_compressed_file_path, 'r')
    header = fts[1].read_header()
    data = fts[1].read()
    assert header['ZCMPTYPE'] == compression_type
    assert data.dtype.kind == np.dtype(dtype).kind
    assert data.dtype.itemsize == np.dtype(dtype).itemsize
    with fits.open(astropy_compressed_file_path) as hdul:
        np.testing.assert_allclose(data, hdul[1].data)