import pickle
import textwrap
from collections import OrderedDict
from itertools import chain, permutations
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from astropy import units as u
from astropy.nddata import NDDataArray
from astropy.nddata import _testing as nd_testing
from astropy.nddata.nddata import NDData
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.utils import NumpyRNGContext
from astropy.utils.compat.optional_deps import HAS_DASK
from astropy.utils.masked import Masked
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS
from .test_nduncertainty import FakeUncertainty

class FakeNumpyArray:
    """
    Class that has a few of the attributes of a numpy array.

    These attributes are checked for by NDData.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def shape(self):
        if False:
            i = 10
            return i + 15
        pass

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        pass

    def __array__(self):
        if False:
            return 10
        pass

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return 'fake'

class MinimalUncertainty:
    """
    Define the minimum attributes acceptable as an uncertainty object.
    """

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._uncertainty = value

    @property
    def uncertainty_type(self):
        if False:
            while True:
                i = 10
        return 'totally and completely fake'

class BadNDDataSubclass(NDData):

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, psf=None):
        if False:
            while True:
                i = 10
        self._data = data
        self._uncertainty = uncertainty
        self._mask = mask
        self._wcs = wcs
        self._psf = psf
        self._unit = unit
        self._meta = meta

def test_uncertainty_setter():
    if False:
        while True:
            i = 10
    nd = NDData([1, 2, 3])
    good_uncertainty = MinimalUncertainty(5)
    nd.uncertainty = good_uncertainty
    assert nd.uncertainty is good_uncertainty
    nd.uncertainty = FakeUncertainty(5)
    assert nd.uncertainty.parent_nddata is nd
    nd = NDData(nd)
    assert isinstance(nd.uncertainty, FakeUncertainty)
    nd.uncertainty = 10
    assert not isinstance(nd.uncertainty, FakeUncertainty)
    assert nd.uncertainty.array == 10

def test_mask_setter():
    if False:
        return 10
    nd = NDData([1, 2, 3])
    nd.mask = True
    assert nd.mask
    nd.mask = False
    assert not nd.mask
    nd = NDData(nd, mask=True)
    assert nd.mask
    nd.mask = False
    assert not nd.mask

def test_nddata_empty():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        NDData()

def test_nddata_init_data_nonarray():
    if False:
        while True:
            i = 10
    inp = [1, 2, 3]
    nd = NDData(inp)
    assert (np.array(inp) == nd.data).all()

def test_nddata_init_data_ndarray():
    if False:
        return 10
    with NumpyRNGContext(123):
        nd = NDData(np.random.random((10, 10)))
    assert nd.data.shape == (10, 10)
    assert nd.data.size == 100
    assert nd.data.dtype == np.dtype(float)
    nd = NDData(np.array([[1, 2, 3], [4, 5, 6]]))
    assert nd.data.size == 6
    assert nd.data.dtype == np.dtype(int)
    a = np.ones((10, 10))
    nd_ref = NDData(a)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] == 0
    a = np.ones((10, 10))
    nd_ref = NDData(a, copy=True)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] != 0

def test_nddata_init_data_maskedarray():
    if False:
        i = 10
        return i + 15
    with NumpyRNGContext(456):
        NDData(np.random.random((10, 10)), mask=np.random.random((10, 10)) > 0.5)
    with NumpyRNGContext(12345):
        a = np.random.randn(100)
        marr = np.ma.masked_where(a > 0, a)
    nd = NDData(marr)
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 123456789
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)
    nd = NDData(marr, copy=True)
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 0
    assert nd.mask[10] != marr.mask[10]
    assert nd.data[11] != marr.data[11]

@pytest.mark.parametrize('data', [np.array([1, 2, 3]), 5])
def test_nddata_init_data_quantity(data):
    if False:
        return 10
    quantity = data * u.adu
    ndd = NDData(quantity)
    assert ndd.unit == quantity.unit
    assert_array_equal(ndd.data, np.array(quantity))
    if ndd.data.size > 1:
        quantity.value[1] = 100
        assert ndd.data[1] == quantity.value[1]
        ndd = NDData(quantity, copy=True)
        quantity.value[1] = 5
        assert ndd.data[1] != quantity.value[1]
    ndd_unit = NDData(data * u.erg, unit=u.J)
    assert ndd_unit.unit == u.J
    np.testing.assert_allclose((ndd_unit.data * ndd_unit.unit).to_value(u.erg), data)

def test_nddata_init_data_masked_quantity():
    if False:
        i = 10
        return i + 15
    a = np.array([2, 3])
    q = a * u.m
    m = False
    mq = Masked(q, mask=m)
    nd = NDData(mq)
    assert_array_equal(nd.data, a)
    assert nd.unit == u.m
    assert not isinstance(nd.data, u.Quantity)
    np.testing.assert_array_equal(nd.mask, np.array(m))

def test_nddata_init_data_nddata():
    if False:
        while True:
            i = 10
    nd1 = NDData(np.array([1]))
    nd2 = NDData(nd1)
    assert nd2.wcs == nd1.wcs
    assert nd2.uncertainty == nd1.uncertainty
    assert nd2.mask == nd1.mask
    assert nd2.unit == nd1.unit
    assert nd2.meta == nd1.meta
    assert nd2.psf == nd1.psf
    nd1 = NDData(np.ones((5, 5)))
    nd2 = NDData(nd1)
    assert nd1.data is nd2.data
    nd2 = NDData(nd1, copy=True)
    nd1.data[2, 3] = 10
    assert nd1.data[2, 3] != nd2.data[2, 3]
    nd1 = NDData(np.array([1]), mask=False, uncertainty=StdDevUncertainty(10), unit=u.s, meta={'dest': 'mordor'}, wcs=WCS(naxis=1), psf=np.array([10]))
    nd2 = NDData(nd1)
    assert nd2.data is nd1.data
    assert nd2.wcs is nd1.wcs
    assert nd2.uncertainty.array == nd1.uncertainty.array
    assert nd2.mask == nd1.mask
    assert nd2.unit == nd1.unit
    assert nd2.meta == nd1.meta
    assert nd2.psf == nd1.psf
    nd3 = NDData(nd1, mask=True, uncertainty=StdDevUncertainty(200), unit=u.km, meta={'observer': 'ME'}, wcs=WCS(naxis=1), psf=np.array([20]))
    assert nd3.data is nd1.data
    assert nd3.wcs is not nd1.wcs
    assert nd3.uncertainty.array != nd1.uncertainty.array
    assert nd3.mask != nd1.mask
    assert nd3.unit != nd1.unit
    assert nd3.meta != nd1.meta
    assert nd3.psf != nd1.psf

def test_nddata_init_data_nddata_subclass():
    if False:
        return 10
    uncert = StdDevUncertainty(3)
    bnd = BadNDDataSubclass(False, True, 3, 2, 'gollum', 100, 12)
    with pytest.raises(TypeError):
        NDData(bnd)
    bnd_good = BadNDDataSubclass(np.array([1, 2]), uncert, 3, HighLevelWCSWrapper(WCS(naxis=1)), {'enemy': 'black knight'}, u.km)
    nd = NDData(bnd_good)
    assert nd.unit == bnd_good.unit
    assert nd.meta == bnd_good.meta
    assert nd.uncertainty == bnd_good.uncertainty
    assert nd.mask == bnd_good.mask
    assert nd.wcs is bnd_good.wcs
    assert nd.data is bnd_good.data

def test_nddata_init_data_fail():
    if False:
        return 10
    with pytest.raises(TypeError):
        NDData({'a': 'dict'})

    class Shape:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = 5

        def __repr__(self):
            if False:
                print('Hello World!')
            return '7'
    with pytest.raises(TypeError):
        NDData(Shape())

def test_nddata_init_data_fakes():
    if False:
        return 10
    ndd1 = NDData(FakeNumpyArray())
    assert isinstance(ndd1.data, FakeNumpyArray)
    ndd2 = NDData(ndd1)
    assert isinstance(ndd2.data, FakeNumpyArray)

def test_param_uncertainty():
    if False:
        print('Hello World!')
    u = StdDevUncertainty(array=np.ones((5, 5)))
    d = NDData(np.ones((5, 5)), uncertainty=u)
    assert d.uncertainty.parent_nddata is d
    u2 = StdDevUncertainty(array=np.ones((5, 5)) * 2)
    d2 = NDData(d, uncertainty=u2)
    assert d2.uncertainty is u2
    assert d2.uncertainty.parent_nddata is d2

def test_param_wcs():
    if False:
        while True:
            i = 10
    nd = NDData([1], wcs=WCS(naxis=1))
    assert nd.wcs is not None
    nd2 = NDData(nd, wcs=WCS(naxis=1))
    assert nd2.wcs is not None and nd2.wcs is not nd.wcs

def test_param_meta():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        NDData([1], meta=3)
    nd = NDData([1, 2, 3], meta={})
    assert len(nd.meta) == 0
    nd = NDData([1, 2, 3])
    assert isinstance(nd.meta, OrderedDict)
    assert len(nd.meta) == 0
    nd2 = NDData(nd, meta={'image': 'sun'})
    assert len(nd2.meta) == 1
    nd3 = NDData(nd2, meta={'image': 'moon'})
    assert len(nd3.meta) == 1
    assert nd3.meta['image'] == 'moon'

def test_param_mask():
    if False:
        print('Hello World!')
    nd = NDData([1], mask=False)
    assert not nd.mask
    nd2 = NDData(nd, mask=True)
    assert nd2.mask
    nd3 = NDData(np.ma.array([1], mask=False), mask=True)
    assert nd3.mask
    mq = np.ma.array(np.array([2, 3]) * u.m, mask=False)
    nd4 = NDData(mq, mask=True)
    assert nd4.mask

def test_param_unit():
    if False:
        return 10
    with pytest.raises(ValueError):
        NDData(np.ones((5, 5)), unit='NotAValidUnit')
    NDData([1, 2, 3], unit='meter')
    q = np.array([1, 2, 3]) * u.m
    nd = NDData(q, unit='cm')
    assert nd.unit != q.unit
    assert nd.unit == u.cm
    mq = np.ma.array(np.array([2, 3]) * u.m, mask=False)
    nd2 = NDData(mq, unit=u.pc)
    assert nd2.unit == u.pc
    nd3 = NDData(nd, unit='km')
    assert nd3.unit == u.km
    mq_astropy = Masked.from_unmasked(q, False)
    nd4 = NDData(mq_astropy, unit='km')
    assert nd4.unit == u.km

def test_pickle_nddata_with_uncertainty():
    if False:
        return 10
    ndd = NDData(np.ones(3), uncertainty=StdDevUncertainty(np.ones(5), unit=u.m), unit=u.m)
    ndd_dumped = pickle.dumps(ndd)
    ndd_restored = pickle.loads(ndd_dumped)
    assert type(ndd_restored.uncertainty) is StdDevUncertainty
    assert ndd_restored.uncertainty.parent_nddata is ndd_restored
    assert ndd_restored.uncertainty.unit == u.m

def test_pickle_uncertainty_only():
    if False:
        return 10
    ndd = NDData(np.ones(3), uncertainty=StdDevUncertainty(np.ones(5), unit=u.m), unit=u.m)
    uncertainty_dumped = pickle.dumps(ndd.uncertainty)
    uncertainty_restored = pickle.loads(uncertainty_dumped)
    np.testing.assert_array_equal(ndd.uncertainty.array, uncertainty_restored.array)
    assert ndd.uncertainty.unit == uncertainty_restored.unit
    assert uncertainty_restored.parent_nddata is None

def test_pickle_nddata_without_uncertainty():
    if False:
        return 10
    ndd = NDData(np.ones(3), unit=u.m)
    dumped = pickle.dumps(ndd)
    ndd_restored = pickle.loads(dumped)
    np.testing.assert_array_equal(ndd.data, ndd_restored.data)
from astropy.utils.metadata.tests.test_metadata import MetaBaseTest

class TestMetaNDData(MetaBaseTest):
    test_class = NDData
    args = np.array([[1.0]])

def test_nddata_str():
    if False:
        print('Hello World!')
    arr1d = NDData(np.array([1, 2, 3]))
    assert str(arr1d) == '[1 2 3]'
    arr2d = NDData(np.array([[1, 2], [3, 4]]))
    assert str(arr2d) == textwrap.dedent('\n        [[1 2]\n         [3 4]]'[1:])
    arr3d = NDData(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert str(arr3d) == textwrap.dedent('\n        [[[1 2]\n          [3 4]]\n\n         [[5 6]\n          [7 8]]]'[1:])
    arr = NDData(np.array([1, 2, 3]), unit='km')
    assert str(arr) == '[1 2 3] km'
    arr = NDData(np.array([1, 2, 3]), unit='erg cm^-2 s^-1 A^-1')
    assert str(arr) == '[1 2 3] erg / (A s cm2)'

def test_nddata_repr():
    if False:
        while True:
            i = 10
    arr1d = NDData(np.array([1, 2, 3]))
    s = repr(arr1d)
    assert s == 'NDData([1, 2, 3])'
    got = eval(s)
    assert np.all(got.data == arr1d.data)
    assert got.unit == arr1d.unit
    arr2d = NDData(np.array([[1, 2], [3, 4]]))
    s = repr(arr2d)
    assert s == 'NDData([[1, 2],\n        [3, 4]])'
    got = eval(s)
    assert np.all(got.data == arr2d.data)
    assert got.unit == arr2d.unit
    arr3d = NDData(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    s = repr(arr3d)
    assert s == 'NDData([[[1, 2],\n         [3, 4]],\n\n        [[5, 6],\n         [7, 8]]])'
    got = eval(s)
    assert np.all(got.data == arr3d.data)
    assert got.unit == arr3d.unit
    arr = NDData(np.array([1, 2, 3]), unit='km')
    s = repr(arr)
    assert s == "NDData([1, 2, 3], unit='km')"
    got = eval(s)
    assert np.all(got.data == arr.data)
    assert got.unit == arr.unit

@pytest.mark.skipif(not HAS_DASK, reason='requires dask to be available')
def test_nddata_repr_dask():
    if False:
        while True:
            i = 10
    import dask.array as da
    arr = NDData(da.arange(3), unit='km')
    s = repr(arr)
    assert s in ('NDData(\n  data=dask.array<arange, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>,\n  unit=Unit("km")\n)', 'NDData(\n  data=dask.array<arange, shape=(3,), dtype=int32, chunksize=(3,), chunktype=numpy.ndarray>,\n  unit=Unit("km")\n)')

def test_slicing_not_supported():
    if False:
        return 10
    ndd = NDData(np.ones((5, 5)))
    with pytest.raises(TypeError):
        ndd[0]

def test_arithmetic_not_supported():
    if False:
        i = 10
        return i + 15
    ndd = NDData(np.ones((5, 5)))
    with pytest.raises(TypeError):
        ndd + ndd

def test_nddata_wcs_setter_error_cases():
    if False:
        print('Hello World!')
    ndd = NDData(np.ones((5, 5)))
    with pytest.raises(TypeError):
        ndd.wcs = 'I am not a WCS'
    naxis = 2
    ndd.wcs = nd_testing._create_wcs_simple(naxis=naxis, ctype=['deg'] * naxis, crpix=[0] * naxis, crval=[10] * naxis, cdelt=[1] * naxis)
    with pytest.raises(ValueError):
        ndd.wcs = nd_testing._create_wcs_simple(naxis=naxis, ctype=['deg'] * naxis, crpix=[0] * naxis, crval=[10] * naxis, cdelt=[1] * naxis)

def test_nddata_wcs_setter_with_low_level_wcs():
    if False:
        i = 10
        return i + 15
    ndd = NDData(np.ones((5, 5)))
    wcs = WCS()
    low_level = SlicedLowLevelWCS(wcs, 5)
    assert not isinstance(low_level, BaseHighLevelWCS)
    ndd.wcs = low_level
    assert isinstance(ndd.wcs, BaseHighLevelWCS)

def test_nddata_init_with_low_level_wcs():
    if False:
        return 10
    wcs = WCS()
    low_level = SlicedLowLevelWCS(wcs, 5)
    ndd = NDData(np.ones((5, 5)), wcs=low_level)
    assert isinstance(ndd.wcs, BaseHighLevelWCS)

class NDDataCustomWCS(NDData):

    @property
    def wcs(self):
        if False:
            return 10
        return WCS()

def test_overriden_wcs():
    if False:
        i = 10
        return i + 15
    NDDataCustomWCS(np.ones((5, 5)))
np.random.seed(42)
collapse_units = [None, u.Jy]
collapse_propagate = [True, False]
collapse_data_shapes = [(4, 3, 2), (6, 5, 4, 3, 2)]
collapse_ignore_masked = [True, False]
collapse_masks = list(chain.from_iterable(([np.zeros(collapse_data_shape).astype(bool)] + [np.random.randint(0, 2, size=collapse_data_shape).astype(bool) for _ in range(10)] for collapse_data_shape in collapse_data_shapes)))
permute = len(collapse_masks) * len(collapse_propagate) * len(collapse_units) * len(collapse_ignore_masked)
collapse_units = permute // len(collapse_units) * collapse_units
collapse_propagate = permute // len(collapse_propagate) * collapse_propagate
collapse_masks = permute // len(collapse_masks) * collapse_masks
collapse_ignore_masked = permute // len(collapse_ignore_masked) * collapse_ignore_masked

@pytest.mark.parametrize('mask, unit, propagate_uncertainties, operation_ignores_mask', zip(collapse_masks, collapse_units, collapse_propagate, collapse_ignore_masked))
def test_collapse(mask, unit, propagate_uncertainties, operation_ignores_mask):
    if False:
        return 10
    axes_permutations = {tuple(axes[:2]) for axes in permutations(range(mask.ndim))}
    axes_permutations.update(set(range(mask.ndim)))
    axes_permutations.update({None})
    cube = np.arange(np.prod(mask.shape)).reshape(mask.shape)
    numpy_cube = np.ma.masked_array(cube, mask=mask)
    ma_cube = Masked(cube, mask=mask)
    ndarr = NDDataArray(cube, uncertainty=StdDevUncertainty(cube), unit=unit, mask=mask)
    for axis in range(cube.ndim):
        assert np.all(np.equal(cube.argmin(axis=axis), 0))
        assert np.all(np.equal(cube.argmax(axis=axis), cube.shape[axis] - 1))
    sum_methods = ['sum', 'mean']
    ext_methods = ['min', 'max']
    all_methods = sum_methods + ext_methods
    for method in all_methods:
        for axes in axes_permutations:
            astropy_method = getattr(ma_cube, method)(axis=axes)
            numpy_method = getattr(numpy_cube, method)(axis=axes)
            nddata_method = getattr(ndarr, method)(axis=axes, propagate_uncertainties=propagate_uncertainties, operation_ignores_mask=operation_ignores_mask)
            astropy_unmasked = astropy_method.base[~astropy_method.mask]
            nddata_unmasked = nddata_method.data[~nddata_method.mask]
            assert unit == nddata_method.unit
            if len(astropy_unmasked) > 0:
                if not operation_ignores_mask:
                    assert np.all(np.equal(astropy_unmasked, nddata_unmasked))
                    assert np.all(np.equal(astropy_method.mask, nddata_method.mask))
                else:
                    assert np.ma.all(np.ma.equal(numpy_method, np.asanyarray(nddata_method)))
            if method in ext_methods and propagate_uncertainties:
                assert np.ma.all(np.ma.equal(astropy_method, nddata_method))