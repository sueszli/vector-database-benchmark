import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from astropy import units as u
from astropy.nddata import NDDataRef
from astropy.nddata import _testing as nd_testing
from astropy.nddata.nduncertainty import IncompatibleUncertaintiesException, InverseVariance, StdDevUncertainty, UnknownUncertainty, VarianceUncertainty
from astropy.units import Quantity, UnitsError
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
NDDataArithmetic = NDDataRef

class StdDevUncertaintyUncorrelated(StdDevUncertainty):

    @property
    def supports_correlated(self):
        if False:
            for i in range(10):
                print('nop')
        return False

@pytest.mark.filterwarnings('ignore:divide by zero encountered.*')
@pytest.mark.parametrize(('data1', 'data2'), [(np.array(5), np.array(10)), (np.array(5), np.arange(10)), (np.array(5), np.arange(10).reshape(2, 5)), (np.arange(10), np.ones(10) * 2), (np.arange(10), np.ones((10, 10)) * 2), (np.arange(10).reshape(2, 5), np.ones((2, 5)) * 3), (np.arange(1000).reshape(20, 5, 10), np.ones((20, 5, 10)) * 3)])
def test_arithmetics_data(data1, data2):
    if False:
        while True:
            i = 10
    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)
    nd3 = nd1.add(nd2)
    assert_array_equal(data1 + data2, nd3.data)
    nd4 = nd1.subtract(nd2)
    assert_array_equal(data1 - data2, nd4.data)
    nd5 = nd1.multiply(nd2)
    assert_array_equal(data1 * data2, nd5.data)
    nd6 = nd1.divide(nd2)
    assert_array_equal(data1 / data2, nd6.data)
    for nd in [nd3, nd4, nd5, nd6]:
        if data1.ndim > data2.ndim:
            assert data1.shape == nd.data.shape
        else:
            assert data2.shape == nd.data.shape
        assert nd.unit is None
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None

def test_arithmetics_data_invalid():
    if False:
        print('Hello World!')
    nd1 = NDDataArithmetic([1, 2, 3])
    nd2 = NDDataArithmetic([1, 2])
    with pytest.raises(ValueError):
        nd1.add(nd2)

@pytest.mark.filterwarnings('ignore:divide by zero encountered.*')
@pytest.mark.parametrize(('data1', 'data2'), [(np.array(5) * u.s, np.array(10) * u.s), (np.array(5) * u.s, np.arange(10) * u.h), (np.array(5) * u.s, np.arange(10).reshape(2, 5) * u.min), (np.arange(10) * u.m / u.s, np.ones(10) * 2 * u.km / u.s), (np.arange(10) * u.m / u.s, np.ones((10, 10)) * 2 * u.m / u.h), (np.arange(10).reshape(2, 5) * u.m / u.s, np.ones((2, 5)) * 3 * u.km / u.h), (np.arange(1000).reshape(20, 5, 10), np.ones((20, 5, 10)) * 3 * u.dimensionless_unscaled), (np.array(5), np.array(10) * u.s / u.h)])
def test_arithmetics_data_unit_identical(data1, data2):
    if False:
        print('Hello World!')
    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)
    nd3 = nd1.add(nd2)
    ref = data1 + data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd3.data)
    assert nd3.unit == ref_unit
    nd4 = nd1.subtract(nd2)
    ref = data1 - data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd4.data)
    assert nd4.unit == ref_unit
    nd5 = nd1.multiply(nd2)
    ref = data1 * data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd5.data)
    assert nd5.unit == ref_unit
    nd6 = nd1.divide(nd2)
    ref = data1 / data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd6.data)
    assert nd6.unit == ref_unit
    for nd in [nd3, nd4, nd5, nd6]:
        if data1.ndim > data2.ndim:
            assert data1.shape == nd.data.shape
        else:
            assert data2.shape == nd.data.shape
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None

@pytest.mark.parametrize(('data1', 'data2'), [(np.array(5) * u.s, np.array(10) * u.m), (np.array(5) * u.Mpc, np.array(10) * u.km / u.s), (np.array(5) * u.Mpc, np.array(10)), (np.array(5), np.array(10) * u.s)])
def test_arithmetics_data_unit_not_identical(data1, data2):
    if False:
        i = 10
        return i + 15
    nd1 = NDDataArithmetic(data1)
    nd2 = NDDataArithmetic(data2)
    with pytest.raises(UnitsError):
        nd1.add(nd2)
    with pytest.raises(UnitsError):
        nd1.subtract(nd2)
    nd3 = nd1.multiply(nd2)
    ref = data1 * data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd3.data)
    assert nd3.unit == ref_unit
    nd4 = nd1.divide(nd2)
    ref = data1 / data2
    (ref_unit, ref_data) = (ref.unit, ref.value)
    assert_array_equal(ref_data, nd4.data)
    assert nd4.unit == ref_unit
    for nd in [nd3, nd4]:
        assert nd.uncertainty is None
        assert nd.mask is None
        assert len(nd.meta) == 0
        assert nd.wcs is None

@pytest.mark.parametrize(('wcs1', 'wcs2'), [(None, None), (None, WCS(naxis=2)), (WCS(naxis=2), None), nd_testing.create_two_equal_wcs(naxis=2), nd_testing.create_two_unequal_wcs(naxis=2)])
def test_arithmetics_data_wcs(wcs1, wcs2):
    if False:
        return 10
    nd1 = NDDataArithmetic(1, wcs=wcs1)
    nd2 = NDDataArithmetic(1, wcs=wcs2)
    if wcs1 is None and wcs2 is None:
        ref_wcs = None
    elif wcs1 is None:
        ref_wcs = wcs2
    elif wcs2 is None:
        ref_wcs = wcs1
    else:
        ref_wcs = wcs1
    nd3 = nd1.add(nd2)
    nd_testing.assert_wcs_seem_equal(ref_wcs, nd3.wcs)
    nd4 = nd1.subtract(nd2)
    nd_testing.assert_wcs_seem_equal(ref_wcs, nd4.wcs)
    nd5 = nd1.multiply(nd2)
    nd_testing.assert_wcs_seem_equal(ref_wcs, nd5.wcs)
    nd6 = nd1.divide(nd2)
    nd_testing.assert_wcs_seem_equal(ref_wcs, nd6.wcs)
    for nd in [nd3, nd4, nd5, nd6]:
        assert nd.unit is None
        assert nd.uncertainty is None
        assert len(nd.meta) == 0
        assert nd.mask is None

@pytest.mark.parametrize(('mask1', 'mask2'), [(None, None), (None, False), (True, None), (False, False), (True, False), (False, True), (True, True), (np.array(False), np.array(True)), (np.array(False), np.array([0, 1, 0, 1, 1], dtype=np.bool_)), (np.array(True), np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.bool_)), (np.array([0, 1, 0, 1, 1], dtype=np.bool_), np.array([1, 1, 0, 0, 1], dtype=np.bool_)), (np.array([0, 1, 0, 1, 1], dtype=np.bool_), np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=np.bool_)), (np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=np.bool_), np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.bool_))])
def test_arithmetics_data_masks(mask1, mask2):
    if False:
        print('Hello World!')
    nd1 = NDDataArithmetic(1, mask=mask1)
    nd2 = NDDataArithmetic(1, mask=mask2)
    if mask1 is None and mask2 is None:
        ref_mask = None
    elif mask1 is None:
        ref_mask = mask2
    elif mask2 is None:
        ref_mask = mask1
    else:
        ref_mask = mask1 | mask2
    nd3 = nd1.add(nd2)
    assert_array_equal(ref_mask, nd3.mask)
    nd4 = nd1.subtract(nd2)
    assert_array_equal(ref_mask, nd4.mask)
    nd5 = nd1.multiply(nd2)
    assert_array_equal(ref_mask, nd5.mask)
    nd6 = nd1.divide(nd2)
    assert_array_equal(ref_mask, nd6.mask)
    for nd in [nd3, nd4, nd5, nd6]:
        assert nd.unit is None
        assert nd.uncertainty is None
        assert len(nd.meta) == 0
        assert nd.wcs is None

@pytest.mark.parametrize(('collapse_axis', 'mask_sum', 'unit'), [(0, [3, 0, 3, 0], 'Jy'), (1, [2, 0, 2, 0], None), (2, [2, 2, 2], 'Jy')])
def test_collapse_masks(collapse_axis, mask_sum, unit):
    if False:
        return 10
    shape = (2, 3, 4)
    data = np.arange(np.prod(shape)).reshape(shape)
    mask = data % 2 == 0
    nd_masked = NDDataArithmetic(data=data, mask=mask, unit=unit)
    nd_nomask = NDDataArithmetic(data=data, unit=unit)
    assert_array_equal(nd_masked.sum(axis=collapse_axis).mask.sum(axis=0), mask_sum)
    assert nd_nomask.sum(axis=collapse_axis).mask is None

def test_arithmetics_data_masks_invalid():
    if False:
        i = 10
        return i + 15
    nd1 = NDDataArithmetic(1, mask=np.array([1, 0], dtype=np.bool_))
    nd2 = NDDataArithmetic(1, mask=np.array([1, 0, 1], dtype=np.bool_))
    with pytest.raises(ValueError):
        nd1.add(nd2)
    with pytest.raises(ValueError):
        nd1.multiply(nd2)
    with pytest.raises(ValueError):
        nd1.subtract(nd2)
    with pytest.raises(ValueError):
        nd1.divide(nd2)

def test_arithmetics_stddevuncertainty_basic():
    if False:
        print('Hello World!')
    nd1 = NDDataArithmetic([1, 2, 3], uncertainty=StdDevUncertainty([1, 1, 3]))
    nd2 = NDDataArithmetic([2, 2, 2], uncertainty=StdDevUncertainty([2, 2, 2]))
    nd3 = nd1.add(nd2)
    nd4 = nd2.add(nd1)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = np.sqrt(np.array([1, 1, 3]) ** 2 + np.array([2, 2, 2]) ** 2)
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.subtract(nd2)
    nd4 = nd2.subtract(nd1)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.multiply(nd2)
    nd4 = nd2.multiply(nd1)
    assert_array_almost_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = np.abs(np.array([2, 4, 6])) * np.sqrt((np.array([1, 1, 3]) / np.array([1, 2, 3])) ** 2 + (np.array([2, 2, 2]) / np.array([2, 2, 2])) ** 2)
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.divide(nd2)
    nd4 = nd2.divide(nd1)
    ref_uncertainty_1 = np.abs(np.array([1 / 2, 2 / 2, 3 / 2])) * np.sqrt((np.array([1, 1, 3]) / np.array([1, 2, 3])) ** 2 + (np.array([2, 2, 2]) / np.array([2, 2, 2])) ** 2)
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty_1)
    ref_uncertainty_2 = np.abs(np.array([2, 1, 2 / 3])) * np.sqrt((np.array([1, 1, 3]) / np.array([1, 2, 3])) ** 2 + (np.array([2, 2, 2]) / np.array([2, 2, 2])) ** 2)
    assert_array_almost_equal(nd4.uncertainty.array, ref_uncertainty_2)

@pytest.mark.parametrize(('cor', 'uncert1', 'data2'), [(-1, [1, 1, 3], [2, 2, 7]), (-0.5, [1, 1, 3], [2, 2, 7]), (-0.25, [1, 1, 3], [2, 2, 7]), (0, [1, 1, 3], [2, 2, 7]), (0.25, [1, 1, 3], [2, 2, 7]), (0.5, [1, 1, 3], [2, 2, 7]), (1, [1, 1, 3], [2, 2, 7]), (-1, [-1, -1, -3], [2, 2, 7]), (-0.5, [-1, -1, -3], [2, 2, 7]), (-0.25, [-1, -1, -3], [2, 2, 7]), (0, [-1, -1, -3], [2, 2, 7]), (0.25, [-1, -1, -3], [2, 2, 7]), (0.5, [-1, -1, -3], [2, 2, 7]), (1, [-1, -1, -3], [2, 2, 7]), (-1, [1, 1, 3], [-2, -3, -2]), (-0.5, [1, 1, 3], [-2, -3, -2]), (-0.25, [1, 1, 3], [-2, -3, -2]), (0, [1, 1, 3], [-2, -3, -2]), (0.25, [1, 1, 3], [-2, -3, -2]), (0.5, [1, 1, 3], [-2, -3, -2]), (1, [1, 1, 3], [-2, -3, -2]), (-1, [-1, -1, -3], [-2, -3, -2]), (-0.5, [-1, -1, -3], [-2, -3, -2]), (-0.25, [-1, -1, -3], [-2, -3, -2]), (0, [-1, -1, -3], [-2, -3, -2]), (0.25, [-1, -1, -3], [-2, -3, -2]), (0.5, [-1, -1, -3], [-2, -3, -2]), (1, [-1, -1, -3], [-2, -3, -2])])
def test_arithmetics_stddevuncertainty_basic_with_correlation(cor, uncert1, data2):
    if False:
        print('Hello World!')
    data1 = np.array([1, 2, 3])
    data2 = np.array(data2)
    uncert1 = np.array(uncert1)
    uncert2 = np.array([2, 2, 2])
    nd1 = NDDataArithmetic(data1, uncertainty=StdDevUncertainty(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=StdDevUncertainty(uncert2))
    nd3 = nd1.add(nd2, uncertainty_correlation=cor)
    nd4 = nd2.add(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = np.sqrt(uncert1 ** 2 + uncert2 ** 2 + 2 * cor * np.abs(uncert1 * uncert2))
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.subtract(nd2, uncertainty_correlation=cor)
    nd4 = nd2.subtract(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = np.sqrt(uncert1 ** 2 + uncert2 ** 2 - 2 * cor * np.abs(uncert1 * uncert2))
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.multiply(nd2, uncertainty_correlation=cor)
    nd4 = nd2.multiply(nd1, uncertainty_correlation=cor)
    assert_array_almost_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = np.abs(data1 * data2) * np.sqrt((uncert1 / data1) ** 2 + (uncert2 / data2) ** 2 + 2 * cor * np.abs(uncert1 * uncert2) / (data1 * data2))
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.divide(nd2, uncertainty_correlation=cor)
    nd4 = nd2.divide(nd1, uncertainty_correlation=cor)
    ref_uncertainty_1 = np.abs(data1 / data2) * np.sqrt((uncert1 / data1) ** 2 + (uncert2 / data2) ** 2 - 2 * cor * np.abs(uncert1 * uncert2) / (data1 * data2))
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty_1)
    ref_uncertainty_2 = np.abs(data2 / data1) * np.sqrt((uncert1 / data1) ** 2 + (uncert2 / data2) ** 2 - 2 * cor * np.abs(uncert1 * uncert2) / (data1 * data2))
    assert_array_almost_equal(nd4.uncertainty.array, ref_uncertainty_2)

@pytest.mark.parametrize(('cor', 'uncert1', 'data2'), [(-1, [1, 1, 3], [2, 2, 7]), (-0.5, [1, 1, 3], [2, 2, 7]), (-0.25, [1, 1, 3], [2, 2, 7]), (0, [1, 1, 3], [2, 2, 7]), (0.25, [1, 1, 3], [2, 2, 7]), (0.5, [1, 1, 3], [2, 2, 7]), (1, [1, 1, 3], [2, 2, 7]), (-1, [-1, -1, -3], [2, 2, 7]), (-0.5, [-1, -1, -3], [2, 2, 7]), (-0.25, [-1, -1, -3], [2, 2, 7]), (0, [-1, -1, -3], [2, 2, 7]), (0.25, [-1, -1, -3], [2, 2, 7]), (0.5, [-1, -1, -3], [2, 2, 7]), (1, [-1, -1, -3], [2, 2, 7]), (-1, [1, 1, 3], [-2, -3, -2]), (-0.5, [1, 1, 3], [-2, -3, -2]), (-0.25, [1, 1, 3], [-2, -3, -2]), (0, [1, 1, 3], [-2, -3, -2]), (0.25, [1, 1, 3], [-2, -3, -2]), (0.5, [1, 1, 3], [-2, -3, -2]), (1, [1, 1, 3], [-2, -3, -2]), (-1, [-1, -1, -3], [-2, -3, -2]), (-0.5, [-1, -1, -3], [-2, -3, -2]), (-0.25, [-1, -1, -3], [-2, -3, -2]), (0, [-1, -1, -3], [-2, -3, -2]), (0.25, [-1, -1, -3], [-2, -3, -2]), (0.5, [-1, -1, -3], [-2, -3, -2]), (1, [-1, -1, -3], [-2, -3, -2])])
def test_arithmetics_varianceuncertainty_basic_with_correlation(cor, uncert1, data2):
    if False:
        while True:
            i = 10
    data1 = np.array([1, 2, 3])
    data2 = np.array(data2)
    uncert1 = np.array(uncert1) ** 2
    uncert2 = np.array([2, 2, 2]) ** 2
    nd1 = NDDataArithmetic(data1, uncertainty=VarianceUncertainty(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=VarianceUncertainty(uncert2))
    nd3 = nd1.add(nd2, uncertainty_correlation=cor)
    nd4 = nd2.add(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = uncert1 + uncert2 + 2 * cor * np.sqrt(uncert1 * uncert2)
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.subtract(nd2, uncertainty_correlation=cor)
    nd4 = nd2.subtract(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = uncert1 + uncert2 - 2 * cor * np.sqrt(uncert1 * uncert2)
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.multiply(nd2, uncertainty_correlation=cor)
    nd4 = nd2.multiply(nd1, uncertainty_correlation=cor)
    assert_array_almost_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = (data1 * data2) ** 2 * (uncert1 / data1 ** 2 + uncert2 / data2 ** 2 + 2 * cor * np.sqrt(uncert1 * uncert2) / (data1 * data2))
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.divide(nd2, uncertainty_correlation=cor)
    nd4 = nd2.divide(nd1, uncertainty_correlation=cor)
    ref_common = uncert1 / data1 ** 2 + uncert2 / data2 ** 2 - 2 * cor * np.sqrt(uncert1 * uncert2) / (data1 * data2)
    ref_uncertainty_1 = (data1 / data2) ** 2 * ref_common
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty_1)
    ref_uncertainty_2 = (data2 / data1) ** 2 * ref_common
    assert_array_almost_equal(nd4.uncertainty.array, ref_uncertainty_2)

@pytest.mark.filterwarnings('ignore:divide by zero encountered.*')
@pytest.mark.parametrize(('cor', 'uncert1', 'data2'), [(-1, [1, 1, 3], [2, 2, 7]), (-0.5, [1, 1, 3], [2, 2, 7]), (-0.25, [1, 1, 3], [2, 2, 7]), (0, [1, 1, 3], [2, 2, 7]), (0.25, [1, 1, 3], [2, 2, 7]), (0.5, [1, 1, 3], [2, 2, 7]), (1, [1, 1, 3], [2, 2, 7]), (-1, [-1, -1, -3], [2, 2, 7]), (-0.5, [-1, -1, -3], [2, 2, 7]), (-0.25, [-1, -1, -3], [2, 2, 7]), (0, [-1, -1, -3], [2, 2, 7]), (0.25, [-1, -1, -3], [2, 2, 7]), (0.5, [-1, -1, -3], [2, 2, 7]), (1, [-1, -1, -3], [2, 2, 7]), (-1, [1, 1, 3], [-2, -3, -2]), (-0.5, [1, 1, 3], [-2, -3, -2]), (-0.25, [1, 1, 3], [-2, -3, -2]), (0, [1, 1, 3], [-2, -3, -2]), (0.25, [1, 1, 3], [-2, -3, -2]), (0.5, [1, 1, 3], [-2, -3, -2]), (1, [1, 1, 3], [-2, -3, -2]), (-1, [-1, -1, -3], [-2, -3, -2]), (-0.5, [-1, -1, -3], [-2, -3, -2]), (-0.25, [-1, -1, -3], [-2, -3, -2]), (0, [-1, -1, -3], [-2, -3, -2]), (0.25, [-1, -1, -3], [-2, -3, -2]), (0.5, [-1, -1, -3], [-2, -3, -2]), (1, [-1, -1, -3], [-2, -3, -2])])
def test_arithmetics_inversevarianceuncertainty_basic_with_correlation(cor, uncert1, data2):
    if False:
        print('Hello World!')
    data1 = np.array([1, 2, 3])
    data2 = np.array(data2)
    uncert1 = 1 / np.array(uncert1) ** 2
    uncert2 = 1 / np.array([2, 2, 2]) ** 2
    nd1 = NDDataArithmetic(data1, uncertainty=InverseVariance(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=InverseVariance(uncert2))
    nd3 = nd1.add(nd2, uncertainty_correlation=cor)
    nd4 = nd2.add(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = 1 / (1 / uncert1 + 1 / uncert2 + 2 * cor / np.sqrt(uncert1 * uncert2))
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.subtract(nd2, uncertainty_correlation=cor)
    nd4 = nd2.subtract(nd1, uncertainty_correlation=cor)
    assert_array_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = 1 / (1 / uncert1 + 1 / uncert2 - 2 * cor / np.sqrt(uncert1 * uncert2))
    assert_array_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.multiply(nd2, uncertainty_correlation=cor)
    nd4 = nd2.multiply(nd1, uncertainty_correlation=cor)
    assert_array_almost_equal(nd3.uncertainty.array, nd4.uncertainty.array)
    ref_uncertainty = 1 / ((data1 * data2) ** 2 * (1 / uncert1 / data1 ** 2 + 1 / uncert2 / data2 ** 2 + 2 * cor / np.sqrt(uncert1 * uncert2) / (data1 * data2)))
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty)
    nd3 = nd1.divide(nd2, uncertainty_correlation=cor)
    nd4 = nd2.divide(nd1, uncertainty_correlation=cor)
    ref_common = 1 / uncert1 / data1 ** 2 + 1 / uncert2 / data2 ** 2 - 2 * cor / np.sqrt(uncert1 * uncert2) / (data1 * data2)
    ref_uncertainty_1 = 1 / ((data1 / data2) ** 2 * ref_common)
    assert_array_almost_equal(nd3.uncertainty.array, ref_uncertainty_1)
    ref_uncertainty_2 = 1 / ((data2 / data1) ** 2 * ref_common)
    assert_array_almost_equal(nd4.uncertainty.array, ref_uncertainty_2)

def test_arithmetics_stddevuncertainty_basic_with_correlation_array():
    if False:
        return 10
    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 1, 1])
    uncert1 = np.array([1, 1, 1])
    uncert2 = np.array([2, 2, 2])
    cor = np.array([0, 0.25, 0])
    nd1 = NDDataArithmetic(data1, uncertainty=StdDevUncertainty(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=StdDevUncertainty(uncert2))
    nd1.add(nd2, uncertainty_correlation=cor)

def test_arithmetics_with_correlation_unsupported():
    if False:
        i = 10
        return i + 15
    data1 = np.array([1, 2, 3])
    data2 = np.array([1, 1, 1])
    uncert1 = np.array([1, 1, 1])
    uncert2 = np.array([2, 2, 2])
    cor = 3
    nd1 = NDDataArithmetic(data1, uncertainty=StdDevUncertaintyUncorrelated(uncert1))
    nd2 = NDDataArithmetic(data2, uncertainty=StdDevUncertaintyUncorrelated(uncert2))
    with pytest.raises(ValueError):
        nd1.add(nd2, uncertainty_correlation=cor)

def test_arithmetics_stddevuncertainty_one_missing():
    if False:
        while True:
            i = 10
    nd1 = NDDataArithmetic([1, -2, 3])
    nd1_ref = NDDataArithmetic([1, -2, 3], uncertainty=StdDevUncertainty([0, 0, 0]))
    nd2 = NDDataArithmetic([2, 2, -2], uncertainty=StdDevUncertainty([2, 2, 2]))
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd2.add(nd1)
    nd3_ref = nd2.add(nd1_ref)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2.subtract(nd1_ref)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2.multiply(nd1_ref)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)
    nd3 = nd2.divide(nd1)
    nd3_ref = nd2.divide(nd1_ref)
    assert_array_equal(nd3.uncertainty.array, nd3_ref.uncertainty.array)
    assert_array_equal(np.abs(nd3.uncertainty.array), nd3.uncertainty.array)

@pytest.mark.filterwarnings('ignore:.*encountered in.*divide.*')
@pytest.mark.parametrize(('uncert1', 'uncert2'), [(np.array([1, 2, 3]) * u.m, None), (np.array([1, 2, 3]) * u.cm, None), (None, np.array([1, 2, 3]) * u.m), (None, np.array([1, 2, 3]) * u.cm), (np.array([1, 2, 3]), np.array([2, 3, 4])), (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.km, np.array([2, 3, 4])) * u.cm])
def test_arithmetics_stddevuncertainty_with_units(uncert1, uncert2):
    if False:
        print('Hello World!')
    data1 = np.array([1, 2, 3]) * u.m
    data2 = np.array([-4, 7, 0]) * u.m
    if uncert1 is not None:
        uncert1 = StdDevUncertainty(uncert1)
        if isinstance(uncert1, Quantity):
            uncert1_ref = uncert1.to_value(data1.unit)
        else:
            uncert1_ref = uncert1
        uncert_ref1 = StdDevUncertainty(uncert1_ref, copy=True)
    else:
        uncert1 = None
        uncert_ref1 = None
    if uncert2 is not None:
        uncert2 = StdDevUncertainty(uncert2)
        if isinstance(uncert2, Quantity):
            uncert2_ref = uncert2.to_value(data2.unit)
        else:
            uncert2_ref = uncert2
        uncert_ref2 = StdDevUncertainty(uncert2_ref, copy=True)
    else:
        uncert2 = None
        uncert_ref2 = None
    nd1 = NDDataArithmetic(data1, uncertainty=uncert1)
    nd2 = NDDataArithmetic(data2, uncertainty=uncert2)
    nd1_ref = NDDataArithmetic(data1, uncertainty=uncert_ref1)
    nd2_ref = NDDataArithmetic(data2, uncertainty=uncert_ref2)
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.add(nd1)
    nd3_ref = nd2_ref.add(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2_ref.subtract(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2_ref.multiply(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.divide(nd1)
    nd3_ref = nd2_ref.divide(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)

@pytest.mark.filterwarnings('ignore:.*encountered in.*divide.*')
@pytest.mark.parametrize(('uncert1', 'uncert2'), [(np.array([1, 2, 3]) * u.m, None), (np.array([1, 2, 3]) * u.cm, None), (None, np.array([1, 2, 3]) * u.m), (None, np.array([1, 2, 3]) * u.cm), (np.array([1, 2, 3]), np.array([2, 3, 4])), (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.km, np.array([2, 3, 4])) * u.cm])
def test_arithmetics_varianceuncertainty_with_units(uncert1, uncert2):
    if False:
        print('Hello World!')
    data1 = np.array([1, 2, 3]) * u.m
    data2 = np.array([-4, 7, 0]) * u.m
    if uncert1 is not None:
        uncert1 = VarianceUncertainty(uncert1 ** 2)
        if isinstance(uncert1, Quantity):
            uncert1_ref = uncert1.to_value(data1.unit ** 2)
        else:
            uncert1_ref = uncert1
        uncert_ref1 = VarianceUncertainty(uncert1_ref, copy=True)
    else:
        uncert1 = None
        uncert_ref1 = None
    if uncert2 is not None:
        uncert2 = VarianceUncertainty(uncert2 ** 2)
        if isinstance(uncert2, Quantity):
            uncert2_ref = uncert2.to_value(data2.unit ** 2)
        else:
            uncert2_ref = uncert2
        uncert_ref2 = VarianceUncertainty(uncert2_ref, copy=True)
    else:
        uncert2 = None
        uncert_ref2 = None
    nd1 = NDDataArithmetic(data1, uncertainty=uncert1)
    nd2 = NDDataArithmetic(data2, uncertainty=uncert2)
    nd1_ref = NDDataArithmetic(data1, uncertainty=uncert_ref1)
    nd2_ref = NDDataArithmetic(data2, uncertainty=uncert_ref2)
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.add(nd1)
    nd3_ref = nd2_ref.add(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2_ref.subtract(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2_ref.multiply(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.divide(nd1)
    nd3_ref = nd2_ref.divide(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)

@pytest.mark.filterwarnings('ignore:.*encountered in.*divide.*')
@pytest.mark.parametrize(('uncert1', 'uncert2'), [(np.array([1, 2, 3]) * u.m, None), (np.array([1, 2, 3]) * u.cm, None), (None, np.array([1, 2, 3]) * u.m), (None, np.array([1, 2, 3]) * u.cm), (np.array([1, 2, 3]), np.array([2, 3, 4])), (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.m, np.array([2, 3, 4])) * u.m, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])), (np.array([1, 2, 3]), np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.cm, np.array([2, 3, 4])) * u.cm, (np.array([1, 2, 3]) * u.km, np.array([2, 3, 4])) * u.cm])
def test_arithmetics_inversevarianceuncertainty_with_units(uncert1, uncert2):
    if False:
        print('Hello World!')
    data1 = np.array([1, 2, 3]) * u.m
    data2 = np.array([-4, 7, 0]) * u.m
    if uncert1 is not None:
        uncert1 = InverseVariance(1 / uncert1 ** 2)
        if isinstance(uncert1, Quantity):
            uncert1_ref = uncert1.to_value(1 / data1.unit ** 2)
        else:
            uncert1_ref = uncert1
        uncert_ref1 = InverseVariance(uncert1_ref, copy=True)
    else:
        uncert1 = None
        uncert_ref1 = None
    if uncert2 is not None:
        uncert2 = InverseVariance(1 / uncert2 ** 2)
        if isinstance(uncert2, Quantity):
            uncert2_ref = uncert2.to_value(1 / data2.unit ** 2)
        else:
            uncert2_ref = uncert2
        uncert_ref2 = InverseVariance(uncert2_ref, copy=True)
    else:
        uncert2 = None
        uncert_ref2 = None
    nd1 = NDDataArithmetic(data1, uncertainty=uncert1)
    nd2 = NDDataArithmetic(data2, uncertainty=uncert2)
    nd1_ref = NDDataArithmetic(data1, uncertainty=uncert_ref1)
    nd2_ref = NDDataArithmetic(data2, uncertainty=uncert_ref2)
    nd3 = nd1.add(nd2)
    nd3_ref = nd1_ref.add(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.add(nd1)
    nd3_ref = nd2_ref.add(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.subtract(nd2)
    nd3_ref = nd1_ref.subtract(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.subtract(nd1)
    nd3_ref = nd2_ref.subtract(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.multiply(nd2)
    nd3_ref = nd1_ref.multiply(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.multiply(nd1)
    nd3_ref = nd2_ref.multiply(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd1.divide(nd2)
    nd3_ref = nd1_ref.divide(nd2_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)
    nd3 = nd2.divide(nd1)
    nd3_ref = nd2_ref.divide(nd1_ref)
    assert nd3.unit == nd3_ref.unit
    assert nd3.uncertainty.unit == nd3_ref.uncertainty.unit
    assert_array_equal(nd3.uncertainty.array, nd3.uncertainty.array)

@pytest.mark.parametrize('use_abbreviation', ['ff', 'first_found'])
def test_arithmetics_handle_switches(use_abbreviation):
    if False:
        for i in range(10):
            print('nop')
    meta1 = {'a': 1}
    meta2 = {'b': 2}
    mask1 = True
    mask2 = False
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    (wcs1, wcs2) = nd_testing.create_two_unequal_wcs(naxis=1)
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]
    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1, uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2, uncertainty=uncertainty2)
    nd3 = NDDataArithmetic(data1)
    nd_ = nd1.add(nd2, propagate_uncertainties=None, handle_meta=None, handle_mask=None, compare_wcs=None)
    assert nd_.wcs is None
    assert len(nd_.meta) == 0
    assert nd_.mask is None
    assert nd_.uncertainty is None
    nd_ = nd3.add(nd2, propagate_uncertainties=False, handle_meta=use_abbreviation, handle_mask=use_abbreviation, compare_wcs=use_abbreviation)
    nd_testing.assert_wcs_seem_equal(nd_.wcs, wcs2)
    assert nd_.meta == meta2
    assert nd_.mask == mask2
    assert_array_equal(nd_.uncertainty.array, uncertainty2.array)
    nd_ = nd1.add(nd3, propagate_uncertainties=False, handle_meta=use_abbreviation, handle_mask=use_abbreviation, compare_wcs=use_abbreviation)
    nd_testing.assert_wcs_seem_equal(nd_.wcs, wcs1)
    assert nd_.meta == meta1
    assert nd_.mask == mask1
    assert_array_equal(nd_.uncertainty.array, uncertainty1.array)

def test_arithmetics_meta_func():
    if False:
        for i in range(10):
            print('nop')

    def meta_fun_func(meta1, meta2, take='first'):
        if False:
            i = 10
            return i + 15
        if take == 'first':
            return meta1
        else:
            return meta2
    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = True
    mask2 = False
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]
    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, uncertainty=uncertainty2)
    nd3 = nd1.add(nd2, handle_meta=meta_fun_func)
    assert nd3.meta['a'] == 1
    assert 'b' not in nd3.meta
    nd4 = nd1.add(nd2, handle_meta=meta_fun_func, meta_take='second')
    assert nd4.meta['a'] == 3
    assert nd4.meta['b'] == 2
    with pytest.raises(KeyError):
        nd1.add(nd2, handle_meta=meta_fun_func, take='second')

def test_arithmetics_wcs_func():
    if False:
        i = 10
        return i + 15

    def wcs_comp_func(wcs1, wcs2, tolerance=0.1):
        if False:
            for i in range(10):
                print('nop')
        if tolerance < 0.01:
            return False
        return True
    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = True
    mask2 = False
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    (wcs1, wcs2) = nd_testing.create_two_equal_wcs(naxis=1)
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]
    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, wcs=wcs1, uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, wcs=wcs2, uncertainty=uncertainty2)
    nd3 = nd1.add(nd2, compare_wcs=wcs_comp_func)
    nd_testing.assert_wcs_seem_equal(nd3.wcs, wcs1)
    with pytest.raises(ValueError):
        nd1.add(nd2, compare_wcs=wcs_comp_func, wcs_tolerance=1e-05)
    with pytest.raises(KeyError):
        nd1.add(nd2, compare_wcs=wcs_comp_func, tolerance=1)

def test_arithmetics_mask_func():
    if False:
        while True:
            i = 10

    def mask_sad_func(mask1, mask2, fun=0):
        if False:
            while True:
                i = 10
        if fun > 0.5:
            return mask2
        else:
            return mask1
    meta1 = {'a': 1}
    meta2 = {'a': 3, 'b': 2}
    mask1 = [True, False, True]
    mask2 = [True, False, False]
    uncertainty1 = StdDevUncertainty([1, 2, 3])
    uncertainty2 = StdDevUncertainty([1, 2, 3])
    data1 = [1, 1, 1]
    data2 = [1, 1, 1]
    nd1 = NDDataArithmetic(data1, meta=meta1, mask=mask1, uncertainty=uncertainty1)
    nd2 = NDDataArithmetic(data2, meta=meta2, mask=mask2, uncertainty=uncertainty2)
    nd3 = nd1.add(nd2, handle_mask=mask_sad_func)
    assert_array_equal(nd3.mask, nd1.mask)
    nd4 = nd1.add(nd2, handle_mask=mask_sad_func, mask_fun=1)
    assert_array_equal(nd4.mask, nd2.mask)
    with pytest.raises(KeyError):
        nd1.add(nd2, handle_mask=mask_sad_func, fun=1)

@pytest.mark.parametrize('meth', ['add', 'subtract', 'divide', 'multiply'])
def test_two_argument_useage(meth):
    if False:
        while True:
            i = 10
    ndd1 = NDDataArithmetic(np.ones((3, 3)))
    ndd2 = NDDataArithmetic(np.ones((3, 3)))
    ndd3 = getattr(NDDataArithmetic, meth)(ndd1, ndd2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)
    ndd3 = getattr(NDDataArithmetic(-100), meth)(ndd1, ndd2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)

@pytest.mark.parametrize('meth', ['add', 'subtract', 'divide', 'multiply'])
def test_two_argument_useage_non_nddata_first_arg(meth):
    if False:
        i = 10
        return i + 15
    data1 = 50
    data2 = 100
    ndd3 = getattr(NDDataArithmetic, meth)(data1, data2)
    ndd1 = NDDataArithmetic(data1)
    ndd2 = NDDataArithmetic(data2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)
    ndd3 = getattr(NDDataArithmetic(-100), meth)(data1, data2)
    ndd4 = getattr(ndd1, meth)(ndd2)
    np.testing.assert_array_equal(ndd3.data, ndd4.data)

def test_arithmetics_unknown_uncertainties():
    if False:
        while True:
            i = 10
    ndd1 = NDDataArithmetic(np.ones((3, 3)), uncertainty=UnknownUncertainty(np.ones((3, 3))))
    ndd2 = NDDataArithmetic(np.ones((3, 3)), uncertainty=UnknownUncertainty(np.ones((3, 3)) * 2))
    with pytest.raises(IncompatibleUncertaintiesException):
        ndd1.add(ndd2)
    ndd3 = ndd1.add(ndd2, propagate_uncertainties=False)
    np.testing.assert_array_equal(ndd1.uncertainty.array, ndd3.uncertainty.array)
    ndd4 = ndd1.add(ndd2, propagate_uncertainties=None)
    assert ndd4.uncertainty is None

def test_psf_warning():
    if False:
        for i in range(10):
            print('nop')
    'Test that math on objects with a psf warn.'
    ndd1 = NDDataArithmetic(np.ones((3, 3)), psf=np.zeros(3))
    ndd2 = NDDataArithmetic(np.ones((3, 3)), psf=None)
    ndd2.add(ndd2)
    with pytest.warns(AstropyUserWarning, match='Not setting psf attribute during add'):
        ndd1.add(ndd2)
    with pytest.warns(AstropyUserWarning, match='Not setting psf attribute during add'):
        ndd2.add(ndd1)
    with pytest.warns(AstropyUserWarning, match='Not setting psf attribute during add'):
        ndd1.add(ndd1)

def test_raise_method_not_supported():
    if False:
        i = 10
        return i + 15
    ndd1 = NDDataArithmetic(np.zeros(3), uncertainty=StdDevUncertainty(np.zeros(3)))
    ndd2 = NDDataArithmetic(np.ones(3), uncertainty=StdDevUncertainty(np.ones(3)))
    result = np.zeros(3)
    correlation = 0
    ndd1.uncertainty.propagate(np.add, ndd2, result, correlation)
    with pytest.raises(ValueError):
        ndd1.uncertainty.propagate(np.mod, ndd2, result, correlation)

def test_nddata_bitmask_arithmetic():
    if False:
        print('Hello World!')
    array = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mask = np.array([[0, 1, 64], [8, 0, 1], [2, 1, 0]])
    nref_nomask = NDDataRef(array)
    nref_masked = NDDataRef(array, mask=mask)
    assert nref_nomask.multiply(1.0, handle_mask=np.bitwise_or).mask is None
    assert nref_nomask.multiply(nref_nomask, handle_mask=np.bitwise_or).mask is None
    np.testing.assert_equal(nref_masked.multiply(1.0, handle_mask=np.bitwise_or).mask, mask)
    np.testing.assert_equal(nref_masked.multiply(nref_masked, handle_mask=np.bitwise_or).mask, mask)
    np.testing.assert_equal(nref_masked.multiply(nref_nomask, handle_mask=np.bitwise_or).mask, mask)
    other_mask = np.array([[64, 1, 0], [2, 1, 0], [8, 0, 2]])
    nref_mask_other = NDDataRef(array, mask=other_mask)
    np.testing.assert_equal(nref_mask_other.multiply(nref_masked, handle_mask=np.bitwise_or).mask, np.bitwise_or(mask, other_mask))