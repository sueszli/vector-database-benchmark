import numpy as np
import pytest
from numpy.testing import assert_array_equal
from astropy import units as u
from astropy.utils.compat.numpycompat import NUMPY_LT_2_0

class TestQuantityArrayCopy:
    """
    Test whether arrays are properly copied/used in place
    """

    def test_copy_on_creation(self):
        if False:
            i = 10
            return i + 15
        v = np.arange(1000.0)
        q_nocopy = u.Quantity(v, 'km/s', copy=False)
        q_copy = u.Quantity(v, 'km/s', copy=True)
        v[0] = -1.0
        assert q_nocopy[0].value == v[0]
        assert q_copy[0].value != v[0]

    def test_to_copies(self):
        if False:
            print('Hello World!')
        q = u.Quantity(np.arange(1.0, 100.0), 'km/s')
        q2 = q.to(u.m / u.s)
        assert np.all(q.value != q2.value)
        q3 = q.to(u.km / u.s)
        assert np.all(q.value == q3.value)
        q[0] = -1.0 * u.km / u.s
        assert q[0].value != q3[0].value

    def test_si_copies(self):
        if False:
            while True:
                i = 10
        q = u.Quantity(np.arange(100.0), 'm/s')
        q2 = q.si
        assert np.all(q.value == q2.value)
        q[0] = -1.0 * u.m / u.s
        assert q[0].value != q2[0].value

    def test_getitem_is_view(self):
        if False:
            while True:
                i = 10
        'Check that [keys] work, and that, like ndarray, it returns\n        a view, so that changing one changes the other.\n\n        Also test that one can add axes (closes #1422)\n        '
        q = u.Quantity(np.arange(100.0), 'm/s')
        q_sel = q[10:20]
        q_sel[0] = -1.0 * u.m / u.s
        assert q_sel[0] == q[10]
        q2 = q[:, np.newaxis]
        q2[10, 0] = -9 * u.m / u.s
        assert np.all(q2.flatten() == q)

    def test_flat(self):
        if False:
            while True:
                i = 10
        q = u.Quantity(np.arange(9.0).reshape(3, 3), 'm/s')
        q_flat = q.flat
        assert q_flat[8] == 8.0 * u.m / u.s
        assert np.all(q_flat[0:2] == np.arange(2.0) * u.m / u.s)
        q_flat_list = list(q.flat)
        assert np.all(u.Quantity(q_flat_list) == u.Quantity(list(q.value.flat), q.unit))
        q_flat[8] = -1.0 * u.km / u.s
        assert q_flat[8] == -1.0 * u.km / u.s
        assert q[2, 2] == -1.0 * u.km / u.s
        q_flat_list[8] = -2 * u.km / u.s
        assert q_flat_list[8] == -2.0 * u.km / u.s
        assert q_flat[8] == -1.0 * u.km / u.s
        assert q[2, 2] == -1.0 * u.km / u.s

class TestQuantityReshapeFuncs:
    """Test different ndarray methods that alter the array shape

    tests: reshape, squeeze, ravel, flatten, transpose, swapaxes
    """

    def test_reshape(self):
        if False:
            while True:
                i = 10
        q = np.arange(6.0) * u.m
        q_reshape = q.reshape(3, 2)
        assert isinstance(q_reshape, u.Quantity)
        assert q_reshape.unit == q.unit
        assert np.all(q_reshape.value == q.value.reshape(3, 2))

    def test_squeeze(self):
        if False:
            while True:
                i = 10
        q = np.arange(6.0).reshape(6, 1) * u.m
        q_squeeze = q.squeeze()
        assert isinstance(q_squeeze, u.Quantity)
        assert q_squeeze.unit == q.unit
        assert np.all(q_squeeze.value == q.value.squeeze())

    def test_ravel(self):
        if False:
            return 10
        q = np.arange(6.0).reshape(3, 2) * u.m
        q_ravel = q.ravel()
        assert isinstance(q_ravel, u.Quantity)
        assert q_ravel.unit == q.unit
        assert np.all(q_ravel.value == q.value.ravel())

    def test_flatten(self):
        if False:
            i = 10
            return i + 15
        q = np.arange(6.0).reshape(3, 2) * u.m
        q_flatten = q.flatten()
        assert isinstance(q_flatten, u.Quantity)
        assert q_flatten.unit == q.unit
        assert np.all(q_flatten.value == q.value.flatten())

    def test_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        q = np.arange(6.0).reshape(3, 2) * u.m
        q_transpose = q.transpose()
        assert isinstance(q_transpose, u.Quantity)
        assert q_transpose.unit == q.unit
        assert np.all(q_transpose.value == q.value.transpose())

    def test_swapaxes(self):
        if False:
            for i in range(10):
                print('nop')
        q = np.arange(6.0).reshape(3, 1, 2) * u.m
        q_swapaxes = q.swapaxes(0, 2)
        assert isinstance(q_swapaxes, u.Quantity)
        assert q_swapaxes.unit == q.unit
        assert np.all(q_swapaxes.value == q.value.swapaxes(0, 2))

    def test_flat_attributes(self):
        if False:
            i = 10
            return i + 15
        "While ``flat`` doesn't make a copy, it changes the shape."
        q = np.arange(6.0).reshape(3, 1, 2) * u.m
        qf = q.flat
        assert len(qf) == 6
        assert qf.base is q
        assert qf.coords == (0, 0, 0)
        assert qf.index == 0
        endindices = [(qf.index, qf.coords) for x in qf][-2]
        assert endindices[0] == 5
        assert endindices[1] == (2, 0, 1)
        q_flat_copy = qf.copy()
        assert all(q_flat_copy == q.flatten())
        assert isinstance(q_flat_copy, u.Quantity)
        assert not np.may_share_memory(q_flat_copy, q)

class TestQuantityStatsFuncs:
    """
    Test statistical functions
    """

    def test_mean(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert_array_equal(np.mean(q1), 3.6 * u.m)
        assert_array_equal(np.mean(q1, keepdims=True), [3.6] * u.m)

    def test_mean_inplace(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        qi = 1.5 * u.s
        qi2 = np.mean(q1, out=qi)
        assert qi2 is qi
        assert qi == 3.6 * u.m

    def test_mean_where(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 7.0]) * u.m
        assert_array_equal(np.mean(q1, where=q1 < 7 * u.m), 3.6 * u.m)

    def test_std(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0]) * u.m
        assert_array_equal(np.std(q1), 0.5 * u.m)
        assert_array_equal(q1.std(axis=-1, keepdims=True), [0.5] * u.m)

    def test_std_inplace(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0]) * u.m
        qi = 1.5 * u.s
        np.std(q1, out=qi)
        assert qi == 0.5 * u.m

    def test_std_where(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 3.0]) * u.m
        assert_array_equal(np.std(q1, where=q1 < 3 * u.m), 0.5 * u.m)

    def test_var(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0]) * u.m
        assert_array_equal(np.var(q1), 0.25 * u.m ** 2)
        assert_array_equal(q1.var(axis=0, keepdims=True), [0.25] * u.m ** 2)

    def test_var_inplace(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0]) * u.m
        qi = 1.5 * u.s
        np.var(q1, out=qi)
        assert qi == 0.25 * u.m ** 2

    def test_var_where(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0, 3.0]) * u.m
        assert_array_equal(np.var(q1, where=q1 < 3 * u.m), 0.25 * u.m ** 2)

    def test_median(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.median(q1) == 4.0 * u.m

    def test_median_inplace(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        qi = 1.5 * u.s
        np.median(q1, out=qi)
        assert qi == 4 * u.m

    def test_min(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.min(q1) == 1.0 * u.m

    def test_min_inplace(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        qi = 1.5 * u.s
        np.min(q1, out=qi)
        assert qi == 1.0 * u.m

    def test_min_where(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.min(q1, initial=10 * u.m, where=q1 > 0 * u.m) == 1.0 * u.m

    def test_argmin(self):
        if False:
            print('Hello World!')
        q1 = np.array([6.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.argmin(q1) == 1

    def test_argmin_keepdims(self):
        if False:
            print('Hello World!')
        q1 = np.array([[6.0, 2.0], [4.0, 5.0]]) * u.m
        assert_array_equal(q1.argmin(axis=0, keepdims=True), np.array([[1, 0]]))

    def test_max(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.max(q1) == 6.0 * u.m

    def test_max_inplace(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        qi = 1.5 * u.s
        np.max(q1, out=qi)
        assert qi == 6.0 * u.m

    def test_max_where(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 7.0]) * u.m
        assert np.max(q1, initial=0 * u.m, where=q1 < 7 * u.m) == 6.0 * u.m

    def test_argmax(self):
        if False:
            while True:
                i = 10
        q1 = np.array([5.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.argmax(q1) == 4

    def test_argmax_keepdims(self):
        if False:
            while True:
                i = 10
        q1 = np.array([[6.0, 2.0], [4.0, 5.0]]) * u.m
        assert_array_equal(q1.argmax(axis=0, keepdims=True), np.array([[0, 1]]))

    def test_clip(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.km / u.m
        c1 = q1.clip(1500, 5.5 * u.Mm / u.km)
        assert np.all(c1 == np.array([1.5, 2.0, 4.0, 5.0, 5.5]) * u.km / u.m)

    def test_clip_inplace(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.km / u.m
        c1 = q1.clip(1500, 5.5 * u.Mm / u.km, out=q1)
        assert np.all(q1 == np.array([1.5, 2.0, 4.0, 5.0, 5.5]) * u.km / u.m)
        c1[0] = 10 * u.Mm / u.mm
        assert np.all(c1.value == q1.value)

    def test_conj(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.km / u.m
        assert np.all(q1.conj() == q1)

    def test_ptp(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        assert np.ptp(q1) == 5.0 * u.m

    def test_ptp_inplace(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 4.0, 5.0, 6.0]) * u.m
        qi = 1.5 * u.s
        np.ptp(q1, out=qi)
        assert qi == 5.0 * u.m

    def test_round(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.253, 2.253, 3.253]) * u.kg
        assert np.all(np.round(q1) == np.array([1, 2, 3]) * u.kg)
        assert np.all(np.round(q1, decimals=2) == np.round(q1.value, decimals=2) * u.kg)
        assert np.all(q1.round(decimals=2) == q1.value.round(decimals=2) * u.kg)

    def test_round_inplace(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.253, 2.253, 3.253]) * u.kg
        qi = np.zeros(3) * u.s
        a = q1.round(decimals=2, out=qi)
        assert a is qi
        assert np.all(q1.round(decimals=2) == qi)

    def test_sum(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 6.0]) * u.m
        assert np.all(q1.sum() == 9.0 * u.m)
        assert np.all(np.sum(q1) == 9.0 * u.m)
        q2 = np.array([[4.0, 5.0, 9.0], [1.0, 1.0, 1.0]]) * u.s
        assert np.all(q2.sum(0) == np.array([5.0, 6.0, 10.0]) * u.s)
        assert np.all(np.sum(q2, 0) == np.array([5.0, 6.0, 10.0]) * u.s)

    def test_sum_inplace(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 6.0]) * u.m
        qi = 1.5 * u.s
        np.sum(q1, out=qi)
        assert qi == 9.0 * u.m

    def test_sum_where(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 6.0, 7.0]) * u.m
        where = q1 < 7 * u.m
        assert np.all(q1.sum(where=where) == 9.0 * u.m)
        assert np.all(np.sum(q1, where=where) == 9.0 * u.m)

    @pytest.mark.parametrize('initial', [0, 0 * u.m, 1 * u.km])
    def test_sum_initial(self, initial):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 6.0, 7.0]) * u.m
        expected = 16 * u.m + initial
        assert q1.sum(initial=initial) == expected
        assert np.sum(q1, initial=initial) == expected

    def test_sum_dimensionless_initial(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, 6.0, 7.0]) * u.one
        assert q1.sum(initial=1000) == 1016 * u.one

    @pytest.mark.parametrize('initial', [10, 1 * u.s])
    def test_sum_initial_exception(self, initial):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 6.0, 7.0]) * u.m
        with pytest.raises(u.UnitsError):
            q1.sum(initial=initial)

    def test_cumsum(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1, 2, 6]) * u.m
        assert np.all(q1.cumsum() == np.array([1, 3, 9]) * u.m)
        assert np.all(np.cumsum(q1) == np.array([1, 3, 9]) * u.m)
        q2 = np.array([4, 5, 9]) * u.s
        assert np.all(q2.cumsum() == np.array([4, 9, 18]) * u.s)
        assert np.all(np.cumsum(q2) == np.array([4, 9, 18]) * u.s)

    def test_cumsum_inplace(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1, 2, 6]) * u.m
        qi = np.ones(3) * u.s
        np.cumsum(q1, out=qi)
        assert np.all(qi == np.array([1, 3, 9]) * u.m)
        q2 = q1
        q1.cumsum(out=q1)
        assert np.all(q2 == qi)

    @pytest.mark.filterwarnings('ignore:The nansum method is deprecated')
    def test_nansum(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, np.nan]) * u.m
        assert np.all(q1.nansum() == 3.0 * u.m)
        assert np.all(np.nansum(q1) == 3.0 * u.m)
        q2 = np.array([[np.nan, 5.0, 9.0], [1.0, np.nan, 1.0]]) * u.s
        assert np.all(q2.nansum(0) == np.array([1.0, 5.0, 10.0]) * u.s)
        assert np.all(np.nansum(q2, 0) == np.array([1.0, 5.0, 10.0]) * u.s)

    @pytest.mark.filterwarnings('ignore:The nansum method is deprecated')
    def test_nansum_inplace(self):
        if False:
            print('Hello World!')
        q1 = np.array([1.0, 2.0, np.nan]) * u.m
        qi = 1.5 * u.s
        qout = q1.nansum(out=qi)
        assert qout is qi
        assert qi == np.nansum(q1.value) * q1.unit
        qi2 = 1.5 * u.s
        qout2 = np.nansum(q1, out=qi2)
        assert qout2 is qi2
        assert qi2 == np.nansum(q1.value) * q1.unit

    @pytest.mark.filterwarnings('ignore:The nansum method is deprecated')
    def test_nansum_where(self):
        if False:
            return 10
        q1 = np.array([1.0, 2.0, np.nan, 4.0]) * u.m
        initial = 0 * u.m
        where = q1 < 4 * u.m
        assert np.all(q1.nansum(initial=initial, where=where) == 3.0 * u.m)
        assert np.all(np.nansum(q1, initial=initial, where=where) == 3.0 * u.m)

    def test_prod(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1, 2, 6]) * u.m
        with pytest.raises(u.UnitsError) as exc:
            q1.prod()
        with pytest.raises(u.UnitsError) as exc:
            np.prod(q1)
        q2 = np.array([3.0, 4.0, 5.0]) * u.Unit(1)
        assert q2.prod() == 60.0 * u.Unit(1)
        assert np.prod(q2) == 60.0 * u.Unit(1)

    def test_cumprod(self):
        if False:
            print('Hello World!')
        q1 = np.array([1, 2, 6]) * u.m
        with pytest.raises(u.UnitsError) as exc:
            q1.cumprod()
        with pytest.raises(u.UnitsError) as exc:
            np.cumprod(q1)
        q2 = np.array([3, 4, 5]) * u.Unit(1)
        assert np.all(q2.cumprod() == np.array([3, 12, 60]) * u.Unit(1))
        assert np.all(np.cumprod(q2) == np.array([3, 12, 60]) * u.Unit(1))

    def test_diff(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1.0, 2.0, 4.0, 10.0]) * u.m
        assert np.all(q1.diff() == np.array([1.0, 2.0, 6.0]) * u.m)
        assert np.all(np.diff(q1) == np.array([1.0, 2.0, 6.0]) * u.m)

    def test_ediff1d(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1.0, 2.0, 4.0, 10.0]) * u.m
        assert np.all(q1.ediff1d() == np.array([1.0, 2.0, 6.0]) * u.m)
        assert np.all(np.ediff1d(q1) == np.array([1.0, 2.0, 6.0]) * u.m)

    def test_dot_meth(self):
        if False:
            i = 10
            return i + 15
        q1 = np.array([1.0, 2.0, 4.0, 10.0]) * u.m
        q2 = np.array([3.0, 4.0, 5.0, 6.0]) * u.s
        q3 = q1.dot(q2)
        assert q3.value == np.dot(q1.value, q2.value)
        assert q3.unit == u.m * u.s

    def test_trace_func(self):
        if False:
            for i in range(10):
                print('nop')
        q = np.array([[1.0, 2.0], [3.0, 4.0]]) * u.m
        assert np.trace(q) == 5.0 * u.m

    def test_trace_meth(self):
        if False:
            while True:
                i = 10
        q1 = np.array([[1.0, 2.0], [3.0, 4.0]]) * u.m
        assert q1.trace() == 5.0 * u.m
        cont = u.Quantity(4.0, u.s)
        q2 = np.array([[3.0, 4.0], [5.0, 6.0]]) * u.m
        q2.trace(out=cont)
        assert cont == 9.0 * u.m

    def test_clip_func(self):
        if False:
            while True:
                i = 10
        q = np.arange(10) * u.m
        assert np.all(np.clip(q, 3 * u.m, 6 * u.m) == np.array([3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0]) * u.m)

    def test_clip_meth(self):
        if False:
            while True:
                i = 10
        expected = np.array([3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0]) * u.m
        q1 = np.arange(10) * u.m
        q3 = q1.clip(3 * u.m, 6 * u.m)
        assert np.all(q1.clip(3 * u.m, 6 * u.m) == expected)
        cont = np.zeros(10) * u.s
        q1.clip(3 * u.m, 6 * u.m, out=cont)
        assert np.all(cont == expected)

class TestArrayConversion:
    """
    Test array conversion methods
    """

    def test_item(self):
        if False:
            print('Hello World!')
        q1 = u.Quantity(np.array([1, 2, 3]), u.m / u.km, dtype=int)
        assert q1.item(1) == 2 * q1.unit
        q1[1] = 1
        assert q1[1] == 1000 * u.m / u.km
        q1[1] = 100 * u.cm / u.km
        assert q1[1] == 1 * u.m / u.km
        with pytest.raises(TypeError):
            q1[1] = 1.5 * u.m / u.km

    @pytest.mark.skipif(not NUMPY_LT_2_0, reason='itemset method removed in numpy 2.0')
    def test_itemset(self):
        if False:
            print('Hello World!')
        q1 = u.Quantity(np.array([1, 2, 3]), u.m / u.km, dtype=int)
        assert q1.item(1) == 2 * q1.unit
        q1.itemset(1, 1)
        assert q1.item(1) == 1000 * u.m / u.km
        q1.itemset(1, 100 * u.cm / u.km)
        assert q1.item(1) == 1 * u.m / u.km
        with pytest.raises(TypeError):
            q1.itemset(1, 1.5 * u.m / u.km)
        with pytest.raises(ValueError):
            q1.itemset()

    def test_take_put(self):
        if False:
            return 10
        q1 = np.array([1, 2, 3]) * u.m / u.km
        assert q1.take(1) == 2 * u.m / u.km
        assert all(q1.take((0, 2)) == np.array([1, 3]) * u.m / u.km)
        q1.put((1, 2), (3, 4))
        assert np.all(q1.take((1, 2)) == np.array([3000, 4000]) * q1.unit)
        q1.put(0, 500 * u.cm / u.km)
        assert q1.item(0) == 5 * u.m / u.km

    def test_slice(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that setitem changes the unit if needed (or ignores it for\n        values where that is allowed; viz., #2695)'
        q2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * u.km / u.m
        q1 = q2.copy()
        q2[0, 0] = 10000.0
        assert q2.unit == q1.unit
        assert q2[0, 0].value == 10.0
        q2[0] = 9.0 * u.Mm / u.km
        assert all(q2.flatten()[:3].value == np.array([9.0, 9.0, 9.0]))
        q2[0, :-1] = 8000.0
        assert all(q2.flatten()[:3].value == np.array([8.0, 8.0, 9.0]))
        with pytest.raises(u.UnitsError):
            q2[1, 1] = 10 * u.s
        q3 = u.Quantity(np.arange(10.0), 'm/s')
        q3[5] = 100.0 * u.cm / u.s
        assert q3[5].value == 1.0
        q3[5] = 0.0
        assert q3[5] == 0.0
        q3[5] = np.inf
        assert np.isinf(q3[5])
        q3[5] = np.nan
        assert np.isnan(q3[5])

    def test_fill(self):
        if False:
            print('Hello World!')
        q1 = np.array([1, 2, 3]) * u.m / u.km
        q1.fill(2)
        assert np.all(q1 == 2000 * u.m / u.km)

    def test_repeat_compress_diagonal(self):
        if False:
            print('Hello World!')
        q1 = np.array([1, 2, 3]) * u.m / u.km
        q2 = q1.repeat(2)
        assert q2.unit == q1.unit
        assert all(q2.value == q1.value.repeat(2))
        q2.sort()
        assert q2.unit == q1.unit
        q2 = q1.compress(np.array([True, True, False, False]))
        assert q2.unit == q1.unit
        assert all(q2.value == q1.value.compress(np.array([True, True, False, False])))
        q1 = np.array([[1, 2], [3, 4]]) * u.m / u.km
        q2 = q1.diagonal()
        assert q2.unit == q1.unit
        assert all(q2.value == q1.value.diagonal())

    def test_view(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1, 2, 3], dtype=np.int64) * u.m / u.km
        q2 = q1.view(np.ndarray)
        assert not hasattr(q2, 'unit')
        q3 = q2.view(u.Quantity)
        assert q3._unit is None
        q4 = np.ma.MaskedArray(q1)
        assert q4._unit is q1._unit
        q5 = q4.view(u.Quantity)
        assert q5.unit is q1.unit

    def test_slice_to_quantity(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Regression test for https://github.com/astropy/astropy/issues/2003\n        '
        a = np.random.uniform(size=(10, 8))
        (x, y, z) = a[:, 1:4].T * u.km / u.s
        total = np.sum(a[:, 1] * u.km / u.s - x)
        assert isinstance(total, u.Quantity)
        assert total == 0.0 * u.km / u.s

    def test_byte_type_view_field_changes(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = np.array([1, 2, 3], dtype=np.int64) * u.m / u.km
        q2 = q1.byteswap()
        assert q2.unit == q1.unit
        assert all(q2.value == q1.value.byteswap())
        q2 = q1.astype(np.float64)
        assert all(q2 == q1)
        assert q2.dtype == np.float64
        q2a = q1.getfield(np.int32, offset=0)
        q2b = q1.byteswap().getfield(np.int32, offset=4)
        assert q2a.unit == q1.unit
        assert all(q2b.byteswap() == q2a)

    def test_sort(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1.0, 5.0, 2.0, 4.0]) * u.km / u.m
        i = q1.argsort()
        assert not hasattr(i, 'unit')
        q1.sort()
        i = q1.searchsorted([1500, 2500])
        assert not hasattr(i, 'unit')
        assert all(i == q1.to(u.dimensionless_unscaled).value.searchsorted([1500, 2500]))

    def test_not_implemented(self):
        if False:
            while True:
                i = 10
        q1 = np.array([1, 2, 3]) * u.m / u.km
        with pytest.raises(NotImplementedError):
            q1.choose([0, 0, 1])
        with pytest.raises(NotImplementedError):
            q1.tolist()
        with pytest.raises(NotImplementedError):
            q1.tostring()
        with pytest.raises(NotImplementedError):
            q1.tobytes()
        with pytest.raises(NotImplementedError):
            q1.tofile(0)
        with pytest.raises(NotImplementedError):
            q1.dump('a.a')
        with pytest.raises(NotImplementedError):
            q1.dumps()

class TestStructuredArray:
    """Structured arrays are not specifically supported, but we should not
    prevent their use unnecessarily.

    Note that these tests use simple units.  Now that structured units are
    supported, it may make sense to deprecate this.
    """

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.ra = np.array(np.arange(12.0).reshape(4, 3)).view(dtype='f8,f8,f8').squeeze()

    def test_creation(self):
        if False:
            i = 10
            return i + 15
        qra = u.Quantity(self.ra, u.m)
        assert np.all(qra[:2].value == self.ra[:2])

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        qra = u.Quantity(self.ra, u.m)
        qra[1] = qra[2]
        assert qra[1] == qra[2]

    def test_assignment_with_non_structured(self):
        if False:
            return 10
        qra = u.Quantity(self.ra, u.m)
        qra[1] = 0
        assert qra[1] == np.zeros(3).view(qra.dtype)

    def test_assignment_with_different_names(self):
        if False:
            return 10
        qra = u.Quantity(self.ra, u.m)
        dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        value = np.array((-1.0, -2.0, -3.0), dtype) << u.km
        qra[1] = value
        assert qra[1] == value
        assert qra[1].value == np.array((-1000.0, -2000.0, -3000.0), qra.dtype)
        assert value.dtype.names == ('x', 'y', 'z')