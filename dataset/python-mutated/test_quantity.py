"""Test the Quantity class and related."""
import copy
import decimal
import numbers
import pickle
from fractions import Fraction
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from astropy import units as u
from astropy.units.quantity import _UNIT_NOT_INITIALISED
from astropy.utils import isiterable, minversion
from astropy.utils.exceptions import AstropyWarning
' The Quantity class will represent a number + unit + uncertainty '

class TestQuantityCreation:

    def test_1(self):
        if False:
            while True:
                i = 10
        quantity = 11.42 * u.meter
        assert isinstance(quantity, u.Quantity)
        quantity = u.meter * 11.42
        assert isinstance(quantity, u.Quantity)
        quantity = 11.42 / u.meter
        assert isinstance(quantity, u.Quantity)
        quantity = u.meter / 11.42
        assert isinstance(quantity, u.Quantity)
        quantity = 11.42 * u.meter / u.second
        assert isinstance(quantity, u.Quantity)
        with pytest.raises(TypeError):
            quantity = 182.234 + u.meter
        with pytest.raises(TypeError):
            quantity = 182.234 - u.meter
        with pytest.raises(TypeError):
            quantity = 182.234 % u.meter

    def test_2(self):
        if False:
            print('Hello World!')
        _ = u.Quantity(11.412, unit=u.meter)
        _ = u.Quantity(21.52, 'cm')
        q3 = u.Quantity(11.412)
        assert q3.unit == u.Unit(1)
        with pytest.raises(TypeError):
            u.Quantity(object(), unit=u.m)

    def test_3(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            u.Quantity(11.412, unit='testingggg')

    def test_nan_inf(self):
        if False:
            for i in range(10):
                print('nop')
        q = u.Quantity('nan', unit='cm')
        assert np.isnan(q.value)
        q = u.Quantity('NaN', unit='cm')
        assert np.isnan(q.value)
        q = u.Quantity('-nan', unit='cm')
        assert np.isnan(q.value)
        q = u.Quantity('nan cm')
        assert np.isnan(q.value)
        assert q.unit == u.cm
        q = u.Quantity('inf', unit='cm')
        assert np.isinf(q.value)
        q = u.Quantity('-inf', unit='cm')
        assert np.isinf(q.value)
        q = u.Quantity('inf cm')
        assert np.isinf(q.value)
        assert q.unit == u.cm
        q = u.Quantity('Infinity', unit='cm')
        assert np.isinf(q.value)
        with pytest.raises(TypeError):
            q = u.Quantity('', unit='cm')
        with pytest.raises(TypeError):
            q = u.Quantity('spam', unit='cm')

    def test_unit_property(self):
        if False:
            while True:
                i = 10
        q1 = u.Quantity(11.4, unit=u.meter)
        with pytest.raises(AttributeError):
            q1.unit = u.cm

    def test_preserve_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that if an explicit dtype is given, it is used, while if not,\n        numbers are converted to float (including decimal.Decimal, which\n        numpy converts to an object; closes #1419)\n        '
        q1 = u.Quantity(12, unit=u.m / u.s, dtype=int)
        assert q1.dtype == int
        q2 = u.Quantity(q1)
        assert q2.dtype == float
        assert q2.value == float(q1.value)
        assert q2.unit == q1.unit
        a3_32 = np.array([1.0, 2.0], dtype=np.float32)
        q3_32 = u.Quantity(a3_32, u.yr)
        assert q3_32.dtype == a3_32.dtype
        a3_16 = np.array([1.0, 2.0], dtype=np.float16)
        q3_16 = u.Quantity(a3_16, u.yr)
        assert q3_16.dtype == a3_16.dtype
        q4 = u.Quantity(decimal.Decimal('10.25'), u.m)
        assert q4.dtype == float
        q5 = u.Quantity(decimal.Decimal('10.25'), u.m, dtype=object)
        assert q5.dtype == object

    def test_numpy_style_dtype_inspect(self):
        if False:
            while True:
                i = 10
        "Test that if ``dtype=None``, NumPy's dtype inspection is used."
        q2 = u.Quantity(12, dtype=None)
        assert np.issubdtype(q2.dtype, np.integer)

    def test_float_dtype_promotion(self):
        if False:
            while True:
                i = 10
        'Test that if ``dtype=numpy.inexact``, the minimum precision is float64.'
        q1 = u.Quantity(12, dtype=np.inexact)
        assert not np.issubdtype(q1.dtype, np.integer)
        assert q1.dtype == np.float64
        q2 = u.Quantity(np.float64(12), dtype=np.inexact)
        assert q2.dtype == np.float64
        q3 = u.Quantity(np.float32(12), dtype=np.inexact)
        assert q3.dtype == np.float32
        if hasattr(np, 'float16'):
            q3 = u.Quantity(np.float16(12), dtype=np.inexact)
            assert q3.dtype == np.float16
        if hasattr(np, 'float128'):
            q4 = u.Quantity(np.float128(12), dtype=np.inexact)
            assert q4.dtype == np.float128

    def test_copy(self):
        if False:
            return 10
        a = np.arange(10.0)
        q0 = u.Quantity(a, unit=u.m / u.s)
        assert q0.base is not a
        q1 = u.Quantity(a, unit=u.m / u.s, copy=False)
        assert q1.base is a
        q2 = u.Quantity(q0)
        assert q2 is not q0
        assert q2.base is not q0.base
        q2 = u.Quantity(q0, copy=False)
        assert q2 is q0
        assert q2.base is q0.base
        q3 = u.Quantity(q0, q0.unit, copy=False)
        assert q3 is q0
        assert q3.base is q0.base
        q4 = u.Quantity(q0, u.cm / u.s, copy=False)
        assert q4 is not q0
        assert q4.base is not q0.base

    def test_subok(self):
        if False:
            i = 10
            return i + 15
        'Test subok can be used to keep class, or to insist on Quantity'

        class MyQuantitySubclass(u.Quantity):
            pass
        myq = MyQuantitySubclass(np.arange(10.0), u.m)
        assert type(u.Quantity(myq)) is u.Quantity
        assert type(u.Quantity(myq, subok=True)) is MyQuantitySubclass
        assert type(u.Quantity(myq, u.km)) is u.Quantity
        assert type(u.Quantity(myq, u.km, subok=True)) is MyQuantitySubclass

    def test_order(self):
        if False:
            print('Hello World!')
        'Test that order is correctly propagated to np.array'
        ac = np.array(np.arange(10.0), order='C')
        qcc = u.Quantity(ac, u.m, order='C')
        assert qcc.flags['C_CONTIGUOUS']
        qcf = u.Quantity(ac, u.m, order='F')
        assert qcf.flags['F_CONTIGUOUS']
        qca = u.Quantity(ac, u.m, order='A')
        assert qca.flags['C_CONTIGUOUS']
        assert u.Quantity(qcc, order='C').flags['C_CONTIGUOUS']
        assert u.Quantity(qcc, order='A').flags['C_CONTIGUOUS']
        assert u.Quantity(qcc, order='F').flags['F_CONTIGUOUS']
        af = np.array(np.arange(10.0), order='F')
        qfc = u.Quantity(af, u.m, order='C')
        assert qfc.flags['C_CONTIGUOUS']
        qff = u.Quantity(ac, u.m, order='F')
        assert qff.flags['F_CONTIGUOUS']
        qfa = u.Quantity(af, u.m, order='A')
        assert qfa.flags['F_CONTIGUOUS']
        assert u.Quantity(qff, order='C').flags['C_CONTIGUOUS']
        assert u.Quantity(qff, order='A').flags['F_CONTIGUOUS']
        assert u.Quantity(qff, order='F').flags['F_CONTIGUOUS']

    def test_ndmin(self):
        if False:
            while True:
                i = 10
        'Test that ndmin is correctly propagated to np.array'
        a = np.arange(10.0)
        q1 = u.Quantity(a, u.m, ndmin=1)
        assert q1.ndim == 1 and q1.shape == (10,)
        q2 = u.Quantity(a, u.m, ndmin=2)
        assert q2.ndim == 2 and q2.shape == (1, 10)
        q3 = u.Quantity(q1, u.m, ndmin=3)
        assert q3.ndim == 3 and q3.shape == (1, 1, 10)
        assert u.Quantity(u.Quantity(1, 'm'), 'm', ndmin=1).ndim == 1
        assert u.Quantity(u.Quantity(1, 'cm'), 'm', ndmin=1).ndim == 1

    def test_non_quantity_with_unit(self):
        if False:
            return 10
        'Test that unit attributes in objects get recognized.'

        class MyQuantityLookalike(np.ndarray):
            pass
        a = np.arange(3.0)
        mylookalike = a.copy().view(MyQuantityLookalike)
        mylookalike.unit = 'm'
        q1 = u.Quantity(mylookalike)
        assert isinstance(q1, u.Quantity)
        assert q1.unit is u.m
        assert np.all(q1.value == a)
        q2 = u.Quantity(mylookalike, u.mm)
        assert q2.unit is u.mm
        assert np.all(q2.value == 1000.0 * a)
        q3 = u.Quantity(mylookalike, copy=False)
        assert np.all(q3.value == mylookalike)
        q3[2] = 0
        assert q3[2] == 0.0
        assert mylookalike[2] == 0.0
        mylookalike = a.copy().view(MyQuantityLookalike)
        mylookalike.unit = u.m
        q4 = u.Quantity(mylookalike, u.mm, copy=False)
        q4[2] = 0
        assert q4[2] == 0.0
        assert mylookalike[2] == 2.0
        mylookalike.unit = 'nonsense'
        with pytest.raises(TypeError):
            u.Quantity(mylookalike)

    def test_creation_via_view(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = 1.0 << u.m
        assert isinstance(q1, u.Quantity)
        assert q1.unit == u.m
        assert q1.value == 1.0
        a2 = np.arange(10.0)
        q2 = a2 << u.m / u.s
        assert isinstance(q2, u.Quantity)
        assert q2.unit == u.m / u.s
        assert np.all(q2.value == a2)
        a2[9] = 0.0
        assert np.all(q2.value == a2)
        q3 = q2 << u.mm / u.s
        assert isinstance(q3, u.Quantity)
        assert q3.unit == u.mm / u.s
        assert np.all(q3.value == a2 * 1000.0)
        a2[8] = 0.0
        assert q3[8].value == 8000.0
        q4 = q2 << q2.unit
        a2[7] = 0.0
        assert np.all(q4.value == a2)
        with pytest.raises(u.UnitsError):
            q2 << u.s
        a2_copy = a2.copy()
        q2 <<= u.mm / u.s
        assert q2.unit == u.mm / u.s
        assert np.all(q2.value == a2)
        assert np.all(q2.value == a2_copy * 1000.0)
        a2[8] = -1.0
        q5 = q2 << 'km/hr'
        assert q5.unit == u.km / u.hr
        assert np.all(q5 == q2)
        not_quite_a_foot = 30.0 * u.cm
        a6 = np.arange(5.0)
        q6 = a6 << not_quite_a_foot
        assert q6.unit == u.Unit(not_quite_a_foot)
        assert np.all(q6.to_value(u.cm) == 30.0 * a6)

    def test_rshift_warns(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError), pytest.warns(AstropyWarning, match='is not implemented') as warning_lines:
            1 >> u.m
        assert len(warning_lines) == 1
        q = 1.0 * u.km
        with pytest.raises(TypeError), pytest.warns(AstropyWarning, match='is not implemented') as warning_lines:
            q >> u.m
        assert len(warning_lines) == 1
        with pytest.raises(TypeError), pytest.warns(AstropyWarning, match='is not implemented') as warning_lines:
            q >>= u.m
        assert len(warning_lines) == 1
        with pytest.raises(TypeError), pytest.warns(AstropyWarning, match='is not implemented') as warning_lines:
            1.0 >> q
        assert len(warning_lines) == 1

class TestQuantityOperations:
    q1 = u.Quantity(11.42, u.meter)
    q2 = u.Quantity(8.0, u.centimeter)

    def test_addition(self):
        if False:
            while True:
                i = 10
        new_quantity = self.q1 + self.q2
        assert new_quantity.value == 11.5
        assert new_quantity.unit == u.meter
        new_quantity = self.q2 + self.q1
        assert new_quantity.value == 1150.0
        assert new_quantity.unit == u.centimeter
        new_q = u.Quantity(1500.1, u.m) + u.Quantity(13.5, u.km)
        assert new_q.unit == u.m
        assert new_q.value == 15000.1

    def test_subtraction(self):
        if False:
            for i in range(10):
                print('nop')
        new_quantity = self.q1 - self.q2
        assert new_quantity.value == 11.34
        assert new_quantity.unit == u.meter
        new_quantity = self.q2 - self.q1
        assert new_quantity.value == -1134.0
        assert new_quantity.unit == u.centimeter

    def test_multiplication(self):
        if False:
            for i in range(10):
                print('nop')
        new_quantity = self.q1 * self.q2
        assert new_quantity.value == 91.36
        assert new_quantity.unit == u.meter * u.centimeter
        new_quantity = self.q2 * self.q1
        assert new_quantity.value == 91.36
        assert new_quantity.unit == u.centimeter * u.meter
        new_quantity = 15.0 * self.q1
        assert new_quantity.value == 171.3
        assert new_quantity.unit == u.meter
        new_quantity = self.q1 * 15.0
        assert new_quantity.value == 171.3
        assert new_quantity.unit == u.meter
        new_quantity = self.q1 * u.s
        assert new_quantity.value == 11.42
        assert new_quantity.unit == u.Unit('m s')
        new_quantity = u.s * self.q1
        assert new_quantity.value == 11.42
        assert new_quantity.unit == u.Unit('m s')

    def test_division(self):
        if False:
            i = 10
            return i + 15
        new_quantity = self.q1 / self.q2
        assert_array_almost_equal(new_quantity.value, 1.4275, decimal=5)
        assert new_quantity.unit == u.meter / u.centimeter
        new_quantity = self.q2 / self.q1
        assert_array_almost_equal(new_quantity.value, 0.7005253940455342, decimal=16)
        assert new_quantity.unit == u.centimeter / u.meter
        q1 = u.Quantity(11.4, unit=u.meter)
        q2 = u.Quantity(10.0, unit=u.second)
        new_quantity = q1 / q2
        assert_array_almost_equal(new_quantity.value, 1.14, decimal=10)
        assert new_quantity.unit == u.meter / u.second
        new_quantity = self.q1 / 10.0
        assert new_quantity.value == 1.142
        assert new_quantity.unit == u.meter
        new_quantity = 11.42 / self.q1
        assert new_quantity.value == 1.0
        assert new_quantity.unit == u.Unit('1/m')
        new_quantity = self.q1 / u.s
        assert new_quantity.value == 11.42
        assert new_quantity.unit == u.Unit('m/s')
        new_quantity = u.s / self.q1
        assert new_quantity.value == 1 / 11.42
        assert new_quantity.unit == u.Unit('s/m')

    def test_commutativity(self):
        if False:
            while True:
                i = 10
        'Regression test for issue #587.'
        new_q = u.Quantity(11.42, 'm*s')
        assert self.q1 * u.s == u.s * self.q1 == new_q
        assert self.q1 / u.s == u.Quantity(11.42, 'm/s')
        assert u.s / self.q1 == u.Quantity(1 / 11.42, 's/m')

    def test_power(self):
        if False:
            print('Hello World!')
        new_quantity = self.q1 ** 2
        assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)
        assert new_quantity.unit == u.Unit('m^2')
        new_quantity = self.q1 ** 3
        assert_array_almost_equal(new_quantity.value, 1489.355288, decimal=7)
        assert new_quantity.unit == u.Unit('m^3')

    def test_matrix_multiplication(self):
        if False:
            print('Hello World!')
        a = np.eye(3)
        q = a * u.m
        result1 = q @ a
        assert np.all(result1 == q)
        result2 = a @ q
        assert np.all(result2 == q)
        result3 = q @ q
        assert np.all(result3 == a * u.m ** 2)
        q2 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]) / u.s
        result4 = q @ q2
        assert np.all(result4 == np.matmul(a, q2.value) * q.unit * q2.unit)

    def test_unary(self):
        if False:
            i = 10
            return i + 15
        new_quantity = -self.q1
        assert new_quantity.value == -self.q1.value
        assert new_quantity.unit == self.q1.unit
        new_quantity = --self.q1
        assert new_quantity.value == self.q1.value
        assert new_quantity.unit == self.q1.unit
        new_quantity = +self.q1
        assert new_quantity.value == self.q1.value
        assert new_quantity.unit == self.q1.unit

    def test_abs(self):
        if False:
            return 10
        q = 1.0 * u.m / u.s
        new_quantity = abs(q)
        assert new_quantity.value == q.value
        assert new_quantity.unit == q.unit
        q = -1.0 * u.m / u.s
        new_quantity = abs(q)
        assert new_quantity.value == -q.value
        assert new_quantity.unit == q.unit

    def test_incompatible_units(self):
        if False:
            i = 10
            return i + 15
        "When trying to add or subtract units that aren't compatible, throw an error"
        q1 = u.Quantity(11.412, unit=u.meter)
        q2 = u.Quantity(21.52, unit=u.second)
        with pytest.raises(u.UnitsError):
            q1 + q2

    def test_non_number_type(self):
        if False:
            i = 10
            return i + 15
        q1 = u.Quantity(11.412, unit=u.meter)
        with pytest.raises(TypeError, match='Unsupported operand type\\(s\\) for ufunc .*'):
            q1 + {'a': 1}
        with pytest.raises(TypeError):
            q1 + u.meter

    def test_dimensionless_operations(self):
        if False:
            while True:
                i = 10
        dq = 3.0 * u.m / u.km
        dq1 = dq + 1.0 * u.mm / u.km
        assert dq1.value == 3.001
        assert dq1.unit == dq.unit
        dq2 = dq + 1.0
        assert dq2.value == 1.003
        assert dq2.unit == u.dimensionless_unscaled
        with pytest.raises(u.UnitsError):
            self.q1 + u.Quantity(0.1, unit=u.Unit(''))
        with pytest.raises(u.UnitsError):
            self.q1 - u.Quantity(0.1, unit=u.Unit(''))
        q = u.Quantity(np.array([1, 2, 3]), u.m / u.km, dtype=int)
        q2 = q + np.array([4, 5, 6])
        assert q2.unit == u.dimensionless_unscaled
        assert_allclose(q2.value, np.array([4.001, 5.002, 6.003]))
        with pytest.raises(TypeError):
            q += np.array([1, 2, 3])
        q = np.array([1, 2, 3]) * u.km / u.m
        q += np.array([4, 5, 6])
        assert q.unit == u.dimensionless_unscaled
        assert np.all(q.value == np.array([1004, 2005, 3006]))

    def test_complicated_operation(self):
        if False:
            for i in range(10):
                print('nop')
        'Perform a more complicated test'
        from astropy.units import imperial
        distance = u.Quantity(15.0, u.meter)
        time = u.Quantity(11.0, u.second)
        velocity = (distance / time).to(imperial.mile / u.hour)
        assert_array_almost_equal(velocity.value, 3.05037, decimal=5)
        G = u.Quantity(6.673e-11, u.m ** 3 / u.kg / u.s ** 2)
        _ = (1.0 / (4.0 * np.pi * G)).to(u.pc ** (-3) / u.s ** (-2) * u.kg)
        side1 = u.Quantity(11.0, u.centimeter)
        side2 = u.Quantity(7.0, u.centimeter)
        area = side1 * side2
        assert_array_almost_equal(area.value, 77.0, decimal=15)
        assert area.unit == u.cm * u.cm

    def test_comparison(self):
        if False:
            print('Hello World!')
        assert 1 / (u.cm * u.cm) == 1 * u.cm ** (-2)
        assert 1 * u.m == 100 * u.cm
        assert 1 * u.m != 1 * u.cm
        unit = u.cm ** 3
        q = 1.0 * unit
        assert q.__eq__(unit) is NotImplemented
        assert unit.__eq__(q) is True
        assert q == unit
        q = 1000.0 * u.mm ** 3
        assert q == unit
        assert not 1.0 * u.cm == 1.0
        assert 1.0 * u.cm != 1.0
        for quantity in (1.0 * u.cm, 1.0 * u.dimensionless_unscaled):
            with pytest.raises(ValueError, match='ambiguous'):
                bool(quantity)

    def test_numeric_converters(self):
        if False:
            print('Hello World!')
        q1 = u.Quantity(1, u.m)
        converter_err_msg = 'only dimensionless scalar quantities can be converted to Python scalars'
        index_err_msg = 'only integer dimensionless scalar quantities can be converted to a Python index'
        with pytest.raises(TypeError) as exc:
            float(q1)
        assert exc.value.args[0] == converter_err_msg
        with pytest.raises(TypeError) as exc:
            int(q1)
        assert exc.value.args[0] == converter_err_msg
        with pytest.raises(TypeError) as exc:
            q1.__index__()
        assert exc.value.args[0] == index_err_msg
        q2 = u.Quantity(1.23, u.m / u.km)
        assert float(q2) == float(q2.to_value(u.dimensionless_unscaled))
        assert int(q2) == int(q2.to_value(u.dimensionless_unscaled))
        with pytest.raises(TypeError) as exc:
            q2.__index__()
        assert exc.value.args[0] == index_err_msg
        q3 = u.Quantity(1.23, u.dimensionless_unscaled)
        assert float(q3) == 1.23
        assert int(q3) == 1
        with pytest.raises(TypeError) as exc:
            q3.__index__()
        assert exc.value.args[0] == index_err_msg
        q4 = u.Quantity(2, u.dimensionless_unscaled, dtype=int)
        assert float(q4) == 2.0
        assert int(q4) == 2
        assert q4.__index__() == 2
        q5 = u.Quantity([1, 2], u.m)
        with pytest.raises(TypeError) as exc:
            float(q5)
        assert exc.value.args[0] == converter_err_msg
        with pytest.raises(TypeError) as exc:
            int(q5)
        assert exc.value.args[0] == converter_err_msg
        with pytest.raises(TypeError) as exc:
            q5.__index__()
        assert exc.value.args[0] == index_err_msg

    @pytest.mark.xfail(reason='list multiplication only works for numpy <=1.10')
    def test_numeric_converter_to_index_in_practice(self):
        if False:
            while True:
                i = 10
        'Test that use of __index__ actually works.'
        q4 = u.Quantity(2, u.dimensionless_unscaled, dtype=int)
        assert q4 * ['a', 'b', 'c'] == ['a', 'b', 'c', 'a', 'b', 'c']

    def test_array_converters(self):
        if False:
            return 10
        q = u.Quantity(1.23, u.m)
        assert np.all(np.array(q) == np.array([1.23]))
        q = u.Quantity([1.0, 2.0, 3.0], u.m)
        assert np.all(np.array(q) == np.array([1.0, 2.0, 3.0]))

def test_quantity_conversion():
    if False:
        while True:
            i = 10
    q1 = u.Quantity(0.1, unit=u.meter)
    value = q1.value
    assert value == 0.1
    value_in_km = q1.to_value(u.kilometer)
    assert value_in_km == 0.0001
    new_quantity = q1.to(u.kilometer)
    assert new_quantity.value == 0.0001
    with pytest.raises(u.UnitsError):
        q1.to(u.zettastokes)
    with pytest.raises(u.UnitsError):
        q1.to_value(u.zettastokes)

def test_quantity_ilshift():
    if False:
        for i in range(10):
            print('nop')
    q = u.Quantity(10, unit=u.one)
    with pytest.raises(u.UnitConversionError):
        q <<= u.rad
    with u.add_enabled_equivalencies(u.dimensionless_angles()):
        q <<= u.rad
    assert np.isclose(q, 10 * u.rad)

def test_regression_12964():
    if False:
        return 10
    x = u.Quantity(10, u.km, dtype=int)
    x <<= u.pc
    assert x.unit is u.pc
    assert x.dtype == np.float64

def test_quantity_value_views():
    if False:
        return 10
    q1 = u.Quantity([1.0, 2.0], unit=u.meter)
    v1 = q1.value
    v1[0] = 0.0
    assert np.all(q1 == [0.0, 2.0] * u.meter)
    v2 = q1.to_value()
    v2[1] = 3.0
    assert np.all(q1 == [0.0, 3.0] * u.meter)
    v3 = q1.to_value('m')
    v3[0] = 1.0
    assert np.all(q1 == [1.0, 3.0] * u.meter)
    q2 = q1.to('m', copy=False)
    q2[0] = 2 * u.meter
    assert np.all(q1 == [2.0, 3.0] * u.meter)
    v4 = q1.to_value('cm')
    v4[0] = 0.0
    assert np.all(q1 == [2.0, 3.0] * u.meter)

def test_quantity_conversion_with_equiv():
    if False:
        while True:
            i = 10
    q1 = u.Quantity(0.1, unit=u.meter)
    v2 = q1.to_value(u.Hz, equivalencies=u.spectral())
    assert_allclose(v2, 2997924580.0)
    q2 = q1.to(u.Hz, equivalencies=u.spectral())
    assert_allclose(q2.value, v2)
    q1 = u.Quantity(0.4, unit=u.arcsecond)
    v2 = q1.to_value(u.au, equivalencies=u.parallax())
    q2 = q1.to(u.au, equivalencies=u.parallax())
    v3 = q2.to_value(u.arcminute, equivalencies=u.parallax())
    q3 = q2.to(u.arcminute, equivalencies=u.parallax())
    assert_allclose(v2, 515662.015)
    assert_allclose(q2.value, v2)
    assert q2.unit == u.au
    assert_allclose(v3, 0.0066666667)
    assert_allclose(q3.value, v3)
    assert q3.unit == u.arcminute

def test_quantity_conversion_equivalency_passed_on():
    if False:
        print('Hello World!')

    class MySpectral(u.Quantity):
        _equivalencies = u.spectral()

        def __quantity_view__(self, obj, unit):
            if False:
                for i in range(10):
                    print('nop')
            return obj.view(MySpectral)

        def __quantity_instance__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return MySpectral(*args, **kwargs)
    q1 = MySpectral([1000, 2000], unit=u.Hz)
    q2 = q1.to(u.nm)
    assert q2.unit == u.nm
    q3 = q2.to(u.Hz)
    assert q3.unit == u.Hz
    assert_allclose(q3.value, q1.value)
    q4 = MySpectral([1000, 2000], unit=u.nm)
    q5 = q4.to(u.Hz).to(u.nm)
    assert q5.unit == u.nm
    assert_allclose(q4.value, q5.value)

def test_self_equivalency():
    if False:
        while True:
            i = 10
    assert u.deg.is_equivalent(0 * u.radian)
    assert u.deg.is_equivalent(1 * u.radian)

def test_si():
    if False:
        i = 10
        return i + 15
    q1 = 10.0 * u.m * u.s ** 2 / (200.0 * u.ms) ** 2
    assert q1.si.value == 250
    assert q1.si.unit == u.m
    q = 10.0 * u.m
    assert q.si.value == 10
    assert q.si.unit == u.m
    q = 10.0 / u.m
    assert q.si.value == 10
    assert q.si.unit == 1 / u.m

def test_cgs():
    if False:
        for i in range(10):
            print('nop')
    q1 = 10.0 * u.cm * u.s ** 2 / (200.0 * u.ms) ** 2
    assert q1.cgs.value == 250
    assert q1.cgs.unit == u.cm
    q = 10.0 * u.m
    assert q.cgs.value == 1000
    assert q.cgs.unit == u.cm
    q = 10.0 / u.cm
    assert q.cgs.value == 10
    assert q.cgs.unit == 1 / u.cm
    q = 10.0 * u.Pa
    assert q.cgs.value == 100
    assert q.cgs.unit == u.barye

class TestQuantityComparison:

    def test_quantity_equality(self):
        if False:
            return 10
        assert u.Quantity(1000, unit='m') == u.Quantity(1, unit='km')
        assert not u.Quantity(1, unit='m') == u.Quantity(1, unit='km')
        assert (u.Quantity(1100, unit=u.m) != u.Quantity(1, unit=u.s)) is True
        assert (u.Quantity(1100, unit=u.m) == u.Quantity(1, unit=u.s)) is False
        assert (u.Quantity(0, unit=u.m) == u.Quantity(0, unit=u.s)) is False
        assert u.Quantity(0, u.m) == 0.0
        assert u.Quantity(1, u.m) != 0.0
        assert u.Quantity(1, u.m) != np.inf
        assert u.Quantity(np.inf, u.m) == np.inf

    def test_quantity_equality_array(self):
        if False:
            for i in range(10):
                print('nop')
        a = u.Quantity([0.0, 1.0, 1000.0], u.m)
        b = u.Quantity(1.0, u.km)
        eq = a == b
        ne = a != b
        assert np.all(eq == [False, False, True])
        assert np.all(eq != ne)
        c = u.Quantity(1.0, u.s)
        eq = a == c
        ne = a != c
        assert eq is False
        assert ne is True
        eq = a == 1.0
        ne = a != 1.0
        assert eq is False
        assert ne is True
        eq = a == 0
        ne = a != 0
        assert np.all(eq == [True, False, False])
        assert np.all(eq != ne)
        d = np.array([0, 1.0, 1000.0])
        eq = a == d
        ne = a != d
        assert eq is False
        assert ne is True

    def test_quantity_comparison(self):
        if False:
            i = 10
            return i + 15
        assert u.Quantity(1100, unit=u.meter) > u.Quantity(1, unit=u.kilometer)
        assert u.Quantity(900, unit=u.meter) < u.Quantity(1, unit=u.kilometer)
        with pytest.raises(u.UnitsError):
            assert u.Quantity(1100, unit=u.meter) > u.Quantity(1, unit=u.second)
        with pytest.raises(u.UnitsError):
            assert u.Quantity(1100, unit=u.meter) < u.Quantity(1, unit=u.second)
        assert u.Quantity(1100, unit=u.meter) >= u.Quantity(1, unit=u.kilometer)
        assert u.Quantity(1000, unit=u.meter) >= u.Quantity(1, unit=u.kilometer)
        assert u.Quantity(900, unit=u.meter) <= u.Quantity(1, unit=u.kilometer)
        assert u.Quantity(1000, unit=u.meter) <= u.Quantity(1, unit=u.kilometer)
        with pytest.raises(u.UnitsError):
            assert u.Quantity(1100, unit=u.meter) >= u.Quantity(1, unit=u.second)
        with pytest.raises(u.UnitsError):
            assert u.Quantity(1100, unit=u.meter) <= u.Quantity(1, unit=u.second)
        assert u.Quantity(1200, unit=u.meter) != u.Quantity(1, unit=u.kilometer)

class TestQuantityDisplay:
    scalarintq = u.Quantity(1, unit='m', dtype=int)
    scalarfloatq = u.Quantity(1.3, unit='m')
    arrq = u.Quantity([1, 2.3, 8.9], unit='m')
    scalar_complex_q = u.Quantity(complex(1.0, 2.0))
    scalar_big_complex_q = u.Quantity(complex(1.0, 2e+27) * 1e+25)
    scalar_big_neg_complex_q = u.Quantity(complex(-1.0, -2e+27) * 1e+36)
    arr_complex_q = u.Quantity(np.arange(3) * (complex(-1.0, -2e+27) * 1e+36))
    big_arr_complex_q = u.Quantity(np.arange(125) * (complex(-1.0, -2e+27) * 1e+36))

    def test_dimensionless_quantity_repr(self):
        if False:
            while True:
                i = 10
        q2 = u.Quantity(1.0, unit='m-1')
        q3 = u.Quantity(1, unit='m-1', dtype=int)
        assert repr(self.scalarintq * q2) == '<Quantity 1.>'
        assert repr(self.arrq * q2) == '<Quantity [1. , 2.3, 8.9]>'
        assert repr(self.scalarintq * q3) == '<Quantity 1>'

    def test_dimensionless_quantity_str(self):
        if False:
            return 10
        q2 = u.Quantity(1.0, unit='m-1')
        q3 = u.Quantity(1, unit='m-1', dtype=int)
        assert str(self.scalarintq * q2) == '1.0'
        assert str(self.scalarintq * q3) == '1'
        assert str(self.arrq * q2) == '[1.  2.3 8.9]'

    def test_dimensionless_quantity_format(self):
        if False:
            return 10
        q1 = u.Quantity(3.14)
        assert format(q1, '.2f') == '3.14'
        assert f'{q1:cds}' == '3.14'

    def test_scalar_quantity_str(self):
        if False:
            i = 10
            return i + 15
        assert str(self.scalarintq) == '1 m'
        assert str(self.scalarfloatq) == '1.3 m'

    def test_scalar_quantity_repr(self):
        if False:
            for i in range(10):
                print('nop')
        assert repr(self.scalarintq) == '<Quantity 1 m>'
        assert repr(self.scalarfloatq) == '<Quantity 1.3 m>'

    def test_array_quantity_str(self):
        if False:
            return 10
        assert str(self.arrq) == '[1.  2.3 8.9] m'

    def test_array_quantity_repr(self):
        if False:
            print('Hello World!')
        assert repr(self.arrq) == '<Quantity [1. , 2.3, 8.9] m>'

    def test_scalar_quantity_format(self):
        if False:
            return 10
        assert format(self.scalarintq, '02d') == '01 m'
        assert format(self.scalarfloatq, '.1f') == '1.3 m'
        assert format(self.scalarfloatq, '.0f') == '1 m'
        assert f'{self.scalarintq:cds}' == '1 m'
        assert f'{self.scalarfloatq:cds}' == '1.3 m'

    def test_uninitialized_unit_format(self):
        if False:
            i = 10
            return i + 15
        bad_quantity = np.arange(10.0).view(u.Quantity)
        assert str(bad_quantity).endswith(_UNIT_NOT_INITIALISED)
        assert repr(bad_quantity).endswith(_UNIT_NOT_INITIALISED + '>')

    def test_to_string(self):
        if False:
            print('Hello World!')
        qscalar = u.Quantity(150000000000000.0, 'm/s')
        assert str(qscalar) == qscalar.to_string()
        res = 'Quantity as KMS: 150000000000.0 km / s'
        assert f'Quantity as KMS: {qscalar.to_string(unit=u.km / u.s)}' == res
        res = 'Quantity as KMS: 1.500e+11 km / s'
        assert f'Quantity as KMS: {qscalar.to_string(precision=3, unit=u.km / u.s)}' == res
        res = '$1.5 \\times 10^{14} \\; \\mathrm{\\frac{m}{s}}$'
        assert qscalar.to_string(format='latex') == res
        assert qscalar.to_string(format='latex', subfmt='inline') == res
        res = '$\\displaystyle 1.5 \\times 10^{14} \\; \\mathrm{\\frac{m}{s}}$'
        assert qscalar.to_string(format='latex', subfmt='display') == res
        res = '$1.5 \\times 10^{14} \\; \\mathrm{m\\,s^{-1}}$'
        assert qscalar.to_string(format='latex_inline') == res
        assert qscalar.to_string(format='latex_inline', subfmt='inline') == res
        res = '$\\displaystyle 1.5 \\times 10^{14} \\; \\mathrm{m\\,s^{-1}}$'
        assert qscalar.to_string(format='latex_inline', subfmt='display') == res
        res = '[0 1 2] (Unit not initialised)'
        assert np.arange(3).view(u.Quantity).to_string() == res

    def test_repr_latex(self):
        if False:
            i = 10
            return i + 15
        from astropy.units.quantity import conf
        q2scalar = u.Quantity(150000000000000.0, 'm/s')
        assert self.scalarintq._repr_latex_() == '$1 \\; \\mathrm{m}$'
        assert self.scalarfloatq._repr_latex_() == '$1.3 \\; \\mathrm{m}$'
        assert q2scalar._repr_latex_() == '$1.5 \\times 10^{14} \\; \\mathrm{\\frac{m}{s}}$'
        assert self.arrq._repr_latex_() == '$[1,~2.3,~8.9] \\; \\mathrm{m}$'
        assert self.scalar_complex_q._repr_latex_() == '$(1+2i) \\; \\mathrm{}$'
        assert self.scalar_big_complex_q._repr_latex_() == '$(1 \\times 10^{25}+2 \\times 10^{52}i) \\; \\mathrm{}$'
        assert self.scalar_big_neg_complex_q._repr_latex_() == '$(-1 \\times 10^{36}-2 \\times 10^{63}i) \\; \\mathrm{}$'
        assert self.arr_complex_q._repr_latex_() == '$[(0-0i),~(-1 \\times 10^{36}-2 \\times 10^{63}i),~(-2 \\times 10^{36}-4 \\times 10^{63}i)] \\; \\mathrm{}$'
        assert '\\dots' in self.big_arr_complex_q._repr_latex_()
        qmed = np.arange(100) * u.m
        qbig = np.arange(1000) * u.m
        qvbig = np.arange(10000) * 1000000000.0 * u.m
        pops = np.get_printoptions()
        oldlat = conf.latex_array_threshold
        try:
            q = u.Quantity(987654321.1234568, 'm/s')
            qa = np.array([7.89123, 123456789.98765433, 0]) * u.cm
            np.set_printoptions(precision=8)
            assert q._repr_latex_() == '$9.8765432 \\times 10^{8} \\; \\mathrm{\\frac{m}{s}}$'
            assert qa._repr_latex_() == '$[7.89123,~1.2345679 \\times 10^{8},~0] \\; \\mathrm{cm}$'
            np.set_printoptions(precision=2)
            assert q._repr_latex_() == '$9.9 \\times 10^{8} \\; \\mathrm{\\frac{m}{s}}$'
            assert qa._repr_latex_() == '$[7.9,~1.2 \\times 10^{8},~0] \\; \\mathrm{cm}$'
            conf.latex_array_threshold = 100
            lsmed = qmed._repr_latex_()
            assert '\\dots' not in lsmed
            lsbig = qbig._repr_latex_()
            assert '\\dots' in lsbig
            lsvbig = qvbig._repr_latex_()
            assert '\\dots' in lsvbig
            conf.latex_array_threshold = 1001
            lsmed = qmed._repr_latex_()
            assert '\\dots' not in lsmed
            lsbig = qbig._repr_latex_()
            assert '\\dots' not in lsbig
            lsvbig = qvbig._repr_latex_()
            assert '\\dots' in lsvbig
            conf.latex_array_threshold = -1
            np.set_printoptions(threshold=99)
            lsmed = qmed._repr_latex_()
            assert '\\dots' in lsmed
            lsbig = qbig._repr_latex_()
            assert '\\dots' in lsbig
            lsvbig = qvbig._repr_latex_()
            assert '\\dots' in lsvbig
            assert lsvbig.endswith(',~1 \\times 10^{13}] \\; \\mathrm{m}$')
        finally:
            np.set_printoptions(**pops)
            conf.latex_array_threshold = oldlat
        qinfnan = [np.inf, -np.inf, np.nan] * u.m
        assert qinfnan._repr_latex_() == '$[\\infty,~-\\infty,~{\\rm NaN}] \\; \\mathrm{m}$'

def test_decompose():
    if False:
        return 10
    q1 = 5 * u.N
    assert q1.decompose() == 5 * u.kg * u.m * u.s ** (-2)

def test_decompose_regression():
    if False:
        i = 10
        return i + 15
    '\n    Regression test for bug #1163\n\n    If decompose was called multiple times on a Quantity with an array and a\n    scale != 1, the result changed every time. This is because the value was\n    being referenced not copied, then modified, which changed the original\n    value.\n    '
    q = np.array([1, 2, 3]) * u.m / (2.0 * u.km)
    assert np.all(q.decompose().value == np.array([0.0005, 0.001, 0.0015]))
    assert np.all(q == np.array([1, 2, 3]) * u.m / (2.0 * u.km))
    assert np.all(q.decompose().value == np.array([0.0005, 0.001, 0.0015]))

def test_arrays():
    if False:
        i = 10
        return i + 15
    '\n    Test using quantites with array values\n    '
    qsec = u.Quantity(np.arange(10), u.second)
    assert isinstance(qsec.value, np.ndarray)
    assert not qsec.isscalar
    assert len(qsec) == len(qsec.value)
    qsecsub25 = qsec[2:5]
    assert qsecsub25.unit == qsec.unit
    assert isinstance(qsecsub25, u.Quantity)
    assert len(qsecsub25) == 3
    qsecnotarray = u.Quantity(10.0, u.second)
    assert qsecnotarray.isscalar
    with pytest.raises(TypeError):
        len(qsecnotarray)
    with pytest.raises(TypeError):
        qsecnotarray[0]
    qseclen0array = u.Quantity(np.array(10), u.second, dtype=int)
    assert qseclen0array.isscalar
    with pytest.raises(TypeError):
        len(qseclen0array)
    with pytest.raises(TypeError):
        qseclen0array[0]
    assert isinstance(qseclen0array.value, numbers.Integral)
    a = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)], dtype=[('x', float), ('y', float), ('z', float)])
    qkpc = u.Quantity(a, u.kpc)
    assert not qkpc.isscalar
    qkpc0 = qkpc[0]
    assert qkpc0.value == a[0]
    assert qkpc0.unit == qkpc.unit
    assert isinstance(qkpc0, u.Quantity)
    assert qkpc0.isscalar
    qkpcx = qkpc['x']
    assert np.all(qkpcx.value == a['x'])
    assert qkpcx.unit == qkpc.unit
    assert isinstance(qkpcx, u.Quantity)
    assert not qkpcx.isscalar
    qkpcx1 = qkpc['x'][1]
    assert qkpcx1.unit == qkpc.unit
    assert isinstance(qkpcx1, u.Quantity)
    assert qkpcx1.isscalar
    qkpc1x = qkpc[1]['x']
    assert qkpc1x.isscalar
    assert qkpc1x == qkpcx1
    qsec = u.Quantity(list(range(10)), u.second)
    assert isinstance(qsec.value, np.ndarray)
    assert_array_equal((qsec * 2).value, np.arange(10) * 2)
    assert_array_equal((qsec / 2).value, np.arange(10) / 2)
    with pytest.raises(u.UnitsError):
        assert_array_equal((qsec + 2).value, np.arange(10) + 2)
    with pytest.raises(u.UnitsError):
        assert_array_equal((qsec - 2).value, np.arange(10) + 2)
    qsec2 = np.arange(10) * u.second
    qsec3 = u.second * np.arange(10)
    assert np.all(qsec == qsec2)
    assert np.all(qsec2 == qsec3)
    with pytest.raises(TypeError):
        float(qsec)
    with pytest.raises(TypeError):
        int(qsec)

def test_array_indexing_slicing():
    if False:
        while True:
            i = 10
    q = np.array([1.0, 2.0, 3.0]) * u.m
    assert q[0] == 1.0 * u.m
    assert np.all(q[0:2] == u.Quantity([1.0, 2.0], u.m))

def test_array_setslice():
    if False:
        return 10
    q = np.array([1.0, 2.0, 3.0]) * u.m
    q[1:2] = np.array([400.0]) * u.cm
    assert np.all(q == np.array([1.0, 4.0, 3.0]) * u.m)

def test_inverse_quantity():
    if False:
        for i in range(10):
            print('nop')
    '\n    Regression test from issue #679\n    '
    q = u.Quantity(4.0, u.meter / u.second)
    qot = q / 2
    toq = 2 / q
    npqot = q / np.array(2)
    assert npqot.value == 2.0
    assert npqot.unit == u.meter / u.second
    assert qot.value == 2.0
    assert qot.unit == u.meter / u.second
    assert toq.value == 0.5
    assert toq.unit == u.second / u.meter

def test_quantity_mutability():
    if False:
        for i in range(10):
            print('nop')
    q = u.Quantity(9.8, u.meter / u.second / u.second)
    with pytest.raises(AttributeError):
        q.value = 3
    with pytest.raises(AttributeError):
        q.unit = u.kg

def test_quantity_initialized_with_quantity():
    if False:
        print('Hello World!')
    q1 = u.Quantity(60, u.second)
    q2 = u.Quantity(q1, u.minute)
    assert q2.value == 1
    q3 = u.Quantity([q1, q2], u.second)
    assert q3[0].value == 60
    assert q3[1].value == 60
    q4 = u.Quantity([q2, q1])
    assert q4.unit == q2.unit
    assert q4[0].value == 1
    assert q4[1].value == 1

def test_quantity_string_unit():
    if False:
        for i in range(10):
            print('nop')
    q1 = 1.0 * u.m / 's'
    assert q1.value == 1
    assert q1.unit == u.m / u.s
    q2 = q1 * 'm'
    assert q2.unit == u.m * u.m / u.s

def test_quantity_invalid_unit_string():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        'foo' * u.m

def test_implicit_conversion():
    if False:
        for i in range(10):
            print('nop')
    q = u.Quantity(1.0, u.meter)
    q._include_easy_conversion_members = True
    assert_allclose(q.centimeter, 100)
    assert_allclose(q.cm, 100)
    assert_allclose(q.parsec, 3.240779289469756e-17)

def test_implicit_conversion_autocomplete():
    if False:
        i = 10
        return i + 15
    q = u.Quantity(1.0, u.meter)
    q._include_easy_conversion_members = True
    q.foo = 42
    attrs = dir(q)
    assert 'centimeter' in attrs
    assert 'cm' in attrs
    assert 'parsec' in attrs
    assert 'foo' in attrs
    assert 'to' in attrs
    assert 'value' in attrs
    assert '__setattr__' in attrs
    with pytest.raises(AttributeError):
        q.l

def test_quantity_iterability():
    if False:
        while True:
            i = 10
    'Regressiont est for issue #878.\n\n    Scalar quantities should not be iterable and should raise a type error on\n    iteration.\n    '
    q1 = [15.0, 17.0] * u.m
    assert isiterable(q1)
    q2 = next(iter(q1))
    assert q2 == 15.0 * u.m
    assert not isiterable(q2)
    pytest.raises(TypeError, iter, q2)

def test_copy():
    if False:
        return 10
    q1 = u.Quantity(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), unit=u.m)
    q2 = q1.copy()
    assert np.all(q1.value == q2.value)
    assert q1.unit == q2.unit
    assert q1.dtype == q2.dtype
    assert q1.value is not q2.value
    q3 = q1.copy(order='F')
    assert q3.flags['F_CONTIGUOUS']
    assert np.all(q1.value == q3.value)
    assert q1.unit == q3.unit
    assert q1.dtype == q3.dtype
    assert q1.value is not q3.value
    q4 = q1.copy(order='C')
    assert q4.flags['C_CONTIGUOUS']
    assert np.all(q1.value == q4.value)
    assert q1.unit == q4.unit
    assert q1.dtype == q4.dtype
    assert q1.value is not q4.value

def test_deepcopy():
    if False:
        i = 10
        return i + 15
    q1 = u.Quantity(np.array([1.0, 2.0, 3.0]), unit=u.m)
    q2 = copy.deepcopy(q1)
    assert isinstance(q2, u.Quantity)
    assert np.all(q1.value == q2.value)
    assert q1.unit == q2.unit
    assert q1.dtype == q2.dtype
    assert q1.value is not q2.value

def test_equality_numpy_scalar():
    if False:
        for i in range(10):
            print('nop')
    '\n    A regression test to ensure that numpy scalars are correctly compared\n    (which originally failed due to the lack of ``__array_priority__``).\n    '
    assert 10 != 10.0 * u.m
    assert np.int64(10) != 10 * u.m
    assert 10 * u.m != np.int64(10)

def test_quantity_pickelability():
    if False:
        i = 10
        return i + 15
    '\n    Testing pickleability of quantity\n    '
    q1 = np.arange(10) * u.m
    q2 = pickle.loads(pickle.dumps(q1))
    assert np.all(q1.value == q2.value)
    assert q1.unit.is_equivalent(q2.unit)
    assert q1.unit == q2.unit

def test_quantity_initialisation_from_string():
    if False:
        i = 10
        return i + 15
    q = u.Quantity('1')
    assert q.unit == u.dimensionless_unscaled
    assert q.value == 1.0
    q = u.Quantity('1.5 m/s')
    assert q.unit == u.m / u.s
    assert q.value == 1.5
    assert u.Unit(q) == u.Unit('1.5 m/s')
    q = u.Quantity('.5 m')
    assert q == u.Quantity(0.5, u.m)
    q = u.Quantity('-1e1km')
    assert q == u.Quantity(-10, u.km)
    q = u.Quantity('-1e+1km')
    assert q == u.Quantity(-10, u.km)
    q = u.Quantity('+.5km')
    assert q == u.Quantity(0.5, u.km)
    q = u.Quantity('+5e-1km')
    assert q == u.Quantity(0.5, u.km)
    q = u.Quantity('5', u.m)
    assert q == u.Quantity(5.0, u.m)
    q = u.Quantity('5 km', u.m)
    assert q.value == 5000.0
    assert q.unit == u.m
    q = u.Quantity('5Em')
    assert q == u.Quantity(5.0, u.Em)
    with pytest.raises(TypeError):
        u.Quantity('')
    with pytest.raises(TypeError):
        u.Quantity('m')
    with pytest.raises(TypeError):
        u.Quantity('1.2.3 deg')
    with pytest.raises(TypeError):
        u.Quantity('1+deg')
    with pytest.raises(TypeError):
        u.Quantity('1-2deg')
    with pytest.raises(TypeError):
        u.Quantity('1.2e-13.3m')
    with pytest.raises(TypeError):
        u.Quantity(['5'])
    with pytest.raises(TypeError):
        u.Quantity(np.array(['5']))
    with pytest.raises(ValueError):
        u.Quantity('5E')
    with pytest.raises(ValueError):
        u.Quantity('5 foo')

def test_unsupported():
    if False:
        return 10
    q1 = np.arange(10) * u.m
    with pytest.raises(TypeError):
        np.bitwise_and(q1, q1)

def test_unit_identity():
    if False:
        while True:
            i = 10
    q = 1.0 * u.hour
    assert q.unit is u.hour

def test_quantity_to_view():
    if False:
        i = 10
        return i + 15
    q1 = np.array([1000, 2000]) * u.m
    q2 = q1.to(u.km)
    assert q1.value[0] == 1000
    assert q2.value[0] == 1

def test_quantity_tuple_power():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        (5.0 * u.m) ** (1, 2)

def test_quantity_fraction_power():
    if False:
        while True:
            i = 10
    q = (25.0 * u.m ** 2) ** Fraction(1, 2)
    assert q.value == 5.0
    assert q.unit == u.m
    assert q.dtype.kind == 'f'

def test_quantity_from_table():
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks that units from tables are respected when converted to a Quantity.\n    This also generically checks the use of *anything* with a `unit` attribute\n    passed into Quantity\n    '
    from astropy.table import Table
    t = Table(data=[np.arange(5), np.arange(5)], names=['a', 'b'])
    t['a'].unit = u.kpc
    qa = u.Quantity(t['a'])
    assert qa.unit == u.kpc
    assert_array_equal(qa.value, t['a'])
    qb = u.Quantity(t['b'])
    assert qb.unit == u.dimensionless_unscaled
    assert_array_equal(qb.value, t['b'])
    qap = u.Quantity(t['a'], u.pc)
    assert qap.unit == u.pc
    assert_array_equal(qap.value, t['a'] * 1000)
    qbp = u.Quantity(t['b'], u.pc)
    assert qbp.unit == u.pc
    assert_array_equal(qbp.value, t['b'])
    t['a'].unit = u.dex(u.cm / u.s ** 2)
    fq = u.Dex(t['a'])
    assert fq.unit == u.dex(u.cm / u.s ** 2)
    assert_array_equal(fq.value, t['a'])
    fq2 = u.Quantity(t['a'], subok=True)
    assert isinstance(fq2, u.Dex)
    assert fq2.unit == u.dex(u.cm / u.s ** 2)
    assert_array_equal(fq2.value, t['a'])
    with pytest.raises(u.UnitTypeError):
        u.Quantity(t['a'])

def test_assign_slice_with_quantity_like():
    if False:
        i = 10
        return i + 15
    from astropy.table import Column, Table
    c = Column(np.arange(10.0), unit=u.mm)
    q = u.Quantity(c)
    q[:2] = c[:2]
    t = Table()
    t['x'] = np.arange(10) * u.mm
    t['y'] = np.ones(10) * u.mm
    assert type(t['x']) is Column
    xy = np.vstack([t['x'], t['y']]).T * u.mm
    ii = [0, 2, 4]
    assert xy[ii, 0].unit == t['x'][ii].unit
    xy[ii, 0] = t['x'][ii]

def test_insert():
    if False:
        return 10
    '\n    Test Quantity.insert method.  This does not test the full capabilities\n    of the underlying np.insert, but hits the key functionality for\n    Quantity.\n    '
    q = [1, 2] * u.m
    q2 = q.insert(0, 1 * u.km)
    assert np.all(q2.value == [1000, 1, 2])
    assert q2.unit is u.m
    assert q2.dtype.kind == 'f'
    if minversion(np, '1.8.0'):
        q2 = q.insert(1, [1, 2] * u.km)
        assert np.all(q2.value == [1, 1000, 2000, 2])
        assert q2.unit is u.m
    with pytest.raises(u.UnitsError):
        q.insert(1, 1.5 * u.s)
    q = [[1, 2], [3, 4]] * u.m
    q2 = q.insert(1, [10, 20] * u.m, axis=0)
    assert np.all(q2.value == [[1, 2], [10, 20], [3, 4]])
    q2 = q.insert(1, [10, 20] * u.m, axis=1)
    assert np.all(q2.value == [[1, 10, 2], [3, 20, 4]])
    q2 = q.insert(1, 10 * u.m, axis=1)
    assert np.all(q2.value == [[1, 10, 2], [3, 10, 4]])

def test_repr_array_of_quantity():
    if False:
        print('Hello World!')
    '\n    Test print/repr of object arrays of Quantity objects with different\n    units.\n\n    Regression test for the issue first reported in\n    https://github.com/astropy/astropy/issues/3777\n    '
    a = np.array([1 * u.m, 2 * u.s], dtype=object)
    assert repr(a) == 'array([<Quantity 1. m>, <Quantity 2. s>], dtype=object)'
    assert str(a) == '[<Quantity 1. m> <Quantity 2. s>]'

class TestSpecificTypeQuantity:

    def setup_method(self):
        if False:
            print('Hello World!')

        class Length(u.SpecificTypeQuantity):
            _equivalent_unit = u.m

        class Length2(Length):
            _default_unit = u.m

        class Length3(Length):
            _unit = u.m
        self.Length = Length
        self.Length2 = Length2
        self.Length3 = Length3

    def test_creation(self):
        if False:
            i = 10
            return i + 15
        l = self.Length(np.arange(10.0) * u.km)
        assert type(l) is self.Length
        with pytest.raises(u.UnitTypeError):
            self.Length(np.arange(10.0) * u.hour)
        with pytest.raises(u.UnitTypeError):
            self.Length(np.arange(10.0))
        l2 = self.Length2(np.arange(5.0))
        assert type(l2) is self.Length2
        assert l2._default_unit is self.Length2._default_unit
        with pytest.raises(u.UnitTypeError):
            self.Length3(np.arange(10.0))

    def test_view(self):
        if False:
            return 10
        l = (np.arange(5.0) * u.km).view(self.Length)
        assert type(l) is self.Length
        with pytest.raises(u.UnitTypeError):
            (np.arange(5.0) * u.s).view(self.Length)
        v = np.arange(5.0).view(self.Length)
        assert type(v) is self.Length
        assert v._unit is None
        l3 = np.ones((2, 2)).view(self.Length3)
        assert type(l3) is self.Length3
        assert l3.unit is self.Length3._unit

    def test_operation_precedence_and_fallback(self):
        if False:
            while True:
                i = 10
        l = self.Length(np.arange(5.0) * u.cm)
        sum1 = l + 1.0 * u.m
        assert type(sum1) is self.Length
        sum2 = 1.0 * u.km + l
        assert type(sum2) is self.Length
        sum3 = l + l
        assert type(sum3) is self.Length
        res1 = l * (1.0 * u.m)
        assert type(res1) is u.Quantity
        res2 = l * l
        assert type(res2) is u.Quantity

def test_unit_class_override():
    if False:
        for i in range(10):
            print('nop')

    class MyQuantity(u.Quantity):
        pass
    my_unit = u.Unit('my_deg', u.deg)
    my_unit._quantity_class = MyQuantity
    q1 = u.Quantity(1.0, my_unit)
    assert type(q1) is u.Quantity
    q2 = u.Quantity(1.0, my_unit, subok=True)
    assert type(q2) is MyQuantity

class QuantityMimic:

    def __init__(self, value, unit):
        if False:
            i = 10
            return i + 15
        self.value = value
        self.unit = unit

    def __array__(self):
        if False:
            print('Hello World!')
        return np.array(self.value)

class QuantityMimic2(QuantityMimic):

    def to(self, unit):
        if False:
            i = 10
            return i + 15
        return u.Quantity(self.value, self.unit).to(unit)

    def to_value(self, unit):
        if False:
            for i in range(10):
                print('nop')
        return u.Quantity(self.value, self.unit).to_value(unit)

class TestQuantityMimics:
    """Test Quantity Mimics that are not ndarray subclasses."""

    @pytest.mark.parametrize('Mimic', (QuantityMimic, QuantityMimic2))
    def test_mimic_input(self, Mimic):
        if False:
            print('Hello World!')
        value = np.arange(10.0)
        mimic = Mimic(value, u.m)
        q = u.Quantity(mimic)
        assert q.unit == u.m
        assert np.all(q.value == value)
        q2 = u.Quantity(mimic, u.cm)
        assert q2.unit == u.cm
        assert np.all(q2.value == 100 * value)

    @pytest.mark.parametrize('Mimic', (QuantityMimic, QuantityMimic2))
    def test_mimic_setting(self, Mimic):
        if False:
            i = 10
            return i + 15
        mimic = Mimic([1.0, 2.0], u.m)
        q = u.Quantity(np.arange(10.0), u.cm)
        q[8:] = mimic
        assert np.all(q[:8].value == np.arange(8.0))
        assert np.all(q[8:].value == [100.0, 200.0])

    def test_mimic_function_unit(self):
        if False:
            i = 10
            return i + 15
        mimic = QuantityMimic([1.0, 2.0], u.dex(u.cm / u.s ** 2))
        d = u.Dex(mimic)
        assert isinstance(d, u.Dex)
        assert d.unit == u.dex(u.cm / u.s ** 2)
        assert np.all(d.value == [1.0, 2.0])
        q = u.Quantity(mimic, subok=True)
        assert isinstance(q, u.Dex)
        assert q.unit == u.dex(u.cm / u.s ** 2)
        assert np.all(q.value == [1.0, 2.0])
        with pytest.raises(u.UnitTypeError):
            u.Quantity(mimic)

def test_masked_quantity_str_repr():
    if False:
        return 10
    "Ensure we don't break masked Quantity representation."
    masked_quantity = np.ma.array([1, 2, 3, 4] * u.kg, mask=[True, False, True, False])
    str(masked_quantity)
    repr(masked_quantity)

class TestQuantitySubclassAboveAndBelow:

    @classmethod
    def setup_class(self):
        if False:
            while True:
                i = 10

        class MyArray(np.ndarray):

            def __array_finalize__(self, obj):
                if False:
                    while True:
                        i = 10
                super_array_finalize = super().__array_finalize__
                if super_array_finalize is not None:
                    super_array_finalize(obj)
                if hasattr(obj, 'my_attr'):
                    self.my_attr = obj.my_attr
        self.MyArray = MyArray
        self.MyQuantity1 = type('MyQuantity1', (u.Quantity, MyArray), dict(my_attr='1'))
        self.MyQuantity2 = type('MyQuantity2', (MyArray, u.Quantity), dict(my_attr='2'))

    def test_setup(self):
        if False:
            return 10
        mq1 = self.MyQuantity1(10, u.m)
        assert isinstance(mq1, self.MyQuantity1)
        assert mq1.my_attr == '1'
        assert mq1.unit is u.m
        mq2 = self.MyQuantity2(10, u.m)
        assert isinstance(mq2, self.MyQuantity2)
        assert mq2.my_attr == '2'
        assert mq2.unit is u.m

    def test_attr_propagation(self):
        if False:
            print('Hello World!')
        mq1 = self.MyQuantity1(10, u.m)
        mq12 = self.MyQuantity2(mq1)
        assert isinstance(mq12, self.MyQuantity2)
        assert not isinstance(mq12, self.MyQuantity1)
        assert mq12.my_attr == '1'
        assert mq12.unit is u.m
        mq2 = self.MyQuantity2(10, u.m)
        mq21 = self.MyQuantity1(mq2)
        assert isinstance(mq21, self.MyQuantity1)
        assert not isinstance(mq21, self.MyQuantity2)
        assert mq21.my_attr == '2'
        assert mq21.unit is u.m