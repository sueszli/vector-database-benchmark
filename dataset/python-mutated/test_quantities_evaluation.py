"""
Tests that relate to evaluating models with quantity parameters
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.modeling.core import Model
from astropy.modeling.models import Gaussian1D, Pix2Sky_TAN, Scale, Shift
from astropy.tests.helper import assert_quantity_allclose
from astropy.units import UnitsError
MESSAGE = "{}: Units of input 'x', {}.*, could not be converted to required input units of {}.*"

def test_evaluate_with_quantities():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test evaluation of a single model with Quantity parameters that do\n    not explicitly require units.\n    '
    g = Gaussian1D(1, 1, 0.1)
    gq = Gaussian1D(1 * u.J, 1 * u.m, 0.1 * u.m)
    assert_quantity_allclose(gq(1 * u.m), g(1) * u.J)
    with pytest.raises(UnitsError, match=MESSAGE.format('Gaussian1D', '', 'm ')):
        gq(1)
    assert_quantity_allclose(gq(0), g(0) * u.J)
    assert_allclose(gq(0.0005 * u.km).value, g(0.5))
    with pytest.raises(UnitsError, match=MESSAGE.format('Gaussian1D', 's', 'm')):
        gq(3 * u.s)
    with pytest.raises(UnitsError, match="Can only apply 'subtract' function to dimensionless quantities .*"):
        g(3 * u.m)

def test_evaluate_with_quantities_and_equivalencies():
    if False:
        print('Hello World!')
    '\n    We now make sure that equivalencies are correctly taken into account\n    '
    g = Gaussian1D(1 * u.Jy, 10 * u.nm, 2 * u.nm)
    with pytest.raises(UnitsError, match=MESSAGE.format('Gaussian1D', 'PHz', 'nm')):
        g(30 * u.PHz)
    assert_quantity_allclose(g(30 * u.PHz, equivalencies={'x': u.spectral()}), g(9.993081933333333 * u.nm))

class MyTestModel(Model):
    n_inputs = 2
    n_outputs = 1

    def evaluate(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        print('a', a)
        print('b', b)
        return a * b

class TestInputUnits:

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.model = MyTestModel()

    def test_evaluate(self):
        if False:
            return 10
        assert_quantity_allclose(self.model(3, 5), 15)
        assert_quantity_allclose(self.model(4 * u.m, 5), 20 * u.m)
        assert_quantity_allclose(self.model(3 * u.deg, 5), 15 * u.deg)

    def test_input_units(self):
        if False:
            for i in range(10):
                print('nop')
        self.model._input_units = {'x': u.deg}
        assert_quantity_allclose(self.model(3 * u.deg, 4), 12 * u.deg)
        assert_quantity_allclose(self.model(4 * u.rad, 2), 8 * u.rad)
        assert_quantity_allclose(self.model(4 * u.rad, 2 * u.s), 8 * u.rad * u.s)
        with pytest.raises(UnitsError, match=MESSAGE.format('MyTestModel', 's', 'deg')):
            self.model(4 * u.s, 3)
        with pytest.raises(UnitsError, match=MESSAGE.format('MyTestModel', '', 'deg')):
            self.model(3, 3)

    def test_input_units_allow_dimensionless(self):
        if False:
            return 10
        self.model._input_units = {'x': u.deg}
        self.model._input_units_allow_dimensionless = True
        assert_quantity_allclose(self.model(3 * u.deg, 4), 12 * u.deg)
        assert_quantity_allclose(self.model(4 * u.rad, 2), 8 * u.rad)
        with pytest.raises(UnitsError, match=MESSAGE.format('MyTestModel', 's', 'deg')):
            self.model(4 * u.s, 3)
        assert_quantity_allclose(self.model(3, 3), 9)

    def test_input_units_strict(self):
        if False:
            i = 10
            return i + 15
        self.model._input_units = {'x': u.deg}
        self.model._input_units_strict = True
        assert_quantity_allclose(self.model(3 * u.deg, 4), 12 * u.deg)
        result = self.model(np.pi * u.rad, 2)
        assert_quantity_allclose(result, 360 * u.deg)
        assert result.unit is u.deg

    def test_input_units_equivalencies(self):
        if False:
            return 10
        self.model._input_units = {'x': u.micron}
        with pytest.raises(UnitsError, match=MESSAGE.format('MyTestModel', 'PHz', 'micron')):
            self.model(3 * u.PHz, 3)
        self.model.input_units_equivalencies = {'x': u.spectral()}
        assert_quantity_allclose(self.model(3 * u.PHz, 3), 3 * (3 * u.PHz).to(u.micron, equivalencies=u.spectral()))

    def test_return_units(self):
        if False:
            print('Hello World!')
        self.model._input_units = {'z': u.deg}
        self.model._return_units = {'z': u.rad}
        result = self.model(3 * u.deg, 4)
        assert_quantity_allclose(result, 12 * u.deg)
        assert result.unit is u.rad

    def test_return_units_scalar(self):
        if False:
            while True:
                i = 10
        self.model._input_units = {'x': u.deg}
        self.model._return_units = u.rad
        result = self.model(3 * u.deg, 4)
        assert_quantity_allclose(result, 12 * u.deg)
        assert result.unit is u.rad

def test_and_input_units():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test units to first model in chain.\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 & s2
    out = cs(10 * u.arcsecond, 20 * u.arcsecond)
    assert_quantity_allclose(out[0], 10 * u.deg + 10 * u.arcsec)
    assert_quantity_allclose(out[1], 10 * u.deg + 20 * u.arcsec)

def test_plus_input_units():
    if False:
        while True:
            i = 10
    '\n    Test units to first model in chain.\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 + s2
    out = cs(10 * u.arcsecond)
    assert_quantity_allclose(out, 20 * u.deg + 20 * u.arcsec)

def test_compound_input_units():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test units to first model in chain.\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 | s2
    out = cs(10 * u.arcsecond)
    assert_quantity_allclose(out, 20 * u.deg + 10 * u.arcsec)

def test_compound_input_units_fail():
    if False:
        print('Hello World!')
    '\n    Test incompatible units to first model in chain.\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 | s2
    with pytest.raises(UnitsError, match=MESSAGE.format('Shift', 'pix', 'deg')):
        cs(10 * u.pix)

def test_compound_incompatible_units_fail():
    if False:
        return 10
    '\n    Test incompatible model units in chain.\n    '
    s1 = Shift(10 * u.pix)
    s2 = Shift(10 * u.deg)
    cs = s1 | s2
    with pytest.raises(UnitsError, match=MESSAGE.format('Shift', 'pix', 'deg')):
        cs(10 * u.pix)

def test_compound_pipe_equiv_call():
    if False:
        return 10
    '\n    Check that equivalencies work when passed to evaluate, for a chained model\n    (which has one input).\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 | s2
    out = cs(10 * u.pix, equivalencies={'x': u.pixel_scale(0.5 * u.deg / u.pix)})
    assert_quantity_allclose(out, 25 * u.deg)

def test_compound_and_equiv_call():
    if False:
        return 10
    '\n    Check that equivalencies work when passed to evaluate, for a composite model\n    with two inputs.\n    '
    s1 = Shift(10 * u.deg)
    s2 = Shift(10 * u.deg)
    cs = s1 & s2
    out = cs(10 * u.pix, 10 * u.pix, equivalencies={'x0': u.pixel_scale(0.5 * u.deg / u.pix), 'x1': u.pixel_scale(0.5 * u.deg / u.pix)})
    assert_quantity_allclose(out[0], 15 * u.deg)
    assert_quantity_allclose(out[1], 15 * u.deg)

def test_compound_input_units_equivalencies():
    if False:
        print('Hello World!')
    '\n    Test setting input_units_equivalencies on one of the models.\n    '
    s1 = Shift(10 * u.deg)
    s1.input_units_equivalencies = {'x': u.pixel_scale(0.5 * u.deg / u.pix)}
    s2 = Shift(10 * u.deg)
    sp = Shift(10 * u.pix)
    cs = s1 | s2
    assert cs.input_units_equivalencies == {'x': u.pixel_scale(0.5 * u.deg / u.pix)}
    out = cs(10 * u.pix)
    assert_quantity_allclose(out, 25 * u.deg)
    cs = sp | s1
    assert cs.input_units_equivalencies is None
    out = cs(10 * u.pix)
    assert_quantity_allclose(out, 20 * u.deg)
    cs = s1 & s2
    assert cs.input_units_equivalencies == {'x0': u.pixel_scale(0.5 * u.deg / u.pix)}
    cs = cs.rename('TestModel')
    out = cs(20 * u.pix, 10 * u.deg)
    assert_quantity_allclose(out, 20 * u.deg)
    with pytest.raises(UnitsError, match=MESSAGE.format('Shift', 'pix', 'deg')):
        out = cs(20 * u.pix, 10 * u.pix)

def test_compound_input_units_strict():
    if False:
        return 10
    '\n    Test setting input_units_strict on one of the models.\n    '

    class ScaleDegrees(Scale):
        input_units = {'x': u.deg}
    s1 = ScaleDegrees(2)
    s2 = Scale(2)
    cs = s1 | s2
    out = cs(10 * u.arcsec)
    assert_quantity_allclose(out, 40 * u.arcsec)
    assert out.unit is u.deg
    cs = s2 | s1
    out = cs(10 * u.arcsec)
    assert_quantity_allclose(out, 40 * u.arcsec)
    assert out.unit is u.deg
    cs = s1 & s2
    out = cs(10 * u.arcsec, 10 * u.arcsec)
    assert_quantity_allclose(out, 20 * u.arcsec)
    assert out[0].unit is u.deg
    assert out[1].unit is u.arcsec

def test_compound_input_units_allow_dimensionless():
    if False:
        return 10
    '\n    Test setting input_units_allow_dimensionless on one of the models.\n    '

    class ScaleDegrees(Scale):
        input_units = {'x': u.deg}
    s1 = ScaleDegrees(2)
    s1._input_units_allow_dimensionless = True
    s2 = Scale(2)
    cs = s1 | s2
    cs = cs.rename('TestModel')
    out = cs(10)
    assert_quantity_allclose(out, 40 * u.one)
    out = cs(10 * u.arcsec)
    assert_quantity_allclose(out, 40 * u.arcsec)
    with pytest.raises(UnitsError, match=MESSAGE.format('ScaleDegrees', 'm', 'deg')):
        out = cs(10 * u.m)
    s1._input_units_allow_dimensionless = False
    cs = s1 | s2
    cs = cs.rename('TestModel')
    with pytest.raises(UnitsError, match=MESSAGE.format('ScaleDegrees', '', 'deg')):
        out = cs(10)
    s1._input_units_allow_dimensionless = True
    cs = s2 | s1
    cs = cs.rename('TestModel')
    out = cs(10)
    assert_quantity_allclose(out, 40 * u.one)
    out = cs(10 * u.arcsec)
    assert_quantity_allclose(out, 40 * u.arcsec)
    with pytest.raises(UnitsError, match=MESSAGE.format('ScaleDegrees', 'm', 'deg')):
        out = cs(10 * u.m)
    s1._input_units_allow_dimensionless = False
    cs = s2 | s1
    with pytest.raises(UnitsError, match=MESSAGE.format('ScaleDegrees', '', 'deg')):
        out = cs(10)
    s1._input_units_allow_dimensionless = True
    s1 = ScaleDegrees(2)
    s1._input_units_allow_dimensionless = True
    s2 = ScaleDegrees(2)
    s2._input_units_allow_dimensionless = False
    cs = s1 & s2
    cs = cs.rename('TestModel')
    out = cs(10, 10 * u.arcsec)
    assert_quantity_allclose(out[0], 20 * u.one)
    assert_quantity_allclose(out[1], 20 * u.arcsec)
    with pytest.raises(UnitsError, match=MESSAGE.format('ScaleDegrees', '', 'deg')):
        out = cs(10, 10)

def test_compound_return_units():
    if False:
        while True:
            i = 10
    '\n    Test that return_units on the first model in the chain is respected for the\n    input to the second.\n    '

    class PassModel(Model):
        n_inputs = 2
        n_outputs = 2

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(*args, **kwargs)

        @property
        def input_units(self):
            if False:
                for i in range(10):
                    print('nop')
            'Input units.'
            return {'x0': u.deg, 'x1': u.deg}

        @property
        def return_units(self):
            if False:
                for i in range(10):
                    print('nop')
            'Output units.'
            return {'x0': u.deg, 'x1': u.deg}

        def evaluate(self, x, y):
            if False:
                print('Hello World!')
            return (x.value, y.value)
    cs = Pix2Sky_TAN() | PassModel()
    assert_quantity_allclose(cs(0 * u.deg, 0 * u.deg), (0, 90) * u.deg)