"""
Tests that relate to fitting models with quantity parameters
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import fitting, models
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.tests.helper import assert_quantity_allclose
from astropy.units import UnitsError
from astropy.utils import NumpyRNGContext
from astropy.utils.compat.optional_deps import HAS_SCIPY
fitters = [fitting.LevMarLSQFitter, fitting.TRFLSQFitter, fitting.LMLSQFitter, fitting.DogBoxLSQFitter]

def _fake_gaussian_data():
    if False:
        print('Hello World!')
    with NumpyRNGContext(12345):
        x = np.linspace(-5.0, 5.0, 2000)
        y = 3 * np.exp(-0.5 * (x - 1.3) ** 2 / 0.8 ** 2)
        y += np.random.normal(0.0, 0.2, x.shape)
    x = x * u.m
    y = y * u.Jy
    return (x, y)
compound_models_no_units = [models.Linear1D() + models.Gaussian1D() + models.Gaussian1D(), models.Linear1D() + models.Gaussian1D() | models.Scale(), models.Linear1D() + models.Gaussian1D() | models.Shift()]

class CustomInputNamesModel(Fittable1DModel):
    n_inputs = 1
    n_outputs = 1
    a = Parameter(default=1.0)
    b = Parameter(default=1.0)

    def __init__(self, a=a, b=b):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(a=a, b=b)
        self.inputs = ('inn',)
        self.outputs = ('out',)

    @staticmethod
    def evaluate(inn, a, b):
        if False:
            return 10
        return a * inn + b

    @property
    def input_units(self):
        if False:
            return 10
        if self.a.unit is None and self.b.unit is None:
            return None
        else:
            return {'inn': self.b.unit / self.a.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if False:
            i = 10
            return i + 15
        return {'a': outputs_unit['out'] / inputs_unit['inn'], 'b': outputs_unit['out']}

def models_with_custom_names():
    if False:
        for i in range(10):
            print('nop')
    line = models.Linear1D(1 * u.m / u.s, 2 * u.m)
    line.inputs = ('inn',)
    line.outputs = ('out',)
    custom_names_model = CustomInputNamesModel(1 * u.m / u.s, 2 * u.m)
    return [line, custom_names_model]

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_simple(fitter):
    if False:
        while True:
            i = 10
    fitter = fitter()
    (x, y) = _fake_gaussian_data()
    g_init = models.Gaussian1D()
    g = fitter(g_init, x, y)
    assert_quantity_allclose(g.amplitude, 3 * u.Jy, rtol=0.05)
    assert_quantity_allclose(g.mean, 1.3 * u.m, rtol=0.05)
    assert_quantity_allclose(g.stddev, 0.8 * u.m, rtol=0.05)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_with_initial_values(fitter):
    if False:
        i = 10
        return i + 15
    fitter = fitter()
    (x, y) = _fake_gaussian_data()
    g_init = models.Gaussian1D(amplitude=1.0 * u.mJy, mean=3 * u.cm, stddev=2 * u.mm)
    g = fitter(g_init, x, y)
    assert_quantity_allclose(g.amplitude, 3 * u.Jy, rtol=0.05)
    assert_quantity_allclose(g.mean, 1.3 * u.m, rtol=0.05)
    assert_quantity_allclose(g.stddev, 0.8 * u.m, rtol=0.05)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_missing_data_units(fitter):
    if False:
        while True:
            i = 10
    "\n    Raise an error if the model has units but the data doesn't\n    "
    fitter = fitter()

    class UnorderedGaussian1D(models.Gaussian1D):

        def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
            if False:
                i = 10
                return i + 15
            return {'amplitude': outputs_unit['y'], 'mean': inputs_unit['x'], 'stddev': inputs_unit['x']}
    g_init = UnorderedGaussian1D(amplitude=1.0 * u.mJy, mean=3 * u.cm, stddev=2 * u.mm)
    MESSAGE = "'cm' .* and '' .* are not convertible"
    with pytest.raises(UnitsError, match=MESSAGE):
        fitter(g_init, [1, 2, 3], [4, 5, 6] * (u.erg / (u.s * u.cm * u.cm * u.Hz)))
    MESSAGE = "'mJy' .* and '' .* are not convertible"
    with pytest.raises(UnitsError, match=MESSAGE):
        fitter(g_init, [1, 2, 3] * u.m, [4, 5, 6])

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_missing_model_units(fitter):
    if False:
        return 10
    "\n    Proceed if the data has units but the model doesn't\n    "
    fitter = fitter()
    (x, y) = _fake_gaussian_data()
    g_init = models.Gaussian1D(amplitude=1.0, mean=3, stddev=2)
    g = fitter(g_init, x, y)
    assert_quantity_allclose(g.amplitude, 3 * u.Jy, rtol=0.05)
    assert_quantity_allclose(g.mean, 1.3 * u.m, rtol=0.05)
    assert_quantity_allclose(g.stddev, 0.8 * u.m, rtol=0.05)
    g_init = models.Gaussian1D(amplitude=1.0, mean=3 * u.m, stddev=2 * u.m)
    g = fitter(g_init, x, y)
    assert_quantity_allclose(g.amplitude, 3 * u.Jy, rtol=0.05)
    assert_quantity_allclose(g.mean, 1.3 * u.m, rtol=0.05)
    assert_quantity_allclose(g.stddev, 0.8 * u.m, rtol=0.05)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_incompatible_units(fitter):
    if False:
        print('Hello World!')
    '\n    Raise an error if the data and model have incompatible units\n    '
    fitter = fitter()
    g_init = models.Gaussian1D(amplitude=1.0 * u.Jy, mean=3 * u.m, stddev=2 * u.cm)
    MESSAGE = "'Hz' .* and 'm' .* are not convertible"
    with pytest.raises(UnitsError, match=MESSAGE):
        fitter(g_init, [1, 2, 3] * u.Hz, [4, 5, 6] * u.Jy)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.filterwarnings('ignore:The fit may be unsuccessful.*')
@pytest.mark.filterwarnings('ignore:divide by zero encountered.*')
@pytest.mark.parametrize('model', compound_models_no_units)
@pytest.mark.parametrize('fitter', fitters)
def test_compound_without_units(model, fitter):
    if False:
        i = 10
        return i + 15
    fitter = fitter()
    x = np.linspace(-5, 5, 10) * u.Angstrom
    with NumpyRNGContext(12345):
        y = np.random.sample(10)
    res_fit = fitter(model, x, y * u.Hz)
    for param_name in res_fit.param_names:
        print(getattr(res_fit, param_name))
    assert all((res_fit[i]._has_units for i in range(3)))
    z = res_fit(x)
    assert isinstance(z, u.Quantity)
    res_fit = fitter(model, np.arange(10) * u.Unit('Angstrom'), y)
    assert all((res_fit[i]._has_units for i in range(3)))
    z = res_fit(x)
    assert isinstance(z, np.ndarray)

@pytest.mark.skip(reason='Flaky and ill-conditioned')
@pytest.mark.parametrize('fitter', fitters)
def test_compound_fitting_with_units(fitter):
    if False:
        i = 10
        return i + 15
    fitter = fitter()
    x = np.linspace(-5, 5, 15) * u.Angstrom
    y = np.linspace(-5, 5, 15) * u.Angstrom
    fitter = fitter()
    m = models.Gaussian2D(10 * u.Hz, 3 * u.Angstrom, 4 * u.Angstrom, 1 * u.Angstrom, 2 * u.Angstrom)
    p = models.Planar2D(3 * u.Hz / u.Angstrom, 4 * u.Hz / u.Angstrom, 1 * u.Hz)
    model = m + p
    z = model(x, y)
    res = fitter(model, x, y, z)
    assert isinstance(res(x, y), np.ndarray)
    assert all((res[i]._has_units for i in range(2)))
    model = models.Gaussian2D() + models.Planar2D()
    res = fitter(model, x, y, z)
    assert isinstance(res(x, y), np.ndarray)
    assert all((res[i]._has_units for i in range(2)))
    model = models.BlackBody(temperature=3000 * u.K) * models.Const1D(amplitude=1.0)
    x = np.linspace(1, 3, 10000) * u.micron
    with NumpyRNGContext(12345):
        n = np.random.normal(3)
    y = model(x)
    res = fitter(model, x, y * (1 + n))
    np.testing.assert_allclose(res.parameters, [3000, 2.1433621, 2.647347], rtol=0.4)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.filterwarnings('ignore:Model is linear in parameters*')
@pytest.mark.parametrize('model', models_with_custom_names())
@pytest.mark.parametrize('fitter', fitters)
def test_fitting_custom_names(model, fitter):
    if False:
        i = 10
        return i + 15
    'Tests fitting of models with custom inputs and outsputs names.'
    fitter = fitter()
    x = np.linspace(0, 10, 100) * u.s
    y = model(x)
    new_model = fitter(model, x, y)
    for param_name in model.param_names:
        assert_quantity_allclose(getattr(new_model, param_name).quantity, getattr(model, param_name).quantity)