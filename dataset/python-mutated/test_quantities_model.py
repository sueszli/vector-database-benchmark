import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.modeling.core import _ModelMeta
from astropy.modeling.models import Gaussian1D, Mapping, Pix2Sky_TAN
from astropy.tests.helper import assert_quantity_allclose

def test_gaussian1d_bounding_box():
    if False:
        while True:
            i = 10
    g = Gaussian1D(mean=3 * u.m, stddev=3 * u.cm, amplitude=3 * u.Jy)
    bbox = g.bounding_box.bounding_box()
    assert_quantity_allclose(bbox[0], 2.835 * u.m)
    assert_quantity_allclose(bbox[1], 3.165 * u.m)

def test_gaussian1d_n_models():
    if False:
        print('Hello World!')
    g = Gaussian1D(amplitude=[1 * u.J, 2.0 * u.J], mean=[1 * u.m, 5000 * u.AA], stddev=[0.1 * u.m, 100 * u.AA], n_models=2)
    assert_quantity_allclose(g(1.01 * u.m), [0.99501248, 0.0] * u.J)
    assert_quantity_allclose(g(u.Quantity([1.01 * u.m, 5010 * u.AA])), [0.99501248, 1.990025] * u.J)
'\nTest the "rules" of model units.\n'

def test_quantity_call():
    if False:
        i = 10
        return i + 15
    '\n    Test that if constructed with Quanties models must be called with quantities.\n    '
    g = Gaussian1D(mean=3 * u.m, stddev=3 * u.cm, amplitude=3 * u.Jy)
    g(10 * u.m)
    MESSAGE = ".* Units of input 'x', .* could not be converted to required input units of m .*"
    with pytest.raises(u.UnitsError, match=MESSAGE):
        g(10)

def test_no_quantity_call():
    if False:
        print('Hello World!')
    '\n    Test that if not constructed with Quantites they can be called without quantities.\n    '
    g = Gaussian1D(mean=3, stddev=3, amplitude=3)
    assert isinstance(g, Gaussian1D)
    g(10)

def test_default_parameters():
    if False:
        i = 10
        return i + 15
    g = Gaussian1D(mean=3 * u.m, stddev=3 * u.cm)
    assert isinstance(g, Gaussian1D)
    g(10 * u.m)

def test_uses_quantity():
    if False:
        while True:
            i = 10
    '\n    Test Quantity\n    '
    g = Gaussian1D(mean=3 * u.m, stddev=3 * u.cm, amplitude=3 * u.Jy)
    assert g.uses_quantity
    g = Gaussian1D(mean=3, stddev=3, amplitude=3)
    assert not g.uses_quantity
    g.mean = 3 * u.m
    assert g.uses_quantity

def test_uses_quantity_compound():
    if False:
        print('Hello World!')
    '\n    Test Quantity\n    '
    g = Gaussian1D(mean=3 * u.m, stddev=3 * u.cm, amplitude=3 * u.Jy)
    g2 = Gaussian1D(mean=5 * u.m, stddev=5 * u.cm, amplitude=5 * u.Jy)
    assert (g | g2).uses_quantity
    g = Gaussian1D(mean=3, stddev=3, amplitude=3)
    g2 = Gaussian1D(mean=5, stddev=5, amplitude=5)
    comp = g | g2
    assert not comp.uses_quantity

def test_uses_quantity_no_param():
    if False:
        print('Hello World!')
    comp = Mapping((0, 1)) | Pix2Sky_TAN()
    assert comp.uses_quantity

def _allmodels():
    if False:
        while True:
            i = 10
    allmodels = []
    for name in dir(models):
        model = getattr(models, name)
        if type(model) is _ModelMeta:
            try:
                m = model()
            except Exception:
                pass
            allmodels.append(m)
    return allmodels

@pytest.mark.parametrize('m', _allmodels())
def test_read_only(m):
    if False:
        print('Hello World!')
    '\n    input_units\n    return_units\n    input_units_allow_dimensionless\n    input_units_strict\n    '
    with pytest.raises(AttributeError):
        m.input_units = {}
    with pytest.raises(AttributeError):
        m.return_units = {}
    with pytest.raises(AttributeError):
        m.input_units_allow_dimensionless = {}
    with pytest.raises(AttributeError):
        m.input_units_strict = {}