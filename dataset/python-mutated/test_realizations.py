import pickle
import pytest
import astropy.cosmology.units as cu
import astropy.units as u
from astropy import cosmology
from astropy.cosmology import parameters, realizations
from astropy.cosmology.realizations import default_cosmology

def test_realizations_in_toplevel_dir():
    if False:
        print('Hello World!')
    'Test the realizations are in ``dir`` of :mod:`astropy.cosmology`.'
    d = dir(cosmology)
    assert set(d) == set(cosmology.__all__)
    for n in parameters.available:
        assert n in d

def test_realizations_in_realizations_dir():
    if False:
        return 10
    'Test the realizations are in ``dir`` of :mod:`astropy.cosmology.realizations`.'
    d = dir(realizations)
    assert set(d) == set(realizations.__all__)
    for n in parameters.available:
        assert n in d

class Test_default_cosmology:
    """Tests for :class:`~astropy.cosmology.realizations.default_cosmology`."""

    def test_get_current(self):
        if False:
            while True:
                i = 10
        'Test :meth:`astropy.cosmology.default_cosmology.get` current value.'
        cosmo = default_cosmology.get()
        assert cosmo is default_cosmology.validate(default_cosmology._value)

    def test_validate_fail(self):
        if False:
            i = 10
            return i + 15
        'Test :meth:`astropy.cosmology.default_cosmology.validate`.'
        with pytest.raises(TypeError, match='must be a string or Cosmology'):
            default_cosmology.validate(TypeError)
        with pytest.raises(ValueError, match='Unknown cosmology'):
            default_cosmology.validate('fail!')
        with pytest.raises(TypeError, match='cannot find a Cosmology'):
            default_cosmology.validate('available')

    def test_validate_default(self):
        if False:
            for i in range(10):
                print('nop')
        'Test method ``validate`` for specific values.'
        value = default_cosmology.validate(None)
        assert value is realizations.Planck18

    @pytest.mark.parametrize('name', parameters.available)
    def test_validate_str(self, name):
        if False:
            print('Hello World!')
        'Test method ``validate`` for string input.'
        value = default_cosmology.validate(name)
        assert value is getattr(realizations, name)

    @pytest.mark.parametrize('name', parameters.available)
    def test_validate_cosmo(self, name):
        if False:
            i = 10
            return i + 15
        'Test method ``validate`` for cosmology instance input.'
        cosmo = getattr(realizations, name)
        value = default_cosmology.validate(cosmo)
        assert value is cosmo

    def test_validate_no_default(self):
        if False:
            for i in range(10):
                print('nop')
        'Test :meth:`astropy.cosmology.default_cosmology.get` to `None`.'
        cosmo = default_cosmology.validate('no_default')
        assert cosmo is None

@pytest.mark.parametrize('name', parameters.available)
def test_pickle_builtin_realizations(name, pickle_protocol):
    if False:
        print('Hello World!')
    '\n    Test in-built realizations can pickle and unpickle.\n    Also a regression test for #12008.\n    '
    original = getattr(cosmology, name)
    f = pickle.dumps(original, protocol=pickle_protocol)
    with u.add_enabled_units(cu):
        unpickled = pickle.loads(f)
    assert unpickled == original
    assert unpickled.meta == original.meta
    unpickled = pickle.loads(f)
    assert unpickled == original
    assert unpickled.meta != original.meta