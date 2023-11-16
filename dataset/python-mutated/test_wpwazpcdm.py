"""Testing :mod:`astropy.cosmology.flrw.wpwazpcdm`."""
import numpy as np
import pytest
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology import FlatwpwaCDM, wpwaCDM
from astropy.cosmology.parameter import Parameter
from astropy.cosmology.tests.test_core import ParameterTestMixin
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.compat.optional_deps import HAS_SCIPY
from .conftest import filter_keys_from_items
from .test_base import FlatFLRWMixinTest, FLRWTest
from .test_w0wacdm import ParameterwaTestMixin
COMOVING_DISTANCE_EXAMPLE_KWARGS = {'wp': -0.9, 'zp': 0.5, 'wa': 0.1, 'Tcmb0': 0.0}

class ParameterwpTestMixin(ParameterTestMixin):
    """Tests for `astropy.cosmology.Parameter` wp on a Cosmology.

    wp is a descriptor, which are tested by mixin, here with ``TestFLRW``.
    These tests expect dicts ``_cls_args`` and ``cls_kwargs`` which give the
    args and kwargs for the cosmology class, respectively. See ``TestFLRW``.
    """

    def test_wp(self, cosmo_cls, cosmo):
        if False:
            while True:
                i = 10
        'Test Parameter ``wp``.'
        wp = cosmo_cls.parameters['wp']
        assert isinstance(wp, Parameter)
        assert 'at the pivot' in wp.__doc__
        assert wp.unit is None
        assert wp.default == -1.0
        assert cosmo.wp is cosmo._wp
        assert cosmo.wp == self.cls_kwargs['wp']

    def test_init_wp(self, cosmo_cls, ba):
        if False:
            return 10
        'Test initialization for values of ``wp``.'
        ba.arguments['wp'] = ba.arguments['wp'] << u.one
        cosmo = cosmo_cls(*ba.args, **ba.kwargs)
        assert cosmo.wp == ba.arguments['wp']
        ba.arguments['wp'] = ba.arguments['wp'].value
        cosmo = cosmo_cls(*ba.args, **ba.kwargs)
        assert cosmo.wp == ba.arguments['wp']
        ba.arguments['wp'] = 10 * u.km
        with pytest.raises(TypeError):
            cosmo_cls(*ba.args, **ba.kwargs)

class ParameterzpTestMixin(ParameterTestMixin):
    """Tests for `astropy.cosmology.Parameter` zp on a Cosmology.

    zp is a descriptor, which are tested by mixin, here with ``TestFLRW``.
    These tests expect dicts ``_cls_args`` and ``cls_kwargs`` which give the
    args and kwargs for the cosmology class, respectively. See ``TestFLRW``.
    """

    def test_zp(self, cosmo_cls, cosmo):
        if False:
            for i in range(10):
                print('nop')
        'Test Parameter ``zp``.'
        zp = cosmo_cls.parameters['zp']
        assert isinstance(zp, Parameter)
        assert 'pivot redshift' in zp.__doc__
        assert zp.unit == cu.redshift
        assert zp.default == 0.0
        assert cosmo.zp is cosmo._zp
        assert cosmo.zp == self.cls_kwargs['zp'] << cu.redshift

    def test_init_zp(self, cosmo_cls, ba):
        if False:
            print('Hello World!')
        'Test initialization for values of ``zp``.'
        ba.arguments['zp'] = ba.arguments['zp'] << u.one
        cosmo = cosmo_cls(*ba.args, **ba.kwargs)
        assert cosmo.zp == ba.arguments['zp']
        ba.arguments['zp'] = ba.arguments['zp'].value
        cosmo = cosmo_cls(*ba.args, **ba.kwargs)
        assert cosmo.zp.value == ba.arguments['zp']
        ba.arguments['zp'] = 10 * u.km
        with pytest.raises(u.UnitConversionError):
            cosmo_cls(*ba.args, **ba.kwargs)

class TestwpwaCDM(FLRWTest, ParameterwpTestMixin, ParameterwaTestMixin, ParameterzpTestMixin):
    """Test :class:`astropy.cosmology.wpwaCDM`."""

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        'Setup for testing.'
        super().setup_class(self)
        self.cls = wpwaCDM
        self.cls_kwargs.update(wp=-0.9, wa=0.2, zp=0.5)

    def test_clone_change_param(self, cosmo):
        if False:
            i = 10
            return i + 15
        'Test method ``.clone()`` changing a(many) Parameter(s).'
        super().test_clone_change_param(cosmo)
        c = cosmo.clone(wp=0.1, wa=0.2, zp=14)
        assert c.wp == 0.1
        assert c.wa == 0.2
        assert c.zp == 14
        for (n, v) in filter_keys_from_items(c.parameters, ('wp', 'wa', 'zp')):
            v_expect = getattr(cosmo, n)
            if v is None:
                assert v is v_expect
            else:
                assert_quantity_allclose(v, v_expect, atol=0.0001 * getattr(v, 'unit', 1))

    def test_w(self, cosmo):
        if False:
            while True:
                i = 10
        'Test :meth:`astropy.cosmology.wpwaCDM.w`.'
        assert u.allclose(cosmo.w(0.5), -0.9)
        assert u.allclose(cosmo.w([0.1, 0.2, 0.5, 1.5, 2.5, 11.5]), [-0.94848485, -0.93333333, -0.9, -0.84666667, -0.82380952, -0.78266667])

    def test_repr(self, cosmo):
        if False:
            return 10
        'Test method ``.__repr__()``.'
        assert repr(cosmo) == "wpwaCDM(name='ABCMeta', H0=<Quantity 70. km / (Mpc s)>, Om0=0.27, Ode0=0.73, wp=-0.9, wa=0.2, zp=<Quantity 0.5 redshift>, Tcmb0=<Quantity 3. K>, Neff=3.04, m_nu=<Quantity [0., 0., 0.] eV>, Ob0=0.03)"

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy required for this test.')
    @pytest.mark.parametrize(('args', 'kwargs', 'expected'), [((75.0, 0.3, 0.6), {}, [2954.68975298, 4599.83254834, 5643.04013201, 6373.36147627] * u.Mpc), ((75.0, 0.25, 0.5), {'zp': 0.4, 'Tcmb0': 3.0, 'Neff': 3, 'm_nu': 0 * u.eV}, [2919.00656215, 4558.0218123, 5615.73412391, 6366.10224229] * u.Mpc), ((75.0, 0.25, 0.5), {'zp': 1.0, 'Tcmb0': 3.0, 'Neff': 4, 'm_nu': 5 * u.eV}, [2629.48489827, 3874.13392319, 4614.31562397, 5116.51184842] * u.Mpc), ((75.0, 0.3, 0.7), {}, [3030.70481348, 4745.82435272, 5828.73710847, 6582.60454542] * u.Mpc), ((75.0, 0.25, 0.75), {'zp': 0.4, 'Tcmb0': 3.0, 'Neff': 3, 'm_nu': 0 * u.eV}, [3113.62199365, 4943.28425668, 6114.45491003, 6934.07461377] * u.Mpc), ((75.0, 0.25, 0.2458794183661), {'zp': 1.0, 'Tcmb0': 3.0, 'Neff': 4, 'm_nu': 5 * u.eV}, [2517.08634022, 3694.21111754, 4402.17802962, 4886.65787948] * u.Mpc)])
    def test_comoving_distance_example(self, cosmo_cls, args, kwargs, expected):
        if False:
            return 10
        'Test :meth:`astropy.cosmology.LambdaCDM.comoving_distance`.\n\n        These do not come from external codes -- they are just internal checks to make\n        sure nothing changes if we muck with the distance calculators.\n        '
        super().test_comoving_distance_example(cosmo_cls, args, {**COMOVING_DISTANCE_EXAMPLE_KWARGS, **kwargs}, expected)

class TestFlatwpwaCDM(FlatFLRWMixinTest, TestwpwaCDM):
    """Test :class:`astropy.cosmology.FlatwpwaCDM`."""

    def setup_class(self):
        if False:
            return 10
        'Setup for testing.'
        super().setup_class(self)
        self.cls = FlatwpwaCDM

    def test_repr(self, cosmo_cls, cosmo):
        if False:
            for i in range(10):
                print('nop')
        'Test method ``.__repr__()``.'
        assert repr(cosmo) == "FlatwpwaCDM(name='ABCMeta', H0=<Quantity 70. km / (Mpc s)>, Om0=0.27, wp=-0.9, wa=0.2, zp=<Quantity 0.5 redshift>, Tcmb0=<Quantity 3. K>, Neff=3.04, m_nu=<Quantity [0., 0., 0.] eV>, Ob0=0.03)"

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy required for this test.')
    @pytest.mark.parametrize(('args', 'kwargs', 'expected'), [((75.0, 0.3), {}, [3030.70481348, 4745.82435272, 5828.73710847, 6582.60454542] * u.Mpc), ((75.0, 0.25), {'zp': 0.4, 'wa': 0.1, 'Tcmb0': 3.0, 'Neff': 3, 'm_nu': 0.0 * u.eV}, [3113.62199365, 4943.28425668, 6114.45491003, 6934.07461377] * u.Mpc), ((75.0, 0.25), {'zp': 1.0, 'Tcmb0': 3.0, 'Neff': 4, 'm_nu': 5 * u.eV}, [2517.08634022, 3694.21111754, 4402.17802962, 4886.65787948] * u.Mpc)])
    def test_comoving_distance_example(self, cosmo_cls, args, kwargs, expected):
        if False:
            while True:
                i = 10
        'Test :meth:`astropy.cosmology.LambdaCDM.comoving_distance`.\n\n        These do not come from external codes -- they are just internal checks to make\n        sure nothing changes if we muck with the distance calculators.\n        '
        super().test_comoving_distance_example(cosmo_cls, args, {**COMOVING_DISTANCE_EXAMPLE_KWARGS, **kwargs}, expected)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy.')
def test_varyde_lumdist_mathematica():
    if False:
        return 10
    'Tests a few varying dark energy EOS models against a Mathematica computation.'
    z = np.array([0.2, 0.4, 0.9, 1.2])
    cosmo = wpwaCDM(H0=70, Om0=0.2, Ode0=0.8, wp=-1.1, wa=0.2, zp=0.5, Tcmb0=0.0)
    assert u.allclose(cosmo.luminosity_distance(z), [1010.81, 2294.45, 6369.45, 9218.95] * u.Mpc, rtol=0.0001)
    cosmo = wpwaCDM(H0=70, Om0=0.2, Ode0=0.8, wp=-1.1, wa=0.2, zp=0.9, Tcmb0=0.0)
    assert u.allclose(cosmo.luminosity_distance(z), [1013.68, 2305.3, 6412.37, 9283.33] * u.Mpc, rtol=0.0001)

@pytest.mark.skipif(not HAS_SCIPY, reason='test requires scipy')
def test_de_densityscale():
    if False:
        print('Hello World!')
    cosmo = wpwaCDM(H0=70, Om0=0.3, Ode0=0.7, wp=-0.9, wa=0.2, zp=0.5)
    z = np.array([0.1, 0.2, 0.5, 1.5, 2.5])
    assert u.allclose(cosmo.de_density_scale(z), [1.012246048, 1.0280102, 1.087439, 1.324988, 1.565746], rtol=0.0001)
    assert u.allclose(cosmo.de_density_scale(3), cosmo.de_density_scale(3.0), rtol=1e-07)
    assert u.allclose(cosmo.de_density_scale([1, 2, 3]), cosmo.de_density_scale([1.0, 2.0, 3.0]), rtol=1e-07)
    cosmo = wpwaCDM(H0=70, Om0=0.3, Ode0=0.7, wp=-0.9, wa=0.2, zp=0.5)
    flatcosmo = FlatwpwaCDM(H0=70, Om0=0.3, wp=-0.9, wa=0.2, zp=0.5)
    assert u.allclose(cosmo.de_density_scale(z), flatcosmo.de_density_scale(z), rtol=1e-07)