"""
Module to test fitting routines
"""
import os.path
import unittest.mock as mk
from importlib.metadata import EntryPoint
from itertools import combinations
from unittest import mock
import numpy as np
import pytest
from numpy import linalg
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from astropy.modeling import models
from astropy.modeling.core import Fittable2DModel, Parameter
from astropy.modeling.fitting import DogBoxLSQFitter, Fitter, FittingWithOutlierRemoval, JointFitter, LevMarLSQFitter, LinearLSQFitter, LMLSQFitter, NonFiniteValueError, SimplexLSQFitter, SLSQPLSQFitter, TRFLSQFitter, _NLLSQFitter, populate_entry_points
from astropy.modeling.optimizers import Optimization
from astropy.stats import sigma_clip
from astropy.utils import NumpyRNGContext
from astropy.utils.compat.optional_deps import HAS_SCIPY
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning
from . import irafutil
if HAS_SCIPY:
    from scipy import optimize
fitters = [SimplexLSQFitter, SLSQPLSQFitter]
non_linear_fitters = [LevMarLSQFitter, TRFLSQFitter, LMLSQFitter, DogBoxLSQFitter]
_RANDOM_SEED = 4919

class TestPolynomial2D:
    """Tests for 2D polynomial fitting."""

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.model = models.Polynomial2D(2)
        (self.y, self.x) = np.mgrid[:5, :5]

        def poly2(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return 1 + 2 * x + 3 * x ** 2 + 4 * y + 5 * y ** 2 + 6 * x * y
        self.z = poly2(self.x, self.y)

    def test_poly2D_fitting(self):
        if False:
            for i in range(10):
                print('nop')
        fitter = LinearLSQFitter()
        v = self.model.fit_deriv(x=self.x, y=self.y)
        p = linalg.lstsq(v, self.z.flatten(), rcond=-1)[0]
        new_model = fitter(self.model, self.x, self.y, self.z)
        assert_allclose(new_model.parameters, p)

    def test_eval(self):
        if False:
            for i in range(10):
                print('nop')
        fitter = LinearLSQFitter()
        new_model = fitter(self.model, self.x, self.y, self.z)
        assert_allclose(new_model(self.x, self.y), self.z)

    @pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_nonlinear_fitting(self, fitter):
        if False:
            while True:
                i = 10
        fitter = fitter()
        self.model.parameters = [0.6, 1.8, 2.9, 3.7, 4.9, 6.7]
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            new_model = fitter(self.model, self.x, self.y, self.z)
        assert_allclose(new_model.parameters, [1, 2, 3, 4, 5, 6])

    @pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
    def test_compare_nonlinear_fitting(self):
        if False:
            print('Hello World!')
        self.model.parameters = [0.6, 1.8, 2.9, 3.7, 4.9, 6.7]
        fit_models = []
        for fitter in non_linear_fitters:
            fitter = fitter()
            with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
                fit_models.append(fitter(self.model, self.x, self.y, self.z))
        for pair in combinations(fit_models, 2):
            assert_allclose(pair[0].parameters, pair[1].parameters)

class TestICheb2D:
    """
    Tests 2D Chebyshev polynomial fitting

    Create a 2D polynomial (z) using Polynomial2DModel and default coefficients
    Fit z using a ICheb2D model
    Evaluate the ICheb2D polynomial and compare with the initial z
    """

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        self.pmodel = models.Polynomial2D(2)
        (self.y, self.x) = np.mgrid[:5, :5]
        self.z = self.pmodel(self.x, self.y)
        self.cheb2 = models.Chebyshev2D(2, 2)
        self.fitter = LinearLSQFitter()

    def test_default_params(self):
        if False:
            return 10
        self.cheb2.parameters = np.arange(9)
        p = np.array([1344.0, 1772.0, 400.0, 1860.0, 2448.0, 552.0, 432.0, 568.0, 128.0])
        z = self.cheb2(self.x, self.y)
        model = self.fitter(self.cheb2, self.x, self.y, z)
        assert_almost_equal(model.parameters, p)

    def test_poly2D_cheb2D(self):
        if False:
            while True:
                i = 10
        model = self.fitter(self.cheb2, self.x, self.y, self.z)
        z1 = model(self.x, self.y)
        assert_almost_equal(self.z, z1)

    @pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_chebyshev2D_nonlinear_fitting(self, fitter):
        if False:
            for i in range(10):
                print('nop')
        fitter = fitter()
        cheb2d = models.Chebyshev2D(2, 2)
        cheb2d.parameters = np.arange(9)
        z = cheb2d(self.x, self.y)
        cheb2d.parameters = [0.1, 0.6, 1.8, 2.9, 3.7, 4.9, 6.7, 7.5, 8.9]
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            model = fitter(cheb2d, self.x, self.y, z)
        assert_allclose(model.parameters, [0, 1, 2, 3, 4, 5, 6, 7, 8], atol=10 ** (-9))

    @pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_chebyshev2D_nonlinear_fitting_with_weights(self, fitter):
        if False:
            while True:
                i = 10
        fitter = fitter()
        cheb2d = models.Chebyshev2D(2, 2)
        cheb2d.parameters = np.arange(9)
        z = cheb2d(self.x, self.y)
        cheb2d.parameters = [0.1, 0.6, 1.8, 2.9, 3.7, 4.9, 6.7, 7.5, 8.9]
        weights = np.ones_like(self.y)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            model = fitter(cheb2d, self.x, self.y, z, weights=weights)
        assert_allclose(model.parameters, [0, 1, 2, 3, 4, 5, 6, 7, 8], atol=10 ** (-9))

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class TestJointFitter:
    """
    Tests the joint fitting routine using 2 gaussian models
    """

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        '\n        Create 2 gaussian models and some data with noise.\n        Create a fitter for the two models keeping the amplitude parameter\n        common for the two models.\n        '
        self.g1 = models.Gaussian1D(10, mean=14.9, stddev=0.3)
        self.g2 = models.Gaussian1D(10, mean=13, stddev=0.4)
        self.jf = JointFitter([self.g1, self.g2], {self.g1: ['amplitude'], self.g2: ['amplitude']}, [9.8])
        self.x = np.arange(10, 20, 0.1)
        y1 = self.g1(self.x)
        y2 = self.g2(self.x)
        with NumpyRNGContext(_RANDOM_SEED):
            n = np.random.randn(100)
        self.ny1 = y1 + 2 * n
        self.ny2 = y2 + 2 * n
        self.jf(self.x, self.ny1, self.x, self.ny2)

    def test_joint_parameter(self):
        if False:
            return 10
        '\n        Tests that the amplitude of the two models is the same\n        '
        assert_allclose(self.jf.fitparams[0], self.g1.parameters[0])
        assert_allclose(self.jf.fitparams[0], self.g2.parameters[0])

    def test_joint_fitter(self):
        if False:
            print('Hello World!')
        '\n        Tests the fitting routine with similar procedure.\n        Compares the fitted parameters.\n        '
        p1 = [14.9, 0.3]
        p2 = [13, 0.4]
        A = 9.8
        p = np.r_[A, p1, p2]

        def model(A, p, x):
            if False:
                for i in range(10):
                    print('nop')
            return A * np.exp(-0.5 / p[1] ** 2 * (x - p[0]) ** 2)

        def errfunc(p, x1, y1, x2, y2):
            if False:
                for i in range(10):
                    print('nop')
            return np.ravel(np.r_[model(p[0], p[1:3], x1) - y1, model(p[0], p[3:], x2) - y2])
        (coeff, _) = optimize.leastsq(errfunc, p, args=(self.x, self.ny1, self.x, self.ny2))
        assert_allclose(coeff, self.jf.fitparams, rtol=10 ** (-2))

class TestLinearLSQFitter:

    def test_compound_model_raises_error(self):
        if False:
            return 10
        'Test that if an user tries to use a compound model, raises an error'
        MESSAGE = 'Model must be simple, not compound'
        with pytest.raises(ValueError, match=MESSAGE):
            init_model1 = models.Polynomial1D(degree=2, c0=[1, 1], n_models=2)
            init_model2 = models.Polynomial1D(degree=2, c0=[1, 1], n_models=2)
            init_model_comp = init_model1 + init_model2
            x = np.arange(10)
            y = init_model_comp(x, model_set_axis=False)
            fitter = LinearLSQFitter()
            fitter(init_model_comp, x, y)

    def test_chebyshev1D(self):
        if False:
            while True:
                i = 10
        'Tests fitting a 1D Chebyshev polynomial to some real world data.'
        test_file = get_pkg_data_filename(os.path.join('data', 'idcompspec.fits'))
        with open(test_file) as f:
            lines = f.read()
            reclist = lines.split('begin')
        record = irafutil.IdentifyRecord(reclist[1])
        coeffs = record.coeff
        order = int(record.fields['order'])
        initial_model = models.Chebyshev1D(order - 1, domain=record.get_range())
        fitter = LinearLSQFitter()
        fitted_model = fitter(initial_model, record.x, record.z)
        assert_allclose(fitted_model.parameters, np.array(coeffs), rtol=0.1)

    def test_linear_fit_model_set(self):
        if False:
            while True:
                i = 10
        'Tests fitting multiple models simultaneously.'
        init_model = models.Polynomial1D(degree=2, c0=[1, 1], n_models=2)
        x = np.arange(10)
        y_expected = init_model(x, model_set_axis=False)
        assert y_expected.shape == (2, 10)
        with NumpyRNGContext(_RANDOM_SEED):
            y = y_expected + np.random.normal(0, 0.01, size=y_expected.shape)
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y)
        assert_allclose(fitted_model(x, model_set_axis=False), y_expected, rtol=0.1)

    def test_linear_fit_2d_model_set(self):
        if False:
            while True:
                i = 10
        'Tests fitted multiple 2-D models simultaneously.'
        init_model = models.Polynomial2D(degree=2, c0_0=[1, 1], n_models=2)
        x = np.arange(10)
        y = np.arange(10)
        z_expected = init_model(x, y, model_set_axis=False)
        assert z_expected.shape == (2, 10)
        with NumpyRNGContext(_RANDOM_SEED):
            z = z_expected + np.random.normal(0, 0.01, size=z_expected.shape)
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y, z)
        assert_allclose(fitted_model(x, y, model_set_axis=False), z_expected, rtol=0.1)

    def test_linear_fit_fixed_parameter(self):
        if False:
            print('Hello World!')
        '\n        Tests fitting a polynomial model with a fixed parameter (issue #6135).\n        '
        init_model = models.Polynomial1D(degree=2, c1=1)
        init_model.c1.fixed = True
        x = np.arange(10)
        y = 2 + x + 0.5 * x * x
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y)
        assert_allclose(fitted_model.parameters, [2.0, 1.0, 0.5], atol=1e-14)

    def test_linear_fit_model_set_fixed_parameter(self):
        if False:
            return 10
        '\n        Tests fitting a polynomial model set with a fixed parameter (#6135).\n        '
        init_model = models.Polynomial1D(degree=2, c1=[1, -2], n_models=2)
        init_model.c1.fixed = True
        x = np.arange(10)
        yy = np.array([2 + x + 0.5 * x * x, -2 * x])
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, yy)
        assert_allclose(fitted_model.c0, [2.0, 0.0], atol=1e-14)
        assert_allclose(fitted_model.c1, [1.0, -2.0], atol=1e-14)
        assert_allclose(fitted_model.c2, [0.5, 0.0], atol=1e-14)

    def test_linear_fit_2d_model_set_fixed_parameters(self):
        if False:
            return 10
        '\n        Tests fitting a 2d polynomial model set with fixed parameters (#6135).\n        '
        init_model = models.Polynomial2D(degree=2, c1_0=[1, 2], c0_1=[-0.5, 1], n_models=2, fixed={'c1_0': True, 'c0_1': True})
        (x, y) = np.mgrid[0:5, 0:5]
        zz = np.array([1 + x - 0.5 * y + 0.1 * x * x, 2 * x + y - 0.2 * y * y])
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y, zz)
        assert_allclose(fitted_model(x, y, model_set_axis=False), zz, atol=1e-14)

    def test_linear_fit_model_set_masked_values(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests model set fitting with masked value(s) (#4824, #6819).\n        '
        init_model = models.Polynomial1D(degree=1, n_models=2)
        x = np.arange(10)
        y = np.ma.masked_array([2 * x + 1, x - 2], mask=np.zeros_like([x, x]))
        y[0, 7] = 100.0
        y.mask[0, 7] = True
        y[1, 1:3] = -100.0
        y.mask[1, 1:3] = True
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y)
        assert_allclose(fitted_model.c0, [1.0, -2.0], atol=1e-14)
        assert_allclose(fitted_model.c1, [2.0, 1.0], atol=1e-14)

    def test_linear_fit_2d_model_set_masked_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests 2D model set fitting with masked value(s) (#4824, #6819).\n        '
        init_model = models.Polynomial2D(1, n_models=2)
        (x, y) = np.mgrid[0:5, 0:5]
        z = np.ma.masked_array([2 * x + 3 * y + 1, x - 0.5 * y - 2], mask=np.zeros_like([x, x]))
        z[0, 3, 1] = -1000.0
        z.mask[0, 3, 1] = True
        fitter = LinearLSQFitter()
        fitted_model = fitter(init_model, x, y, z)
        assert_allclose(fitted_model.c0_0, [1.0, -2.0], atol=1e-14)
        assert_allclose(fitted_model.c1_0, [2.0, 1.0], atol=1e-14)
        assert_allclose(fitted_model.c0_1, [3.0, -0.5], atol=1e-14)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class TestNonLinearFitters:
    """Tests non-linear least squares fitting and the SLSQP algorithm."""

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        self.initial_values = [100, 5, 1]
        self.xdata = np.arange(0, 10, 0.1)
        sigma = 4.0 * np.ones_like(self.xdata)
        with NumpyRNGContext(_RANDOM_SEED):
            yerror = np.random.normal(0, sigma)

        def func(p, x):
            if False:
                print('Hello World!')
            return p[0] * np.exp(-0.5 / p[2] ** 2 * (x - p[1]) ** 2)
        self.ydata = func(self.initial_values, self.xdata) + yerror
        self.gauss = models.Gaussian1D(100, 5, stddev=1)

    @pytest.mark.parametrize('fitter0', non_linear_fitters)
    @pytest.mark.parametrize('fitter1', non_linear_fitters)
    def test_estimated_vs_analytic_deriv(self, fitter0, fitter1):
        if False:
            while True:
                i = 10
        '\n        Runs `LevMarLSQFitter` and `TRFLSQFitter` with estimated and\n        analytic derivatives of a `Gaussian1D`.\n        '
        fitter0 = fitter0()
        model = fitter0(self.gauss, self.xdata, self.ydata)
        g1e = models.Gaussian1D(100, 5.0, stddev=1)
        fitter1 = fitter1()
        emodel = fitter1(g1e, self.xdata, self.ydata, estimate_jacobian=True)
        assert_allclose(model.parameters, emodel.parameters, rtol=10 ** (-3))

    @pytest.mark.parametrize('fitter0', non_linear_fitters)
    @pytest.mark.parametrize('fitter1', non_linear_fitters)
    def test_estimated_vs_analytic_deriv_with_weights(self, fitter0, fitter1):
        if False:
            print('Hello World!')
        '\n        Runs `LevMarLSQFitter` and `TRFLSQFitter` with estimated and\n        analytic derivatives of a `Gaussian1D`.\n        '
        weights = 1.0 / (self.ydata / 10.0)
        fitter0 = fitter0()
        model = fitter0(self.gauss, self.xdata, self.ydata, weights=weights)
        g1e = models.Gaussian1D(100, 5.0, stddev=1)
        fitter1 = fitter1()
        emodel = fitter1(g1e, self.xdata, self.ydata, weights=weights, estimate_jacobian=True)
        assert_allclose(model.parameters, emodel.parameters, rtol=10 ** (-3))

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_with_optimize(self, fitter):
        if False:
            while True:
                i = 10
        '\n        Tests results from `LevMarLSQFitter` and `TRFLSQFitter` against\n        `scipy.optimize.leastsq`.\n        '
        fitter = fitter()
        model = fitter(self.gauss, self.xdata, self.ydata, estimate_jacobian=True)

        def func(p, x):
            if False:
                print('Hello World!')
            return p[0] * np.exp(-0.5 / p[2] ** 2 * (x - p[1]) ** 2)

        def errfunc(p, x, y):
            if False:
                while True:
                    i = 10
            return func(p, x) - y
        result = optimize.leastsq(errfunc, self.initial_values, args=(self.xdata, self.ydata))
        assert_allclose(model.parameters, result[0], rtol=10 ** (-3))

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_with_weights(self, fitter):
        if False:
            return 10
        '\n        Tests results from `LevMarLSQFitter` and `TRFLSQFitter` with weights.\n        '
        fitter = fitter()
        model = fitter(self.gauss, self.xdata, self.ydata, estimate_jacobian=True)
        withw = fitter(self.gauss, self.xdata, self.ydata, estimate_jacobian=True, weights=np.ones_like(self.xdata))
        assert_allclose(model.parameters, withw.parameters, rtol=10 ** (-4))
        weights = np.zeros_like(self.xdata)
        weights[::2] = 1.0
        mask = weights >= 1.0
        model = fitter(self.gauss, self.xdata[mask], self.ydata[mask], estimate_jacobian=True)
        withw = fitter(self.gauss, self.xdata, self.ydata, estimate_jacobian=True, weights=weights)
        assert_allclose(model.parameters, withw.parameters, rtol=10 ** (-4))

    @pytest.mark.filterwarnings('ignore:.* Maximum number of iterations reached')
    @pytest.mark.filterwarnings('ignore:Values in x were outside bounds during a minimize step, clipping to bounds')
    @pytest.mark.parametrize('fitter_class', fitters)
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_fitter_against_LevMar(self, fitter_class, fitter):
        if False:
            i = 10
            return i + 15
        '\n        Tests results from non-linear fitters against `LevMarLSQFitter`\n        and `TRFLSQFitter`\n        '
        fitter = fitter()
        fitter_cls = fitter_class()
        new_model = fitter_cls(self.gauss, self.xdata, self.ydata)
        model = fitter(self.gauss, self.xdata, self.ydata)
        assert_allclose(model.parameters, new_model.parameters, rtol=10 ** (-4))

    @pytest.mark.filterwarnings('ignore:Values in x were outside bounds during a minimize step, clipping to bounds')
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_LSQ_SLSQP_with_constraints(self, fitter):
        if False:
            print('Hello World!')
        '\n        Runs `LevMarLSQFitter`/`TRFLSQFitter` and `SLSQPLSQFitter` on a\n        model with constraints.\n        '
        fitter = fitter()
        g1 = models.Gaussian1D(100, 5, stddev=1)
        g1.mean.fixed = True
        fslsqp = SLSQPLSQFitter()
        slsqp_model = fslsqp(g1, self.xdata, self.ydata)
        model = fitter(g1, self.xdata, self.ydata)
        assert_allclose(model.parameters, slsqp_model.parameters, rtol=10 ** (-4))

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_non_linear_lsq_fitter_with_weights(self, fitter):
        if False:
            while True:
                i = 10
        '\n        Tests that issue #11581 has been solved.\n        '
        fitter = fitter()
        np.random.seed(42)
        norder = 2
        fitter2 = LinearLSQFitter()
        model = models.Polynomial1D(norder)
        npts = 10000
        c = [2.0, -10.0, 7.0]
        tw = np.random.uniform(0.0, 10.0, npts)
        tx = np.random.uniform(0.0, 10.0, npts)
        ty = c[0] + c[1] * tx + c[2] * tx ** 2
        ty += np.random.normal(0.0, 1.5, npts)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            tf1 = fitter(model, tx, ty, weights=tw)
        tf2 = fitter2(model, tx, ty, weights=tw)
        assert_allclose(tf1.parameters, tf2.parameters, atol=10 ** (-16))
        assert_allclose(tf1.parameters, c, rtol=10 ** (-2), atol=10 ** (-2))
        model = models.Gaussian1D()
        if isinstance(fitter, (TRFLSQFitter, LMLSQFitter)):
            with pytest.warns(AstropyUserWarning, match='The fit may be unsuccessful; *.'):
                fitter(model, tx, ty, weights=tw)
        else:
            fitter(model, tx, ty, weights=tw)
        model = models.Polynomial2D(norder)
        nxpts = 100
        nypts = 150
        npts = nxpts * nypts
        c = [1.0, 4.0, 7.0, -8.0, -9.0, -3.0]
        tw = np.random.uniform(0.0, 10.0, npts).reshape(nxpts, nypts)
        tx = np.random.uniform(0.0, 10.0, npts).reshape(nxpts, nypts)
        ty = np.random.uniform(0.0, 10.0, npts).reshape(nxpts, nypts)
        tz = c[0] + c[1] * tx + c[2] * tx ** 2 + c[3] * ty + c[4] * ty ** 2 + c[5] * tx * ty
        tz += np.random.normal(0.0, 1.5, npts).reshape(nxpts, nypts)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            tf1 = fitter(model, tx, ty, tz, weights=tw)
        tf2 = fitter2(model, tx, ty, tz, weights=tw)
        assert_allclose(tf1.parameters, tf2.parameters, atol=10 ** (-16))
        assert_allclose(tf1.parameters, c, rtol=10 ** (-2), atol=10 ** (-2))

    def test_simplex_lsq_fitter(self):
        if False:
            for i in range(10):
                print('nop')
        'A basic test for the `SimplexLSQ` fitter.'

        class Rosenbrock(Fittable2DModel):
            a = Parameter()
            b = Parameter()

            @staticmethod
            def evaluate(x, y, a, b):
                if False:
                    i = 10
                    return i + 15
                return (a - x) ** 2 + b * (y - x ** 2) ** 2
        x = y = np.linspace(-3.0, 3.0, 100)
        with NumpyRNGContext(_RANDOM_SEED):
            z = Rosenbrock.evaluate(x, y, 1.0, 100.0)
            z += np.random.normal(0.0, 0.1, size=z.shape)
        fitter = SimplexLSQFitter()
        r_i = Rosenbrock(1, 100)
        r_f = fitter(r_i, x, y, z)
        assert_allclose(r_f.parameters, [1.0, 100.0], rtol=0.01)

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_param_cov(self, fitter):
        if False:
            while True:
                i = 10
        "\n        Tests that the 'param_cov' fit_info entry gets the right answer for\n        *linear* least squares, where the answer is exact\n        "
        fitter = fitter()
        a = 2
        b = 100
        with NumpyRNGContext(_RANDOM_SEED):
            x = np.linspace(0, 1, 100)
            y = x * a + b + np.random.randn(len(x))
        X = np.vstack([x, np.ones(len(x))]).T
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y.T)
        s2 = np.sum((y - np.matmul(X, beta).ravel()) ** 2) / (len(y) - len(beta))
        olscov = np.linalg.inv(np.matmul(X.T, X)) * s2
        mod = models.Linear1D(a, b)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            fmod = fitter(mod, x, y)
        assert_allclose(fmod.parameters, beta.ravel())
        assert_allclose(olscov, fitter.fit_info['param_cov'])

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_param_cov_with_uncertainties(self, fitter):
        if False:
            print('Hello World!')
        "\n        Tests that the 'param_cov' fit_info entry gets the right answer for\n        *linear* least squares, where the answer is exact\n        "
        fitter = fitter()
        a = 2
        b = 100
        with NumpyRNGContext(_RANDOM_SEED):
            x = np.linspace(0, 1, 100)
            y = x * a + b + np.random.normal(size=len(x))
            sigma = np.random.normal(loc=1, scale=0.1, size=len(x))
        X = np.vstack([x, np.ones(len(x))]).T
        inv_N = np.linalg.inv(np.diag(sigma) ** 2)
        cov = np.linalg.inv(X.T @ inv_N @ X)
        beta = cov @ X.T @ inv_N @ y.T
        mod = models.Linear1D(a, b)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            fmod = fitter(mod, x, y, weights=sigma ** (-1))
        assert_allclose(fmod.parameters, beta.ravel())
        assert_allclose(cov, fitter.fit_info['param_cov'])

class TestEntryPoint:
    """Tests population of fitting with entry point fitters"""

    def successfulimport(self):
        if False:
            return 10

        class goodclass(Fitter):
            __name__ = 'GoodClass'
        return goodclass

    def raiseimporterror(self):
        if False:
            while True:
                i = 10
        raise ImportError

    def returnbadfunc(self):
        if False:
            print('Hello World!')

        def badfunc():
            if False:
                i = 10
                return i + 15
            pass
        return badfunc

    def returnbadclass(self):
        if False:
            while True:
                i = 10

        class badclass:
            pass
        return badclass

    def test_working(self):
        if False:
            print('Hello World!')
        'This should work fine'
        mock_entry_working = mock.create_autospec(EntryPoint)
        mock_entry_working.name = 'Working'
        mock_entry_working.load = self.successfulimport
        populate_entry_points([mock_entry_working])

    def test_import_error(self):
        if False:
            print('Hello World!')
        'This raises an import error on load to test that it is handled correctly'
        mock_entry_importerror = mock.create_autospec(EntryPoint)
        mock_entry_importerror.name = 'IErr'
        mock_entry_importerror.load = self.raiseimporterror
        with pytest.warns(AstropyUserWarning, match='.*ImportError.*'):
            populate_entry_points([mock_entry_importerror])

    def test_bad_func(self):
        if False:
            print('Hello World!')
        'This returns a function which fails the type check'
        mock_entry_badfunc = mock.create_autospec(EntryPoint)
        mock_entry_badfunc.name = 'BadFunc'
        mock_entry_badfunc.load = self.returnbadfunc
        with pytest.warns(AstropyUserWarning, match='.*Class.*'):
            populate_entry_points([mock_entry_badfunc])

    def test_bad_class(self):
        if False:
            return 10
        "This returns a class which doesn't inherient from fitter"
        mock_entry_badclass = mock.create_autospec(EntryPoint)
        mock_entry_badclass.name = 'BadClass'
        mock_entry_badclass.load = self.returnbadclass
        with pytest.warns(AstropyUserWarning, match='.*BadClass.*'):
            populate_entry_points([mock_entry_badclass])

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class Test1DFittingWithOutlierRemoval:

    def setup_class(self):
        if False:
            while True:
                i = 10
        self.x = np.linspace(-5.0, 5.0, 200)
        self.model_params = (3.0, 1.3, 0.8)

        def func(p, x):
            if False:
                for i in range(10):
                    print('nop')
            return p[0] * np.exp(-0.5 * (x - p[1]) ** 2 / p[2] ** 2)
        self.y = func(self.model_params, self.x)

    @pytest.mark.filterwarnings('ignore:The fit may be unsuccessful')
    @pytest.mark.filterwarnings('ignore:Values in x were outside bounds during a minimize step, clipping to bounds')
    @pytest.mark.parametrize('fitter', non_linear_fitters + fitters)
    def test_with_fitters_and_sigma_clip(self, fitter):
        if False:
            return 10
        import scipy.stats as stats
        fitter = fitter()
        np.random.seed(0)
        c = stats.bernoulli.rvs(0.25, size=self.x.shape)
        y = self.y + (np.random.normal(0.0, 0.2, self.x.shape) + c * np.random.normal(3.0, 5.0, self.x.shape))
        g_init = models.Gaussian1D(amplitude=1.0, mean=0, stddev=1.0)
        fit = FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
        (fitted_model, _) = fit(g_init, self.x, y)
        assert_allclose(fitted_model.parameters, self.model_params, rtol=0.1)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class Test2DFittingWithOutlierRemoval:

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        (self.y, self.x) = np.mgrid[-3:3:128j, -3:3:128j]
        self.model_params = (3.0, 1.0, 0.0, 0.8, 0.8)

        def Gaussian_2D(p, pos):
            if False:
                while True:
                    i = 10
            return p[0] * np.exp(-0.5 * (pos[0] - p[2]) ** 2 / p[4] ** 2 - 0.5 * (pos[1] - p[1]) ** 2 / p[3] ** 2)
        self.z = Gaussian_2D(self.model_params, np.array([self.y, self.x]))

    def initial_guess(self, data, pos):
        if False:
            while True:
                i = 10
        y = pos[0]
        x = pos[1]
        'computes the centroid of the data as the initial guess for the\n        center position'
        wx = x * data
        wy = y * data
        total_intensity = np.sum(data)
        x_mean = np.sum(wx) / total_intensity
        y_mean = np.sum(wy) / total_intensity
        x_to_pixel = x[0].size / (x[x[0].size - 1][x[0].size - 1] - x[0][0])
        y_to_pixel = y[0].size / (y[y[0].size - 1][y[0].size - 1] - y[0][0])
        x_pos = np.around(x_mean * x_to_pixel + x[0].size / 2.0).astype(int)
        y_pos = np.around(y_mean * y_to_pixel + y[0].size / 2.0).astype(int)
        amplitude = data[y_pos][x_pos]
        return (amplitude, x_mean, y_mean)

    @pytest.mark.filterwarnings('ignore:The fit may be unsuccessful')
    @pytest.mark.filterwarnings('ignore:Values in x were outside bounds during a minimize step, clipping to bounds')
    @pytest.mark.parametrize('fitter', non_linear_fitters + fitters)
    def test_with_fitters_and_sigma_clip(self, fitter):
        if False:
            return 10
        import scipy.stats as stats
        fitter = fitter()
        np.random.seed(0)
        c = stats.bernoulli.rvs(0.25, size=self.z.shape)
        z = self.z + (np.random.normal(0.0, 0.2, self.z.shape) + c * np.random.normal(self.z, 2.0, self.z.shape))
        guess = self.initial_guess(self.z, np.array([self.y, self.x]))
        g2_init = models.Gaussian2D(amplitude=guess[0], x_mean=guess[1], y_mean=guess[2], x_stddev=0.75, y_stddev=1.25)
        fit = FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
        (fitted_model, _) = fit(g2_init, self.x, self.y, z)
        assert_allclose(fitted_model.parameters[0:5], self.model_params, atol=0.1)

def test_1d_set_fitting_with_outlier_removal():
    if False:
        for i in range(10):
            print('nop')
    'Test model set fitting with outlier removal (issue #6819)'
    poly_set = models.Polynomial1D(2, n_models=2)
    fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, sigma=2.5, niter=3, cenfunc=np.ma.mean, stdfunc=np.ma.std)
    x = np.arange(10)
    y = np.array([2.5 * x - 4, 2 * x * x + x + 10])
    y[1, 5] = -1000
    (poly_set, filt_y) = fitter(poly_set, x, y)
    assert_allclose(poly_set.c0, [-4.0, 10.0], atol=1e-14)
    assert_allclose(poly_set.c1, [2.5, 1.0], atol=1e-14)
    assert_allclose(poly_set.c2, [0.0, 2.0], atol=1e-14)

def test_2d_set_axis_2_fitting_with_outlier_removal():
    if False:
        return 10
    'Test fitting 2D model set (axis 2) with outlier removal (issue #6819)'
    poly_set = models.Polynomial2D(1, n_models=2, model_set_axis=2)
    fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, sigma=2.5, niter=3, cenfunc=np.ma.mean, stdfunc=np.ma.std)
    (y, x) = np.mgrid[0:5, 0:5]
    z = np.rollaxis(np.array([x + y, 1 - 0.1 * x + 0.2 * y]), 0, 3)
    z[3, 3:5, 0] = 100.0
    (poly_set, filt_z) = fitter(poly_set, x, y, z)
    assert_allclose(poly_set.c0_0, [[[0.0, 1.0]]], atol=1e-14)
    assert_allclose(poly_set.c1_0, [[[1.0, -0.1]]], atol=1e-14)
    assert_allclose(poly_set.c0_1, [[[1.0, 0.2]]], atol=1e-14)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class TestWeightedFittingWithOutlierRemoval:
    """Issue #7020"""

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        (self.y, self.x) = np.mgrid[0:20, 0:20]
        self.z = np.mod(self.x + self.y, 2) * 2 - 1
        self.weights = np.mod(self.x + self.y, 2) * 2 + 1
        self.z[0, 0] = 1000.0
        self.z[0, 1] = 1000.0
        self.x1d = self.x.flatten()
        self.z1d = self.z.flatten()
        self.weights1d = self.weights.flatten()

    def test_1d_without_weights_without_sigma_clip(self):
        if False:
            for i in range(10):
                print('nop')
        model = models.Polynomial1D(0)
        fitter = LinearLSQFitter()
        fit = fitter(model, self.x1d, self.z1d)
        assert_allclose(fit.parameters[0], self.z1d.mean(), atol=10 ** (-2))

    def test_1d_without_weights_with_sigma_clip(self):
        if False:
            return 10
        model = models.Polynomial1D(0)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        (fit, mask) = fitter(model, self.x1d, self.z1d)
        assert (~mask).sum() == self.z1d.size - 2
        assert mask[0] and mask[1]
        assert_allclose(fit.parameters[0], 0.0, atol=10 ** (-2))

    def test_1d_with_weights_without_sigma_clip(self):
        if False:
            while True:
                i = 10
        model = models.Polynomial1D(0)
        fitter = LinearLSQFitter()
        fit = fitter(model, self.x1d, self.z1d, weights=self.weights1d)
        assert fit.parameters[0] > 1.0

    def test_1d_with_weights_with_sigma_clip(self):
        if False:
            i = 10
            return i + 15
        '\n        smoke test for #7020 - fails without fitting.py\n        patch because weights does not propagate\n        '
        model = models.Polynomial1D(0)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        (fit, filtered) = fitter(model, self.x1d, self.z1d, weights=self.weights1d)
        assert fit.parameters[0] > 10 ** (-2)
        assert fit.parameters[0] < 1.0

    def test_1d_set_with_common_weights_with_sigma_clip(self):
        if False:
            print('Hello World!')
        'added for #6819 (1D model set with weights in common)'
        model = models.Polynomial1D(0, n_models=2)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        z1d = np.array([self.z1d, self.z1d])
        (fit, filtered) = fitter(model, self.x1d, z1d, weights=self.weights1d)
        assert_allclose(fit.parameters, [0.8, 0.8], atol=1e-14)

    def test_1d_set_with_weights_with_sigma_clip(self):
        if False:
            for i in range(10):
                print('nop')
        '1D model set with separate weights'
        model = models.Polynomial1D(0, n_models=2)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        z1d = np.array([self.z1d, self.z1d])
        weights = np.array([self.weights1d, self.weights1d])
        (fit, filtered) = fitter(model, self.x1d, z1d, weights=weights)
        assert_allclose(fit.parameters, [0.8, 0.8], atol=1e-14)

    def test_2d_without_weights_without_sigma_clip(self):
        if False:
            while True:
                i = 10
        model = models.Polynomial2D(0)
        fitter = LinearLSQFitter()
        fit = fitter(model, self.x, self.y, self.z)
        assert_allclose(fit.parameters[0], self.z.mean(), atol=10 ** (-2))

    def test_2d_without_weights_with_sigma_clip(self):
        if False:
            print('Hello World!')
        model = models.Polynomial2D(0)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        (fit, mask) = fitter(model, self.x, self.y, self.z)
        assert (~mask).sum() == self.z.size - 2
        assert mask[0, 0] and mask[0, 1]
        assert_allclose(fit.parameters[0], 0.0, atol=10 ** (-2))

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_2d_with_weights_without_sigma_clip(self, fitter):
        if False:
            while True:
                i = 10
        fitter = fitter()
        model = models.Polynomial2D(0)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            fit = fitter(model, self.x, self.y, self.z, weights=self.weights)
        assert fit.parameters[0] > 1.0

    def test_2d_linear_with_weights_without_sigma_clip(self):
        if False:
            i = 10
            return i + 15
        model = models.Polynomial2D(0)
        fitter = LinearLSQFitter()
        fit = fitter(model, self.x, self.y, self.z, weights=self.weights)
        assert fit.parameters[0] > 1.0

    @pytest.mark.parametrize('base_fitter', non_linear_fitters)
    def test_2d_with_weights_with_sigma_clip(self, base_fitter):
        if False:
            print('Hello World!')
        'smoke test for #7020 - fails without fitting.py patch because\n        weights does not propagate'
        base_fitter = base_fitter()
        model = models.Polynomial2D(0)
        fitter = FittingWithOutlierRemoval(base_fitter, sigma_clip, niter=3, sigma=3.0)
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            (fit, _) = fitter(model, self.x, self.y, self.z, weights=self.weights)
        assert fit.parameters[0] > 10 ** (-2)
        assert fit.parameters[0] < 1.0

    def test_2d_linear_with_weights_with_sigma_clip(self):
        if False:
            for i in range(10):
                print('nop')
        'same as test above with a linear fitter.'
        model = models.Polynomial2D(0)
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        (fit, _) = fitter(model, self.x, self.y, self.z, weights=self.weights)
        assert fit.parameters[0] > 10 ** (-2)
        assert fit.parameters[0] < 1.0

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', non_linear_fitters)
def test_fitters_with_weights(fitter):
    if False:
        i = 10
        return i + 15
    'Issue #5737'
    fitter = fitter()
    if isinstance(fitter, _NLLSQFitter):
        pytest.xfail('This test is poorly designed and causes issues for scipy.optimize.least_squares based fitters')
    (Xin, Yin) = np.mgrid[0:21, 0:21]
    with NumpyRNGContext(_RANDOM_SEED):
        zsig = np.random.normal(0, 0.01, size=Xin.shape)
    g2 = models.Gaussian2D(10, 10, 9, 2, 3)
    z = g2(Xin, Yin)
    gmod = fitter(models.Gaussian2D(15, 7, 8, 1.3, 1.2), Xin, Yin, z + zsig)
    assert_allclose(gmod.parameters, g2.parameters, atol=10 ** (-2))
    p2 = models.Polynomial2D(3)
    p2.parameters = np.arange(10) / 1.2
    z = p2(Xin, Yin)
    with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
        pmod = fitter(models.Polynomial2D(3), Xin, Yin, z + zsig)
    assert_allclose(pmod.parameters, p2.parameters, atol=10 ** (-2))

def test_linear_fitter_with_weights():
    if False:
        i = 10
        return i + 15
    'Regression test for #7035'
    (Xin, Yin) = np.mgrid[0:21, 0:21]
    fitter = LinearLSQFitter()
    with NumpyRNGContext(_RANDOM_SEED):
        zsig = np.random.normal(0, 0.01, size=Xin.shape)
    p2 = models.Polynomial2D(3)
    p2.parameters = np.arange(10) / 1.2
    z = p2(Xin, Yin)
    pmod = fitter(models.Polynomial2D(3), Xin, Yin, z + zsig, weights=zsig ** (-2))
    assert_allclose(pmod.parameters, p2.parameters, atol=10 ** (-2))

@pytest.mark.parametrize('fixed, warns', [({}, True), ({'c1_0': True}, True), ({'c0_1': True}, False)])
def test_polynomial_poorly_conditioned(fixed, warns):
    if False:
        for i in range(10):
            print('nop')
    p0 = models.Polynomial2D(degree=1, c0_0=3, c1_0=5, c0_1=0, fixed=fixed)
    fitter = LinearLSQFitter()
    x = [1, 2, 3, 4, 5]
    y = [1, 1, 1, 1, 1]
    values = p0(x, y)
    if warns:
        with pytest.warns(AstropyUserWarning, match='The fit may be poorly conditioned'):
            p = fitter(p0, x, y, values)
    else:
        p = fitter(p0, x, y, values)
        assert np.allclose(p0.parameters, p.parameters, rtol=0, atol=1e-14)

def test_linear_fitter_with_weights_flat():
    if False:
        for i in range(10):
            print('nop')
    'Same as the above #7035 test but with flattened inputs'
    (Xin, Yin) = np.mgrid[0:21, 0:21]
    (Xin, Yin) = (Xin.flatten(), Yin.flatten())
    fitter = LinearLSQFitter()
    with NumpyRNGContext(_RANDOM_SEED):
        zsig = np.random.normal(0, 0.01, size=Xin.shape)
    p2 = models.Polynomial2D(3)
    p2.parameters = np.arange(10) / 1.2
    z = p2(Xin, Yin)
    pmod = fitter(models.Polynomial2D(3), Xin, Yin, z + zsig, weights=zsig ** (-2))
    assert_allclose(pmod.parameters, p2.parameters, atol=10 ** (-2))

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.filterwarnings('ignore:The fit may be unsuccessful')
@pytest.mark.parametrize('fitter', non_linear_fitters + fitters)
def test_fitters_interface(fitter):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that ``**kwargs`` work with all optimizers.\n    This is a basic smoke test.\n    '
    fitter = fitter()
    model = models.Gaussian1D(10, 4, 0.3)
    x = np.arange(21)
    y = model(x)
    if isinstance(fitter, SimplexLSQFitter):
        kwargs = {'maxiter': 79, 'verblevel': 1, 'acc': 1e-06}
    else:
        kwargs = {'maxiter': 77, 'verblevel': 1, 'epsilon': 0.01, 'acc': 1e-06}
    if isinstance(fitter, (LevMarLSQFitter, _NLLSQFitter)):
        kwargs.pop('verblevel')
    _ = fitter(model, x, y, **kwargs)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter_class', [SLSQPLSQFitter, SimplexLSQFitter])
def test_optimizers(fitter_class):
    if False:
        print('Hello World!')
    fitter = fitter_class()
    assert fitter._opt_method.maxiter == 100
    fitter._opt_method.maxiter = 1000
    assert fitter._opt_method.maxiter == 1000
    assert fitter._opt_method.eps == np.sqrt(np.finfo(float).eps)
    fitter._opt_method.eps = 1e-16
    assert fitter._opt_method.eps == 1e-16
    assert fitter._opt_method.acc == 1e-07
    fitter._opt_method.acc = 1e-16
    assert fitter._opt_method.acc == 1e-16
    assert repr(fitter._opt_method) == f'{fitter._opt_method.__class__.__name__}()'
    fitparams = mk.MagicMock()
    final_func_val = mk.MagicMock()
    numiter = mk.MagicMock()
    funcalls = mk.MagicMock()
    exit_mode = 1
    mess = mk.MagicMock()
    xtol = mk.MagicMock()
    if fitter_class == SLSQPLSQFitter:
        return_value = (fitparams, final_func_val, numiter, exit_mode, mess)
        fit_info = {'final_func_val': final_func_val, 'numiter': numiter, 'exit_mode': exit_mode, 'message': mess}
    else:
        return_value = (fitparams, final_func_val, numiter, funcalls, exit_mode)
        fit_info = {'final_func_val': final_func_val, 'numiter': numiter, 'exit_mode': exit_mode, 'num_function_calls': funcalls}
    with mk.patch.object(fitter._opt_method.__class__, 'opt_method', return_value=return_value):
        with pytest.warns(AstropyUserWarning, match='The fit may be unsuccessful; .*'):
            assert (fitparams, fit_info) == fitter._opt_method(mk.MagicMock(), mk.MagicMock(), mk.MagicMock(), xtol=xtol)
        assert fit_info == fitter._opt_method.fit_info
        if isinstance(fitter, SLSQPLSQFitter):
            assert fitter._opt_method.acc == 1e-16
        else:
            assert fitter._opt_method.acc == xtol

@mk.patch.multiple(Optimization, __abstractmethods__=set())
def test_Optimization_abstract_call():
    if False:
        while True:
            i = 10
    optimization = Optimization(mk.MagicMock())
    MESSAGE = 'Subclasses should implement this method'
    with pytest.raises(NotImplementedError, match=MESSAGE):
        optimization()

def test_fitting_with_outlier_removal_niter():
    if False:
        i = 10
        return i + 15
    '\n    Test that FittingWithOutlierRemoval stops prior to reaching niter if the\n    set of masked points has converged and correctly reports the actual number\n    of iterations performed.\n    '
    x = np.arange(25)
    with NumpyRNGContext(_RANDOM_SEED):
        y = np.random.normal(loc=10.0, scale=1.0, size=(2, 25))
    y[0, 14] = 100.0
    fitter = FittingWithOutlierRemoval(fitter=LinearLSQFitter(), outlier_func=sigma_clip, niter=5, sigma_lower=3.0, sigma_upper=3.0, maxiters=1)
    (model, mask) = fitter(models.Chebyshev1D(2, n_models=2), x, y)
    assert_equal(np.where(mask), [[0], [14]])
    assert fitter.fit_info['niter'] == 2
    fitter = FittingWithOutlierRemoval(fitter=LinearLSQFitter(), outlier_func=sigma_clip, niter=0, sigma_lower=3.0, sigma_upper=3.0, maxiters=1)
    (model, mask) = fitter(models.Chebyshev1D(2), x, y[0])
    assert mask.sum() == 0
    assert fitter.fit_info['niter'] == 0

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
class TestFittingUncertanties:
    """
    Test that parameter covariance is calculated correctly for the fitters
    that do so (currently LevMarLSQFitter, LinearLSQFitter).
    """
    example_1D_models = [models.Polynomial1D(2), models.Linear1D()]
    example_1D_sets = [models.Polynomial1D(2, n_models=2, model_set_axis=False), models.Linear1D(n_models=2, slope=[1.0, 1.0], intercept=[0, 0])]

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(619)
        self.x = np.arange(10)
        self.x_grid = np.random.randint(0, 100, size=100).reshape(10, 10)
        self.y_grid = np.random.randint(0, 100, size=100).reshape(10, 10)
        self.rand_grid = np.random.random(100).reshape(10, 10)
        self.rand = self.rand_grid[0]

    @pytest.mark.parametrize(('single_model', 'model_set'), list(zip(example_1D_models, example_1D_sets)))
    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_1d_models(self, single_model, model_set, fitter):
        if False:
            while True:
                i = 10
        'Test that fitting uncertainties are computed correctly for 1D models\n        and 1D model sets. Use covariance/stds given by LevMarLSQFitter as\n        a benchmark since they are returned by the numpy fitter.\n        '
        fitter = fitter(calc_uncertainties=True)
        linlsq_fitter = LinearLSQFitter(calc_uncertainties=True)
        y = single_model(self.x) + self.rand
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            fit_model = fitter(single_model, self.x, y)
        cov_model = fit_model.cov_matrix.cov_matrix
        fit_model_linlsq = linlsq_fitter(single_model, self.x, y)
        cov_model_linlsq = fit_model_linlsq.cov_matrix.cov_matrix
        assert_allclose(cov_model_linlsq, cov_model)
        assert_allclose(np.sqrt(np.diag(cov_model_linlsq)), fit_model_linlsq.stds.stds)
        y = model_set(self.x, model_set_axis=False) + np.array([self.rand, self.rand])
        fit_1d_set_linlsq = linlsq_fitter(model_set, self.x, y)
        cov_1d_set_linlsq = [j.cov_matrix for j in fit_1d_set_linlsq.cov_matrix]
        assert_allclose(cov_1d_set_linlsq[0], cov_model)
        assert_allclose(np.sqrt(np.diag(cov_1d_set_linlsq[0])), fit_1d_set_linlsq.stds[0].stds)

    @pytest.mark.parametrize('fitter', non_linear_fitters)
    def test_2d_models(self, fitter):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that fitting uncertainties are computed correctly for 2D models\n        and 2D model sets. Use covariance/stds given by LevMarLSQFitter as\n        a benchmark since they are returned by the numpy fitter.\n        '
        fitter = fitter(calc_uncertainties=True)
        linlsq_fitter = LinearLSQFitter(calc_uncertainties=True)
        single_model = models.Polynomial2D(2, c0_0=2)
        model_set = models.Polynomial2D(degree=2, n_models=2, c0_0=[2, 3], model_set_axis=False)
        z_grid = single_model(self.x_grid, self.y_grid) + self.rand_grid
        with pytest.warns(AstropyUserWarning, match='Model is linear in parameters'):
            fit_model = fitter(single_model, self.x_grid, self.y_grid, z_grid)
        cov_model = fit_model.cov_matrix.cov_matrix
        fit_model_linlsq = linlsq_fitter(single_model, self.x_grid, self.y_grid, z_grid)
        cov_model_linlsq = fit_model_linlsq.cov_matrix.cov_matrix
        assert_allclose(cov_model, cov_model_linlsq)
        assert_allclose(np.sqrt(np.diag(cov_model_linlsq)), fit_model_linlsq.stds.stds)
        z_grid = model_set(self.x_grid, self.y_grid) + np.array((self.rand_grid, self.rand_grid))
        fit_2d_set_linlsq = linlsq_fitter(model_set, self.x_grid, self.y_grid, z_grid)
        cov_2d_set_linlsq = [j.cov_matrix for j in fit_2d_set_linlsq.cov_matrix]
        assert_allclose(cov_2d_set_linlsq[0], cov_model)
        assert_allclose(np.sqrt(np.diag(cov_2d_set_linlsq[0])), fit_2d_set_linlsq.stds[0].stds)

    def test_covariance_std_printing_indexing(self, capsys):
        if False:
            i = 10
            return i + 15
        '\n        Test printing methods and indexing.\n        '
        fitter = LinearLSQFitter(calc_uncertainties=True)
        mod = models.Linear1D()
        fit_mod = fitter(mod, self.x, mod(self.x) + self.rand)
        print(fit_mod.cov_matrix)
        captured = capsys.readouterr()
        assert 'slope    | 0.001' in captured.out
        assert 'intercept| -0.005,  0.03' in captured.out
        print(fit_mod.stds)
        captured = capsys.readouterr()
        assert 'slope    | 0.032' in captured.out
        assert 'intercept| 0.173' in captured.out
        print(fit_mod.cov_matrix.pprint(round_val=5, max_lines=1))
        captured = capsys.readouterr()
        assert 'slope    | 0.00105' in captured.out
        assert 'intercept' not in captured.out
        print(fit_mod.stds.pprint(max_lines=1, round_val=5))
        captured = capsys.readouterr()
        assert 'slope    | 0.03241' in captured.out
        assert 'intercept' not in captured.out
        assert fit_mod.cov_matrix[0, 0] == fit_mod.cov_matrix['slope', 'slope']
        assert fit_mod.stds[1] == fit_mod.stds['intercept']

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', non_linear_fitters)
@pytest.mark.parametrize('weights', [np.ones(8), None])
def test_non_finite_error(fitter, weights):
    if False:
        i = 10
        return i + 15
    'Regression test error introduced to solve issues #3575 and #12809'
    x = np.array([1, 2, 3, 4, 5, np.nan, 7, np.inf])
    y = np.array([9, np.nan, 11, np.nan, 13, np.nan, 15, 16])
    m_init = models.Gaussian1D()
    fit = fitter()
    with pytest.raises(NonFiniteValueError, match='Objective function has encountered.*'):
        fit(m_init, x, y, weights=weights)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', non_linear_fitters)
@pytest.mark.parametrize('weights', [np.ones(8), None])
def test_non_finite_filter_1D(fitter, weights):
    if False:
        print('Hello World!')
    'Regression test filter introduced to remove non-finte values from data'
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([9, np.nan, 11, np.nan, 13, np.nan, 15, np.inf])
    m_init = models.Gaussian1D()
    fit = fitter()
    if weights is not None:
        weights[[1, 4]] = np.nan
    with pytest.warns(AstropyUserWarning, match='Non-Finite input data has been removed by the fitter'):
        fit(m_init, x, y, filter_non_finite=True, weights=weights)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.parametrize('fitter', non_linear_fitters)
@pytest.mark.parametrize('weights', [np.ones((10, 10)), None])
def test_non_finite_filter_2D(fitter, weights):
    if False:
        i = 10
        return i + 15
    'Regression test filter introduced to remove non-finte values from data'
    (x, y) = np.mgrid[0:10, 0:10]
    m_true = models.Gaussian2D(amplitude=1, x_mean=5, y_mean=5, x_stddev=2, y_stddev=2)
    with NumpyRNGContext(_RANDOM_SEED):
        z = m_true(x, y) + np.random.rand(*x.shape)
    z[0, 0] = np.nan
    z[3, 3] = np.inf
    z[7, 5] = -np.inf
    if weights is not None:
        weights[1, 1] = np.nan
        weights[4, 3] = np.inf
    m_init = models.Gaussian2D()
    fit = fitter()
    with pytest.warns(AstropyUserWarning, match='Non-Finite input data has been removed by the fitter'):
        fit(m_init, x, y, z, filter_non_finite=True, weights=weights)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
@pytest.mark.filterwarnings('ignore:Model is linear in parameters*')
@pytest.mark.parametrize('fitter', non_linear_fitters)
def test_non_linear_fit_zero_degree_polynomial_with_weights(fitter):
    if False:
        while True:
            i = 10
    '\n    Regression test for issue #13617\n\n        Issue:\n            Weighted non-linear weighted fits of O-degree polynomials cause an error\n            to be raised by scipy.\n\n        Fix:\n            There should be no error raised in this circumstance\n    '
    model = models.Polynomial1D(0, c0=0)
    fitter = fitter()
    x = np.arange(10, dtype=float)
    y = np.ones((10,))
    weights = np.ones((10,))
    fit = fitter(model, x, y)
    assert_almost_equal(fit.c0, 1.0)
    fit = fitter(model, x, y, weights=weights)
    assert_almost_equal(fit.c0, 1.0)