"""
Tests for the alpha selection visualizations.
"""
import sys
import pytest
import numpy as np
from tests.base import VisualTestCase
from numpy.testing import assert_array_equal
from yellowbrick.datasets import load_energy, load_concrete
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.regressor.alphas import AlphaSelection, alphas
from yellowbrick.regressor.alphas import ManualAlphaSelection, manual_alphas
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import ElasticNet, ElasticNetCV

class TestAlphaSelection(VisualTestCase):
    """
    Test the AlphaSelection visualizer
    """

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_similar_image(self):
        if False:
            print('Hello World!')
        '\n        Integration test with image similarity comparison\n        '
        visualizer = AlphaSelection(LassoCV(random_state=0))
        (X, y) = make_regression(random_state=0)
        visualizer.fit(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    @pytest.mark.parametrize('model', [SVR, Ridge, Lasso, LassoLars, ElasticNet])
    def test_regressor_nocv(self, model):
        if False:
            while True:
                i = 10
        '\n        Ensure only "CV" regressors are allowed\n        '
        with pytest.raises(YellowbrickTypeError):
            AlphaSelection(model())

    @pytest.mark.parametrize('model', [RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV])
    def test_regressor_cv(self, model):
        if False:
            i = 10
            return i + 15
        '\n        Ensure "CV" regressors are allowed\n        '
        try:
            AlphaSelection(model())
        except YellowbrickTypeError:
            pytest.fail('could not instantiate RegressorCV on alpha selection')

    @pytest.mark.parametrize('model', [SVC, KMeans, PCA])
    def test_only_regressors(self, model):
        if False:
            print('Hello World!')
        '\n        Assert AlphaSelection only works with regressors\n        '
        with pytest.raises(YellowbrickTypeError):
            AlphaSelection(model())

    def test_store_cv_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that store_cv_values is true on RidgeCV\n        '
        model = AlphaSelection(RidgeCV())
        assert model.estimator.store_cv_values
        model = AlphaSelection(RidgeCV(store_cv_values=True))
        assert model.estimator.store_cv_values
        model = AlphaSelection(RidgeCV(store_cv_values=False))
        assert model.estimator.store_cv_values

    @pytest.mark.parametrize('model', [RidgeCV, LassoCV, ElasticNetCV])
    def test_get_alphas_param(self, model):
        if False:
            return 10
        '\n        Assert that we can get the alphas from original CV models\n        '
        alphas = np.logspace(-10, -2, 100)
        try:
            model = AlphaSelection(model(alphas=alphas))
            malphas = model._find_alphas_param()
            assert_array_equal(alphas, malphas)
        except YellowbrickValueError:
            pytest.fail('could not find alphas on {}'.format(model.name))

    def test_get_alphas_param_lassolars(self):
        if False:
            while True:
                i = 10
        '\n        Assert that we can get alphas from lasso lars.\n        '
        (X, y) = make_regression()
        model = AlphaSelection(LassoLarsCV())
        model.fit(X, y)
        try:
            malphas = model._find_alphas_param()
            assert len(malphas) > 0
        except YellowbrickValueError:
            pytest.fail('could not find alphas on {}'.format(model.name))

    @pytest.mark.parametrize('model', [RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV])
    def test_get_errors_param(self, model):
        if False:
            i = 10
            return i + 15
        '\n        Test known models we can get the cv errors for alpha selection\n        '
        try:
            model = AlphaSelection(model())
            (X, y) = make_regression()
            model.fit(X, y)
            errors = model._find_errors_param()
            assert len(errors) > 0
        except YellowbrickValueError:
            pytest.fail('could not find errors on {}'.format(model.name))

    def test_score(self):
        if False:
            while True:
                i = 10
        '\n        Assert the score method returns an R2 value\n        '
        visualizer = AlphaSelection(RidgeCV())
        (X, y) = make_regression(random_state=352)
        visualizer.fit(X, y)
        assert visualizer.score(X, y) == pytest.approx(0.9999780266590336)

    def test_quick_method(self):
        if False:
            while True:
                i = 10
        '\n        Test the quick method producing a valid visualization\n        '
        (X, y) = load_energy(return_dataset=True).to_numpy()
        visualizer = alphas(LassoCV(random_state=0), X, y, is_fitted=False, show=False)
        assert isinstance(visualizer, AlphaSelection)
        self.assert_images_similar(visualizer, tol=0.1)

    def test_within_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_concrete()
        alphas = np.logspace(-10, 1, 400)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('alpha', AlphaSelection(LassoCV(random_state=42, alphas=alphas)))])
        model.fit(X, y)
        model['alpha'].finalize()
        self.assert_images_similar(model['alpha'], tol=2.0)

class TestManualAlphaSelection(VisualTestCase):
    """
    Test the ManualAlphaSelection visualizer
    """

    def test_similar_image_manual(self):
        if False:
            print('Hello World!')
        '\n        Integration test with image similarity comparison\n        '
        visualizer = ManualAlphaSelection(Lasso(random_state=0), cv=5)
        (X, y) = make_regression(random_state=0)
        visualizer.fit(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1)

    @pytest.mark.parametrize('model', [RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV])
    def test_manual_with_cv(self, model):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure only non-CV regressors are allowed\n        '
        with pytest.raises(YellowbrickTypeError):
            ManualAlphaSelection(model())

    @pytest.mark.parametrize('model', [SVR, Ridge, Lasso, LassoLars, ElasticNet])
    def test_manual_no_cv(self, model):
        if False:
            while True:
                i = 10
        '\n        Ensure non-CV regressors are allowed\n        '
        try:
            ManualAlphaSelection(model())
        except YellowbrickTypeError:
            pytest.fail('could not instantiate Regressor on alpha selection')

    def test_quick_method_manual(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the manual alphas quick method producing a valid visualization\n        '
        (X, y) = load_energy(return_dataset=True).to_numpy()
        visualizer = manual_alphas(ElasticNet(random_state=0), X, y, cv=3, is_fitted=False, show=False)
        assert isinstance(visualizer, ManualAlphaSelection)
        self.assert_images_similar(visualizer, tol=0.5)

    def test_manual_within_pipeline(self):
        if False:
            while True:
                i = 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_concrete()
        alpha_values = np.logspace(1, 4, 50)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('alpha', ManualAlphaSelection(Ridge(random_state=42), alphas=alpha_values, cv=12, scoring='neg_mean_squared_error'))])
        model.fit(X, y)
        model['alpha'].finalize()
        self.assert_images_similar(model['alpha'], tol=2.0)