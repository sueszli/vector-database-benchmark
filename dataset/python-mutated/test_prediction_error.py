"""
Ensure that the regressor prediction error visualization works.
"""
import pytest
import matplotlib.pyplot as plt
import numpy as np
from unittest import mock
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from yellowbrick.datasets import load_energy, load_concrete
from yellowbrick.regressor.prediction_error import PredictionError, prediction_error
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.fixture(scope='class')
def data(request):
    if False:
        while True:
            i = 10
    '\n    Creates a fixture of train and test splits for the sklearn digits dataset\n    For ease of use returns a Dataset named tuple composed of two Split tuples.\n    '
    (X, y) = make_regression(n_samples=500, n_features=22, n_informative=8, random_state=42, noise=0.2, bias=0.2)
    (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=11)
    request.cls.data = Dataset(Split(X_train, X_test), Split(y_train, y_test))

@pytest.mark.usefixtures('data')
class TestPredictionError(VisualTestCase):
    """
    Test the PredictionError visualizer
    """

    @pytest.mark.filterwarnings('ignore:Stochastic Optimizer')
    @pytest.mark.filterwarnings('ignore:internal gelsd driver lwork query error')
    def test_prediction_error(self):
        if False:
            print('Hello World!')
        '\n        Test image similarity of prediction error on random data\n        '
        (_, ax) = plt.subplots()
        model = MLPRegressor(random_state=229)
        visualizer = PredictionError(model, ax=ax)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    @pytest.mark.skipif(pd is None, reason='pandas is required')
    def test_prediction_error_pandas(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Pandas real world dataset with image similarity on Ridge\n        '
        (_, ax) = plt.subplots()
        data = load_energy(return_dataset=True)
        (X, y) = data.to_pandas()
        splits = tts(X, y, test_size=0.2, random_state=8873)
        (X_train, X_test, y_train, y_test) = splits
        visualizer = PredictionError(Ridge(random_state=22), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    def test_prediction_error_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test NumPy real world dataset with image similarity on Ridge\n        '
        (_, ax) = plt.subplots()
        data = load_energy(return_dataset=True)
        (X, y) = data.to_numpy()
        splits = tts(X, y, test_size=0.2, random_state=8873)
        (X_train, X_test, y_train, y_test) = splits
        visualizer = PredictionError(Ridge(random_state=22), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    def test_score(self):
        if False:
            return 10
        '\n        Assert returns R2 score\n        '
        visualizer = PredictionError(LinearRegression())
        visualizer.fit(self.data.X.train, self.data.y.train)
        score = visualizer.score(self.data.X.test, self.data.y.test)
        assert score == pytest.approx(0.9999983124154965)
        assert visualizer.score_ == score

    def test_peplot_shared_limits(self):
        if False:
            print('Hello World!')
        '\n        Test shared limits on the peplot\n        '
        visualizer = PredictionError(LinearRegression(), shared_limits=False)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()
        xlim = tuple(map(int, visualizer.ax.get_xlim()))
        ylim = tuple(map(int, visualizer.ax.get_ylim()))
        assert xlim == ylim

    @pytest.mark.filterwarnings('ignore:internal gelsd driver lwork query error')
    def test_peplot_no_shared_limits(self):
        if False:
            print('Hello World!')
        '\n        Test image similarity with no shared limits on the peplot\n        '
        visualizer = PredictionError(Ridge(random_state=43), shared_limits=False)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()
        xlim = tuple(map(int, visualizer.ax.get_xlim()))
        ylim = tuple(map(int, visualizer.ax.get_ylim()))
        assert not xlim == ylim
        self.assert_images_similar(visualizer, tol=1.0, remove_legend=True)

    def test_peplot_no_lines(self):
        if False:
            return 10
        '\n        Test image similarity with no lines drawn on the plot\n        '
        visualizer = PredictionError(Lasso(random_state=23, alpha=10), bestfit=False, identity=False)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=1.0, remove_legend=True)

    def test_alpha_param(self):
        if False:
            print('Hello World!')
        '\n        Test that the user can supply an alpha param on instantiation\n        '
        model = Lasso(random_state=23, alpha=10)
        visualizer = PredictionError(model, bestfit=False, identity=False, alpha=0.7)
        assert visualizer.alpha == 0.7
        visualizer.ax = mock.MagicMock(autospec=True)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        (_, scatter_kwargs) = visualizer.ax.scatter.call_args
        assert 'alpha' in scatter_kwargs
        assert scatter_kwargs['alpha'] == 0.7

    def test_is_fitted_param(self):
        if False:
            return 10
        "\n        Test that the user can supply an is_fitted param and it's state is maintained\n        "
        model = Lasso(random_state=23, alpha=10)
        visualizer = PredictionError(model, bestfit=False, identity=False, is_fitted=False)
        assert visualizer.is_fitted == False

    @pytest.mark.xfail(reason='third test fails with AssertionError: Expected fit\n        to be called once. Called 0 times.')
    def test_peplot_with_fitted(self):
        if False:
            print('Hello World!')
        '\n        Test that PredictionError properly handles an already-fitted model\n        '
        (X, y) = load_energy(return_dataset=True).to_numpy()
        model = Ridge().fit(X, y)
        with mock.patch.object(model, 'fit') as mockfit:
            oz = PredictionError(model)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = PredictionError(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = PredictionError(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_prediction_error_quick_method(self):
        if False:
            return 10
        '\n        Image similarity test using the residuals plot quick method\n        '
        (_, ax) = plt.subplots()
        model = Lasso(random_state=19)
        oz = prediction_error(model, self.data.X.train, self.data.y.train, ax=ax, show=False)
        assert isinstance(oz, PredictionError)
        self.assert_images_similar(oz)

    def test_within_pipeline(self):
        if False:
            return 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_concrete()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=42)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('pe', PredictionError(Lasso()))])
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        model['pe'].finalize()
        self.assert_images_similar(model['pe'], tol=2.0)

    def test_within_pipeline_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_concrete()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=42)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('pe', PredictionError(Lasso()))])
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        model['pe'].finalize()
        self.assert_images_similar(model['pe'], tol=2.0)

    def test_pipeline_as_model_input(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_concrete()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=42)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('lasso', Lasso())])
        oz = PredictionError(model)
        oz.fit(X_train, y_train)
        oz.score(X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=2.0)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_concrete()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=42)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('lasso', Lasso())])
        oz = prediction_error(model, X_train, y_train, X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=2.0)