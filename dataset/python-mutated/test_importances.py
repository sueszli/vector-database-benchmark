"""
Test the feature importance visualizers
"""
import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from yellowbrick.exceptions import NotFitted
from yellowbrick.model_selection.importances import *
from yellowbrick.datasets import load_occupancy, load_concrete
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from unittest import mock
from tests.base import VisualTestCase
try:
    import pandas as pd
except ImportError:
    pd = None

class TestFeatureImportancesVisualizer(VisualTestCase):
    """
    Test FeatureImportances visualizer
    """

    def test_integration_feature_importances(self):
        if False:
            print('Hello World!')
        '\n        Integration test of visualizer with feature importances param\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot()
        clf = GradientBoostingClassifier(random_state=42)
        viz = FeatureImportances(clf, ax=ax)
        viz.fit(X, y)
        viz.finalize()
        self.assert_images_similar(viz, tol=13.0)

    def test_integration_coef(self):
        if False:
            print('Hello World!')
        '\n        Integration test of visualizer with coef param\n        '
        dataset = load_concrete(return_dataset=True)
        (X, y) = dataset.to_numpy()
        features = dataset.meta['features']
        fig = plt.figure()
        ax = fig.add_subplot()
        reg = Lasso(random_state=42)
        features = list(map(lambda s: s.title(), features))
        viz = FeatureImportances(reg, ax=ax, labels=features, relative=False)
        viz.fit(X, y)
        viz.finalize()
        self.assert_images_similar(viz, tol=16.2)

    def test_integration_quick_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Integration test of quick method\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot()
        clf = RandomForestClassifier(random_state=42)
        g = feature_importances(clf, X, y, ax=ax, show=False)
        self.assert_images_similar(g, tol=15.0)

    def test_fit_no_importances_model(self):
        if False:
            while True:
                i = 10
        '\n        Fitting a model without feature importances raises an exception\n        '
        X = np.random.rand(100, 42)
        y = np.random.rand(100)
        visualizer = FeatureImportances(MockEstimator())
        expected_error = 'could not find feature importances param on MockEstimator'
        with pytest.raises(YellowbrickTypeError, match=expected_error):
            visualizer.fit(X, y)

    def test_fit_sorted_params(self):
        if False:
            return 10
        '\n        On fit, sorted features_ and feature_importances_ params are created\n        '
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])
        names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
        model = MockEstimator()
        model.make_importance_param(value=coefs)
        visualizer = FeatureImportances(model, labels=names)
        visualizer.fit(np.random.rand(100, len(names)), np.random.rand(100))
        assert hasattr(visualizer, 'features_')
        assert hasattr(visualizer, 'feature_importances_')
        sort_idx = np.argsort(coefs)
        npt.assert_array_equal(names[sort_idx], visualizer.features_)
        npt.assert_array_equal(coefs[sort_idx], visualizer.feature_importances_)

    def test_fit_relative(self):
        if False:
            return 10
        '\n        Test fit computes relative importances\n        '
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])
        model = MockEstimator()
        model.make_importance_param(value=coefs)
        visualizer = FeatureImportances(model, relative=True)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))
        expected = 100.0 * coefs / coefs.max()
        expected.sort()
        npt.assert_array_equal(visualizer.feature_importances_, expected)

    def test_fit_not_relative(self):
        if False:
            i = 10
            return i + 15
        '\n        Test fit stores unmodified importances\n        '
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])
        model = MockEstimator()
        model.make_importance_param(value=coefs)
        visualizer = FeatureImportances(model, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))
        coefs.sort()
        npt.assert_array_equal(visualizer.feature_importances_, coefs)

    def test_fit_absolute(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test fit with absolute values\n        '
        coefs = np.array([0.4, 0.2, -0.08, 0.07, 0.16, 0.23, -0.38, 0.1, -0.05])
        model = MockEstimator()
        model.make_importance_param(value=coefs)
        visualizer = FeatureImportances(model, absolute=True, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))
        expected = np.array([0.05, 0.07, 0.08, 0.1, 0.16, 0.2, 0.23, 0.38, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)
        visualizer = FeatureImportances(model, absolute=False, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))
        expected = np.array([-0.38, -0.08, -0.05, 0.07, 0.1, 0.16, 0.2, 0.23, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)

    def test_multi_coefs(self):
        if False:
            return 10
        '\n        Test fit with multidimensional coefficients and stack warning\n        '
        coefs = np.array([[0.4, 0.2, -0.08, 0.07, 0.16, 0.23, -0.38, 0.1, -0.05], [0.41, 0.12, -0.1, 0.1, 0.14, 0.21, 0.01, 0.31, -0.15], [0.31, 0.2, -0.01, 0.1, 0.22, 0.23, 0.01, 0.12, -0.15]])
        model = MockEstimator()
        model.make_importance_param(value=coefs)
        visualizer = FeatureImportances(model, stack=False)
        with pytest.warns(YellowbrickWarning):
            visualizer.fit(np.random.rand(100, len(np.mean(coefs, axis=0))), np.random.rand(100))
        npt.assert_equal(visualizer.feature_importances_.ndim, 1)

    def test_multi_coefs_stacked(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test stack plot with multidimensional coefficients\n        '
        (X, y) = load_iris(return_X_y=True)
        viz = FeatureImportances(LogisticRegression(solver='liblinear', random_state=222), stack=True)
        viz.fit(X, y)
        viz.finalize()
        npt.assert_equal(viz.feature_importances_.shape, (3, 4))
        self.assert_images_similar(viz, tol=17.5)

    def test_stack_param_incorrectly_used_throws_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Test incorrectly using stack param on a dataset with two classes which\n        does not return a coef_ array in the shape of (n_classes, n_features)\n        '
        (X, y) = load_occupancy()
        viz = FeatureImportances(LogisticRegression(solver='liblinear', random_state=222), stack=True)
        expected_error = 'The model used does not return coef_ array'
        with pytest.raises(YellowbrickValueError, match=expected_error):
            viz.fit(X, y)

    @pytest.mark.skipif(pd is None, reason='pandas is required for this test')
    def test_fit_dataframe(self):
        if False:
            while True:
                i = 10
        '\n        Ensure feature names are extracted from DataFrame columns\n        '
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        df = pd.DataFrame(np.random.rand(100, 6), columns=labels)
        s = pd.Series(np.random.rand(100), name='target')
        assert df.shape == (100, 6)
        model = MockEstimator()
        model.make_importance_param(value=np.linspace(0, 1, 6))
        visualizer = FeatureImportances(model)
        visualizer.fit(df, s)
        assert hasattr(visualizer, 'features_')
        npt.assert_array_equal(visualizer.features_, np.array(df.columns))

    def test_fit_makes_labels(self):
        if False:
            return 10
        '\n        Assert that the fit process makes label indices\n        '
        model = MockEstimator()
        model.make_importance_param(value=np.linspace(0, 1, 10))
        visualizer = FeatureImportances(model)
        visualizer.fit(np.random.rand(100, 10), np.random.rand(100))
        assert hasattr(visualizer, 'features_')
        npt.assert_array_equal(np.arange(10), visualizer.features_)

    def test_fit_calls_draw(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert that fit calls draw\n        '
        model = MockEstimator()
        model.make_importance_param('coef_')
        visualizer = FeatureImportances(model)
        with mock.patch.object(visualizer, 'draw') as mdraw:
            visualizer.fit(np.random.rand(100, 42), np.random.rand(100))
            mdraw.assert_called_once()

    def test_draw_raises_unfitted(self):
        if False:
            return 10
        '\n        Assert draw raises exception when not fitted\n        '
        visualizer = FeatureImportances(Lasso())
        with pytest.raises(NotFitted):
            visualizer.draw()

    def test_find_importances_param(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the expected parameters can be found\n        '
        params = ('feature_importances_', 'coef_')
        for param in params:
            model = MockEstimator()
            model.make_importance_param(param, 'foo')
            visualizer = FeatureImportances(model)
            assert hasattr(model, param), "expected '{}' missing".format(param)
            for oparam in params:
                if oparam == param:
                    continue
                assert not hasattr(model, oparam), "unexpected '{}'".format(oparam)
            importances = visualizer._find_importances_param()
            assert importances == 'foo'

    def test_find_importances_param_priority(self):
        if False:
            while True:
                i = 10
        '\n        With both feature_importances_ and coef_, one has priority\n        '
        model = MockEstimator()
        model.make_importance_param('feature_importances_', 'foo')
        model.make_importance_param('coef_', 'bar')
        visualizer = FeatureImportances(model)
        assert hasattr(model, 'feature_importances_')
        assert hasattr(model, 'coef_')
        importances = visualizer._find_importances_param()
        assert importances == 'foo'

    def test_find_importances_param_not_found(self):
        if False:
            i = 10
            return i + 15
        '\n        Raises an exception when importances param not found\n        '
        model = MockEstimator()
        visualizer = FeatureImportances(model)
        assert not hasattr(model, 'feature_importances_')
        assert not hasattr(model, 'coef_')
        with pytest.raises(YellowbrickTypeError):
            visualizer._find_importances_param()

    def test_find_classes_param_not_found(self):
        if False:
            return 10
        '\n        Raises an exception when classes param not found\n        '
        model = MockClassifier()
        visualizer = FeatureImportances(model)
        assert not hasattr(model, 'classes_')
        e = 'could not find classes_ param on {}'.format(visualizer.estimator.__class__.__name__)
        with pytest.raises(YellowbrickTypeError, match=e):
            visualizer._find_classes_param()

    def test_xlabel(self):
        if False:
            while True:
                i = 10
        '\n        Check the various xlabels are sensical\n        '
        model = MockEstimator()
        model.make_importance_param('feature_importances_')
        visualizer = FeatureImportances(model, xlabel='foo', relative=True)
        assert visualizer._get_xlabel() == 'foo', 'could not set user xlabel'
        visualizer.set_params(xlabel=None)
        assert 'relative' in visualizer._get_xlabel()
        visualizer.set_params(relative=False)
        assert 'relative' not in visualizer._get_xlabel()
        model = MockEstimator()
        model.make_importance_param('coef_')
        visualizer = FeatureImportances(model, xlabel='baz', relative=True)
        assert visualizer._get_xlabel() == 'baz', 'could not set user xlabel'
        visualizer.set_params(xlabel=None)
        assert 'coefficient' in visualizer._get_xlabel()
        assert 'relative' in visualizer._get_xlabel()
        visualizer.set_params(relative=False)
        assert 'coefficient' in visualizer._get_xlabel()
        assert 'relative' not in visualizer._get_xlabel()

    def test_is_fitted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test identification if is fitted\n        '
        visualizer = FeatureImportances(Lasso())
        assert not visualizer._is_fitted()
        visualizer.features_ = 'foo'
        assert not visualizer._is_fitted()
        visualizer.feature_importances_ = 'bar'
        assert visualizer._is_fitted()
        del visualizer.features_
        assert not visualizer._is_fitted()

    def test_with_fitted(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer properly handles an already-fitted model\n        '
        (X, y) = load_concrete(return_dataset=True).to_numpy()
        model = Lasso().fit(X, y)
        with mock.patch.object(model, 'fit') as mockfit:
            oz = FeatureImportances(model)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = FeatureImportances(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = FeatureImportances(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    def test_topn_stacked(self):
        if False:
            return 10
        "\n        Test stack plot with only the three most important features by sum of\n        each feature's importance across all classes\n        "
        (X, y) = load_iris(return_X_y=True)
        viz = FeatureImportances(LogisticRegression(solver='liblinear', random_state=222), stack=True, topn=3)
        viz.fit(X, y)
        viz.finalize()
        npt.assert_equal(viz.feature_importances_.shape, (3, 3))
        self.assert_images_similar(viz, tol=17.5)

    def test_topn_negative_stacked(self):
        if False:
            i = 10
            return i + 15
        "\n        Test stack plot with only the three least important features by sum of\n        each feature's importance across all classes\n        "
        (X, y) = load_iris(return_X_y=True)
        viz = FeatureImportances(LogisticRegression(solver='liblinear', random_state=222), stack=True, topn=-3)
        viz.fit(X, y)
        viz.finalize()
        npt.assert_equal(viz.feature_importances_.shape, (3, 3))
        self.assert_images_similar(viz, tol=17.5)

    def test_topn(self):
        if False:
            i = 10
            return i + 15
        '\n        Test plot with only top three important features by absolute value\n        '
        (X, y) = load_iris(return_X_y=True)
        viz = FeatureImportances(GradientBoostingClassifier(random_state=42), topn=3)
        viz.fit(X, y)
        viz.finalize()
        self.assert_images_similar(viz, tol=17.5)

    def test_topn_negative(self):
        if False:
            return 10
        '\n        Test plot with only the three least important features by absolute value\n        '
        (X, y) = load_iris(return_X_y=True)
        viz = FeatureImportances(GradientBoostingClassifier(random_state=42), topn=-3)
        viz.fit(X, y)
        viz.finalize()
        self.assert_images_similar(viz, tol=17.5)

    def test_within_pipeline(self):
        if False:
            return 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        dataset = load_concrete(return_dataset=True)
        (X, y) = dataset.to_data()
        features = dataset.meta['features']
        features = list(map(lambda s: s.title(), features))
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('fi', FeatureImportances(Lasso(random_state=42), labels=features, relative=False))])
        model.fit(X, y)
        model['fi'].finalize()
        self.assert_images_similar(model['fi'], tol=17.5)

    def test_within_pipeline_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        dataset = load_concrete(return_dataset=True)
        (X, y) = dataset.to_data()
        features = dataset.meta['features']
        features = list(map(lambda s: s.title(), features))
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('fi', feature_importances(Lasso(random_state=42), X, y, labels=features, relative=False, show=False))])
        self.assert_images_similar(model['fi'], tol=17.5)

class MockEstimator(BaseEstimator):
    """
    Creates params when fit is called on demand.
    """

    def make_importance_param(self, name='feature_importances_', value=None):
        if False:
            i = 10
            return i + 15
        if value is None:
            value = np.random.rand(42)
        setattr(self, name, value)

    def fit(self, X, y=None, **kwargs):
        if False:
            return 10
        return self

class MockClassifier(BaseEstimator, ClassifierMixin):
    """
    Creates empty classifier.
    """
    pass