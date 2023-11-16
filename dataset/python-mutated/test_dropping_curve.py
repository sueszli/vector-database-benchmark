"""
Tests for the DroppingCurve visualizer
"""
import sys
import pytest
import numpy as np
from unittest.mock import patch
from tests.base import VisualTestCase
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from yellowbrick.datasets import load_mushroom
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.model_selection import DroppingCurve, dropping_curve
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.mark.usefixtures('classification', 'regression')
class TestDroppingCurve(VisualTestCase):
    """
    Test the DroppingCurve visualizer
    """

    @patch.object(DroppingCurve, 'draw')
    def test_fit(self, mock_draw):
        if False:
            i = 10
            return i + 15
        '\n        Assert that fit returns self and creates expected properties\n        '
        (X, y) = self.classification
        params = ('train_scores_', 'train_scores_mean_', 'train_scores_std_', 'valid_scores_', 'valid_scores_mean_', 'valid_scores_std_')
        oz = DroppingCurve(MultinomialNB(), feature_sizes=np.linspace(0.05, 1, 20))
        for param in params:
            assert not hasattr(oz, param)
        assert oz.fit(X, y) is oz
        mock_draw.assert_called_once()
        for param in params:
            assert hasattr(oz, param)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_classifier(self):
        if False:
            print('Hello World!')
        '\n        Test image closeness on a classification dataset with MultinomialNB\n        '
        (X, y) = self.classification
        cv = ShuffleSplit(3, random_state=288)
        oz = DroppingCurve(KNeighborsClassifier(), cv=cv, feature_sizes=np.linspace(0.05, 1, 20), random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_regression(self):
        if False:
            print('Hello World!')
        '\n        Test image closeness on a regression dataset with a DecisionTree\n        '
        (X, y) = self.regression
        cv = ShuffleSplit(3, random_state=938)
        param_range = np.arange(3, 10)
        oz = DroppingCurve(DecisionTreeRegressor(random_state=23), param_name='max_depth', param_range=param_range, cv=cv, scoring='r2', random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=12.0)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_quick_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test validation curve quick method with image closeness on SVC\n        '
        (X, y) = self.classification
        pr = np.logspace(-6, -1, 3)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=321)
        viz = dropping_curve(SVC(), X, y, logx=True, param_name='gamma', param_range=pr, cv=cv, show=False, random_state=42)
        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test on mushroom dataset with pandas DataFrame and Series and NB\n        '
        data = load_mushroom(return_dataset=True)
        (X, y) = data.to_pandas()
        X = pd.get_dummies(X)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)
        oz = DroppingCurve(MultinomialNB(), cv=cv, random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_numpy_integration(self):
        if False:
            print('Hello World!')
        '\n        Test on mushroom dataset with NumPy arrays\n        '
        data = load_mushroom(return_dataset=True)
        (X, y) = data.to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)
        pr = np.linspace(0.1, 3.0, 6)
        oz = DroppingCurve(BernoulliNB(), cv=cv, param_range=pr, param_name='alpha', random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_bad_train_sizes(self):
        if False:
            print('Hello World!')
        '\n        Test learning curve with bad input for feature size.\n        '
        with pytest.raises(YellowbrickValueError):
            DroppingCurve(SVC(), param_name='gamma', feature_sizes=100)

    def test_within_pipeline(self):
        if False:
            return 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_mushroom(return_dataset=True).to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        model = Pipeline([('minmax', MinMaxScaler()), ('matrix', DroppingCurve(BernoulliNB(), random_state=42))])
        model.fit(X, y)
        model['matrix'].finalize()
        self.assert_images_similar(model['matrix'], tol=12)

    def test_within_pipeline_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        (X, y) = load_mushroom(return_dataset=True).to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        model = Pipeline([('minmax', MinMaxScaler()), ('matrix', dropping_curve(BernoulliNB(), X, y, show=False, random_state=42))])
        self.assert_images_similar(model['matrix'], tol=12)

    def test_pipeline_as_model_input(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_mushroom(return_dataset=True).to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        model = Pipeline([('minmax', MinMaxScaler()), ('nb', BernoulliNB())])
        oz = DroppingCurve(model, random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=12)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        within a quickmethod\n        '
        (X, y) = load_mushroom(return_dataset=True).to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        model = Pipeline([('minmax', MinMaxScaler()), ('nb', BernoulliNB())])
        oz = dropping_curve(model, X, y, show=False, random_state=42)
        self.assert_images_similar(oz, tol=12)

    def test_get_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure dropping curve get params works correctly\n        '
        oz = DroppingCurve(MultinomialNB())
        params = oz.get_params()
        assert len(params) > 0