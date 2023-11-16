"""
Tests for the ValidationCurve visualizer
"""
import sys
import pytest
import numpy as np
from unittest.mock import patch
from tests.base import VisualTestCase
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from yellowbrick.datasets import load_mushroom
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.model_selection.validation_curve import *
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.mark.usefixtures('classification', 'regression', 'clusters')
class TestValidationCurve(VisualTestCase):
    """
    Test the ValidationCurve visualizer
    """

    @patch.object(ValidationCurve, 'draw')
    def test_fit(self, mock_draw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that fit returns self and creates expected properties\n        '
        (X, y) = self.classification
        params = ('train_scores_', 'train_scores_mean_', 'train_scores_std_', 'test_scores_', 'test_scores_mean_', 'test_scores_std_')
        oz = ValidationCurve(SVC(), param_name='gamma', param_range=np.logspace(-6, -1, 5))
        for param in params:
            assert not hasattr(oz, param)
        assert oz.fit(X, y) is oz
        mock_draw.assert_called_once()
        for param in params:
            assert hasattr(oz, param)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_classifier(self):
        if False:
            i = 10
            return i + 15
        '\n        Test image closeness on a classification dataset with kNN\n        '
        (X, y) = self.classification
        cv = ShuffleSplit(3, random_state=288)
        param_range = np.arange(3, 10)
        oz = ValidationCurve(KNeighborsClassifier(), param_name='n_neighbors', param_range=param_range, cv=cv, scoring='f1_weighted')
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_regression(self):
        if False:
            return 10
        '\n        Test image closeness on a regression dataset with a DecisionTree\n        '
        (X, y) = self.regression
        cv = ShuffleSplit(3, random_state=938)
        param_range = np.arange(3, 10)
        oz = ValidationCurve(DecisionTreeRegressor(random_state=23), param_name='max_depth', param_range=param_range, cv=cv, scoring='r2')
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=12.0)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_quick_method(self):
        if False:
            while True:
                i = 10
        '\n        Test validation curve quick method with image closeness on SVC\n        '
        (X, y) = self.classification
        pr = np.logspace(-6, -1, 3)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=321)
        viz = validation_curve(SVC(), X, y, logx=True, param_name='gamma', param_range=pr, cv=cv, show=False)
        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            return 10
        '\n        Test on mushroom dataset with pandas DataFrame and Series and NB\n        '
        data = load_mushroom(return_dataset=True)
        (X, y) = data.to_pandas()
        X = pd.get_dummies(X)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)
        pr = np.linspace(0.1, 3.0, 6)
        oz = ValidationCurve(BernoulliNB(), cv=cv, param_range=pr, param_name='alpha')
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
        oz = ValidationCurve(BernoulliNB(), cv=cv, param_range=pr, param_name='alpha')
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    @patch.object(ValidationCurve, 'draw')
    def test_reshape_scores(self, mock_draw):
        if False:
            print('Hello World!')
        '\n        Test supplying an alternate CV methodology and train_sizes\n        '
        (X, y) = self.classification
        pr = np.logspace(-6, -1, 3)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=14)
        oz = ValidationCurve(SVC(), param_name='gamma', param_range=pr, cv=cv)
        oz.fit(X, y)
        assert oz.train_scores_.shape == (3, 5)
        assert oz.test_scores_.shape == (3, 5)

    def test_bad_train_sizes(self):
        if False:
            while True:
                i = 10
        '\n        Test learning curve with bad input for training size.\n        '
        with pytest.raises(YellowbrickValueError):
            ValidationCurve(SVC(), param_name='gamma', param_range=100)