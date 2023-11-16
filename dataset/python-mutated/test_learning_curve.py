"""
Tests for the LearningCurve visualizer
"""
import sys
import pytest
import numpy as np
from unittest.mock import patch
from tests.base import VisualTestCase
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from yellowbrick.datasets import load_mushroom, load_game
from yellowbrick.model_selection.learning_curve import *
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.mark.usefixtures('classification', 'regression', 'clusters')
class TestLearningCurve(VisualTestCase):
    """
    Test the LearningCurve visualizer
    """

    @patch.object(LearningCurve, 'draw')
    def test_fit(self, mock_draw):
        if False:
            print('Hello World!')
        '\n        Assert that fit returns self and creates expected properties\n        '
        (X, y) = self.classification
        params = ('train_sizes_', 'train_scores_', 'train_scores_mean_', 'train_scores_std_', 'test_scores_', 'test_scores_mean_', 'test_scores_std_')
        oz = LearningCurve(GaussianNB(), random_state=12)
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
        '\n        Test image closeness on a classification dataset\n        '
        (X, y) = self.classification
        oz = LearningCurve(RandomForestClassifier(random_state=21), random_state=12).fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_regressor(self):
        if False:
            i = 10
            return i + 15
        '\n        Test image closeness on a regression dataset\n        '
        (X, y) = self.regression
        oz = LearningCurve(Ridge(), random_state=18)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_clusters(self):
        if False:
            i = 10
            return i + 15
        '\n        Test image closeness on a clustering dataset\n        '
        (X, y) = self.clusters
        oz = LearningCurve(MiniBatchKMeans(random_state=281), random_state=182).fit(X)
        oz.finalize()
        self.assert_images_similar(oz, tol=10)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_quick_method(self):
        if False:
            print('Hello World!')
        '\n        Test the learning curve quick method acts as expected\n        '
        (X, y) = self.classification
        train_sizes = np.linspace(0.1, 1.0, 8)
        viz = learning_curve(GaussianNB(), X, y, train_sizes=train_sizes, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=34), scoring='f1_macro', random_state=43, show=False)
        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            print('Hello World!')
        '\n        Test on a real dataset with pandas DataFrame and Series\n        '
        data = load_mushroom(return_dataset=True)
        (X, y) = data.to_pandas()
        X = pd.get_dummies(X)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=32)
        oz = LearningCurve(GaussianNB(), cv=cv, random_state=23)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_numpy_integration(self):
        if False:
            print('Hello World!')
        '\n        Test on a real dataset with NumPy arrays\n        '
        data = load_mushroom(return_dataset=True)
        (X, y) = data.to_numpy()
        X = OneHotEncoder().fit_transform(X).toarray()
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=32)
        oz = LearningCurve(GaussianNB(), cv=cv, random_state=23)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz)

    @patch.object(LearningCurve, 'draw')
    def test_reshape_scores(self, mock_draw):
        if False:
            print('Hello World!')
        '\n        Test supplying an alternate CV methodology and train_sizes\n        '
        (X, y) = self.classification
        cv = ShuffleSplit(n_splits=12, test_size=0.2, random_state=14)
        oz = LearningCurve(LinearSVC(), cv=cv, train_sizes=[0.5, 0.75, 1.0])
        oz.fit(X, y)
        assert oz.train_scores_.shape == (3, 12)
        assert oz.test_scores_.shape == (3, 12)

    def test_bad_train_sizes(self):
        if False:
            while True:
                i = 10
        '\n        Test learning curve with bad input for training size.\n        '
        with pytest.raises(YellowbrickValueError):
            LearningCurve(LinearSVC(), train_sizes=10000)

    def test_within_pipeline(self):
        if False:
            while True:
                i = 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_game()
        X = OneHotEncoder().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('lc', LearningCurve(MultinomialNB(), cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4, random_state=42))])
        model.fit(X, y)
        model['lc'].finalize()
        self.assert_images_similar(model['lc'], tol=2.0)

    def test_within_pipeline_quickmethod(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        (X, y) = load_game()
        X = OneHotEncoder().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('lc', learning_curve(MultinomialNB(), X, y, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4, random_state=42))])
        model['lc'].finalize()
        self.assert_images_similar(model['lc'], tol=2.0)

    def test_pipeline_as_model_input(self):
        if False:
            return 10
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_game()
        X = OneHotEncoder().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('nb', MultinomialNB())])
        oz = LearningCurve(model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4, random_state=42)
        oz.fit(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=2.0)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            while True:
                i = 10
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        within a quickmethod\n        '
        (X, y) = load_game()
        X = OneHotEncoder().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('nb', MultinomialNB())])
        oz = learning_curve(model, X, y, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4, random_state=42)
        self.assert_images_similar(oz, tol=2.0)