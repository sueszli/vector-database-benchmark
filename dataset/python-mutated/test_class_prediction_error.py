"""
Testing for the ClassPredictionError visualizer
"""
import pytest
import matplotlib.pyplot as plt
from yellowbrick.exceptions import ModelError
from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier.class_prediction_error import *
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import patch
from tests.base import VisualTestCase
try:
    import pandas as pd
except ImportError:
    pd = None

class TestClassPredictionError(VisualTestCase):
    """
    Test ClassPredictionError visualizer
    """

    @pytest.mark.filterwarnings('ignore:could not determine class_counts_')
    def test_numpy_integration(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors during class prediction error integration with NumPy arrays\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        classes = ['unoccupied', 'occupied']
        model = SVC(random_state=42)
        model.fit(X, y)
        visualizer = ClassPredictionError(model, classes=classes)
        visualizer.score(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=12.5, windows_tol=13.3)

    @pytest.mark.filterwarnings('ignore:could not determine class_counts_')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            return 10
        '\n        Assert no errors during class prediction error integration with Pandas\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        model = SVC(random_state=42)
        model.fit(X, y)
        visualizer = ClassPredictionError(model, classes=classes)
        visualizer.score(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=12.5, windows_tol=13.3)

    def test_class_prediction_error_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the ClassPredictionError quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot()
        clf = SVC(random_state=42)
        viz = class_prediction_error(clf, X, y, ax=ax, show=False)
        self.assert_images_similar(viz, tol=16, windows_tol=16)

    def test_class_prediction_error_quickmethod_X_test_only(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the ClassPredictionError quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        fig = plt.figure()
        ax = fig.add_subplot()
        clf = LinearSVC(random_state=42)
        with pytest.raises(YellowbrickValueError, match='must specify both X_test and y_test or neither'):
            class_prediction_error(clf, X_train=X_train, y_train=y_train, X_test=X_test, ax=ax, show=False)

    def test_class_prediction_error_quickmethod_X_test_and_y_test(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the ClassPredictionError quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        fig = plt.figure()
        ax = fig.add_subplot()
        clf = SVC(random_state=42)
        viz = class_prediction_error(clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, ax=ax, show=False)
        self.assert_images_similar(viz, tol=13, windows_tol=13)

    @pytest.mark.filterwarnings('ignore:could not determine class_counts_')
    def test_classes_greater_than_indices(self):
        if False:
            return 10
        '\n        A model error should be raised when there are more classes in fit than score\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        classes = ['unoccupied', 'occupied', 'partytime']
        model = LinearSVC(random_state=42)
        model.fit(X, y)
        with pytest.raises(ModelError):
            visualizer = ClassPredictionError(model, classes=classes)
            visualizer.score(X, y)

    def test_classes_less_than_indices(self):
        if False:
            print('Hello World!')
        '\n        Assert error when there is an attempt to filter classes\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        classes = ['unoccupied']
        model = LinearSVC(random_state=42)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            visualizer = ClassPredictionError(model, classes=classes)
            visualizer.score(X, y)

    @pytest.mark.skip(reason='not implemented yet')
    def test_no_classes_provided(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors when no classes are provided\n        '
        pass

    def test_class_type(self):
        if False:
            print('Hello World!')
        '\n        Test class must be either binary or multiclass type\n        '
        (X, y) = make_multilabel_classification()
        model = RandomForestClassifier()
        model.fit(X, y)
        with pytest.raises(YellowbrickValueError):
            visualizer = ClassPredictionError(model)
            visualizer.score(X, y)

    def test_score_returns_score(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that ClassPredictionError score() returns a score between 0 and 1\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        visualizer = ClassPredictionError(LinearSVC(random_state=42))
        visualizer.fit(X, y)
        s = visualizer.score(X, y)
        assert 0 <= s <= 1

    def test_with_fitted(self):
        if False:
            return 10
        '\n        Test that visualizer properly handles an already-fitted model\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        model = RandomForestClassifier().fit(X, y)
        classes = ['unoccupied', 'occupied']
        with patch.object(model, 'fit') as mockfit:
            oz = ClassPredictionError(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with patch.object(model, 'fit') as mockfit:
            oz = ClassPredictionError(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with patch.object(model, 'fit') as mockfit:
            oz = ClassPredictionError(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    def test_within_pipeline(self):
        if False:
            return 10
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('cpe', ClassPredictionError(SVC(random_state=42), classes=classes))])
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        model['cpe'].finalize()
        self.assert_images_similar(model['cpe'], tol=12.5, windows_tol=13.3)

    def test_within_pipeline_quickmethod(self):
        if False:
            while True:
                i = 10
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('cpe', class_prediction_error(SVC(random_state=42), X_train, y_train, X_test, y_test, classes=['vacant', 'occupied'], show=False))])
        self.assert_images_similar(model['cpe'], tol=12.5, windows_tol=13.3)

    def test_pipeline_as_model_input(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = ClassPredictionError(model, classes=classes)
        oz.fit(X_train, y_train)
        oz.score(X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=12.5, windows_tol=13.3)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        within a quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = class_prediction_error(model, X_train, y_train, X_test, y_test, classes=['vacant', 'occupied'], show=False)
        self.assert_images_similar(oz, tol=12.5, windows_tol=13.3)