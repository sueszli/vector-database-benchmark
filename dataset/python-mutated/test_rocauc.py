"""
Testing for the ROCAUC visualizer
"""
import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch
from tests.base import VisualTestCase
from yellowbrick.classifier.rocauc import *
from yellowbrick.exceptions import ModelError
from yellowbrick.datasets import load_occupancy
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
try:
    import pandas as pd
except ImportError:
    pd = None

class FakeClassifier(BaseEstimator, ClassifierMixin):
    """
    A fake classifier for testing noops on the visualizer.
    """
    pass

def assert_valid_rocauc_scores(visualizer, nscores=4):
    if False:
        while True:
            i = 10
    '\n    Assertion helper to ensure scores are correctly computed\n    '
    __tracebackhide__ = True
    assert len(visualizer.fpr.keys()) == nscores
    assert len(visualizer.tpr.keys()) == nscores
    assert len(visualizer.roc_auc.keys()) == nscores
    for k in (0, 1, 'micro', 'macro'):
        assert k in visualizer.fpr
        assert k in visualizer.tpr
        assert k in visualizer.roc_auc
        assert len(visualizer.fpr[k]) == len(visualizer.tpr[k])
        assert 0.0 < visualizer.roc_auc[k] < 1.0

@pytest.mark.usefixtures('binary', 'multiclass')
class TestROCAUC(VisualTestCase):
    """
    Test ROCAUC visualizer
    """

    def test_binary_probability(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ROCAUC with a binary classifier with a predict_proba function\n        '
        visualizer = ROCAUC(RandomForestClassifier(random_state=42))
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert 0 <= s <= 1
        assert_valid_rocauc_scores(visualizer)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_probability_decision(self):
        if False:
            return 10
        '\n        Test ROCAUC with a binary classifier with both decision & predict_proba\n        '
        visualizer = ROCAUC(AdaBoostClassifier())
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert 0 <= s <= 1
        assert_valid_rocauc_scores(visualizer)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_probability_decision_single_curve(self):
        if False:
            print('Hello World!')
        '\n        Test ROCAUC binary classifier with both decision & predict_proba with per_class=False\n        '
        visualizer = ROCAUC(AdaBoostClassifier(), micro=False, macro=False, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert 0 <= s <= 1
        assert len(visualizer.fpr.keys()) == 1
        assert len(visualizer.tpr.keys()) == 1
        assert len(visualizer.roc_auc.keys()) == 1
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_decision(self):
        if False:
            while True:
                i = 10
        '\n        Test ROCAUC with a binary classifier with a decision_function\n        '
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=False, macro=False, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert 0 <= s <= 1
        assert len(visualizer.fpr.keys()) == 1
        assert len(visualizer.tpr.keys()) == 1
        assert len(visualizer.roc_auc.keys()) == 1
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=10)

    def test_binary_decision_per_class(self):
        if False:
            while True:
                i = 10
        '\n        Test ROCAUC with a binary classifier with a decision_function\n        '
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=False, macro=False, per_class=True)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert 0 <= s <= 1
        assert len(visualizer.fpr.keys()) == 2
        assert len(visualizer.tpr.keys()) == 2
        assert len(visualizer.roc_auc.keys()) == 2
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=10)

    def test_binary_micro_error(self):
        if False:
            while True:
                i = 10
        '\n        Test ROCAUC to see if _binary_decision with micro = True raises an error\n        '
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=True, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_binary_macro_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ROCAUC to see if _binary_decision with macro = True raises an error\n        '
        visualizer = ROCAUC(LinearSVC(random_state=42), macro=True, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_multiclass_rocauc(self):
        if False:
            print('Hello World!')
        '\n        Test ROCAUC with a multiclass classifier\n        '
        visualizer = ROCAUC(GaussianNB())
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)
        s = visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        assert 0 <= s <= 1
        assert_valid_rocauc_scores(visualizer, nscores=8)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_classes(self):
        if False:
            return 10
        '\n        Test ROCAUC without per-class curves\n        '
        visualizer = ROCAUC(GaussianNB(), per_class=False)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)
        s = visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        assert s == pytest.approx(0.77303, abs=0.0001)
        for c in (0, 1):
            assert c in visualizer.fpr
            assert c in visualizer.tpr
            assert c in visualizer.roc_auc
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_curves(self):
        if False:
            return 10
        '\n        Test ROCAUC with no curves specified at all\n        '
        visualizer = ROCAUC(GaussianNB(), per_class=False, macro=False, micro=False)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)
        with pytest.raises(YellowbrickValueError, match='no curves will be drawn'):
            visualizer.score(self.multiclass.X.test, self.multiclass.y.test)

    def test_rocauc_quickmethod(self):
        if False:
            print('Hello World!')
        '\n        Test the ROCAUC quick method\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        model = LogisticRegression()
        visualizer = roc_auc(model, X, y, show=False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            return 10
        '\n        Test the ROCAUC with Pandas dataframe\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        splits = tts(X, y, test_size=0.2, random_state=4512)
        (X_train, X_test, y_train, y_test) = splits
        visualizer = ROCAUC(GaussianNB())
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_rocauc_no_micro(self):
        if False:
            return 10
        '\n        Test ROCAUC without a micro average\n        '
        visualizer = ROCAUC(LogisticRegression(), micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8661, abs=0.0001)
        assert 'micro' not in visualizer.fpr
        assert 'micro' not in visualizer.tpr
        assert 'micro' not in visualizer.roc_auc
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_macro(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ROCAUC without a macro average\n        '
        visualizer = ROCAUC(LogisticRegression(), macro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8573, abs=0.0001)
        assert 'macro' not in visualizer.fpr
        assert 'macro' not in visualizer.tpr
        assert 'macro' not in visualizer.roc_auc
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_macro_no_micro(self):
        if False:
            return 10
        '\n        Test ROCAUC without a macro or micro average\n        '
        visualizer = ROCAUC(LogisticRegression(), macro=False, micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8)
        assert 'macro' not in visualizer.fpr
        assert 'macro' not in visualizer.tpr
        assert 'macro' not in visualizer.roc_auc
        assert 'micro' not in visualizer.fpr
        assert 'micro' not in visualizer.tpr
        assert 'micro' not in visualizer.roc_auc
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_label_encoded(self):
        if False:
            print('Hello World!')
        '\n        Test ROCAUC with a target specifying a list of classes as strings\n        '
        class_labels = ['a', 'b', 'c', 'd', 'e', 'f']
        visualizer = ROCAUC(LogisticRegression(), classes=class_labels)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)
        visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        assert list(visualizer.classes_) == class_labels

    def test_rocauc_not_label_encoded(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ROCAUC with a target whose classes are unencoded strings before scoring\n        '
        classes = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
        y_train = np.array([classes[yi] for yi in self.multiclass.y.train])
        y_test = np.array([classes[yi] for yi in self.multiclass.y.test])
        visualizer = ROCAUC(LogisticRegression())
        visualizer.fit(self.multiclass.X.train, y_train)
        assert set(y_train) == set(y_test)

    def test_binary_decision_function_rocauc(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ROCAUC with binary classifiers that have a decision function\n        '
        model = LinearSVC()
        with pytest.raises(AttributeError):
            model.predict_proba
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        first_ten_expected = np.asarray([-0.092, 0.019, -0.751, -0.838, 0.183, -0.344, -1.019, 2.203, 1.415, -0.529])
        y_scores = visualizer._get_y_scores(self.binary.X.train)
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_binary_false_decision_function_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test binary decision_function model raises error when the binary param is False\n        '
        visualizer = ROCAUC(LinearSVC(random_state=42), binary=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_multi_decision_function_rocauc(self):
        if False:
            return 10
        '\n        Test ROCAUC with multiclass classifiers that have a decision function\n        '
        model = LinearSVC()
        with pytest.raises(AttributeError):
            model.predict_proba
        visualizer = ROCAUC(model)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)
        first_five_expected = [[-0.37, -0.543, -1.059, -0.466, -0.743, -1.156], [-0.445, -0.693, -0.362, -1.002, -0.815, -0.878], [-1.058, -0.808, -0.291, -0.767, -0.651, -0.586], [-0.446, -1.255, -0.489, -0.961, -0.807, -0.126], [-1.066, -0.493, -0.639, -0.442, -0.639, -1.017]]
        y_scores = visualizer._get_y_scores(self.multiclass.X.train)
        npt.assert_array_almost_equal(y_scores[:5], first_five_expected, decimal=1)

    def test_predict_proba_rocauc(self):
        if False:
            return 10
        '\n        Test ROCAUC with classifiers that utilize predict_proba\n        '
        model = GaussianNB()
        with pytest.raises(AttributeError):
            model.decision_function
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)
        first_ten_expected = np.asarray([[0.595, 0.405], [0.161, 0.839], [0.99, 0.01], [0.833, 0.167], [0.766, 0.234], [0.996, 0.004], [0.592, 0.408], [0.007, 0.993], [0.035, 0.965], [0.764, 0.236]])
        y_scores = visualizer._get_y_scores(self.binary.X.train)
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_no_scoring_function(self):
        if False:
            return 10
        '\n        Test ROCAUC with classifiers that have no scoring method\n        '
        visualizer = ROCAUC(FakeClassifier())
        with pytest.raises(ModelError):
            visualizer._get_y_scores(self.binary.X.train)

    def test_with_fitted(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer properly handles an already-fitted model\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        model = GaussianNB().fit(X, y)
        classes = ['unoccupied', 'occupied']
        with patch.object(model, 'fit') as mockfit:
            oz = ROCAUC(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with patch.object(model, 'fit') as mockfit:
            oz = ROCAUC(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with patch.object(model, 'fit') as mockfit:
            oz = ROCAUC(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    def test_binary_meta_param(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the binary meta param with ROCAUC\n        '
        oz = ROCAUC(GaussianNB(), binary=False)
        assert oz.micro is True
        assert oz.macro is True
        assert oz.per_class is True
        oz = ROCAUC(GaussianNB(), binary=True)
        assert oz.micro is False
        assert oz.macro is False
        assert oz.per_class is False

    def test_within_pipeline(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('matrix', ROCAUC(SVC(random_state=42), classes=classes, binary=True))])
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        model['matrix'].finalize()
        self.assert_images_similar(model['matrix'], tol=12)

    def test_within_pipeline_quickmethod(self):
        if False:
            return 10
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('matrix', roc_auc(SVC(random_state=42), X_train, y_train, X_test, y_test, classes=['vacant', 'occupied'], show=False, binary=True))])
        self.assert_images_similar(model['matrix'], tol=12)

    def test_pipeline_as_model_input(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = ROCAUC(model, classes=classes, binary=True)
        oz.fit(X_train, y_train)
        oz.score(X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=12)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        within a quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = roc_auc(model, X_train, y_train, X_test, y_test, classes=['vacant', 'occupied'], show=False, binary=True)
        self.assert_images_similar(oz, tol=12)