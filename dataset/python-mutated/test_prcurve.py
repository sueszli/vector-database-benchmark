"""
Tests for the Precision-Recall curves visualizer
"""
import sys
import pytest
import matplotlib
from yellowbrick.exceptions import *
from yellowbrick.classifier.prcurve import *
from yellowbrick.datasets import load_occupancy
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from .test_rocauc import FakeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
try:
    import pandas as pd
except ImportError:
    pd = None
LEARNED_FIELDS = ('target_type_', 'score_', 'precision_', 'recall_')

def assert_not_fitted(oz):
    if False:
        while True:
            i = 10
    for field in LEARNED_FIELDS:
        assert not hasattr(oz, field)

def assert_fitted(oz):
    if False:
        for i in range(10):
            print('nop')
    for field in LEARNED_FIELDS:
        assert hasattr(oz, field)

@pytest.mark.usefixtures('binary', 'multiclass')
class TestPrecisionRecallCurve(VisualTestCase):
    """
    Test the PrecisionRecallCurve visualizer
    """

    def test_fit_continuous(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Should not allow any target type other than binary or multiclass\n        '
        (X, y) = make_regression()
        with pytest.raises(YellowbrickValueError, match='does not support target type'):
            oz = PrecisionRecallCurve(LinearSVC())
            oz.fit(X, y)

    def test_ensure_fit(self):
        if False:
            while True:
                i = 10
        '\n        Requires visualizer to be fit\n        '
        with pytest.raises(NotFitted, match='this PrecisionRecallCurve instance is not fitted yet'):
            oz = PrecisionRecallCurve(RidgeClassifier())
            oz.score(self.binary.X.test, self.binary.y.test)

    def test_binary_probability(self):
        if False:
            return 10
        '\n        Visual similarity of binary classifier with predict_proba function\n        '
        oz = PrecisionRecallCurve(RandomForestClassifier(random_state=12))
        assert_not_fitted(oz)
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)
        oz.finalize()
        tol = 1.5 if sys.platform == 'win32' else 1.0
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_binary_probability_decision(self):
        if False:
            return 10
        '\n        Visual similarity of binary classifier with both predict_proba & decision\n        '
        oz = PrecisionRecallCurve(AdaBoostClassifier(), iso_f1_curves=True)
        assert_not_fitted(oz)
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_binary_decision(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Visual similarity of binary classifier with a decision function\n        '
        oz = PrecisionRecallCurve(LinearSVC(random_state=232))
        assert_not_fitted(oz)
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)
        self.assert_images_similar(oz, tol=3.5)

    def test_multiclass_decision(self):
        if False:
            print('Hello World!')
        '\n        Visual similarity of multiclass classifier with a decision function\n        '
        oz = PrecisionRecallCurve(RidgeClassifier(random_state=993))
        assert_not_fitted(oz)
        assert oz.fit(self.multiclass.X.train, self.multiclass.y.train) is oz
        s = oz.score(self.multiclass.X.test, self.multiclass.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == MULTICLASS
        assert isinstance(oz.score_, dict)
        assert oz.score_[MICRO] == s
        assert isinstance(oz.precision_, dict)
        assert isinstance(oz.recall_, dict)
        assert len(oz.score_) == len(oz.classes_) + 1
        assert len(oz.precision_) == len(oz.classes_) + 1
        assert len(oz.recall_) == len(oz.classes_) + 1
        oz.finalize()
        tol = 1.25 if sys.platform == 'win32' else 1.0
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_multiclass_probability(self):
        if False:
            while True:
                i = 10
        '\n        Visual similarity of multiclass classifier with predict_proba function\n        '
        oz = PrecisionRecallCurve(GaussianNB(), per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False)
        assert_not_fitted(oz)
        assert oz.fit(self.multiclass.X.train, self.multiclass.y.train) is oz
        s = oz.score(self.multiclass.X.test, self.multiclass.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == MULTICLASS
        assert isinstance(oz.score_, dict)
        assert oz.score_[MICRO] == s
        assert isinstance(oz.precision_, dict)
        assert isinstance(oz.recall_, dict)
        assert len(oz.score_) == len(oz.classes_) + 1
        assert len(oz.precision_) == len(oz.classes_) + 1
        assert len(oz.recall_) == len(oz.classes_) + 1
        oz.finalize()
        self.assert_images_similar(oz)

    def test_multiclass_probability_with_class_labels(self):
        if False:
            for i in range(10):
                print('nop')
        'Visual similarity of multiclass classifier with class labels.'
        oz = PrecisionRecallCurve(GaussianNB(), per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=['a', 'b', 'c', 'd', 'e', 'f'])
        assert_not_fitted(oz)
        assert oz.fit(self.multiclass.X.train, self.multiclass.y.train) is oz
        s = oz.score(self.multiclass.X.test, self.multiclass.y.test)
        assert_fitted(oz)
        assert 0.0 <= s <= 1.0
        assert oz.target_type_ == MULTICLASS
        assert isinstance(oz.score_, dict)
        assert oz.score_[MICRO] == s
        assert isinstance(oz.precision_, dict)
        assert isinstance(oz.recall_, dict)
        assert len(oz.score_) == len(oz.classes_) + 1
        assert len(oz.precision_) == len(oz.classes_) + 1
        assert len(oz.recall_) == len(oz.classes_) + 1
        oz.finalize()
        assert oz.ax.get_xlabel() == 'Recall'
        oz.ax.set_xlabel('')
        assert oz.ax.get_ylabel() == 'Precision'
        oz.ax.set_ylabel('')
        assert oz.ax.get_title() == 'Precision-Recall Curve for GaussianNB'
        oz.ax.set_title('')
        expected_legend_txt = ['PR for class a (area=0.42)', 'PR for class b (area=0.36)', 'PR for class c (area=0.44)', 'PR for class d (area=0.52)', 'PR for class e (area=0.37)', 'PR for class f (area=0.49)']
        assert [x.get_text() for x in oz.ax.legend().get_texts()] == expected_legend_txt
        oz.ax.get_legend().remove()
        for child in oz.ax.get_children():
            if isinstance(child, matplotlib.text.Annotation):
                oz.ax.texts.remove(child)
        tol = 6.6 if sys.platform == 'win32' else 1.0
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.filterwarnings('ignore:From version 0.21')
    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_quick_method(self):
        if False:
            return 10
        '\n        Test the precision_recall_curve quick method with numpy arrays.\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        model = DecisionTreeClassifier(random_state=23, max_depth=2)
        oz = precision_recall_curve(model, X, y, per_class=False, micro=True, fill_area=False, iso_f1_curves=True, ap_score=False, show=False)
        assert isinstance(oz, PrecisionRecallCurve)
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            print('Hello World!')
        '\n        Test the precision_recall_curve with Pandas dataframes\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        model = DecisionTreeClassifier(random_state=14)
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=555)
        oz = PrecisionRecallCurve(model, per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=['unoccupied', 'occupied'])
        oz.fit(X_train, y_train)
        oz.score(X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=5.5)

    def test_no_scoring_function(self):
        if False:
            while True:
                i = 10
        '\n        Test get y scores with classifiers that have no scoring method\n        '
        oz = PrecisionRecallCurve(FakeClassifier())
        with pytest.raises(ModelError, match='requires .* predict_proba or decision_function'):
            oz._get_y_scores(self.binary.X.train)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_custom_iso_f1_scores(self):
        if False:
            return 10
        '\n        Test using custom ISO F1 Values\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        vals = (0.1, 0.6, 0.3, 0.9, 0.9)
        viz = PrecisionRecallCurve(RandomForestClassifier(random_state=27), iso_f1_curves=True, iso_f1_values=vals)
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=555)
        assert viz.fit(X_train, y_train) is viz
        viz.score(X_test, y_test)
        viz.finalize()
        self.assert_images_similar(viz)

    def test_quick_method_with_test_set(self):
        if False:
            print('Hello World!')
        '\n        Test quick method when both train and test data is supplied\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=555)
        viz = precision_recall_curve(RandomForestClassifier(random_state=72), X_train, y_train, X_test, y_test, show=False)
        self.assert_images_similar(viz)

    def test_missing_test_data_in_quick_method(self):
        if False:
            while True:
                i = 10
        '\n        Test quick method when test data is missing.\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=55555)
        emsg = 'both X_test and y_test are required if one is specified'
        with pytest.raises(YellowbrickValueError, match=emsg):
            precision_recall_curve(RandomForestClassifier(), X_train, y_train, y_test=y_test, show=False)
        with pytest.raises(YellowbrickValueError, match=emsg):
            precision_recall_curve(RandomForestClassifier(), X_train, y_train, X_test, show=False)

    def test_per_class_and_micro(self):
        if False:
            print('Hello World!')
        '\n        Test if both per_class and micro set to True, user gets micro ignored warning\n        '
        msg = 'micro=True is ignored;specify per_class=False to draw a PR curve after micro-averaging'
        with pytest.warns(YellowbrickWarning, match=msg):
            PrecisionRecallCurve(RidgeClassifier(random_state=13), micro=True, per_class=True, show=False)

    def test_within_pipeline(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('prc', PrecisionRecallCurve(SVC(random_state=42), per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=classes))])
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        model['prc'].finalize()
        self.assert_images_similar(model['prc'], tol=5.5)

    def test_within_pipeline_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer quickmethod can be accessed within a\n        sklearn pipeline\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('prc', precision_recall_curve(SVC(random_state=42), X_train, y_train, X_test, y_test, per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=['unoccupied', 'occupied'], show=False))])
        self.assert_images_similar(model['prc'], tol=5.5)

    def test_pipeline_as_model_input(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        classes = ['unoccupied', 'occupied']
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = PrecisionRecallCurve(model, per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=classes)
        oz.fit(X_train, y_train)
        oz.score(X_test, y_test)
        oz.finalize()
        self.assert_images_similar(oz, tol=5.5)

    def test_pipeline_as_model_input_quickmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that visualizer can handle sklearn pipeline as model input\n        within a quickmethod\n        '
        (X, y) = load_occupancy(return_dataset=True).to_pandas()
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, shuffle=True, random_state=42)
        model = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(random_state=42))])
        oz = precision_recall_curve(model, X_train, y_train, X_test, y_test, per_class=True, micro=False, fill_area=False, iso_f1_curves=True, ap_score=False, classes=['unoccupied', 'occupied'], show=False)
        self.assert_images_similar(oz, tol=5.5)