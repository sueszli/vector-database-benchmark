import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, _CalibratedClassifier, _sigmoid_calibration, _SigmoidCalibration, calibration_curve
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold, LeaveOneOut, check_cv, cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import _convert_container, assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
N_SAMPLES = 200

@pytest.fixture(scope='module')
def data():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(n_samples=N_SAMPLES, n_features=6, random_state=42)
    return (X, y)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration(data, method, csr_container, ensemble):
    if False:
        while True:
            i = 10
    n_samples = N_SAMPLES // 2
    (X, y) = data
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)
    X -= X.min()
    (X_train, y_train, sw_train) = (X[:n_samples], y[:n_samples], sample_weight[:n_samples])
    (X_test, y_test) = (X[n_samples:], y[n_samples:])
    clf = MultinomialNB(force_alpha=True).fit(X_train, y_train, sample_weight=sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
    cal_clf = CalibratedClassifierCV(clf, cv=y.size + 1, ensemble=ensemble)
    with pytest.raises(ValueError):
        cal_clf.fit(X, y)
    for (this_X_train, this_X_test) in [(X_train, X_test), (csr_container(X_train), csr_container(X_test))]:
        cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
        cal_clf.fit(this_X_train, y_train, sample_weight=sw_train)
        prob_pos_cal_clf = cal_clf.predict_proba(this_X_test)[:, 1]
        assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(y_test, prob_pos_cal_clf)
        cal_clf.fit(this_X_train, y_train + 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)
        cal_clf.fit(this_X_train, 2 * y_train - 1, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        assert_array_almost_equal(prob_pos_cal_clf, prob_pos_cal_clf_relabeled)
        cal_clf.fit(this_X_train, (y_train + 1) % 2, sample_weight=sw_train)
        prob_pos_cal_clf_relabeled = cal_clf.predict_proba(this_X_test)[:, 1]
        if method == 'sigmoid':
            assert_array_almost_equal(prob_pos_cal_clf, 1 - prob_pos_cal_clf_relabeled)
        else:
            assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss((y_test + 1) % 2, prob_pos_cal_clf_relabeled)

def test_calibration_default_estimator(data):
    if False:
        return 10
    (X, y) = data
    calib_clf = CalibratedClassifierCV(cv=2)
    calib_clf.fit(X, y)
    base_est = calib_clf.calibrated_classifiers_[0].estimator
    assert isinstance(base_est, LinearSVC)

@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration_cv_splitter(data, ensemble):
    if False:
        i = 10
        return i + 15
    (X, y) = data
    splits = 5
    kfold = KFold(n_splits=splits)
    calib_clf = CalibratedClassifierCV(cv=kfold, ensemble=ensemble)
    assert isinstance(calib_clf.cv, KFold)
    assert calib_clf.cv.n_splits == splits
    calib_clf.fit(X, y)
    expected_n_clf = splits if ensemble else 1
    assert len(calib_clf.calibrated_classifiers_) == expected_n_clf

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_sample_weight(data, method, ensemble):
    if False:
        return 10
    n_samples = N_SAMPLES // 2
    (X, y) = data
    sample_weight = np.random.RandomState(seed=42).uniform(size=len(y))
    (X_train, y_train, sw_train) = (X[:n_samples], y[:n_samples], sample_weight[:n_samples])
    X_test = X[n_samples:]
    estimator = LinearSVC(dual='auto', random_state=42)
    calibrated_clf = CalibratedClassifierCV(estimator, method=method, ensemble=ensemble)
    calibrated_clf.fit(X_train, y_train, sample_weight=sw_train)
    probs_with_sw = calibrated_clf.predict_proba(X_test)
    calibrated_clf.fit(X_train, y_train)
    probs_without_sw = calibrated_clf.predict_proba(X_test)
    diff = np.linalg.norm(probs_with_sw - probs_without_sw)
    assert diff > 0.1

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_parallel_execution(data, method, ensemble):
    if False:
        while True:
            i = 10
    'Test parallel calibration'
    (X, y) = data
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42)
    estimator = make_pipeline(StandardScaler(), LinearSVC(dual='auto', random_state=42))
    cal_clf_parallel = CalibratedClassifierCV(estimator, method=method, n_jobs=2, ensemble=ensemble)
    cal_clf_parallel.fit(X_train, y_train)
    probs_parallel = cal_clf_parallel.predict_proba(X_test)
    cal_clf_sequential = CalibratedClassifierCV(estimator, method=method, n_jobs=1, ensemble=ensemble)
    cal_clf_sequential.fit(X_train, y_train)
    probs_sequential = cal_clf_sequential.predict_proba(X_test)
    assert_allclose(probs_parallel, probs_sequential)

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
@pytest.mark.parametrize('seed', range(2))
def test_calibration_multiclass(method, ensemble, seed):
    if False:
        while True:
            i = 10

    def multiclass_brier(y_true, proba_pred, n_classes):
        if False:
            i = 10
            return i + 15
        Y_onehot = np.eye(n_classes)[y_true]
        return np.sum((Y_onehot - proba_pred) ** 2) / Y_onehot.shape[0]
    clf = LinearSVC(dual='auto', random_state=7)
    (X, y) = make_blobs(n_samples=500, n_features=100, random_state=seed, centers=10, cluster_std=15.0)
    y[y > 2] = 2
    n_classes = np.unique(y).shape[0]
    (X_train, y_train) = (X[::2], y[::2])
    (X_test, y_test) = (X[1::2], y[1::2])
    clf.fit(X_train, y_train)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    probas = cal_clf.predict_proba(X_test)
    assert_allclose(np.sum(probas, axis=1), np.ones(len(X_test)))
    assert 0.65 < clf.score(X_test, y_test) < 0.95
    assert cal_clf.score(X_test, y_test) > 0.95 * clf.score(X_test, y_test)
    uncalibrated_brier = multiclass_brier(y_test, softmax(clf.decision_function(X_test)), n_classes=n_classes)
    calibrated_brier = multiclass_brier(y_test, probas, n_classes=n_classes)
    assert calibrated_brier < 1.1 * uncalibrated_brier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    uncalibrated_brier = multiclass_brier(y_test, clf_probs, n_classes=n_classes)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=5, ensemble=ensemble)
    cal_clf.fit(X_train, y_train)
    cal_clf_probs = cal_clf.predict_proba(X_test)
    calibrated_brier = multiclass_brier(y_test, cal_clf_probs, n_classes=n_classes)
    assert calibrated_brier < 1.1 * uncalibrated_brier

def test_calibration_zero_probability():
    if False:
        return 10

    class ZeroCalibrator:

        def predict(self, X):
            if False:
                return 10
            return np.zeros(X.shape[0])
    (X, y) = make_blobs(n_samples=50, n_features=10, random_state=7, centers=10, cluster_std=15.0)
    clf = DummyClassifier().fit(X, y)
    calibrator = ZeroCalibrator()
    cal_clf = _CalibratedClassifier(estimator=clf, calibrators=[calibrator], classes=clf.classes_)
    probas = cal_clf.predict_proba(X)
    assert_allclose(probas, 1.0 / clf.n_classes_)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_calibration_prefit(csr_container):
    if False:
        for i in range(10):
            print('nop')
    'Test calibration for prefitted classifiers'
    n_samples = 50
    (X, y) = make_classification(n_samples=3 * n_samples, n_features=6, random_state=42)
    sample_weight = np.random.RandomState(seed=42).uniform(size=y.size)
    X -= X.min()
    (X_train, y_train, sw_train) = (X[:n_samples], y[:n_samples], sample_weight[:n_samples])
    (X_calib, y_calib, sw_calib) = (X[n_samples:2 * n_samples], y[n_samples:2 * n_samples], sample_weight[n_samples:2 * n_samples])
    (X_test, y_test) = (X[2 * n_samples:], y[2 * n_samples:])
    clf = MultinomialNB(force_alpha=True)
    unfit_clf = CalibratedClassifierCV(clf, cv='prefit')
    with pytest.raises(NotFittedError):
        unfit_clf.fit(X_calib, y_calib)
    clf.fit(X_train, y_train, sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
    for (this_X_calib, this_X_test) in [(X_calib, X_test), (csr_container(X_calib), csr_container(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            cal_clf = CalibratedClassifierCV(clf, method=method, cv='prefit')
            for sw in [sw_calib, None]:
                cal_clf.fit(this_X_calib, y_calib, sample_weight=sw)
                y_prob = cal_clf.predict_proba(this_X_test)
                y_pred = cal_clf.predict(this_X_test)
                prob_pos_cal_clf = y_prob[:, 1]
                assert_array_equal(y_pred, np.array([0, 1])[np.argmax(y_prob, axis=1)])
                assert brier_score_loss(y_test, prob_pos_clf) > brier_score_loss(y_test, prob_pos_cal_clf)

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
def test_calibration_ensemble_false(data, method):
    if False:
        print('Hello World!')
    (X, y) = data
    clf = LinearSVC(dual='auto', random_state=7)
    cal_clf = CalibratedClassifierCV(clf, method=method, cv=3, ensemble=False)
    cal_clf.fit(X, y)
    cal_probas = cal_clf.predict_proba(X)
    unbiased_preds = cross_val_predict(clf, X, y, cv=3, method='decision_function')
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    else:
        calibrator = _SigmoidCalibration()
    calibrator.fit(unbiased_preds, y)
    clf.fit(X, y)
    clf_df = clf.decision_function(X)
    manual_probas = calibrator.predict(clf_df)
    assert_allclose(cal_probas[:, 1], manual_probas)

def test_sigmoid_calibration():
    if False:
        print('Hello World!')
    'Test calibration values with Platt sigmoid model'
    exF = np.array([5, -4, 1.0])
    exY = np.array([1, -1, -1])
    AB_lin_libsvm = np.array([-0.20261354391187855, 0.6523631498001051])
    assert_array_almost_equal(AB_lin_libsvm, _sigmoid_calibration(exF, exY), 3)
    lin_prob = 1.0 / (1.0 + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
    sk_prob = _SigmoidCalibration().fit(exF, exY).predict(exF)
    assert_array_almost_equal(lin_prob, sk_prob, 6)
    with pytest.raises(ValueError):
        _SigmoidCalibration().fit(np.vstack((exF, exF)), exY)

def test_calibration_curve():
    if False:
        print('Hello World!')
    'Check calibration_curve function'
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    (prob_true, prob_pred) = calibration_curve(y_true, y_pred, n_bins=2)
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) == 2
    assert_almost_equal(prob_true, [0, 1])
    assert_almost_equal(prob_pred, [0.1, 0.9])
    with pytest.raises(ValueError):
        calibration_curve([1], [-0.1])
    y_true2 = np.array([0, 0, 0, 0, 1, 1])
    y_pred2 = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    (prob_true_quantile, prob_pred_quantile) = calibration_curve(y_true2, y_pred2, n_bins=2, strategy='quantile')
    assert len(prob_true_quantile) == len(prob_pred_quantile)
    assert len(prob_true_quantile) == 2
    assert_almost_equal(prob_true_quantile, [0, 2 / 3])
    assert_almost_equal(prob_pred_quantile, [0.1, 0.8])
    with pytest.raises(ValueError):
        calibration_curve(y_true2, y_pred2, strategy='percentile')

@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration_nan_imputer(ensemble):
    if False:
        i = 10
        return i + 15
    'Test that calibration can accept nan'
    (X, y) = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X[0, 0] = np.nan
    clf = Pipeline([('imputer', SimpleImputer()), ('rf', RandomForestClassifier(n_estimators=1))])
    clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic', ensemble=ensemble)
    clf_c.fit(X, y)
    clf_c.predict(X)

@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration_prob_sum(ensemble):
    if False:
        for i in range(10):
            print('nop')
    num_classes = 2
    (X, y) = make_classification(n_samples=10, n_features=5, n_classes=num_classes)
    clf = LinearSVC(dual='auto', C=1.0, random_state=7)
    clf_prob = CalibratedClassifierCV(clf, method='sigmoid', cv=LeaveOneOut(), ensemble=ensemble)
    clf_prob.fit(X, y)
    probs = clf_prob.predict_proba(X)
    assert_array_almost_equal(probs.sum(axis=1), np.ones(probs.shape[0]))

@pytest.mark.parametrize('ensemble', [True, False])
def test_calibration_less_classes(ensemble):
    if False:
        return 10
    X = np.random.randn(10, 5)
    y = np.arange(10)
    clf = LinearSVC(dual='auto', C=1.0, random_state=7)
    cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=LeaveOneOut(), ensemble=ensemble)
    cal_clf.fit(X, y)
    for (i, calibrated_classifier) in enumerate(cal_clf.calibrated_classifiers_):
        proba = calibrated_classifier.predict_proba(X)
        if ensemble:
            assert_array_equal(proba[:, i], np.zeros(len(y)))
            assert np.all(proba[:, :i] > 0)
            assert np.all(proba[:, i + 1:] > 0)
        else:
            assert np.allclose(proba, 1 / proba.shape[0])

@pytest.mark.parametrize('X', [np.random.RandomState(42).randn(15, 5, 2), np.random.RandomState(42).randn(15, 5, 2, 6)])
def test_calibration_accepts_ndarray(X):
    if False:
        while True:
            i = 10
    'Test that calibration accepts n-dimensional arrays as input'
    y = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]

    class MockTensorClassifier(BaseEstimator):
        """A toy estimator that accepts tensor inputs"""

        def fit(self, X, y):
            if False:
                i = 10
                return i + 15
            self.classes_ = np.unique(y)
            return self

        def decision_function(self, X):
            if False:
                print('Hello World!')
            return X.reshape(X.shape[0], -1).sum(axis=1)
    calibrated_clf = CalibratedClassifierCV(MockTensorClassifier())
    calibrated_clf.fit(X, y)

@pytest.fixture
def dict_data():
    if False:
        print('Hello World!')
    dict_data = [{'state': 'NY', 'age': 'adult'}, {'state': 'TX', 'age': 'adult'}, {'state': 'VT', 'age': 'child'}]
    text_labels = [1, 0, 1]
    return (dict_data, text_labels)

@pytest.fixture
def dict_data_pipeline(dict_data):
    if False:
        while True:
            i = 10
    (X, y) = dict_data
    pipeline_prefit = Pipeline([('vectorizer', DictVectorizer()), ('clf', RandomForestClassifier())])
    return pipeline_prefit.fit(X, y)

def test_calibration_dict_pipeline(dict_data, dict_data_pipeline):
    if False:
        i = 10
        return i + 15
    'Test that calibration works in prefit pipeline with transformer\n\n    `X` is not array-like, sparse matrix or dataframe at the start.\n    See https://github.com/scikit-learn/scikit-learn/issues/8710\n\n    Also test it can predict without running into validation errors.\n    See https://github.com/scikit-learn/scikit-learn/issues/19637\n    '
    (X, y) = dict_data
    clf = dict_data_pipeline
    calib_clf = CalibratedClassifierCV(clf, cv='prefit')
    calib_clf.fit(X, y)
    assert_array_equal(calib_clf.classes_, clf.classes_)
    assert not hasattr(clf, 'n_features_in_')
    assert not hasattr(calib_clf, 'n_features_in_')
    calib_clf.predict(X)
    calib_clf.predict_proba(X)

@pytest.mark.parametrize('clf, cv', [pytest.param(LinearSVC(dual='auto', C=1), 2), pytest.param(LinearSVC(dual='auto', C=1), 'prefit')])
def test_calibration_attributes(clf, cv):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    if cv == 'prefit':
        clf = clf.fit(X, y)
    calib_clf = CalibratedClassifierCV(clf, cv=cv)
    calib_clf.fit(X, y)
    if cv == 'prefit':
        assert_array_equal(calib_clf.classes_, clf.classes_)
        assert calib_clf.n_features_in_ == clf.n_features_in_
    else:
        classes = LabelEncoder().fit(y).classes_
        assert_array_equal(calib_clf.classes_, classes)
        assert calib_clf.n_features_in_ == X.shape[1]

def test_calibration_inconsistent_prefit_n_features_in():
    if False:
        i = 10
        return i + 15
    (X, y) = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    clf = LinearSVC(dual='auto', C=1).fit(X, y)
    calib_clf = CalibratedClassifierCV(clf, cv='prefit')
    msg = 'X has 3 features, but LinearSVC is expecting 5 features as input.'
    with pytest.raises(ValueError, match=msg):
        calib_clf.fit(X[:, :3], y)

def test_calibration_votingclassifier():
    if False:
        return 10
    (X, y) = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=7)
    vote = VotingClassifier(estimators=[('lr' + str(i), LogisticRegression()) for i in range(3)], voting='soft')
    vote.fit(X, y)
    calib_clf = CalibratedClassifierCV(estimator=vote, cv='prefit')
    calib_clf.fit(X, y)

@pytest.fixture(scope='module')
def iris_data():
    if False:
        i = 10
        return i + 15
    return load_iris(return_X_y=True)

@pytest.fixture(scope='module')
def iris_data_binary(iris_data):
    if False:
        print('Hello World!')
    (X, y) = iris_data
    return (X[y < 2], y[y < 2])

@pytest.mark.parametrize('n_bins', [5, 10])
@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_calibration_display_compute(pyplot, iris_data_binary, n_bins, strategy):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y, n_bins=n_bins, strategy=strategy, alpha=0.8)
    y_prob = lr.predict_proba(X)[:, 1]
    (prob_true, prob_pred) = calibration_curve(y, y_prob, n_bins=n_bins, strategy=strategy)
    assert_allclose(viz.prob_true, prob_true)
    assert_allclose(viz.prob_pred, prob_pred)
    assert_allclose(viz.y_prob, y_prob)
    assert viz.estimator_name == 'LogisticRegression'
    import matplotlib as mpl
    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert viz.line_.get_alpha() == 0.8
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)
    assert viz.ax_.get_xlabel() == 'Mean predicted probability (Positive class: 1)'
    assert viz.ax_.get_ylabel() == 'Fraction of positives (Positive class: 1)'
    expected_legend_labels = ['LogisticRegression', 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

def test_plot_calibration_curve_pipeline(pyplot, iris_data_binary):
    if False:
        print('Hello World!')
    (X, y) = iris_data_binary
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(X, y)
    viz = CalibrationDisplay.from_estimator(clf, X, y)
    expected_legend_labels = [viz.estimator_name, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

@pytest.mark.parametrize('name, expected_label', [(None, '_line1'), ('my_est', 'my_est')])
def test_calibration_display_default_labels(pyplot, name, expected_label):
    if False:
        while True:
            i = 10
    prob_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.2, 0.8, 0.8, 0.4])
    y_prob = np.array([])
    viz = CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=name)
    viz.plot()
    expected_legend_labels = [] if name is None else [name]
    expected_legend_labels.append('Perfectly calibrated')
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

def test_calibration_display_label_class_plot(pyplot):
    if False:
        return 10
    prob_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.2, 0.8, 0.8, 0.4])
    y_prob = np.array([])
    name = 'name one'
    viz = CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=name)
    assert viz.estimator_name == name
    name = 'name two'
    viz.plot(name=name)
    expected_legend_labels = [name, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_calibration_display_name_multiple_calls(constructor_name, pyplot, iris_data_binary):
    if False:
        while True:
            i = 10
    (X, y) = iris_data_binary
    clf_name = 'my hand-crafted name'
    clf = LogisticRegression().fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    constructor = getattr(CalibrationDisplay, constructor_name)
    params = (clf, X, y) if constructor_name == 'from_estimator' else (y, y_prob)
    viz = constructor(*params, name=clf_name)
    assert viz.estimator_name == clf_name
    pyplot.close('all')
    viz.plot()
    expected_legend_labels = [clf_name, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels
    pyplot.close('all')
    clf_name = 'another_name'
    viz.plot(name=clf_name)
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

def test_calibration_display_ref_line(pyplot, iris_data_binary):
    if False:
        print('Hello World!')
    (X, y) = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    dt = DecisionTreeClassifier().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y)
    viz2 = CalibrationDisplay.from_estimator(dt, X, y, ax=viz.ax_)
    labels = viz2.ax_.get_legend_handles_labels()[1]
    assert labels.count('Perfectly calibrated') == 1

@pytest.mark.parametrize('dtype_y_str', [str, object])
def test_calibration_curve_pos_label_error_str(dtype_y_str):
    if False:
        for i in range(10):
            print('nop')
    'Check error message when a `pos_label` is not specified with `str` targets.'
    rng = np.random.RandomState(42)
    y1 = np.array(['spam'] * 3 + ['eggs'] * 2, dtype=dtype_y_str)
    y2 = rng.randint(0, 2, size=y1.size)
    err_msg = "y_true takes value in {'eggs', 'spam'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly"
    with pytest.raises(ValueError, match=err_msg):
        calibration_curve(y1, y2)

@pytest.mark.parametrize('dtype_y_str', [str, object])
def test_calibration_curve_pos_label(dtype_y_str):
    if False:
        while True:
            i = 10
    'Check the behaviour when passing explicitly `pos_label`.'
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    classes = np.array(['spam', 'egg'], dtype=dtype_y_str)
    y_true_str = classes[y_true]
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    (prob_true, _) = calibration_curve(y_true, y_pred, n_bins=4)
    assert_allclose(prob_true, [0, 0.5, 1, 1])
    (prob_true, _) = calibration_curve(y_true_str, y_pred, n_bins=4, pos_label='egg')
    assert_allclose(prob_true, [0, 0.5, 1, 1])
    (prob_true, _) = calibration_curve(y_true, 1 - y_pred, n_bins=4, pos_label=0)
    assert_allclose(prob_true, [0, 0, 0.5, 1])
    (prob_true, _) = calibration_curve(y_true_str, 1 - y_pred, n_bins=4, pos_label='spam')
    assert_allclose(prob_true, [0, 0, 0.5, 1])

@pytest.mark.parametrize('pos_label, expected_pos_label', [(None, 1), (0, 0), (1, 1)])
def test_calibration_display_pos_label(pyplot, iris_data_binary, pos_label, expected_pos_label):
    if False:
        for i in range(10):
            print('nop')
    'Check the behaviour of `pos_label` in the `CalibrationDisplay`.'
    (X, y) = iris_data_binary
    lr = LogisticRegression().fit(X, y)
    viz = CalibrationDisplay.from_estimator(lr, X, y, pos_label=pos_label)
    y_prob = lr.predict_proba(X)[:, expected_pos_label]
    (prob_true, prob_pred) = calibration_curve(y, y_prob, pos_label=pos_label)
    assert_allclose(viz.prob_true, prob_true)
    assert_allclose(viz.prob_pred, prob_pred)
    assert_allclose(viz.y_prob, y_prob)
    assert viz.ax_.get_xlabel() == f'Mean predicted probability (Positive class: {expected_pos_label})'
    assert viz.ax_.get_ylabel() == f'Fraction of positives (Positive class: {expected_pos_label})'
    expected_legend_labels = [lr.__class__.__name__, 'Perfectly calibrated']
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibrated_classifier_cv_double_sample_weights_equivalence(method, ensemble):
    if False:
        while True:
            i = 10
    'Check that passing repeating twice the dataset `X` is equivalent to\n    passing a `sample_weight` with a factor 2.'
    (X, y) = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    (X, y) = (X[:100], y[:100])
    sample_weight = np.ones_like(y) * 2
    X_twice = np.zeros((X.shape[0] * 2, X.shape[1]), dtype=X.dtype)
    X_twice[::2, :] = X
    X_twice[1::2, :] = X
    y_twice = np.zeros(y.shape[0] * 2, dtype=y.dtype)
    y_twice[::2] = y
    y_twice[1::2] = y
    estimator = LogisticRegression()
    calibrated_clf_without_weights = CalibratedClassifierCV(estimator, method=method, ensemble=ensemble, cv=2)
    calibrated_clf_with_weights = clone(calibrated_clf_without_weights)
    calibrated_clf_with_weights.fit(X, y, sample_weight=sample_weight)
    calibrated_clf_without_weights.fit(X_twice, y_twice)
    for (est_with_weights, est_without_weights) in zip(calibrated_clf_with_weights.calibrated_classifiers_, calibrated_clf_without_weights.calibrated_classifiers_):
        assert_allclose(est_with_weights.estimator.coef_, est_without_weights.estimator.coef_)
    y_pred_with_weights = calibrated_clf_with_weights.predict_proba(X)
    y_pred_without_weights = calibrated_clf_without_weights.predict_proba(X)
    assert_allclose(y_pred_with_weights, y_pred_without_weights)

@pytest.mark.parametrize('fit_params_type', ['list', 'array'])
def test_calibration_with_fit_params(fit_params_type, data):
    if False:
        return 10
    'Tests that fit_params are passed to the underlying base estimator.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/12384\n    '
    (X, y) = data
    fit_params = {'a': _convert_container(y, fit_params_type), 'b': _convert_container(y, fit_params_type)}
    clf = CheckingClassifier(expected_fit_params=['a', 'b'])
    pc_clf = CalibratedClassifierCV(clf)
    pc_clf.fit(X, y, **fit_params)

@pytest.mark.parametrize('sample_weight', [[1.0] * N_SAMPLES, np.ones(N_SAMPLES)])
def test_calibration_with_sample_weight_base_estimator(sample_weight, data):
    if False:
        return 10
    'Tests that sample_weight is passed to the underlying base\n    estimator.\n    '
    (X, y) = data
    clf = CheckingClassifier(expected_sample_weight=True)
    pc_clf = CalibratedClassifierCV(clf)
    pc_clf.fit(X, y, sample_weight=sample_weight)

def test_calibration_without_sample_weight_base_estimator(data):
    if False:
        while True:
            i = 10
    "Check that even if the estimator doesn't support\n    sample_weight, fitting with sample_weight still works.\n\n    There should be a warning, since the sample_weight is not passed\n    on to the estimator.\n    "
    (X, y) = data
    sample_weight = np.ones_like(y)

    class ClfWithoutSampleWeight(CheckingClassifier):

        def fit(self, X, y, **fit_params):
            if False:
                while True:
                    i = 10
            assert 'sample_weight' not in fit_params
            return super().fit(X, y, **fit_params)
    clf = ClfWithoutSampleWeight()
    pc_clf = CalibratedClassifierCV(clf)
    with pytest.warns(UserWarning):
        pc_clf.fit(X, y, sample_weight=sample_weight)

@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_calibrated_classifier_cv_zeros_sample_weights_equivalence(method, ensemble):
    if False:
        return 10
    'Check that passing removing some sample from the dataset `X` is\n    equivalent to passing a `sample_weight` with a factor 0.'
    (X, y) = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X = np.vstack((X[:40], X[50:90]))
    y = np.hstack((y[:40], y[50:90]))
    sample_weight = np.zeros_like(y)
    sample_weight[::2] = 1
    estimator = LogisticRegression()
    calibrated_clf_without_weights = CalibratedClassifierCV(estimator, method=method, ensemble=ensemble, cv=2)
    calibrated_clf_with_weights = clone(calibrated_clf_without_weights)
    calibrated_clf_with_weights.fit(X, y, sample_weight=sample_weight)
    calibrated_clf_without_weights.fit(X[::2], y[::2])
    for (est_with_weights, est_without_weights) in zip(calibrated_clf_with_weights.calibrated_classifiers_, calibrated_clf_without_weights.calibrated_classifiers_):
        assert_allclose(est_with_weights.estimator.coef_, est_without_weights.estimator.coef_)
    y_pred_with_weights = calibrated_clf_with_weights.predict_proba(X)
    y_pred_without_weights = calibrated_clf_without_weights.predict_proba(X)
    assert_allclose(y_pred_with_weights, y_pred_without_weights)

def test_calibrated_classifier_error_base_estimator(data):
    if False:
        while True:
            i = 10
    'Check that we raise an error is a user set both `base_estimator` and\n    `estimator`.'
    calibrated_classifier = CalibratedClassifierCV(base_estimator=LogisticRegression(), estimator=LogisticRegression())
    with pytest.raises(ValueError, match='Both `base_estimator` and `estimator`'):
        calibrated_classifier.fit(*data)

def test_calibrated_classifier_deprecation_base_estimator(data):
    if False:
        while True:
            i = 10
    'Check that we raise a warning regarding the deprecation of\n    `base_estimator`.'
    calibrated_classifier = CalibratedClassifierCV(base_estimator=LogisticRegression())
    warn_msg = '`base_estimator` was renamed to `estimator`'
    with pytest.warns(FutureWarning, match=warn_msg):
        calibrated_classifier.fit(*data)

def test_calibration_with_non_sample_aligned_fit_param(data):
    if False:
        for i in range(10):
            print('nop')
    'Check that CalibratedClassifierCV does not enforce sample alignment\n    for fit parameters.'

    class TestClassifier(LogisticRegression):

        def fit(self, X, y, sample_weight=None, fit_param=None):
            if False:
                print('Hello World!')
            assert fit_param is not None
            return super().fit(X, y, sample_weight=sample_weight)
    CalibratedClassifierCV(estimator=TestClassifier()).fit(*data, fit_param=np.ones(len(data[1]) + 1))

def test_calibrated_classifier_cv_works_with_large_confidence_scores(global_random_seed):
    if False:
        return 10
    'Test that :class:`CalibratedClassifierCV` works with large confidence\n    scores when using the `sigmoid` method, particularly with the\n    :class:`SGDClassifier`.\n\n    Non-regression test for issue #26766.\n    '
    prob = 0.67
    n = 1000
    random_noise = np.random.default_rng(global_random_seed).normal(size=n)
    y = np.array([1] * int(n * prob) + [0] * (n - int(n * prob)))
    X = 100000.0 * y.reshape((-1, 1)) + random_noise
    cv = check_cv(cv=None, y=y, classifier=True)
    indices = cv.split(X, y)
    for (train, test) in indices:
        (X_train, y_train) = (X[train], y[train])
        X_test = X[test]
        sgd_clf = SGDClassifier(loss='squared_hinge', random_state=global_random_seed)
        sgd_clf.fit(X_train, y_train)
        predictions = sgd_clf.decision_function(X_test)
        assert (predictions > 10000.0).any()
    clf_sigmoid = CalibratedClassifierCV(SGDClassifier(loss='squared_hinge', random_state=global_random_seed), method='sigmoid')
    score_sigmoid = cross_val_score(clf_sigmoid, X, y, scoring='roc_auc')
    clf_isotonic = CalibratedClassifierCV(SGDClassifier(loss='squared_hinge', random_state=global_random_seed), method='isotonic')
    score_isotonic = cross_val_score(clf_isotonic, X, y, scoring='roc_auc')
    assert_allclose(score_sigmoid, score_isotonic)

def test_sigmoid_calibration_max_abs_prediction_threshold(global_random_seed):
    if False:
        print('Hello World!')
    random_state = np.random.RandomState(seed=global_random_seed)
    n = 100
    y = random_state.randint(0, 2, size=n)
    predictions_small = random_state.uniform(low=-2, high=2, size=100)
    threshold_1 = 0.1
    (a1, b1) = _sigmoid_calibration(predictions=predictions_small, y=y, max_abs_prediction_threshold=threshold_1)
    threshold_2 = 10
    (a2, b2) = _sigmoid_calibration(predictions=predictions_small, y=y, max_abs_prediction_threshold=threshold_2)
    (a3, b3) = _sigmoid_calibration(predictions=predictions_small, y=y)
    atol = 1e-06
    assert_allclose(a1, a2, atol=atol)
    assert_allclose(a2, a3, atol=atol)
    assert_allclose(b1, b2, atol=atol)
    assert_allclose(b2, b3, atol=atol)