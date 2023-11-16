"""
Test the prepredict estimator.
"""
import pytest
from io import BytesIO
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification, make_regression, make_blobs
from yellowbrick.contrib.prepredict import *
from yellowbrick.regressor import PredictionError
from yellowbrick.classifier import ClassificationReport
import numpy as np
np.random.seed()

@pytest.fixture(scope='class')
def multiclass(request):
    if False:
        return 10
    '\n    Creates a random multiclass classification dataset fixture\n    '
    (X, y) = make_classification(n_samples=500, n_features=20, n_informative=8, n_redundant=2, n_classes=6, n_clusters_per_class=3, random_state=87)
    (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=93)
    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.multiclass = dataset

@pytest.fixture(scope='class')
def continuous(request):
    if False:
        return 10
    '\n    Creates a random continuous regression dataset fixture\n    '
    (X, y) = make_regression(n_samples=500, n_features=22, n_informative=8, random_state=42, noise=0.2, bias=0.2)
    (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=11)
    request.cls.continuous = Dataset(Split(X_train, X_test), Split(y_train, y_test))

@pytest.fixture(scope='class')
def blobs(request):
    if False:
        while True:
            i = 10
    '\n    Create a random blobs clustering dataset fixture\n    '
    (X, y) = make_blobs(n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42)
    request.cls.blobs = Dataset(X, y)

@pytest.mark.usefixtures('multiclass')
@pytest.mark.usefixtures('continuous')
@pytest.mark.usefixtures('blobs')
class TestPrePrePredictEstimator(VisualTestCase):
    """
    Pre-predict contrib tests.
    """

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='image comparison failure on Conda 3.8 and 3.9 with RMS 19.307')
    def test_prepredict_classifier(self):
        if False:
            while True:
                i = 10
        '\n        Test the prepredict estimator with classification report\n        '
        (X, y) = (self.multiclass.X, self.multiclass.y)
        y_pred = GaussianNB().fit(X.train, y.train).predict(X.test)
        estimator = PrePredict(y_pred, CLASSIFIER)
        assert estimator.fit(X.train, y.train) is estimator
        assert estimator.predict(X.train) is y_pred
        assert estimator.score(X.test, y.test) == pytest.approx(0.41, rel=0.001)
        viz = ClassificationReport(estimator)
        viz.fit(None, y.train)
        viz.score(None, y.test)
        viz.finalize()
        self.assert_images_similar(viz)

    def test_prepredict_regressor(self):
        if False:
            return 10
        '\n        Test the prepredict estimator with a prediction error plot\n        '
        (X, y) = (self.continuous.X, self.continuous.y)
        y_pred = LinearRegression().fit(X.train, y.train).predict(X.test)
        estimator = PrePredict(y_pred, REGRESSOR)
        assert estimator.fit(X.train, y.train) is estimator
        assert estimator.predict(X.train) is y_pred
        assert estimator.score(X.test, y.test) == pytest.approx(0.9999983124154966, rel=0.01)
        viz = PredictionError(estimator)
        viz.fit(X.train, y.train)
        viz.score(X.test, y.test)
        viz.finalize()
        self.assert_images_similar(viz, tol=10.0)

    def test_prepredict_clusterer(self):
        if False:
            while True:
                i = 10
        '\n        Test the prepredict estimator with a silhouette visualizer\n        '
        X = self.blobs.X
        y_pred = MiniBatchKMeans(random_state=831).fit(X).predict(X)
        estimator = PrePredict(y_pred, CLUSTERER)
        assert estimator.fit(X) is estimator
        assert estimator.predict(X) is y_pred
        assert estimator.score(X) == pytest.approx(0.5477478541994333, rel=0.01)

    def test_load(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the various ways that prepredict loads data\n        '
        ppe = PrePredict(lambda : self.multiclass.y.test)
        assert ppe._load() is self.multiclass.y.test
        f = BytesIO()
        np.save(f, self.continuous.y.test)
        f.seek(0)
        ppe = PrePredict(f)
        assert np.array_equal(ppe._load(), self.continuous.y.test)