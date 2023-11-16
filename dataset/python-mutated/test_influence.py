"""
Test the regressor influence visualizers.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from tests.base import VisualTestCase
from tests.fixtures import Dataset
from sklearn.datasets import make_regression
from yellowbrick.regressor.influence import *
from yellowbrick.datasets import load_concrete
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.fixture(scope='class')
def data(request):
    if False:
        print('Hello World!')
    '\n    Creates a random regression fixture that has a R2 score below 0.85 and several\n    outliers that best demonstrate the effectiveness of influence visualizers.\n    '
    (X, y) = make_regression(n_samples=100, n_features=14, n_informative=6, bias=1.2, noise=49.8, tail_strength=0.6, random_state=637)
    request.cls.data = Dataset(X, y)
LEARNED_FIELDS = ('distance_', 'p_values_', 'influence_threshold_', 'outlier_percentage_')

def assert_not_fitted(oz):
    if False:
        while True:
            i = 10
    for field in LEARNED_FIELDS:
        assert not hasattr(oz, field)

def assert_fitted(oz):
    if False:
        while True:
            i = 10
    for field in LEARNED_FIELDS:
        assert hasattr(oz, field)

@pytest.mark.usefixtures('data')
class TestCooksDistance(VisualTestCase):
    """
    CooksDistance visual test cases
    """

    def test_cooks_distance(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test image similarity of Cook's Distance on a random dataset\n        "
        (_, ax) = plt.subplots()
        viz = CooksDistance(ax=ax)
        assert_not_fitted(viz)
        assert viz.fit(self.data.X, self.data.y) is viz
        assert_fitted(viz)
        assert viz.distance_.shape == (self.data.X.shape[0],)
        assert viz.p_values_.shape == viz.distance_.shape
        assert 0.0 <= viz.influence_threshold_ <= 4.0
        assert 0.0 <= viz.outlier_percentage_ <= 100.0
        self.assert_images_similar(viz)

    def test_cooks_distance_quickmethod(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the cooks_distance quick method on a random dataset\n        '
        (_, ax) = plt.subplots()
        viz = cooks_distance(self.data.X, self.data.y, ax=ax, draw_threshold=False, linefmt='r-', markerfmt='ro', show=False)
        assert_fitted(viz)
        self.assert_images_similar(viz)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test on the concrete dataset with pandas DataFrame and Series\n        '
        data = load_concrete(return_dataset=True)
        (X, y) = data.to_pandas()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        (_, ax) = plt.subplots()
        viz = CooksDistance(ax=ax).fit(X, y)
        assert_fitted(viz)
        assert viz.distance_.sum() == pytest.approx(1.2911900571300652)
        assert viz.p_values_.sum() == pytest.approx(1029.9999525376425)
        assert viz.influence_threshold_ == pytest.approx(0.003883495145631068)
        assert viz.outlier_percentage_ == pytest.approx(7.3786407766990285)
        viz.finalize()
        self.assert_images_similar(viz)

    def test_numpy_integration(self):
        if False:
            return 10
        '\n        Test on concrete dataset with numpy arrays\n        '
        data = load_concrete(return_dataset=True)
        (X, y) = data.to_numpy()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        (_, ax) = plt.subplots()
        viz = CooksDistance(ax=ax).fit(X, y)
        assert_fitted(viz)
        assert viz.distance_.sum() == pytest.approx(1.2911900571300652)
        assert viz.p_values_.sum() == pytest.approx(1029.9999525376425)
        assert viz.influence_threshold_ == pytest.approx(0.003883495145631068)
        assert viz.outlier_percentage_ == pytest.approx(7.3786407766990285)
        viz.finalize()
        self.assert_images_similar(viz)