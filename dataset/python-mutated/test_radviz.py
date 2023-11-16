"""
Test the RadViz feature analysis visualizers
"""
import sys
import pytest
import numpy.testing as npt
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from ..fixtures import Dataset
from sklearn.datasets import make_classification
from yellowbrick.datasets import load_occupancy
from yellowbrick.features.radviz import *
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.fixture(scope='class')
def dataset(request):
    if False:
        return 10
    '\n    Creates a random multiclass classification dataset fixture\n    '
    (X, y) = make_classification(n_samples=200, n_features=5, n_informative=4, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=451, flip_y=0, class_sep=3, scale=np.array([1.0, 2.0, 100.0, 20.0, 1.0]))
    dataset = Dataset(X, y)
    request.cls.dataset = dataset

@pytest.mark.usefixtures('dataset')
class TestRadViz(VisualTestCase):
    """
    Test the RadViz visualizer
    """

    def test_normalize_x(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the static normalization method on the RadViz class\n        '
        X = np.array([[2.318, 2.727, 4.26, 7.212, 4.792], [2.315, 2.726, 4.295, 7.14, 4.783], [2.315, 2.724, 4.26, 7.135, 4.779], [2.11, 3.609, 4.33, 7.985, 5.595], [2.11, 3.626, 4.33, 8.203, 5.621], [2.11, 3.62, 4.47, 8.21, 5.612]])
        Xe = np.array([[1.0, 0.00332594, 0.0, 0.07162791, 0.01543943], [0.98557692, 0.00221729, 0.16666667, 0.00465116, 0.00475059], [0.98557692, 0.0, 0.0, 0.0, 0.0], [0.0, 0.98115299, 0.33333333, 0.79069767, 0.96912114], [0.0, 1.0, 0.33333333, 0.99348837, 1.0], [0.0, 0.99334812, 1.0, 1.0, 0.98931116]])
        Xp = RadViz.normalize(X)
        npt.assert_array_almost_equal(Xp, Xe)

    def test_radviz(self):
        if False:
            while True:
                i = 10
        '\n        Assert image similarity on test dataset\n        '
        visualizer = RadViz()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_radviz_alpha(self):
        if False:
            while True:
                i = 10
        '\n        Assert image similarity with alpha transparency\n        '
        visualizer = RadViz(alpha=0.5)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_integrated_radviz_with_pandas(self):
        if False:
            i = 10
            return i + 15
        '\n        Test RadViz with Pandas on the occupancy dataset\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        visualizer = RadViz()
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_integrated_radviz_with_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test RadViz with numpy on the occupancy dataset\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        visualizer = RadViz()
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_integrated_radviz_pandas_classes_features(self):
        if False:
            while True:
                i = 10
        '\n        Test RadViz with classes and features specified using Pandas\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        features = ['temperature', 'relative humidity', 'light']
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        X = X[features]
        y = y.astype(int)
        visualizer = RadViz(features=features, classes=classes)
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_integrated_radviz_numpy_classes_features(self):
        if False:
            print('Hello World!')
        '\n        Test RadViz with classes and features specified using numpy\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        features = data.meta['features'][0:3]
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        X = X[:, :3]
        y = y.astype(int)
        visualizer = RadViz(features=features, classes=classes)
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1)

    def test_radviz_quick_method(self):
        if False:
            return 10
        '\n        Test RadViz quick method with colors being set.\n        '
        visualizer = radviz(*self.dataset, colors=['cyan', 'magenta', 'yellow'], show=False)
        self.assert_images_similar(visualizer)