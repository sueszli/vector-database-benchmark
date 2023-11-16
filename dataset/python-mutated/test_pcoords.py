"""
Testing for the parallel coordinates feature visualizers
"""
import pytest
import numpy as np
from yellowbrick.datasets import load_occupancy
from yellowbrick.features.pcoords import *
from tests.base import VisualTestCase
from ..fixtures import Dataset
from sklearn.datasets import make_classification
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.fixture(scope='class')
def dataset(request):
    if False:
        print('Hello World!')
    '\n    Creates a random multiclass classification dataset fixture\n    '
    (X, y) = make_classification(n_samples=200, n_features=5, n_informative=4, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=451, flip_y=0, class_sep=3, scale=np.array([1.0, 2.0, 100.0, 20.0, 1.0]))
    dataset = Dataset(X, y)
    request.cls.dataset = dataset

@pytest.mark.usefixtures('dataset')
class TestParallelCoordinates(VisualTestCase):
    """
    Test the ParallelCoordinates visualizer
    """

    def test_parallel_coords(self):
        if False:
            return 10
        '\n        Test images closeness on random 3 class dataset\n        '
        visualizer = ParallelCoordinates()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_parallel_coords_fast(self):
        if False:
            print('Hello World!')
        '\n        Test images closeness on random 3 class dataset in fast mode\n        '
        visualizer = ParallelCoordinates(fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_alpha(self):
        if False:
            print('Hello World!')
        '\n        Test image closeness on opaque alpha for random 3 class dataset\n        '
        visualizer = ParallelCoordinates(alpha=1.0)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_alpha_fast(self):
        if False:
            return 10
        '\n        Test image closeness on opaque alpha for random 3 class dataset in fast mode\n        '
        visualizer = ParallelCoordinates(alpha=1.0, fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_labels(self):
        if False:
            i = 10
            return i + 15
        '\n        Test image closeness when class and feature labels are supplied\n        '
        visualizer = ParallelCoordinates(classes=['a', 'b', 'c'], features=['f1', 'f2', 'f3', 'f4', 'f5'])
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_labels_fast(self):
        if False:
            return 10
        '\n        Test image closeness when class and feature labels are supplied in fast mode\n        '
        visualizer = ParallelCoordinates(classes=['a', 'b', 'c'], features=['f1', 'f2', 'f3', 'f4', 'f5'], fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_normalized_l2(self):
        if False:
            while True:
                i = 10
        '\n        Test image closeness on l2 normalized 3 class dataset\n        '
        visualizer = ParallelCoordinates(normalize='l2')
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_l2_fast(self):
        if False:
            i = 10
            return i + 15
        '\n        Test image closeness on l2 normalized 3 class dataset in fast mode\n        '
        visualizer = ParallelCoordinates(normalize='l2', fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_minmax(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test image closeness on minmax normalized 3 class dataset\n        '
        visualizer = ParallelCoordinates(normalize='minmax')
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_minmax_fast(self):
        if False:
            print('Hello World!')
        '\n        Test image closeness on minmax normalized 3 class dataset in fast mode\n        '
        visualizer = ParallelCoordinates(normalize='minmax', fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_parallel_coordinates_quickmethod(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the quick method producing a valid visualization\n        '
        (X, y) = load_occupancy(return_dataset=True).to_numpy()
        visualizer = parallel_coordinates(X, y, sample=100, show=False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration_sampled(self):
        if False:
            return 10
        '\n        Test on a real dataset with pandas DataFrame and Series sampled for speed\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        oz = ParallelCoordinates(sample=0.05, shuffle=True, random_state=4291, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    def test_numpy_integration_sampled(self):
        if False:
            print('Hello World!')
        '\n        Ensure visualizer works in default case with numpy arrays and sampling\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        oz = ParallelCoordinates(sample=0.05, shuffle=True, random_state=4291, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_integration_fast(self):
        if False:
            return 10
        '\n        Test on a real dataset with pandas DataFrame and Series in fast mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        oz = ParallelCoordinates(fast=True, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    def test_numpy_integration_fast(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure visualizer works in default case with numpy arrays and fast mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        classes = [k for (k, _) in sorted(data.meta['labels'].items(), key=lambda i: i[1])]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        oz = ParallelCoordinates(fast=True, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    def test_normalized_invalid_arg(self):
        if False:
            i = 10
            return i + 15
        "\n        Invalid argument to 'normalize' should raise\n        "
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(normalize='foo')

    def test_sample_int(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Assert no errors occur using integer 'sample' argument\n        "
        visualizer = ParallelCoordinates(sample=10)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_shuffle(self):
        if False:
            while True:
                i = 10
        "\n        Assert no errors occur using integer 'sample' argument and shuffle, with different random_state args\n        "
        visualizer = ParallelCoordinates(sample=3, shuffle=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=3, shuffle=True, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=3, shuffle=True, random_state=np.random.RandomState())
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_shuffle_false(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Assert no errors occur using integer 'sample' argument and shuffle, with different random_state args\n        "
        visualizer = ParallelCoordinates(sample=3, shuffle=False)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=3, shuffle=False, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=3, shuffle=False, random_state=np.random.RandomState())
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_invalid(self):
        if False:
            i = 10
            return i + 15
        '\n        Negative int values should raise exception\n        '
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=-1)

    def test_sample_float(self):
        if False:
            while True:
                i = 10
        "\n        Assert no errors occur using float 'sample' argument\n        "
        visualizer = ParallelCoordinates(sample=0.5)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_shuffle(self):
        if False:
            return 10
        "\n        Assert no errors occur using float 'sample' argument and shuffle, with different random_state args\n        "
        visualizer = ParallelCoordinates(sample=0.5, shuffle=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=0.5, shuffle=True, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=0.5, shuffle=True, random_state=np.random.RandomState())
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_shuffle_false(self):
        if False:
            return 10
        "\n        Assert no errors occur using float 'sample' argument and shuffle, with different random_state args\n        "
        visualizer = ParallelCoordinates(sample=0.5, shuffle=False)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=0.5, shuffle=False, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer = ParallelCoordinates(sample=0.5, shuffle=False, random_state=np.random.RandomState())
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_invalid(self):
        if False:
            return 10
        "\n        Float values for 'sample' argument outside [0,1] should raise.\n        "
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=-0.2)
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=1.1)

    def test_sample_invalid_type(self):
        if False:
            return 10
        "\n        Non-numeric values for 'sample' argument should raise.\n        "
        with pytest.raises(YellowbrickTypeError):
            ParallelCoordinates(sample='foo')

    @staticmethod
    def test_static_subsample():
        if False:
            i = 10
            return i + 15
        '\n        Assert output of subsampling method against expectations\n        '
        ntotal = 100
        ncols = 50
        y = np.arange(ntotal)
        X = np.ones((ntotal, ncols)) * y.reshape(ntotal, 1)
        visualizer = ParallelCoordinates(sample=1.0, random_state=None, shuffle=False)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X)
        assert np.array_equal(yprime, y)
        visualizer = ParallelCoordinates(sample=200, random_state=None, shuffle=False)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X)
        assert np.array_equal(yprime, y)
        sample = 50
        visualizer = ParallelCoordinates(sample=sample, random_state=None, shuffle=False)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[:sample, :])
        assert np.array_equal(yprime, y[:sample])
        sample = 50
        visualizer = ParallelCoordinates(sample=sample, random_state=None, shuffle=True)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == sample
        assert len(yprime) == sample
        visualizer = ParallelCoordinates(sample=0.5, random_state=None, shuffle=False)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[:int(ntotal / 2), :])
        assert np.array_equal(yprime, y[:int(ntotal / 2)])
        sample = 0.5
        visualizer = ParallelCoordinates(sample=sample, random_state=None, shuffle=True)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample
        sample = 0.25
        visualizer = ParallelCoordinates(sample=sample, random_state=444, shuffle=True)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample
        sample = 0.99
        visualizer = ParallelCoordinates(sample=sample, random_state=np.random.RandomState(), shuffle=True)
        (Xprime, yprime) = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample