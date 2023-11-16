"""
Tests for the KElbowVisualizer
"""
import sys
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, csr_matrix
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tests.fixtures import Dataset
from tests.base import VisualTestCase
from yellowbrick.datasets import load_hobbies
from yellowbrick.cluster.elbow import distortion_score
from yellowbrick.cluster.elbow import KElbowVisualizer, kelbow_visualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning
from tests.base import IS_WINDOWS_OR_CONDA
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.fixture(scope='class')
def clusters(request):
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[-0.40020753, -4.67055317, -0.27191127, -1.49156318], [0.37143349, -4.89391622, -1.23893945, 0.48318165], [8.625142, -1.2372284, 1.39301471, 4.3394457], [7.65803596, -2.21017215, 1.99175714, 3.71004654], [0.89319875, -5.37152317, 1.50313598, 1.95284886], [2.68362166, -5.78810913, -0.41233406, 1.94638989], [7.63541182, -1.99606076, 0.9241231, 4.53478238], [9.04699415, -0.74540679, 0.98042851, 5.99569071], [1.02552122, -5.73874278, -1.74804915, -0.07831216], [7.18135665, -3.49473178, 1.14300963, 4.46065816], [0.58812902, -4.66559815, -0.72831685, 1.40171779], [1.48620862, -5.9963108, 0.19145963, -1.11369256], [7.6625556, -1.21328083, 2.06361094, 6.2643551], [9.45050727, -1.36536078, 1.31154384, 3.89103468], [6.88203724, -1.62040255, 3.89961049, 2.12865388], [5.60842705, -2.10693356, 1.93328514, 3.90825432], [2.35150936, -6.62836131, -1.84278374, 0.51540886], [1.17446451, -5.62506058, -2.18420699, 1.21385128]])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    request.cls.clusters = Dataset(X, y)

@pytest.mark.usefixtures('clusters')
class TestKElbowHelper(object):
    """
    Helper functions for K-Elbow Visualizer
    """

    def test_distortion_score(self):
        if False:
            while True:
                i = 10
        '\n        Test the distortion score metric function\n        '
        score = distortion_score(self.clusters.X, self.clusters.y)
        assert score == pytest.approx(69.10006514142941)

    @pytest.mark.parametrize('func', [csc_matrix, csr_matrix], ids=['csc', 'csr'])
    def test_distortion_score_sparse_matrix_input(self, func):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the distortion score metric on a sparse array\n        '
        score = distortion_score(func(self.clusters.X), self.clusters.y)
        assert score == pytest.approx(69.10006514142938)

    @pytest.mark.skipif(pd is None, reason='pandas is required')
    def test_distortion_score_pandas_input(self):
        if False:
            while True:
                i = 10
        '\n        Test the distortion score metric on pandas DataFrame and Series\n        '
        df = pd.DataFrame(self.clusters.X)
        s = pd.Series(self.clusters.y)
        score = distortion_score(df, s)
        assert score == pytest.approx(69.10006514142941)

    def test_distortion_score_empty_clusters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure no ValueError is thrown when there are empty clusters #1185\n        '
        X = np.array([[1, 2], [3, 4], [5, 6]])
        valuea = distortion_score(X, np.array([1, 3, 3]))
        valueb = distortion_score(X, np.array([0, 1, 1]))
        assert valuea == valueb

@pytest.mark.usefixtures('clusters')
class TestKElbowVisualizer(VisualTestCase):
    """
    K-Elbow Visualizer Tests
    """

    @pytest.mark.xfail(reason='images not close due to timing lines')
    @pytest.mark.filterwarnings("ignore:No 'knee'")
    def test_integrated_kmeans_elbow(self):
        if False:
            print('Hello World!')
        '\n        Test no exceptions for kmeans k-elbow visualizer on blobs dataset\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42)
        try:
            (_, ax) = plt.subplots()
            visualizer = KElbowVisualizer(KMeans(random_state=42), k=4, ax=ax)
            visualizer.fit(X)
            visualizer.finalize()
            self.assert_images_similar(visualizer)
        except Exception as e:
            pytest.fail('error during k-elbow: {}'.format(e))

    @pytest.mark.xfail(reason='images not close due to timing lines')
    @pytest.mark.filterwarnings("ignore:No 'knee'")
    def test_integrated_mini_batch_kmeans_elbow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test no exceptions for mini-batch kmeans k-elbow visualizer\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42)
        try:
            (_, ax) = plt.subplots()
            visualizer = KElbowVisualizer(MiniBatchKMeans(random_state=42), k=4, ax=ax)
            visualizer.fit(X)
            visualizer.finalize()
            self.assert_images_similar(visualizer)
        except Exception as e:
            pytest.fail('error during k-elbow: {}'.format(e))

    @pytest.mark.skip(reason='takes over 20 seconds to run')
    def test_topic_modeling_k_means(self):
        if False:
            return 10
        '\n        Test topic modeling k-means on the hobbies corpus\n        '
        corpus = load_hobbies()
        tfidf = TfidfVectorizer()
        docs = tfidf.fit_transform(corpus.data)
        visualizer = KElbowVisualizer(KMeans(), k=(4, 8))
        visualizer.fit(docs)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_invalid_k(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert that invalid values of K raise exceptions\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42)
        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k=(1, 2, 3, 'foo', 5)).fit(X)
        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k='foo').fit(X)

    def test_valid_k(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert that valid values of K generate correct k_values_\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42)
        visualizer = KElbowVisualizer(KMeans(), k=8).fit(X)
        assert visualizer.k_values_ == list(np.arange(2, 8 + 1))
        visualizer = KElbowVisualizer(KMeans(), k=(4, 12)).fit(X)
        assert visualizer.k_values_ == list(np.arange(4, 12))
        visualizer = KElbowVisualizer(KMeans(), k=np.arange(10, 100, 10)).fit(X)
        assert visualizer.k_values_ == list(np.arange(10, 100, 10))
        visualizer = KElbowVisualizer(KMeans(), k=[10, 20, 30, 40, 50, 60, 70, 80, 90]).fit(X)
        assert visualizer.k_values_ == list(np.arange(10, 100, 10))

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_distortion_metric(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the distortion metric of the k-elbow visualizer\n        '
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=5, metric='distortion', timings=False, locate_elbow=False)
        visualizer.fit(self.clusters.X)
        expected = np.array([69.100065, 54.081571, 43.146921, 34.978487])
        assert len(visualizer.k_scores_) == 4
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.03)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_silhouette_metric(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the silhouette metric of the k-elbow visualizer\n        '
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=5, metric='silhouette', timings=False, locate_elbow=False)
        visualizer.fit(self.clusters.X)
        expected = np.array([0.691636, 0.456646, 0.255174, 0.239842])
        assert len(visualizer.k_scores_) == 4
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_calinski_harabasz_metric(self):
        if False:
            return 10
        '\n        Test the calinski-harabasz metric of the k-elbow visualizer\n        '
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=5, metric='calinski_harabasz', timings=False, locate_elbow=False)
        visualizer.fit(self.clusters.X)
        assert len(visualizer.k_scores_) == 4
        assert visualizer.elbow_value_ is None
        expected = np.array([81.662726, 50.992378, 40.952179, 35.939494])
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_distance_metric(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the manhattan distance metric of the distortion metric of the k-elbow visualizer\n        '
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=5, metric='distortion', distance_metric='manhattan', timings=False, locate_elbow=False)
        visualizer.fit(self.clusters.X)
        assert len(visualizer.k_scores_) == 4
        assert visualizer.elbow_value_ is None
        expected = np.array([189.060129, 154.096223, 124.271208, 107.087566])
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='computation of k_scores_ varies by 2.867 max absolute difference')
    def test_locate_elbow(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the addition of locate_elbow to an image\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=5, centers=3, shuffle=True, random_state=42)
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=6, metric='calinski_harabasz', timings=False, locate_elbow=True)
        visualizer.fit(X)
        assert len(visualizer.k_scores_) == 5
        assert visualizer.elbow_value_ == 3
        expected = np.array([4286.5, 12463.4, 8766.8, 6950.1, 5863.6])
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5, windows_tol=2.2)
        assert_array_almost_equal(visualizer.k_scores_, expected, decimal=1)

    def test_no_knee(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert that a warning is issued if there is no knee detected\n        '
        (X, y) = make_blobs(n_samples=1000, centers=3, n_features=12, random_state=12)
        message = "No 'knee' or 'elbow point' detected This could be due to bad clustering, no actual clusters being formed etc."
        with pytest.warns(YellowbrickWarning, match=message):
            visualizer = KElbowVisualizer(KMeans(random_state=12), k=(4, 12), locate_elbow=True)
            visualizer.fit(X)

    def test_bad_metric(self):
        if False:
            print('Hello World!')
        '\n        Assert KElbow raises an exception when a bad metric is supplied\n        '
        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k=5, metric='foo')

    def test_bad_distance_metric(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert KElbow raises an exception when a bad distance metric is supplied\n        '
        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k=5, distance_metric='foo')

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_timings(self):
        if False:
            while True:
                i = 10
        '\n        Test the twinx double axes with k-elbow timings\n        '
        visualizer = KElbowVisualizer(KMeans(random_state=0), k=5, timings=True, locate_elbow=False)
        visualizer.fit(self.clusters.X)
        assert len(visualizer.k_timers_) == 4
        assert all([t > 0 for t in visualizer.k_timers_])
        assert hasattr(visualizer, 'axes')
        assert len(visualizer.axes) == 2
        visualizer.axes[1].remove()
        visualizer.k_timers_ = [0.01084589958190918, 0.011144161224365234, 0.017028093338012695, 0.010634183883666992]
        visualizer.k_values_ = [2, 3, 4, 5]
        visualizer.draw()
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_sample_weights(self):
        if False:
            i = 10
            return i + 15
        "\n        Test that passing in sample weights correctly influences the clusterer's fit\n        "
        seed = 1234
        (X, y) = make_blobs(n_samples=[5, 30, 30, 30, 30], n_features=5, random_state=seed, shuffle=False)
        visualizer = KElbowVisualizer(KMeans(random_state=seed), k=(2, 12), timings=False)
        visualizer.fit(X)
        assert visualizer.elbow_value_ == 5
        weights = np.concatenate([np.ones(5) * 0.0001, np.ones(120)])
        visualizer.fit(X, sample_weight=weights)
        assert visualizer.elbow_value_ == 4

    @pytest.mark.xfail(reason='images not close due to timing lines')
    def test_quick_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the quick method producing a valid visualization\n        '
        (X, y) = make_blobs(n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=2)
        model = MiniBatchKMeans(3, random_state=43)
        oz = kelbow_visualizer(model, X, show=False)
        assert isinstance(oz, KElbowVisualizer)
        self.assert_images_similar(oz)

    def test_quick_method_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the quick method correctly consumes the user-provided parameters\n        '
        (X, y) = make_blobs(centers=3)
        custom_title = 'My custom title'
        model = KMeans(3, random_state=13)
        oz = kelbow_visualizer(model, X, sample_weight=np.ones(X.shape[0]), title=custom_title, show=False)
        assert oz.title == custom_title

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows')
    def test_set_colors_manually(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the silhouette metric of the k-elbow visualizer\n        '
        oz = KElbowVisualizer(KMeans(random_state=0), k=5)
        oz.metric_color = 'r'
        oz.timing_color = 'y'
        oz.vline_color = 'c'
        oz.k_values_ = [1, 2, 3, 4, 5, 6, 7, 8]
        oz.k_timers_ = [6.2, 8.3, 10.1, 15.8, 21.2, 27.9, 38.2, 44.9]
        oz.k_scores_ = [0.8, 0.7, 0.55, 0.48, 0.4, 0.38, 0.35, 0.3]
        oz.elbow_value_ = 5
        oz.elbow_score_ = 0.4
        oz.draw()
        oz.finalize()
        self.assert_images_similar(oz, tol=3.2)

    def test_get_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure the get params works for sklearn-compatibility\n        '
        oz = KElbowVisualizer(KMeans(random_state=0), k=5)
        params = oz.get_params()
        assert len(params) > 0