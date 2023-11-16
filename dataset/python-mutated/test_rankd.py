"""
Test the Rankd feature analysis visualizers
"""
import pytest
import numpy as np
import numpy.testing as npt
from yellowbrick.features.rankd import RankDBase
from yellowbrick.features.rankd import kendalltau
from yellowbrick.features.rankd import Rank1D, rank1d
from yellowbrick.features.rankd import Rank2D, rank2d
from yellowbrick.exceptions import YellowbrickValueError
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from yellowbrick.datasets import load_occupancy, load_credit, load_energy
try:
    import pandas as pd
except ImportError:
    pd = None

class TestKendallTau(object):
    """
    Test the Kendall-Tau correlation metric
    """

    def test_kendalltau(self):
        if False:
            while True:
                i = 10
        '\n        Test results returned match expectations\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        expected = np.array([[1.0, -1.0, -0.2724275, -0.7361443, 0.7385489, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.2724275, 0.7361443, -0.7385489, 0.0, 0.0, 0.0], [-0.2724275, 0.2724275, 1.0, -0.15192004, 0.19528337, 0.0, 0.0, 0.0], [-0.73614431, 0.73614431, -0.15192004, 1.0, -0.87518995, 0.0, 0.0, 0.0], [0.73854895, -0.73854895, 0.19528337, -0.87518995, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.15430335], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15430335, 1.0]])
        actual = kendalltau(X)
        npt.assert_almost_equal(expected, actual)

    def test_kendalltau_shape(self):
        if False:
            print('Hello World!')
        '\n        Assert that a square correlation matrix is returned\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        corr = kendalltau(X)
        assert corr.shape[0] == corr.shape[1]
        for ((i, j), val) in np.ndenumerate(corr):
            assert corr[j][i] == pytest.approx(val)

    def test_kendalltau_1D(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that a 2D matrix is required as input\n        '
        with pytest.raises(IndexError, match='tuple index out of range'):
            X = 0.1 * np.arange(10)
            kendalltau(X)

class TestRankDBase(VisualTestCase):
    """
    Test the RankDBase Visualizer
    """

    def test_rankdbase_unknown_algorithm(self):
        if False:
            return 10
        '\n        Assert that unknown algorithms raise an exception\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        with pytest.raises(YellowbrickValueError, match='.* is unrecognized ranking method') as e:
            oz = RankDBase(algorithm='unknown')
            oz.fit_transform(X)
            assert str(e.value) == "'unknown' is unrecognized ranking method"

class TestRank1D(VisualTestCase):
    """
    Test the Rank1D visualizer
    """

    def test_rank1d_unknown_algorithm(self):
        if False:
            print('Hello World!')
        '\n        Test that an error is raised for Rank1D with an unknown algorithm\n        '
        (X, _) = load_energy()
        msg = "'oscar' is unrecognized ranking method"
        with pytest.raises(YellowbrickValueError, match=msg):
            Rank1D(algorithm='Oscar').transform(X)

    def test_rank1d_shapiro(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank1D using shapiro metric\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(algorithm='shapiro')
        npt.assert_array_equal(oz.fit_transform(X), X)
        expected = np.array([0.93340671, 0.94967198, 0.92689574, 0.7459445, 0.63657606, 0.85603625, 0.84349269, 0.91551381])
        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1],)
        npt.assert_array_almost_equal(oz.ranks_, expected)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_vertical(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank1D using vertical orientation\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(orient='v')
        npt.assert_array_equal(oz.fit_transform(X), X)
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_horizontal(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank1D using horizontal orientation\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(orient='h')
        npt.assert_array_equal(oz.fit_transform(X), X)
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.filterwarnings('ignore:p-value')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_rank1d_integrated_pandas(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank1D on occupancy dataset with pandas DataFrame and Series\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        features = data.meta['features']
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        oz = Rank1D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.filterwarnings('ignore:p-value')
    def test_rank1d_integrated_numpy(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank1D on occupancy dataset with default numpy data structures\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        features = data.meta['features']
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        oz = Rank1D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.filterwarnings('ignore:p-value may not be accurate')
    def test_rank1d_quick_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Rank1d quick method\n        '
        (X, y) = load_credit()
        viz = rank1d(X, y, show=False)
        assert isinstance(viz, Rank1D)
        self.assert_images_similar(viz, tol=0.1)

class TestRank2D(VisualTestCase):
    """
    Test the Rank2D visualizer
    """

    def test_rank2d_unknown_algorithm(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that an error is raised for Rank2D with an unknown algorithm\n        '
        (X, _) = load_energy()
        msg = "'oscar' is unrecognized ranking method"
        with pytest.raises(YellowbrickValueError, match=msg):
            Rank2D(algorithm='Oscar').transform(X)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_pearson(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank2D using pearson metric\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm='pearson')
        npt.assert_array_equal(oz.fit_transform(X), X)
        expected = np.array([[1.0, -0.991901462, -0.20378168, -0.868823408, 0.827747317, 0.0, 1.11706815e-16, -1.1293567e-16], [-0.991901462, 1.0, 0.195501633, 0.880719517, -0.858147673, 0.0, -2.26567708e-16, -3.55861251e-16], [-0.20378168, 0.195501633, 1.0, -0.292316466, 0.280975743, 0.0, 7.87010445e-18, 0.0], [-0.868823408, 0.880719517, -0.292316466, 1.0, -0.972512237, 0.0, -3.2755331e-16, 2.20057668e-16], [0.827747317, -0.858147673, 0.280975743, -0.972512237, 1.0, 0.0, -1.24094525e-18, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.42798319e-19, 0.0], [1.11706815e-16, -2.26567708e-16, 7.87010445e-18, -3.2755331e-16, -1.24094525e-18, -2.42798319e-19, 1.0, 0.212964221], [-1.1293567e-16, -3.55861251e-16, 0.0, 2.20057668e-16, 0.0, 0.0, 0.212964221, 1.0]])
        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.5)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_covariance(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank2D using covariance metric\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm='covariance')
        npt.assert_array_equal(oz.fit_transform(X), X)
        expected = np.array([[0.0111888744, -9.24206867, -0.940391134, -4.15083877, 0.153324641, 0.0, 1.57414282e-18, -1.85278419e-17], [-9.24206867, 7759.16384, 751.290743, 3503.93655, -132.370274, 0.0, -2.65874531e-15, -4.86170571e-14], [-0.940391134, 751.290743, 1903.26988, -575.98957, 21.4654498, 0.0, 4.57406096e-17, 0.0], [-4.15083877, 3503.93655, -575.98957, 2039.96306, -76.9178618, 0.0, -1.97089918e-15, 1.54151644e-14], [0.153324641, -132.370274, 21.4654498, -76.9178618, 3.06649283, 0.0, -2.89497529e-19, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.25162973, -3.61871912e-20, 0.0], [1.57414282e-18, -2.65874531e-15, 4.57406096e-17, -1.97089918e-15, -2.89497529e-19, -3.61871912e-20, 0.0177477184, 0.0440026076], [-1.85278419e-17, -4.86170571e-14, 0.0, 1.54151644e-14, 0.0, 0.0, 0.0440026076, 2.40547588]])
        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected, decimal=5)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_spearman(self):
        if False:
            return 10
        '\n        Test Rank2D using spearman metric\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm='spearman')
        npt.assert_array_equal(oz.fit_transform(X), X)
        expected = np.array([[1.0, -1.0, -0.25580533, -0.8708862, 0.86904819, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.25580533, 0.8708862, -0.86904819, 0.0, 0.0, 0.0], [-0.25580533, 0.25580533, 1.0, -0.19345677, 0.22076336, 0.0, 0.0, 0.0], [-0.8708862, 0.8708862, -0.19345677, 1.0, -0.93704257, 0.0, 0.0, 0.0], [0.86904819, -0.86904819, 0.22076336, -0.93704257, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.18759162], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18759162, 1.0]])
        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_kendalltau(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Rank2D using kendalltau metric\n        '
        (X, _) = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm='kendalltau')
        npt.assert_array_equal(oz.fit_transform(X), X)
        expected = np.array([[1.0, -1.0, -0.2724275, -0.73614431, 0.73854895, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.2724275, 0.73614431, -0.73854895, 0.0, 0.0, 0.0], [-0.2724275, 0.2724275, 1.0, -0.15192004, 0.19528337, 0.0, 0.0, 0.0], [-0.73614431, 0.73614431, -0.15192004, 1.0, -0.87518995, 0.0, 0.0, 0.0], [0.73854895, -0.73854895, 0.19528337, -0.87518995, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.15430335], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15430335, 1.0]])
        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_rank2d_integrated_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Rank2D on occupancy dataset with pandas DataFrame and Series\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        features = data.meta['features']
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        oz = Rank2D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_integrated_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Rank2D on occupancy dataset with numpy ndarray\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        features = data.meta['features']
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        oz = Rank2D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_rank2d_quick_method(self):
        if False:
            while True:
                i = 10
        '\n        Test Rank2D quick method\n        '
        (X, y) = load_occupancy()
        oz = rank2d(X, y, algorithm='spearman', colormap='RdYlGn_r', show=False)
        assert isinstance(oz, Rank2D)
        self.assert_images_similar(oz, tol=0.1)