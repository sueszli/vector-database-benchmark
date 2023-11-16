"""
Tests for the intercluster distance map visualizer.
"""
import pytest
import matplotlib as mpl
import numpy as np
from yellowbrick.cluster.icdm import *
from yellowbrick.datasets import load_nfl
from yellowbrick.exceptions import YellowbrickValueError
from unittest import mock
from tests.fixtures import Dataset
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans, AffinityPropagation, MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    import pandas as pd
except ImportError:
    pd = None
MPL_VERS_MAJ = int(mpl.__version__.split('.')[0])

@pytest.fixture(scope='class')
def blobs12(request):
    if False:
        while True:
            i = 10
    '\n    Creates a fixture of 1000 instances in 12 clusters with 16 features.\n    '
    (X, y) = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True, random_state=2121)
    request.cls.blobs12 = Dataset(X, y)

@pytest.fixture(scope='class')
def blobs4(request):
    if False:
        print('Hello World!')
    '\n    Creates a fixture of 400 instances in 4 clusters with 16 features.\n    '
    (X, y) = make_blobs(centers=4, n_samples=400, n_features=16, shuffle=True, random_state=1212)
    request.cls.blobs4 = Dataset(X, y)

def assert_fitted(oz):
    if False:
        i = 10
        return i + 15
    for param in ('cluster_centers_', 'embedded_centers_', 'scores_', 'fit_time_'):
        assert hasattr(oz, param)

def assert_not_fitted(oz):
    if False:
        for i in range(10):
            print('nop')
    for param in ('embedded_centers_', 'scores_', 'fit_time_'):
        assert not hasattr(oz, param)

@pytest.mark.usefixtures('blobs12', 'blobs4')
class TestInterclusterDistance(VisualTestCase):
    """
    Test the InterclusterDistance visualizer
    """

    def test_only_valid_embeddings(self):
        if False:
            while True:
                i = 10
        '\n        Should raise an exception on invalid embedding\n        '
        with pytest.raises(YellowbrickValueError, match="unknown embedding 'foo'"):
            InterclusterDistance(KMeans(), embedding='foo')
        icdm = InterclusterDistance(KMeans())
        icdm.embedding = 'foo'
        with pytest.raises(YellowbrickValueError, match="unknown embedding 'foo'"):
            icdm.transformer

    def test_only_valid_scoring(self):
        if False:
            print('Hello World!')
        '\n        Should raise an exception on invalid scoring\n        '
        with pytest.raises(YellowbrickValueError, match="unknown scoring 'foo'"):
            InterclusterDistance(KMeans(), scoring='foo')
        icdm = InterclusterDistance(KMeans())
        icdm.scoring = 'foo'
        with pytest.raises(YellowbrickValueError, match="unknown scoring method 'foo'"):
            icdm._score_clusters(None)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_kmeans_mds(self):
        if False:
            return 10
        '\n        Visual similarity with KMeans and MDS scaling\n        '
        model = KMeans(9, random_state=38)
        oz = InterclusterDistance(model, random_state=83, embedding='mds')
        assert_not_fitted(oz)
        assert oz.fit(self.blobs12.X) is oz
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9
        oz.finalize()
        self.assert_images_similar(oz, tol=0.5)

    @pytest.mark.filterwarnings('ignore:the matrix subclass is not the recommended way')
    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_affinity_tsne_no_legend(self):
        if False:
            i = 10
            return i + 15
        '\n        Visual similarity with AffinityPropagation, TSNE scaling, and no legend\n        '
        model = AffinityPropagation()
        oz = InterclusterDistance(model, random_state=763, embedding='tsne', legend=False)
        assert_not_fitted(oz)
        assert oz.fit(self.blobs4.X) is oz
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.skip(reason='LDA not implemented yet')
    def test_lda_mds(self):
        if False:
            print('Hello World!')
        '\n        Visual similarity with LDA and MDS scaling\n        '
        model = LDA(9, random_state=6667)
        oz = InterclusterDistance(model, random_state=2332, embedding='mds')
        assert_not_fitted(oz)
        assert oz.fit(self.blobs12.X) is oz
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.skip(reason='agglomerative not implemented yet')
    @pytest.mark.filterwarnings('ignore:Using a non-tuple sequence')
    @pytest.mark.filterwarnings('ignore:the matrix subclass is not the recommended way')
    def test_birch_tsne(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Visual similarity with Birch and MDS scaling\n        '
        oz = InterclusterDistance(Birch(n_clusters=9), random_state=83, embedding='mds')
        assert_not_fitted(oz)
        assert oz.fit(self.blobs12.X) is oz
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.skip(reason='agglomerative not implemented yet')
    def test_ward_mds_no_legend(self):
        if False:
            i = 10
            return i + 15
        '\n        Visual similarity with Ward, TSNE scaling, and no legend\n        '
        model = AgglomerativeClustering(n_clusters=9)
        oz = InterclusterDistance(model, random_state=83, embedding='tsne', legend=False)
        assert_not_fitted(oz)
        assert oz.fit(self.blobs12.X) is oz
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_quick_method(self):
        if False:
            return 10
        '\n        Test the quick method producing a valid visualization\n        '
        model = MiniBatchKMeans(3, random_state=343)
        oz = intercluster_distance(model, self.blobs4.X, random_state=93, legend=False, show=False)
        assert isinstance(oz, InterclusterDistance)
        self.assert_images_similar(oz)

    @pytest.mark.skipif(MPL_VERS_MAJ >= 2, reason='test requires mpl earlier than 2.0.2')
    def test_legend_matplotlib_version(self, mock_toolkit):
        if False:
            for i in range(10):
                print('nop')
        '\n        ValueError is raised when matplotlib version is incorrect and legend=True\n        '
        with pytest.raises(ImportError):
            from mpl_toolkits.axes_grid1 import inset_locator
            assert not inset_locator
        with pytest.raises(YellowbrickValueError, match='requires matplotlib 2.0.2'):
            InterclusterDistance(KMeans(), legend=True)

    @pytest.mark.skipif(MPL_VERS_MAJ >= 2, reason='test requires mpl earlier than 2.0.2')
    def test_no_legend_matplotlib_version(self, mock_toolkit):
        if False:
            for i in range(10):
                print('nop')
        '\n        No error is raised when matplotlib version is incorrect and legend=False\n        '
        with pytest.raises(ImportError):
            from mpl_toolkits.axes_grid1 import inset_locator
            assert not inset_locator
        InterclusterDistance(KMeans(), legend=False)

    @pytest.mark.xfail(reason='third test fails with AssertionError: Expected fit\n        to be called once. Called 0 times.')
    def test_with_fitted(self):
        if False:
            return 10
        '\n        Test that visualizer properly handles an already-fitted model\n        '
        (X, y) = load_nfl(return_dataset=True).to_numpy()
        model = KMeans().fit(X, y)
        with mock.patch.object(model, 'fit') as mockfit:
            oz = ICDM(model)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = ICDM(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()
        with mock.patch.object(model, 'fit') as mockfit:
            oz = ICDM(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_within_pipeline(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_nfl()
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('icdm', InterclusterDistance(KMeans(5, random_state=42), random_state=42))])
        model.fit(X)
        model['icdm'].finalize()
        self.assert_images_similar(model['icdm'], tol=2.0)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_within_pipeline_quickmethod(self):
        if False:
            print('Hello World!')
        '\n        Test that visualizer can be accessed within a sklearn pipeline\n        '
        (X, y) = load_nfl()
        model = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('icdm', intercluster_distance(KMeans(5, random_state=42), X, random_state=42))])
        model['icdm'].finalize()
        self.assert_images_similar(model['icdm'], tol=2.0)