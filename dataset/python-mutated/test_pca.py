"""
Tests for the PCA based feature visualizer.
"""
import sys
import pytest
import numpy as np
import numpy.testing as npt
from unittest import mock
from tests.base import VisualTestCase, IS_WINDOWS_OR_CONDA
from yellowbrick.features.pca import *
from yellowbrick.exceptions import YellowbrickError, NotFitted
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None

@pytest.mark.usefixtures('discrete', 'continuous')
class TestPCA(VisualTestCase):
    """
    Test the PCA visualizer
    """

    def test_single(self):
        if False:
            i = 10
            return i + 15
        '\n        Test single target.\n        '
        visualizer = PCA(random_state=1998)
        visualizer.fit(self.continuous.X)
        visualizer.transform(self.continuous.X)
        assert not hasattr(visualizer, 'classes_')
        assert not hasattr(visualizer, 'range_')
        self.assert_images_similar(visualizer)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 10.205 on miniconda')
    def test_continuous(self):
        if False:
            return 10
        '\n        Test continuous target\n        '
        visualizer = PCA(colormap='YlOrRd', random_state=2019)
        assert not hasattr(visualizer, 'range_')
        visualizer.fit(*self.continuous)
        visualizer.transform(*self.continuous)
        assert hasattr(visualizer, 'range_')
        assert not hasattr(visualizer, 'classes_')
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer, windows_tol=10.5)

    def test_discrete(self):
        if False:
            print('Hello World!')
        '\n        Test discrete target.\n        '
        classes = ['a', 'b', 'c', 'd', 'e']
        colors = ['r', 'b', 'g', 'm', 'c']
        visualizer = PCA(colors=colors, classes=classes, random_state=83)
        assert not hasattr(visualizer, 'classes_')
        visualizer.fit(*self.discrete)
        assert hasattr(visualizer, 'classes_')
        assert not hasattr(visualizer, 'range_')
        visualizer.transform(*self.discrete)
        npt.assert_array_equal(visualizer.classes_, classes)
        self.assert_images_similar(visualizer)

    def test_fit(self):
        if False:
            print('Hello World!')
        '\n        Test that fit returns self.\n        '
        pca = PCA()
        assert pca.fit(*self.discrete) is pca

    @pytest.mark.parametrize('n_components', [2, 3])
    def test_transform(self, n_components):
        if False:
            while True:
                i = 10
        Xprime = PCA(projection=n_components).fit_transform(*self.continuous)
        assert Xprime.shape == (500, n_components)

    def test_transform_without_fit(self):
        if False:
            print('Hello World!')
        '\n        Test that appropriate error is raised when transform called without fit.\n        '
        oz = PCA(projection=3)
        msg = 'instance is not fitted yet, please call fit'
        with pytest.raises(NotFitted, match=msg):
            oz.transform(*self.continuous)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 12.115 on miniconda')
    def test_pca_decomposition_quick_method(self):
        if False:
            return 10
        '\n        Test the quick method PCA visualizer\n        '
        visualizer = pca_decomposition(*self.discrete, projection=2, scale=True, random_state=28, show=False)
        self.assert_images_similar(visualizer, windows_tol=12.5)

    def test_scale_true_2d(self):
        if False:
            print('Hello World!')
        '\n        Test the PCA visualizer 2 dimensions scaled.\n        '
        params = {'scale': True, 'projection': 2, 'random_state': 9932}
        visualizer = PCA(**params).fit(*self.discrete)
        pca_array = visualizer.transform(*self.discrete)
        self.assert_images_similar(visualizer)
        assert pca_array.shape == (self.discrete.X.shape[0], 2)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 8.828 on miniconda')
    def test_scale_false_2d(self):
        if False:
            return 10
        '\n        Test the PCA visualizer 2 dimensions non-scaled.\n        '
        params = {'scale': False, 'projection': 2, 'random_state': 1229}
        visualizer = PCA(**params).fit(*self.continuous)
        pca_array = visualizer.transform(*self.continuous)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer, tol=0.03, windows_tol=8.5)
        assert pca_array.shape == (self.continuous.X.shape[0], 2)

    def test_biplot_2d(self):
        if False:
            print('Hello World!')
        '\n        Test the PCA 2D biplot (proj_features).\n        '
        params = {'features': list('ABCDEFGHIKLM'), 'random_state': 67, 'proj_features': True, 'projection': 2}
        visualizer = PCA(**params).fit(self.discrete.X)
        pca_array = visualizer.transform(self.discrete.X)
        self.assert_images_similar(visualizer, tol=5)
        assert pca_array.shape == (self.discrete.X.shape[0], 2)

    def test_scale_true_3d(self):
        if False:
            while True:
                i = 10
        '\n        Test the PCA visualizer 3 dimensions scaled.\n        '
        params = {'scale': True, 'projection': 3, 'random_state': 7382}
        visualizer = PCA(**params).fit(self.discrete.X)
        pca_array = visualizer.transform(self.discrete.X)
        self.assert_images_similar(visualizer)
        assert pca_array.shape == (self.discrete.X.shape[0], 3)

    def test_scale_false_3d(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the PCA visualizer 3 dimensions non-scaled.\n        '
        params = {'scale': False, 'projection': 3, 'random_state': 98}
        visualizer = PCA(**params).fit(self.discrete.X)
        pca_array = visualizer.transform(self.discrete.X)
        self.assert_images_similar(visualizer)
        assert pca_array.shape == (self.discrete.X.shape[0], 3)

    @pytest.mark.xfail(sys.platform == 'win32', reason='images not close on windows (RMSE=3)')
    def test_biplot_3d(self):
        if False:
            while True:
                i = 10
        '\n        Test the PCA 3D biplot (proj_features).\n        '
        params = {'features': list('ABCDEFGHIKLM'), 'random_state': 800, 'proj_features': True, 'projection': 3}
        visualizer = PCA(**params).fit(*self.discrete)
        pca_array = visualizer.transform(*self.discrete)
        self.assert_images_similar(visualizer, tol=5)
        assert pca_array.shape == (self.discrete.X.shape[0], 3)

    def test_scale_true_4d_exception(self):
        if False:
            print('Hello World!')
        '\n        Test PCA visualizer 4 dimensions scaled (catch YellowbrickError).\n        '
        params = {'scale': True, 'projection': 4}
        msg = 'Projection dimensions must be either 2 or 3'
        with pytest.raises(YellowbrickError, match=msg):
            PCA(**params)

    def test_scale_true_3d_exception(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test PCA visualizer 3 dims scaled on 2 dim data set (catch ValueError).\n        '
        X = np.random.normal(loc=2, size=(100, 2))
        params = {'scale': True, 'projection': 3}
        e = 'n_components=3 must be between 0 and min\\(n_samples, n_features\\)=2'
        with pytest.raises(ValueError, match=e):
            pca = PCA(**params)
            pca.fit(X)

    @mock.patch('yellowbrick.features.pca.plt.sca', autospec=True)
    def test_alpha_param(self, mock_sca):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the user can supply an alpha param on instantiation\n        '
        params = {'alpha': 0.3, 'projection': 2, 'random_state': 9932}
        visualizer = PCA(**params).fit(self.discrete.X)
        pca_array = visualizer.transform(self.discrete.X)
        assert visualizer.alpha == 0.3
        visualizer.ax = mock.MagicMock()
        visualizer.fit(self.discrete.X)
        visualizer.transform(self.discrete.X)
        (_, scatter_kwargs) = visualizer.ax.scatter.call_args
        assert 'alpha' in scatter_kwargs
        assert scatter_kwargs['alpha'] == 0.3
        assert pca_array.shape == (self.discrete.X.shape[0], 2)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 7.332 on miniconda')
    def test_colorbar(self):
        if False:
            print('Hello World!')
        "\n        Test the PCA visualizer's colorbar features.\n        "
        params = {'scale': True, 'projection': 2, 'random_state': 7382, 'color': self.discrete.y, 'colorbar': True}
        visualizer = PCA(**params).fit(*self.continuous)
        visualizer.transform(self.continuous.X, self.continuous.y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer, windows_tol=7.5)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 14.515 on miniconda')
    def test_heatmap(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the PCA visualizer's heatmap features.\n        "
        params = {'scale': True, 'projection': 2, 'random_state': 7382, 'color': self.discrete.y, 'heatmap': True}
        visualizer = PCA(**params).fit(self.discrete.X, self.discrete.y)
        visualizer.transform(self.discrete.X, self.discrete.y)
        visualizer.finalize()
        visualizer.lax.set_xticks([])
        visualizer.lax.set_yticks([])
        visualizer.lax.set_xticks([], minor=True)
        visualizer.uax.set_xticklabels([])
        self.assert_images_similar(visualizer, windows_tol=14.5)

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='RMS of 10.987 on miniconda')
    def test_colorbar_heatmap(self):
        if False:
            return 10
        '\n        Test the PCA visualizer with both colorbar and heatmap.\n        '
        params = {'scale': True, 'projection': 2, 'random_state': 7382, 'color': self.discrete.y, 'colorbar': True, 'heatmap': True}
        visualizer = PCA(**params).fit(self.continuous.X, self.continuous.y)
        visualizer.transform(self.continuous.X, self.continuous.y)
        visualizer.finalize()
        visualizer.lax.set_xticks([])
        visualizer.lax.set_yticks([])
        visualizer.lax.set_xticks([], minor=True)
        visualizer.uax.set_xticklabels([])
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer, windows_tol=10.5)

    def test_3d_heatmap_enabled_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert an exception if colorbar and heatmap is enabled with 3-dimensions\n        '
        with pytest.raises(YellowbrickValueError):
            PCA(projection=3, heatmap=True)

    @pytest.mark.skipif(make_axes_locatable is not None, reason='requires matplotlib <= 2.0.1')
    def test_matplotlib_version_error():
        if False:
            i = 10
            return i + 15
        '\n        Assert an exception is raised with incompatible matplotlib versions\n        '
        with pytest.raises(YellowbrickValueError):
            PCA(colorbar=True, heatmap=True)