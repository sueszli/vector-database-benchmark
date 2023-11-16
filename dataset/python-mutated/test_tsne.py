"""
Tests for the TSNE visual corpus embedding mechanism.
"""
import pytest
from unittest import mock
from yellowbrick.text.tsne import *
from tests.base import VisualTestCase
from yellowbrick.datasets import load_hobbies
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import pandas
except ImportError:
    pandas = None
corpus = load_hobbies()

class TestTSNE(VisualTestCase):
    """
    TSNEVisualizer tests
    """

    def test_bad_decomposition(self):
        if False:
            while True:
                i = 10
        '\n        Ensure an error is raised when a bad decompose argument is specified\n        '
        with pytest.raises(YellowbrickValueError):
            TSNEVisualizer(decompose='bob')

    def test_make_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify the pipeline creation step for TSNE\n        '
        tsne = TSNEVisualizer()
        assert tsne.transformer_ is not None
        svdp = tsne.make_transformer('svd', 90)
        assert len(svdp.steps) == 2
        pcap = tsne.make_transformer('pca')
        assert len(pcap.steps) == 2
        none = tsne.make_transformer(None)
        assert len(none.steps) == 1

    def test_integrated_tsne(self):
        if False:
            i = 10
            return i + 15
        '\n        Check tSNE integrated visualization on the hobbies corpus\n        '
        tfidf = TfidfVectorizer()
        docs = tfidf.fit_transform(corpus.data)
        labels = corpus.target
        tsne = TSNEVisualizer(random_state=8392, colormap='Set1', alpha=1.0)
        tsne.fit_transform(docs, labels)
        self.assert_images_similar(tsne, tol=50)

    def test_sklearn_tsne_size(self):
        if False:
            while True:
                i = 10
        "\n        Check to make sure sklearn's TSNE doesn't use the size param\n        "
        with pytest.raises(TypeError):
            TSNE(size=(100, 100))

    def test_sklearn_tsne_title(self):
        if False:
            i = 10
            return i + 15
        "\n        Check to make sure sklearn's TSNE doesn't use the title param\n        "
        with pytest.raises(TypeError):
            TSNE(title='custom_title')

    def test_custom_title_tsne(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check tSNE can accept a custom title (string) from the user\n        '
        tsne = TSNEVisualizer(title='custom_title')
        assert tsne.title == 'custom_title'

    def test_custom_size_tsne(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check tSNE can accept a custom size (tuple of pixels) from the user\n        '
        tsne = TSNEVisualizer(size=(100, 50))
        assert tsne._size == (100, 50)

    def test_custom_colors_tsne(self):
        if False:
            return 10
        '\n        Check tSNE accepts and properly handles custom colors from user\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=5, random_state=42)
        purple_blues = ['indigo', 'orchid', 'plum', 'navy', 'purple', 'blue']
        purple_tsne = TSNEVisualizer(colors=purple_blues, random_state=87)
        assert purple_tsne.colors == purple_blues
        purple_tsne.fit(X, y)
        assert len(purple_tsne.color_values_) == len(purple_tsne.classes_)
        assert purple_tsne.color_values_ == purple_blues[:len(purple_tsne.classes_)]
        greens = ['green', 'lime', 'teal']
        green_tsne = TSNEVisualizer(colors=greens, random_state=87)
        assert green_tsne.colors == greens
        green_tsne.fit(X, y)
        assert len(green_tsne.color_values_) == len(green_tsne.classes_)
        assert green_tsne.color_values_ == ['green', 'lime', 'teal', 'green', 'lime']

    def test_make_classification_tsne(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test tSNE integrated visualization on a sklearn classifier dataset\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=42)
        tsne = TSNEVisualizer(random_state=87)
        tsne.fit(X, y)
        self.assert_images_similar(tsne, tol=0.1)

    def test_make_classification_tsne_class_labels(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test tSNE integrated visualization with class labels specified\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=42)
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b', 'c'])
        tsne.fit(X, y)
        self.assert_images_similar(tsne, tol=0.1)

    def test_tsne_mismtached_labels(self):
        if False:
            while True:
                i = 10
        "\n        Assert exception is raised when number of labels doesn't match\n        "
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=42)
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b'])
        with pytest.raises(YellowbrickValueError):
            tsne.fit(X, y)
        tsne = TSNEVisualizer(random_state=87, labels=['a', 'b', 'c', 'd'])
        with pytest.raises(YellowbrickValueError):
            tsne.fit(X, y)

    def test_no_target_tsne(self):
        if False:
            i = 10
            return i + 15
        '\n        Test tSNE when no target or classes are specified\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=6897)
        tsne = TSNEVisualizer(random_state=64)
        tsne.fit(X)
        self.assert_images_similar(tsne, tol=0.1)

    @pytest.mark.skipif(pandas is None, reason='test requires pandas')
    def test_visualizer_with_pandas(self):
        if False:
            while True:
                i = 10
        '\n        Test tSNE when passed a pandas DataFrame and series\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=3020)
        X = pandas.DataFrame(X)
        y = pandas.Series(y)
        tsne = TSNEVisualizer(random_state=64)
        tsne.fit(X, y)
        self.assert_images_similar(tsne, tol=0.1)

    def test_alpha_param(self):
        if False:
            return 10
        '\n        Test that the user can supply an alpha param on instantiation\n        '
        (X, y) = make_classification(n_samples=200, n_features=100, n_informative=20, n_redundant=10, n_classes=3, random_state=42)
        tsne = TSNEVisualizer(random_state=64, alpha=0.5)
        assert tsne.alpha == 0.5
        tsne.ax = mock.MagicMock(autospec=True)
        tsne.fit(X, y)
        (_, scatter_kwargs) = tsne.ax.scatter.call_args
        assert 'alpha' in scatter_kwargs
        assert scatter_kwargs['alpha'] == 0.5

    def test_quick_method(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for tsne quick  method with hobbies dataset\n        '
        corpus = load_hobbies()
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus.data)
        y = corpus.target
        viz = tsne(X, y, show=False)
        self.assert_images_similar(viz, tol=50)