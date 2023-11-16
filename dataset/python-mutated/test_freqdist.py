"""
Tests for the frequency distribution text visualization
"""
import pytest
from yellowbrick.datasets import load_hobbies
from yellowbrick.text.freqdist import *
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from sklearn.feature_extraction.text import CountVectorizer
corpus = load_hobbies()

class TestFreqDist(VisualTestCase):

    @pytest.mark.xfail(IS_WINDOWS_OR_CONDA, reason='font rendering different in OS and/or Python; see #892')
    def test_integrated_freqdist(self):
        if False:
            return 10
        '\n        Assert no errors occur during freqdist integration\n        '
        vectorizer = CountVectorizer()
        docs = vectorizer.fit_transform(corpus.data)
        features = vectorizer.get_feature_names()
        visualizer = FreqDistVisualizer(features)
        visualizer.fit(docs)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)

    def test_freqdist_quickmethod(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert no errors occur during freqdist quickmethod\n        '
        vectorizer = CountVectorizer()
        docs = vectorizer.fit_transform(corpus.data)
        features = vectorizer.get_feature_names()
        viz = freqdist(features, docs, show=False)
        assert isinstance(viz, FreqDistVisualizer)
        self.assert_images_similar(viz, tol=1.5)