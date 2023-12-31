# tests.test_cluster.test_silhouette
# Tests for the SilhouetteVisualizer
#
# Author:   Benjamin Bengfort
# Created:  Mon Mar 27 10:01:37 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_silhouette.py [57b563b] benjamin@bengfort.com $

"""
Tests for the SilhouetteVisualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

from unittest import mock
from tests.base import VisualTestCase

from yellowbrick.cluster.silhouette import SilhouetteVisualizer, silhouette_visualizer


##########################################################################
## SilhouetteVisualizer Test Cases
##########################################################################


class TestSilhouetteVisualizer(VisualTestCase):
    """
    Silhouette Visualizer Tests
    """

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_integrated_kmeans_silhouette(self):
        """
        Test no exceptions for kmeans silhouette visualizer on blobs dataset
        """
        # NOTE see #182: cannot use occupancy dataset because of memory usage

        # Generate a blobs data set
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = SilhouetteVisualizer(KMeans(random_state=0), ax=ax)
        visualizer.fit(X)
        visualizer.finalize()

        self.assert_images_similar(visualizer, remove_legend=True)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_integrated_mini_batch_kmeans_silhouette(self):
        """
        Test no exceptions for mini-batch kmeans silhouette visualizer
        """
        # NOTE see #182: cannot use occupancy dataset because of memory usage

        # Generate a blobs data set
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = SilhouetteVisualizer(MiniBatchKMeans(random_state=0), ax=ax)
        visualizer.fit(X)
        visualizer.finalize()

        self.assert_images_similar(visualizer, remove_legend=True)

    @pytest.mark.skip(reason="no negative silhouette example available yet")
    def test_negative_silhouette_score(self):
        """
        Ensure negative silhouette scores are correctly displayed by the visualizer.
        """
        raise NotImplementedError("no negative silhouette example available")

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_colormap_silhouette(self):
        """
        Test no exceptions for modifying the colormap in a silhouette visualizer
        """
        # Generate a blobs data set
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = SilhouetteVisualizer(
            MiniBatchKMeans(random_state=0), ax=ax, colormap="gnuplot"
        )
        visualizer.fit(X)
        visualizer.finalize()

        self.assert_images_similar(visualizer, remove_legend=True)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_colors_silhouette(self):
        """
        Test no exceptions for modifying the colors in a silhouette visualizer
        with a list of color names
        """
        # Generate a blobs data set
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = SilhouetteVisualizer(
            MiniBatchKMeans(random_state=0),
            ax=ax,
            colors=["red", "green", "blue", "indigo", "cyan", "lavender"],
        )
        visualizer.fit(X)
        visualizer.finalize()

        self.assert_images_similar(visualizer, remove_legend=True)

    def test_colormap_as_colors_silhouette(self):
        """
        Test no exceptions for modifying the colors in a silhouette visualizer
        by using a matplotlib colormap as colors
        """
        # Generate a blobs data set
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = SilhouetteVisualizer(
            MiniBatchKMeans(random_state=0), ax=ax, colors="cool"
        )
        visualizer.fit(X)
        visualizer.finalize()

        tol = (
            3.2 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.143
        self.assert_images_similar(visualizer, remove_legend=True, tol=tol)

    def test_quick_method(self):
        """
        Test the quick method producing a valid visualization
        """
        X, y = make_blobs(
            n_samples=1000, n_features=12, centers=8, shuffle=False, random_state=0
        )

        model = MiniBatchKMeans(3, random_state=343)
        oz = silhouette_visualizer(model, X, show=False)
        assert isinstance(oz, SilhouetteVisualizer)

        self.assert_images_similar(oz)

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = make_blobs(
            n_samples=100, n_features=5, centers=3, shuffle=False, random_state=112
        )
        model = MiniBatchKMeans().fit(X)
        labels = model.predict(X)

        with mock.patch.object(model, "fit") as mockfit:
            oz = SilhouetteVisualizer(model)
            oz.fit(X)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = SilhouetteVisualizer(model, is_fitted=True)
            oz.fit(X)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit_predict", return_value=labels) as mockfit:
            oz = SilhouetteVisualizer(model, is_fitted=False)
            oz.fit(X)
            mockfit.assert_called_once_with(X, None)

    @pytest.mark.parametrize(
        "model",
        [SpectralClustering, AgglomerativeClustering],
    )
    def test_clusterer_without_predict(self, model):
        """
        Assert that clustering estimators that don't implement
        a predict() method utilize fit_predict()
        """
        X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        try:
            visualizer = SilhouetteVisualizer(model(n_clusters=2))
            visualizer.fit(X)
            visualizer.finalize()
        except AttributeError:
            self.fail("could not use fit or fit_predict methods")
