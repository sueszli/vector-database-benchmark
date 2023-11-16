"""
Implements TSNE visualizations of documents in 2D space.
"""
import numpy as np
from collections import defaultdict
from yellowbrick.draw import manual_legend
from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, PCA

def tsne(X, y=None, ax=None, decompose='svd', decompose_by=50, labels=None, colors=None, colormap=None, alpha=0.7, show=True, **kwargs):
    if False:
        return 10
    '\n    Display a projection of a vectorized corpus in two dimensions using TSNE,\n    a nonlinear dimensionality reduction method that is particularly well\n    suited to embedding in two or three dimensions for visualization as a\n    scatter plot. TSNE is widely used in text analysis to show clusters or\n    groups of documents or utterances and their relative proximities.\n\n    Parameters\n    ----------\n\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features representing the corpus of\n        vectorized documents to visualize with tsne.\n\n    y : ndarray or Series of length n\n        An optional array or series of target or class values for instances.\n        If this is specified, then the points will be colored according to\n        their class. Often cluster labels are passed in to color the documents\n        in cluster space, so this method is used both for classification and\n        clustering methods.\n\n    ax : matplotlib axes\n        The axes to plot the figure on.\n\n    decompose : string or None\n        A preliminary decomposition is often used prior to TSNE to make the\n        projection faster. Specify `"svd"` for sparse data or `"pca"` for\n        dense data. If decompose is None, the original data set will be used.\n\n    decompose_by : int\n        Specify the number of components for preliminary decomposition, by\n        default this is 50; the more components, the slower TSNE will be.\n\n    labels : list of strings\n        The names of the classes in the target, used to create a legend.\n\n    colors : list or tuple of colors\n        Specify the colors for each individual class\n\n    colormap : string or matplotlib cmap\n        Sequential colormap for continuous target\n\n    alpha : float, default: 0.7\n        Specify a transparency where 1 is completely opaque and 0 is completely\n        transparent. This property makes densely clustered points more visible.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Pass any additional keyword arguments to the TSNE transformer.\n\n    Example\n    --------\n    >>> from yellowbrick.text.tsne import tsne\n    >>> from sklearn.feature_extraction.text import TfidfVectorizer\n    >>> from yellowbrick.datasets import load_hobbies\n    >>> corpus = load_hobbies()\n    >>> tfidf = TfidfVectorizer()\n    >>> X = tfidf.fit_transform(corpus.data)\n    >>> y = corpus.target\n    >>> tsne(X, y)\n\n    Returns\n    -------\n    visualizer: TSNEVisualizer\n        Returns the fitted, finalized visualizer\n    '
    visualizer = TSNEVisualizer(ax=ax, decompose=decompose, decompose_by=decompose_by, labels=labels, colors=colors, colormap=colormap, alpha=alpha, **kwargs)
    visualizer.fit(X, y, **kwargs)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer

class TSNEVisualizer(TextVisualizer):
    """
    Display a projection of a vectorized corpus in two dimensions using TSNE,
    a nonlinear dimensionality reduction method that is particularly well
    suited to embedding in two or three dimensions for visualization as a
    scatter plot. TSNE is widely used in text analysis to show clusters or
    groups of documents or utterances and their relative proximities.

    TSNE will return a scatter plot of the vectorized corpus, such that each
    point represents a document or utterance. The distance between two points
    in the visual space is embedded using the probability distribution of
    pairwise similarities in the higher dimensionality; thus TSNE shows
    clusters of similar documents and the relationships between groups of
    documents as a scatter plot.

    TSNE can be used with either clustering or classification; by specifying
    the ``classes`` argument, points will be colored based on their similar
    traits. For example, by passing ``cluster.labels_`` as ``y`` in ``fit()``, all
    points in the same cluster will be grouped together. This extends the
    neighbor embedding with more information about similarity, and can allow
    better interpretation of both clusters and classes.

    For more, see https://lvdmaaten.github.io/tsne/

    Parameters
    ----------

    ax : matplotlib axes
        The axes to plot the figure on.

    decompose : string or None, default: ``'svd'``
        A preliminary decomposition is often used prior to TSNE to make the
        projection faster. Specify ``"svd"`` for sparse data or ``"pca"`` for
        dense data. If None, the original data set will be used.

    decompose_by : int, default: 50
        Specify the number of components for preliminary decomposition, by
        default this is 50; the more components, the slower TSNE will be.

    labels : list of strings
        The names of the classes in the target, used to create a legend.
        Labels must match names of classes in sorted order.

    colors : list or tuple of colors
        Specify the colors for each individual class

    colormap : string or matplotlib cmap
        Sequential colormap for continuous target

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. The random state is applied to the preliminary
        decomposition as well as tSNE.

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Pass any additional keyword arguments to the TSNE transformer.
    """
    NULL_CLASS = None

    def __init__(self, ax=None, decompose='svd', decompose_by=50, labels=None, classes=None, colors=None, colormap=None, random_state=None, alpha=0.7, **kwargs):
        if False:
            while True:
                i = 10
        self.alpha = alpha
        self.labels = labels
        self.colors = colors
        self.colormap = colormap
        self.random_state = random_state
        tsne_kwargs = {key: kwargs.pop(key) for key in TSNE().get_params() if key in kwargs}
        self.transformer_ = self.make_transformer(decompose, decompose_by, tsne_kwargs)
        super(TSNEVisualizer, self).__init__(ax=ax, **kwargs)

    def make_transformer(self, decompose='svd', decompose_by=50, tsne_kwargs={}):
        if False:
            while True:
                i = 10
        '\n        Creates an internal transformer pipeline to project the data set into\n        2D space using TSNE, applying an pre-decomposition technique ahead of\n        embedding if necessary. This method will reset the transformer on the\n        class, and can be used to explore different decompositions.\n\n        Parameters\n        ----------\n\n        decompose : string or None, default: ``\'svd\'``\n            A preliminary decomposition is often used prior to TSNE to make\n            the projection faster. Specify ``"svd"`` for sparse data or ``"pca"``\n            for dense data. If decompose is None, the original data set will\n            be used.\n\n        decompose_by : int, default: 50\n            Specify the number of components for preliminary decomposition, by\n            default this is 50; the more components, the slower TSNE will be.\n\n        Returns\n        -------\n\n        transformer : Pipeline\n            Pipelined transformer for TSNE projections\n        '
        decompositions = {'svd': TruncatedSVD, 'pca': PCA}
        if decompose and decompose.lower() not in decompositions:
            raise YellowbrickValueError("'{}' is not a valid decomposition, use {}, or None".format(decompose, ', '.join(decompositions.keys())))
        steps = []
        if decompose:
            klass = decompositions[decompose]
            steps.append((decompose, klass(n_components=decompose_by, random_state=self.random_state)))
        steps.append(('tsne', TSNE(n_components=2, random_state=self.random_state, **tsne_kwargs)))
        return Pipeline(steps)

    def fit(self, X, y=None, **kwargs):
        if False:
            return 10
        '\n        The fit method is the primary drawing input for the TSNE projection\n        since the visualization requires both X and an optional y value. The\n        fit method expects an array of numeric vectors, so text documents must\n        be vectorized before passing them to this method.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features representing the corpus of\n            vectorized documents to visualize with tsne.\n\n        y : ndarray or Series of length n\n            An optional array or series of target or class values for\n            instances. If this is specified, then the points will be colored\n            according to their class. Often cluster labels are passed in to\n            color the documents in cluster space, so this method is used both\n            for classification and clustering methods.\n\n        kwargs : dict\n            Pass generic arguments to the drawing method\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the transformer/visualizer\n        '
        if y is not None:
            self.classes_ = np.unique(y)
        elif y is None and self.labels is not None:
            self.classes_ = np.array([self.labels[0]])
        else:
            self.classes_ = np.array([self.NULL_CLASS])
        vecs = self.transformer_.fit_transform(X)
        self.n_instances_ = vecs.shape[0]
        self.draw(vecs, y, **kwargs)
        return self

    def draw(self, points, target=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Called from the fit method, this method draws the TSNE scatter plot,\n        from a set of decomposed points in 2 dimensions. This method also\n        accepts a third dimension, target, which is used to specify the colors\n        of each of the points. If the target is not specified, then the points\n        are plotted as a single cloud to show similar documents.\n        '
        labels = self.labels if self.labels is not None else self.classes_
        if len(labels) != len(self.classes_):
            raise YellowbrickValueError('number of supplied labels ({}) does not match the number of classes ({})'.format(len(labels), len(self.classes_)))
        self.color_values_ = resolve_colors(n_colors=len(labels), colormap=self.colormap, colors=self.colors)
        colors = dict(zip(labels, self.color_values_))
        labels = dict(zip(self.classes_, labels))
        series = defaultdict(lambda : {'x': [], 'y': []})
        if target is not None:
            for (t, point) in zip(target, points):
                label = labels[t]
                series[label]['x'].append(point[0])
                series[label]['y'].append(point[1])
        else:
            label = self.classes_[0]
            for (x, y) in points:
                series[label]['x'].append(x)
                series[label]['y'].append(y)
        for (label, points) in series.items():
            self.ax.scatter(points['x'], points['y'], c=colors[label], alpha=self.alpha, label=label)
        return self.ax

    def finalize(self, **kwargs):
        if False:
            return 10
        '\n        Finalize the drawing by adding a title and legend, and removing the\n        axes objects that do not convey information about TNSE.\n        '
        self.set_title('TSNE Projection of {} Documents'.format(self.n_instances_))
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        if not all(self.classes_ == np.array([self.NULL_CLASS])):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            manual_legend(self, self.classes_, self.color_values_, loc='center left', bbox_to_anchor=(1, 0.5))