"""
Implements Intercluster Distance Map visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.manifold import MDS, TSNE
from yellowbrick.utils.timer import Timer
from yellowbrick.utils.decorators import memoized
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.cluster.base import ClusteringScoreVisualizer
from yellowbrick.utils.helpers import prop_to_size, check_fitted
try:
    from mpl_toolkits.axes_grid1 import inset_locator
except ImportError:
    inset_locator = None
__all__ = ['InterclusterDistance', 'intercluster_distance', 'VALID_EMBEDDING', 'VALID_SCORING', 'ICDM']
VALID_EMBEDDING = {'mds', 'tsne'}
VALID_SCORING = {'membership'}

class InterclusterDistance(ClusteringScoreVisualizer):
    """
    Intercluster distance maps display an embedding of the cluster centers in
    2 dimensions with the distance to other centers preserved. E.g. the closer
    to centers are in the visualization, the closer they are in the original
    feature space. The clusters are sized according to a scoring metric. By
    default, they are sized by membership, e.g. the number of instances that
    belong to each center. This gives a sense of the relative importance of
    clusters. Note however, that because two clusters overlap in the 2D space,
    it does not imply that they overlap in the original feature space.

    Parameters
    ----------
    estimator : a Scikit-Learn clusterer
        Should be an instance of a centroidal clustering algorithm (or a
        hierarchical algorithm with a specified number of clusters). Also
        accepts some other models like LDA for text clustering.
        If it is not a clusterer, an exception is raised. If the estimator
        is not fitted, it is fit when the visualizer is fitted, unless
        otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    min_size : int, default: 400
        The size, in points, of the smallest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    max_size : int, default: 25000
        The size, in points, of the largest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    embedding : default: 'mds'
        The algorithm used to embed the cluster centers in 2 dimensional space
        so that the distance between clusters is represented equivalently to
        their relationship in feature spaceself.
        Embedding algorithm options include:

        - **mds**: multidimensional scaling
        - **tsne**: stochastic neighbor embedding

    scoring : default: 'membership'
        The scoring method used to determine the size of the clusters drawn on
        the graph so that the relative importance of clusters can be viewed.
        Scoring method options include:

        - **membership**: number of instances belonging to each cluster

    legend : bool, default: True
        Whether or not to draw the size legend onto the graph, omit the legend
        to more easily see clusters that overlap.

    legend_loc : str, default: "lower left"
        The location of the legend on the graph, used to move the legend out
        of the way of clusters into open space. The same legend location
        options for matplotlib are used here.

        .. seealso:: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend

    legend_size : float, default: 1.5
        The size, in inches, of the size legend to inset into the graph.

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic embedding algorithms.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If 'auto' (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Attributes
    ----------
    cluster_centers_ : array of shape (n_clusters, n_features)
        The computed cluster centers from the underlying model.

    embedded_centers_ : array of shape (n_clusters, 2)
        The positions of all the cluster centers on the graph.

    scores_ : array of shape (n_clusters,)
        The scores of each cluster that determine its size on the graph.

    fit_time_ : Timer
        The time it took to fit the clustering model and perform the embedding.

    Notes
    -----
    Currently the only two embeddings supported are MDS and TSNE. Soon to
    follow will be PCoA and a customized version of PCoA for LDA. The only
    supported scoring metric is membership, but in the future, silhouette
    scores and cluster diameter will be added.

    In terms of algorithm support, right now any clustering algorithm that has
    a learned ``cluster_centers_`` and ``labels_`` attribute will work with
    the visualizer. In the future, we will update this to work with hierarchical
    clusterers that have ``n_components`` and LDA.
    """

    def __init__(self, estimator, ax=None, min_size=400, max_size=25000, embedding='mds', scoring='membership', legend=True, legend_loc='lower left', legend_size=1.5, random_state=None, is_fitted='auto', **kwargs):
        if False:
            return 10
        super(InterclusterDistance, self).__init__(estimator, ax=ax, **kwargs)
        validate_embedding(embedding)
        validate_scoring(scoring)
        self.scoring = scoring
        self.embedding = embedding
        self.random_state = random_state
        self.legend = legend
        self.min_size = min_size
        self.max_size = max_size
        self.legend_loc = legend_loc
        self.legend_size = legend_size
        self.facecolor = '#2e719344'
        self.edgecolor = '#2e719399'
        if self.legend:
            self.lax

    @memoized
    def lax(self):
        if False:
            while True:
                i = 10
        '\n        Returns the legend axes, creating it only on demand by creating a 2"\n        by 2" inset axes that has no grid, ticks, spines or face frame (e.g\n        is mostly invisible). The legend can then be drawn on this axes.\n        '
        if inset_locator is None:
            raise YellowbrickValueError('intercluster distance map legend requires matplotlib 2.0.2 or later please upgrade matplotlib or set legend=False ')
        lax = inset_locator.inset_axes(self.ax, width=self.legend_size, height=self.legend_size, loc=self.legend_loc)
        lax.set_frame_on(False)
        lax.set_facecolor('none')
        lax.grid(False)
        lax.set_xlim(-1.4, 1.4)
        lax.set_ylim(-1.4, 1.4)
        lax.set_xticks([])
        lax.set_yticks([])
        for name in lax.spines:
            lax.spines[name].set_visible(False)
        return lax

    @memoized
    def transformer(self):
        if False:
            while True:
                i = 10
        "\n        Creates the internal transformer that maps the cluster center's high\n        dimensional space to its two dimensional space.\n        "
        ttype = self.embedding.lower()
        if ttype == 'mds':
            return MDS(n_components=2, random_state=self.random_state)
        if ttype == 'tsne':
            return TSNE(n_components=2, random_state=self.random_state)
        raise YellowbrickValueError("unknown embedding '{}'".format(ttype))

    @property
    def cluster_centers_(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Searches for or creates cluster centers for the specified clustering\n        algorithm. This algorithm ensures that that the centers are\n        appropriately drawn and scaled so that distance between clusters are\n        maintained.\n        '
        for attr in ('cluster_centers_',):
            try:
                return getattr(self.estimator, attr)
            except AttributeError:
                continue
        raise AttributeError('could not find or make cluster_centers_ for {}'.format(self.estimator.__class__.__name__))

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        '\n        Fit the clustering model, computing the centers then embeds the centers\n        into 2D space using the embedding method specified.\n        '
        with Timer() as self.fit_time_:
            if not check_fitted(self.estimator, is_fitted_by=self.is_fitted):
                self.estimator.fit(X, y)
        C = self.cluster_centers_
        self.embedded_centers_ = self.transformer.fit_transform(C)
        self.scores_ = self._score_clusters(X, y)
        self.draw()
        return self

    def draw(self):
        if False:
            return 10
        '\n        Draw the embedded centers with their sizes on the visualization.\n        '
        sizes = self._get_cluster_sizes()
        self.ax.scatter(self.embedded_centers_[:, 0], self.embedded_centers_[:, 1], s=sizes, c=self.facecolor, edgecolor=self.edgecolor, linewidth=1)
        for (i, pt) in enumerate(self.embedded_centers_):
            self.ax.text(s=str(i), x=pt[0], y=pt[1], va='center', ha='center', fontweight='bold')
        plt.sca(self.ax)
        return self.ax

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finalize the visualization to create an "origin grid" feel instead of\n        the default matplotlib feel. Set the title, remove spines, and label\n        the grid with components. This function also adds a legend from the\n        sizes if required.\n        '
        self.set_title('{} Intercluster Distance Map (via {})'.format(self.estimator.__class__.__name__, self.embedding.upper()))
        self.ax.set_xticks([0])
        self.ax.set_yticks([0])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlabel('PC2')
        self.ax.set_ylabel('PC1')
        if self.legend:
            self._make_size_legend()

    def _score_clusters(self, X, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Determines the "scores" of the cluster, the metric that determines the\n        size of the cluster visualized on the visualization.\n        '
        stype = self.scoring.lower()
        if stype == 'membership':
            return np.bincount(self.estimator.labels_)
        raise YellowbrickValueError("unknown scoring method '{}'".format(stype))

    def _get_cluster_sizes(self):
        if False:
            return 10
        '\n        Returns the marker size (in points, e.g. area of the circle) based on\n        the scores, using the prop_to_size scaling mechanism.\n        '
        return prop_to_size(self.scores_, mi=self.min_size, ma=self.max_size, log=False, power=0.5)

    def _make_size_legend(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw a legend that shows relative sizes of the clusters at the 25th,\n        50th, and 75th percentile based on the current scoring metric.\n        '
        areas = self._get_cluster_sizes()
        radii = np.sqrt(areas / np.pi)
        scaled = np.interp(radii, (radii.min(), radii.max()), (0.1, 1))
        indices = np.array([percentile_index(self.scores_, p) for p in (25, 50, 75)])
        for idx in indices:
            center = (-0.3, 1 - scaled[idx])
            c = Circle(center, scaled[idx], facecolor='none', edgecolor='#2e7193', linewidth=1.5, linestyle='--')
            self.lax.add_patch(c)
            self.lax.annotate(self.scores_[idx], (-0.3, 1 - 2 * scaled[idx]), xytext=(1, 1 - 2 * scaled[idx]), arrowprops=dict(arrowstyle='wedge', color='#2e7193'), va='center', ha='center')
        self.lax.text(s='membership', x=0, y=1.2, va='center', ha='center')
        plt.sca(self.ax)
ICDM = InterclusterDistance

def percentile_index(a, q):
    if False:
        while True:
            i = 10
    '\n    Returns the index of the value at the Qth percentile in array a.\n    '
    return np.where(a == np.percentile(a, q, interpolation='nearest'))[0][0]

def validate_string_param(s, valid, param_name='param'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Raises a well formatted exception if s is not in valid, otherwise does not\n    raise an exception. Uses ``param_name`` to identify the parameter.\n    '
    if s.lower() not in valid:
        raise YellowbrickValueError("unknown {} '{}', chose from '{}'".format(param_name, s, ', '.join(valid)))

def validate_embedding(param):
    if False:
        i = 10
        return i + 15
    '\n    Raises an exception if the param is not in VALID_EMBEDDING\n    '
    validate_string_param(param, VALID_EMBEDDING, 'embedding')

def validate_scoring(param):
    if False:
        while True:
            i = 10
    '\n    Raises an exception if the param is not in VALID_SCORING\n    '
    validate_string_param(param, VALID_SCORING, 'scoring')

def intercluster_distance(estimator, X, y=None, ax=None, min_size=400, max_size=25000, embedding='mds', scoring='membership', legend=True, legend_loc='lower left', legend_size=1.5, random_state=None, is_fitted='auto', show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Quick Method:\n    Intercluster distance maps display an embedding of the cluster centers in\n    2 dimensions with the distance to other centers preserved. E.g. the closer\n    to centers are in the visualization, the closer they are in the original\n    feature space. The clusters are sized according to a scoring metric. By\n    default, they are sized by membership, e.g. the number of instances that\n    belong to each center. This gives a sense of the relative importance of\n    clusters. Note however, that because two clusters overlap in the 2D space,\n    it does not imply that they overlap in the original feature space.\n\n    Parameters\n    ----------\n    estimator : a Scikit-Learn clusterer\n        Should be an instance of a centroidal clustering algorithm (or a\n        hierarchical algorithm with a specified number of clusters). Also\n        accepts some other models like LDA for text clustering.\n        If it is not a clusterer, an exception is raised. If the estimator\n        is not fitted, it is fit when the visualizer is fitted, unless\n        otherwise specified by ``is_fitted``.\n\n    X : array-like of shape (n, m)\n        A matrix or data frame with n instances and m features\n\n    y : array-like of shape (n,), optional\n        A vector or series representing the target for each instance\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    min_size : int, default: 400\n        The size, in points, of the smallest cluster drawn on the graph.\n        Cluster sizes will be scaled between the min and max sizes.\n\n    max_size : int, default: 25000\n        The size, in points, of the largest cluster drawn on the graph.\n        Cluster sizes will be scaled between the min and max sizes.\n\n    embedding : default: \'mds\'\n        The algorithm used to embed the cluster centers in 2 dimensional space\n        so that the distance between clusters is represented equivalently to\n        their relationship in feature spaceself.\n        Embedding algorithm options include:\n\n        - **mds**: multidimensional scaling\n        - **tsne**: stochastic neighbor embedding\n\n    scoring : default: \'membership\'\n        The scoring method used to determine the size of the clusters drawn on\n        the graph so that the relative importance of clusters can be viewed.\n        Scoring method options include:\n\n        - **membership**: number of instances belonging to each cluster\n\n    legend : bool, default: True\n        Whether or not to draw the size legend onto the graph, omit the legend\n        to more easily see clusters that overlap.\n\n    legend_loc : str, default: "lower left"\n        The location of the legend on the graph, used to move the legend out\n        of the way of clusters into open space. The same legend location\n        options for matplotlib are used here.\n\n        .. seealso:: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend\n\n    legend_size : float, default: 1.5\n        The size, in inches, of the size legend to inset into the graph.\n\n    random_state : int or RandomState, default: None\n        Fixes the random state for stochastic embedding algorithms.\n\n    is_fitted : bool or str, default=\'auto\'\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If \'auto\' (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments passed to the base class and may influence the\n        feature visualization properties.\n\n    Returns\n    -------\n    viz : InterclusterDistance\n        The intercluster distance visualizer, fitted and finalized.\n    '
    oz = InterclusterDistance(estimator, ax=ax, min_size=min_size, max_size=max_size, embedding=embedding, scoring=scoring, legend=legend, legend_loc=legend_loc, legend_size=legend_size, random_state=random_state, is_fitted=is_fitted, **kwargs)
    oz.fit(X, y)
    if show:
        oz.show()
    else:
        oz.finalize()
    return oz