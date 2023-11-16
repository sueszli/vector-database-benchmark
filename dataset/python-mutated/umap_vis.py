"""
Implements UMAP visualizations of documents in 2D space.
"""
import warnings
import numpy as np
from collections import defaultdict
from yellowbrick.draw import manual_legend
from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.pipeline import Pipeline
try:
    from umap import UMAP
except ImportError:
    UMAP = None
except (RuntimeError, AttributeError):
    UMAP = None
    warnings.warn('Error Importing UMAP.  UMAP does not support python 2.7 on Windows 32 bit.')

def umap(X, y=None, ax=None, classes=None, colors=None, colormap=None, alpha=0.7, show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display a projection of a vectorized corpus in two dimensions using UMAP (Uniform\n    Manifold Approximation and Projection), a nonlinear dimensionality reduction method\n    that is particularly well suited to embedding in two or three dimensions for\n    visualization as a scatter plot. UMAP is a relatively new technique but is often\n    used to visualize clusters or groups of data points and their relative proximities.\n    It typically is fast, scalable, and can be applied directly to sparse matrices\n    eliminating the need to run a ``TruncatedSVD`` as a pre-processing step.\n\n    The current default for UMAP is Euclidean distance. Hellinger distance would be a\n    more appropriate distance function to use with CountVectorize data. That will be\n    released in a forthcoming version of UMAP. In the meantime cosine distance is likely\n    a better text default that Euclidean and can be set using the keyword argument\n    ``metric='cosine'``.\n\n    Parameters\n    ----------\n\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features representing the corpus of\n        vectorized documents to visualize with umap.\n\n    y : ndarray or Series of length n\n        An optional array or series of target or class values for instances.\n        If this is specified, then the points will be colored according to\n        their class. Often cluster labels are passed in to color the documents\n        in cluster space, so this method is used both for classification and\n        clustering methods.\n\n    ax : matplotlib axes\n        The axes to plot the figure on.\n\n    classes : list of strings\n        The names of the classes in the target, used to create a legend.\n\n    colors : list or tuple of colors\n        Specify the colors for each individual class\n\n    colormap : string or matplotlib cmap\n        Sequential colormap for continuous target\n\n    alpha : float, default: 0.7\n        Specify a transparency where 1 is completely opaque and 0 is completely\n        transparent. This property makes densely clustered points more visible.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs : dict\n        Pass any additional keyword arguments to the UMAP transformer.\n\n    -------\n    visualizer: UMAPVisualizer\n        Returns the fitted, finalized visualizer\n    "
    visualizer = UMAPVisualizer(ax=ax, classes=classes, colors=colors, colormap=colormap, alpha=alpha, **kwargs)
    visualizer.fit(X, y, **kwargs)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer

class UMAPVisualizer(TextVisualizer):
    """
    Display a projection of a vectorized corpus in two dimensions using UMAP (Uniform
    Manifold Approximation and Projection), a nonlinear dimensionality reduction method
    that is particularly well suited to embedding in two or three dimensions for
    visualization as a scatter plot. UMAP is a relatively new technique but is often
    used to visualize clusters or groups of data points and their relative proximities.
    It typically is fast, scalable, and can be applied directly to sparse matrices
    eliminating the need to run a ``TruncatedSVD`` as a pre-processing step.

    The current default for UMAP is Euclidean distance. Hellinger distance would be a
    more appropriate distance function to use with CountVectorize data. That will be
    released in a forthcoming version of UMAP. In the meantime cosine distance is likely
    a better text default that Euclidean and can be set using the keyword argument
    ``metric='cosine'``.

    For more, see https://github.com/lmcinnes/umap

    Parameters
    ----------

    ax : matplotlib axes
        The axes to plot the figure on.

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
        decomposition as well as UMAP.

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Pass any additional keyword arguments to the UMAP transformer.

    Examples
    --------

    >>> model = MyVisualizer(metric='cosine')
    >>> model.fit(X)
    >>> model.show()

    """
    NULL_CLASS = None

    def __init__(self, ax=None, labels=None, classes=None, colors=None, colormap=None, random_state=None, alpha=0.7, **kwargs):
        if False:
            print('Hello World!')
        if UMAP is None:
            raise YellowbrickValueError("umap package doesn't seem to be installed.Please install UMAP via: pip install umap-learn")
        self.alpha = alpha
        self.labels = labels
        self.colors = colors
        self.colormap = colormap
        self.random_state = random_state
        umap_kwargs = {key: kwargs.pop(key) for key in UMAP().get_params() if key in kwargs}
        self.transformer_ = self.make_transformer(umap_kwargs)
        super(UMAPVisualizer, self).__init__(ax=ax, **kwargs)

    def make_transformer(self, umap_kwargs={}):
        if False:
            while True:
                i = 10
        '\n        Creates an internal transformer pipeline to project the data set into\n        2D space using UMAP. This method will reset the transformer on the\n        class.\n\n        Parameters\n        ----------\n        umap_kwargs : dict\n            Keyword arguments for the internal UMAP transformer\n\n        Returns\n        -------\n        transformer : Pipeline\n            Pipelined transformer for UMAP projections\n        '
        steps = []
        steps.append(('umap', UMAP(n_components=2, random_state=self.random_state, **umap_kwargs)))
        return Pipeline(steps)

    def fit(self, X, y=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        The fit method is the primary drawing input for the UMAP projection\n        since the visualization requires both X and an optional y value. The\n        fit method expects an array of numeric vectors, so text documents must\n        be vectorized before passing them to this method.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features representing the corpus of\n            vectorized documents to visualize with UMAP.\n\n        y : ndarray or Series of length n\n            An optional array or series of target or class values for\n            instances. If this is specified, then the points will be colored\n            according to their class. Often cluster labels are passed in to\n            color the documents in cluster space, so this method is used both\n            for classification and clustering methods.\n\n        kwargs : dict\n            Pass generic arguments to the drawing method\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the transformer/visualizer\n        '
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
            while True:
                i = 10
        '\n        Called from the fit method, this method draws the UMAP scatter plot,\n        from a set of decomposed points in 2 dimensions. This method also\n        accepts a third dimension, target, which is used to specify the colors\n        of each of the points. If the target is not specified, then the points\n        are plotted as a single cloud to show similar documents.\n        '
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
        '\n        Finalize the drawing by adding a title and legend, and removing the\n        axes objects that do not convey information about UMAP.\n        '
        self.set_title('UMAP Projection of {} Documents'.format(self.n_instances_))
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        if not all(self.classes_ == np.array([self.NULL_CLASS])):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            manual_legend(self, self.classes_, self.color_values_, loc='center left', bbox_to_anchor=(1, 0.5))