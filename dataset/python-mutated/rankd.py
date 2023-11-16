"""
Implements 1D (histograms) and 2D (joint plot) feature rankings.
"""
import warnings
import numpy as np
import matplotlib as mpl
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import kendalltau as sp_kendalltau
from yellowbrick.utils import is_dataframe
from yellowbrick.features.base import MultiFeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning
__all__ = ['rank1d', 'rank2d', 'Rank1D', 'Rank2D']

def kendalltau(X):
    if False:
        print('Hello World!')
    '\n    Accepts a matrix X and returns a correlation matrix so that each column\n    is the variable and each row is the observations.\n\n    Parameters\n    ----------\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    '
    corrs = np.zeros((X.shape[1], X.shape[1]))
    for (idx, cola) in enumerate(X.T):
        for (jdx, colb) in enumerate(X.T):
            corrs[idx, jdx] = sp_kendalltau(cola, colb)[0]
    return corrs

class RankDBase(MultiFeatureVisualizer):
    """
    Base visualizer for Rank1D and Rank2D

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    algorithm : string
        The ranking algorithm to use; options and defaults vary by subclass

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the
        plot.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An n-dimensional, symmetric array of rank scores, where n is the
        number of features. E.g. for 1D ranking, it is (n,), for a
        2D ranking it is (n,n) and so forth.

    Examples
    --------

    >>> visualizer = Rank2D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    ranking_methods = {}

    def __init__(self, ax=None, fig=None, algorithm=None, features=None, show_feature_names=True, **kwargs):
        if False:
            return 10
        '\n        Initialize the class with the options required to rank and\n        order features as well as visualize the result.\n        '
        super(RankDBase, self).__init__(ax=ax, fig=fig, features=features, **kwargs)
        self.ranking_ = algorithm
        self.show_feature_names_ = show_feature_names

    def transform(self, X, **kwargs):
        if False:
            while True:
                i = 10
        "\n        The transform method is the primary drawing hook for ranking classes.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        kwargs : dict\n            Pass generic arguments to the drawing method\n\n        Returns\n        -------\n        X : ndarray\n            Typically a transformed matrix, X' is returned. However, this\n            method performs no transformation on the original data, instead\n            simply ranking the features that are in the input data and returns\n            the original data, unmodified.\n        "
        self.ranks_ = self.rank(X)
        self.draw(**kwargs)
        return X

    def rank(self, X, algorithm=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the feature ranking.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        algorithm : str or None\n            The ranking mechanism to use, or None for the default\n\n        Returns\n        -------\n        ranks : ndarray\n            An n-dimensional, symmetric array of rank scores, where n is the\n            number of features. E.g. for 1D ranking, it is (n,), for a\n            2D ranking it is (n,n) and so forth.\n        '
        algorithm = algorithm or self.ranking_
        algorithm = algorithm.lower()
        if algorithm not in self.ranking_methods:
            raise YellowbrickValueError("'{}' is unrecognized ranking method".format(algorithm))
        if is_dataframe(X):
            X = X.values
        return self.ranking_methods[algorithm](X)

    def finalize(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Sets a title on the RankD plot.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        if mpl.__version__ == '3.1.1':
            msg = 'RankD plots may be clipped when using matplotlib v3.1.1, upgrade to matplotlib v3.1.2 or later to fix the plots.'
            warnings.warn(msg, YellowbrickWarning)
        self.set_title('{} Ranking of {} Features'.format(self.ranking_.title(), len(self.features_)))

class Rank1D(RankDBase):
    """
    Rank1D computes a score for each feature in the data set with a specific
    metric or algorithm (e.g. Shapiro-Wilk) then returns the features ranked
    as a bar plot.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    algorithm : one of {'shapiro', }, default: 'shapiro'
        The ranking algorithm to use, default is 'Shapiro-Wilk.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    orient : 'h' or 'v', default='h'
        Specifies a horizontal or vertical bar chart.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the x and y ticks in the
        plot.

    color: string
        Specify color for barchart

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,), where n is the
        number of features. It is computed during `fit`.

    Examples
    --------
    >>> visualizer = Rank1D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()
    """
    ranking_methods = {'shapiro': lambda X: np.array([shapiro(x)[0] for x in X.T])}

    def __init__(self, ax=None, algorithm='shapiro', features=None, orient='h', show_feature_names=True, color=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the class with the options required to rank and\n        order features as well as visualize the result.\n        '
        super(Rank1D, self).__init__(ax=ax, algorithm=algorithm, features=features, show_feature_names=show_feature_names, **kwargs)
        self.color = color
        self.orientation_ = orient

    def draw(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Draws the bar plot of the ranking array of features.\n        '
        if self.orientation_ == 'h':
            self.ax.barh(np.arange(len(self.ranks_)), self.ranks_, color=self.color)
            self.ax.set_yticks(np.arange(len(self.ranks_)))
            if self.show_feature_names_:
                self.ax.set_yticklabels(self.features_)
            else:
                self.ax.set_yticklabels([])
            self.ax.invert_yaxis()
            self.ax.yaxis.grid(False)
        elif self.orientation_ == 'v':
            self.ax.bar(np.arange(len(self.ranks_)), self.ranks_, color=self.color)
            self.ax.set_xticks(np.arange(len(self.ranks_)))
            if self.show_feature_names_:
                self.ax.set_xticklabels(self.features_, rotation=90)
            else:
                self.ax.set_xticklabels([])
            self.ax.xaxis.grid(False)
        else:
            raise YellowbrickValueError("Orientation must be 'h' or 'v'")

class Rank2D(RankDBase):
    """
    Rank2D performs pairwise comparisons of each feature in the data set with
    a specific metric or algorithm (e.g. Pearson correlation) then returns
    them ranked as a lower left triangle diagram.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    algorithm : str, default: 'pearson'
        The ranking algorithm to use, one of: 'pearson', 'covariance', 'spearman',
        or 'kendalltau'.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or colormap to
        color them on a continuous scale.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the plot.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,n), where n is the
        number of features. It is computed during `fit`.

    Examples
    --------

    >>> visualizer = Rank2D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    ranking_methods = {'pearson': lambda X: np.corrcoef(X.transpose()), 'covariance': lambda X: np.cov(X.transpose()), 'spearman': lambda X: spearmanr(X, axis=0)[0], 'kendalltau': lambda X: kendalltau(X)}

    def __init__(self, ax=None, algorithm='pearson', features=None, colormap='RdBu_r', show_feature_names=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize the class with the options required to rank and\n        order features as well as visualize the result.\n        '
        super(Rank2D, self).__init__(ax=ax, algorithm=algorithm, features=features, show_feature_names=show_feature_names, **kwargs)
        self.colormap = colormap

    def draw(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Draws the heatmap of the ranking matrix of variables.\n        '
        self.ax.set_aspect('equal')
        mask = np.zeros_like(self.ranks_, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        data = np.ma.masked_where(mask, self.ranks_)
        mesh = self.ax.pcolormesh(data, cmap=self.colormap, vmin=-1, vmax=1)
        self.ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_yticks(np.arange(len(self.ranks_)) + 0.5)
        if self.show_feature_names_:
            self.ax.set_xticklabels(self.features_, rotation=90)
            self.ax.set_yticklabels(self.features_)
        else:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])

def rank1d(X, y=None, ax=None, algorithm='shapiro', features=None, orient='h', show_feature_names=True, color=None, show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Scores each feature with the algorithm and ranks them in a bar plot.\n\n    This helper function is a quick wrapper to utilize the Rank1D Visualizer\n    (Transformer) for one-off analysis.\n\n    Parameters\n    ----------\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    y : ndarray or Series of length n\n        An array or series of target or class values\n\n    ax : matplotlib axes\n        the axis to plot the figure on.\n\n    algorithm : one of {'shapiro', }, default: 'shapiro'\n        The ranking algorithm to use, default is 'Shapiro-Wilk.\n\n    features : list\n        A list of feature names to use.\n        If a DataFrame is passed to fit and features is None, feature\n        names are selected as the columns of the DataFrame.\n\n    orient : 'h' or 'v'\n        Specifies a horizontal or vertical bar chart.\n\n    show_feature_names : boolean, default: True\n        If True, the feature names are used to label the axis ticks in the\n        plot.\n\n    color: string\n        Specify color for barchart\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n     kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    viz : Rank1D\n        Returns the fitted, finalized visualizer.\n\n    "
    visualizer = Rank1D(ax=ax, algorithm=algorithm, features=features, orient=orient, show_feature_names=show_feature_names, color=color, **kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer

def rank2d(X, y=None, ax=None, algorithm='pearson', features=None, colormap='RdBu_r', show_feature_names=True, show=True, **kwargs):
    if False:
        return 10
    "Rank2D quick method\n\n    Rank2D performs pairwise comparisons of each feature in the data set with\n    a specific metric or algorithm (e.g. Pearson correlation) then returns\n    them ranked as a lower left triangle diagram.\n\n    Parameters\n    ----------\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features to perform the pairwise compairsons on.\n\n    y : ndarray or Series of length n, default: None\n        An array or series of target or class values, optional (not used).\n\n    ax : matplotlib Axes, default: None\n        The axis to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    algorithm : str, default: 'pearson'\n        The ranking algorithm to use, one of: 'pearson', 'covariance', 'spearman',\n        or 'kendalltau'.\n\n    features : list\n        A list of feature names to use.\n        If a DataFrame is passed to fit and features is None, feature names are\n        selected as the columns of the DataFrame.\n\n    colormap : string or cmap, default: 'RdBu_r'\n        optional string or matplotlib cmap to colorize lines\n        Use either color to colorize the lines on a per class basis or colormap to\n        color them on a continuous scale.\n\n    show_feature_names : boolean, default: True\n        If True, the feature names are used to label the axis ticks in the plot.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    viz : Rank2D\n        Returns the fitted, finalized visualizer that created the Rank2D heatmap.\n    "
    viz = Rank2D(ax=ax, algorithm=algorithm, features=features, colormap=colormap, show_feature_names=show_feature_names, **kwargs)
    viz.fit(X, y)
    viz.transform(X)
    if show:
        viz.show()
    else:
        viz.finalize()
    return viz