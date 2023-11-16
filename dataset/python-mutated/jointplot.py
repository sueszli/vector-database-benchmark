import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None
from .base import FeatureVisualizer
from ..utils.types import is_dataframe
from ..exceptions import YellowbrickValueError
from scipy.stats import pearsonr, spearmanr, kendalltau
FACECOLOR = '#FAFAFA'
HISTCOLOR = '#6897bb'
__all__ = ['JointPlot', 'JointPlotVisualizer', 'joint_plot']

class JointPlot(FeatureVisualizer):
    """
    Joint plots are useful for machine learning on multi-dimensional data, allowing for
    the visualization of complex interactions between different data dimensions, their
    varying distributions, and even their relationships to the target variable for
    prediction.

    The Yellowbrick ``JointPlot`` can be used both for pairwise feature analysis and
    feature-to-target plots. For pairwise feature analysis, the ``columns`` argument can
    be used to specify the index of the two desired columns in ``X``. If ``y`` is also
    specified, the plot can be colored with a heatmap or by class. For feature-to-target
    plots, the user can provide either ``X`` and ``y`` as 1D vectors, or a ``columns``
    argument with an index to a single feature in ``X`` to be plotted against ``y``.

    Histograms can be included by setting the ``hist`` argument to ``True`` for a
    frequency distribution, or to ``"density"`` for a probability density function. Note
    that histograms requires matplotlib 2.0.2 or greater.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes will be
        used (or generated if required). This is considered the base axes where the
        the primary joint plot is drawn. It will be shifted and two additional axes
        added above (xhax) and to the right (yhax) if hist=True.

    columns : int, str, [int, int], [str, str], default: None
        Determines what data is plotted in the joint plot and acts as a selection index
        into the data passed to ``fit(X, y)``. This data therefore must be indexable by
        the column type (e.g. an int for a numpy array or a string for a DataFrame).

        If None is specified then either both X and y must be 1D vectors and they will
        be plotted against each other or X must be a 2D array with only 2 columns. If a
        single index is specified then the data is indexed as ``X[columns]`` and plotted
        jointly with the target variable, y. If two indices are specified then they are
        both selected from X, additionally in this case, if y is specified, then it is
        used to plot the color of points.

        Note that these names are also used as the x and y axes labels if they aren't
        specified in the joint_kws argument.

    correlation : str, default: 'pearson'
        The algorithm used to compute the relationship between the variables in the
        joint plot, one of: 'pearson', 'covariance', 'spearman', 'kendalltau'.

    kind : str in {'scatter', 'hex'}, default: 'scatter'
        The type of plot to render in the joint axes. Note that when kind='hex' the
        target cannot be plotted by color.

    hist : {True, False, None, 'density', 'frequency'}, default: True
        Draw histograms showing the distribution of the variables plotted jointly.
        If set to 'density', the probability density function will be plotted.
        If set to True or 'frequency' then the frequency will be plotted.
        Requires Matplotlib >= 2.0.2.

    alpha : float, default: 0.65
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    {joint, hist}_kws : dict, default: None
        Additional keyword arguments for the plot components.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    corr_ : float
        The correlation or relationship of the data in the joint plot, specified by the
        correlation algorithm.

    Examples
    --------

    >>> viz = JointPlot(columns=["temp", "humidity"])
    >>> viz.fit(X, y)
    >>> viz.show()
    """
    correlation_methods = {'pearson': lambda x, y: pearsonr(x, y)[0], 'spearman': lambda x, y: spearmanr(x, y)[0], 'covariance': lambda x, y: np.cov(x, y)[0, 1], 'kendalltau': lambda x, y: kendalltau(x, y)[0]}

    def __init__(self, ax=None, columns=None, correlation='pearson', kind='scatter', hist=True, alpha=0.65, joint_kws=None, hist_kws=None, **kwargs):
        if False:
            print('Hello World!')
        super(JointPlot, self).__init__(ax=ax, **kwargs)
        (self._xhax, self._yhax) = (None, None)
        self.columns = columns
        if self.columns is not None and (not isinstance(self.columns, (int, str))):
            self.columns = tuple(self.columns)
            if len(self.columns) > 2:
                raise YellowbrickValueError("'{}' contains too many indices or is invalid for joint plot - specify either a single int or str index or two columns as a list".format(columns))
        self.correlation = correlation
        if self.correlation not in self.correlation_methods:
            raise YellowbrickValueError("'{}' is an invalid correlation method, use one of {}".format(self.correlation, ', '.join(self.correlation_methods.keys())))
        self.kind = kind
        if self.kind not in {'scatter', 'hex', 'hexbin'}:
            raise YellowbrickValueError("'{}' is invalid joint plot kind, use 'scatter' or 'hex'".format(self.kind))
        self.hist = hist
        if self.hist not in {True, 'density', 'frequency', None, False}:
            raise YellowbrickValueError("'{}' is an invalid argument for hist, use None, True, False, 'density', or 'frequency'".format(hist))
        if self.hist in {True, 'density', 'frequency'}:
            self._layout()
        self.alpha = alpha
        self.joint_kws = joint_kws
        self.hist_kws = hist_kws

    @property
    def xhax(self):
        if False:
            while True:
                i = 10
        '\n        The axes of the histogram for the top of the JointPlot (X-axis)\n        '
        if self._xhax is None:
            raise AttributeError('this visualizer does not have a histogram for the X axis')
        return self._xhax

    @property
    def yhax(self):
        if False:
            print('Hello World!')
        '\n        The axes of the histogram for the right of the JointPlot (Y-axis)\n        '
        if self._yhax is None:
            raise AttributeError('this visualizer does not have a histogram for the Y axis')
        return self._yhax

    def _layout(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates the grid layout for the joint plot, adding new axes for the histograms\n        if necessary and modifying the aspect ratio. Does not modify the axes or the\n        layout if self.hist is False or None.\n        '
        if not self.hist:
            self.ax
            return
        if make_axes_locatable is None:
            raise YellowbrickValueError('joint plot histograms requires matplotlib 2.0.2 or greater please upgrade matplotlib or set hist=False on the visualizer')
        divider = make_axes_locatable(self.ax)
        self._xhax = divider.append_axes('top', size=1, pad=0.1, sharex=self.ax)
        self._yhax = divider.append_axes('right', size=1, pad=0.1, sharey=self.ax)
        self._xhax.xaxis.tick_top()
        self._yhax.yaxis.tick_right()
        self._xhax.grid(False, axis='y')
        self._yhax.grid(False, axis='x')

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        '\n        Fits the JointPlot, creating a correlative visualization between the columns\n        specified during initialization and the data and target passed into fit:\n\n            - If self.columns is None then X and y must both be specified as 1D arrays\n              or X must be a 2D array with only 2 columns.\n            - If self.columns is a single int or str, that column is selected to be\n              visualized against the target y.\n            - If self.columns is two ints or strs, those columns are visualized against\n              each other. If y is specified then it is used to color the points.\n\n        This is the main entry point into the joint plot visualization.\n\n        Parameters\n        ----------\n        X : array-like\n            An array-like object of either 1 or 2 dimensions depending on self.columns.\n            Usually this is a 2D table with shape (n, m)\n\n        y : array-like, default: None\n            An vector or 1D array that has the same length as X. May be used to either\n            directly plot data or to color data points.\n        '
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if y is not None and isinstance(y, (list, tuple)):
            y = np.array(y)
        if self.columns is None:
            if y is None and (X.ndim != 2 or X.shape[1] != 2) or (y is not None and (X.ndim != 1 or y.ndim != 1)):
                raise YellowbrickValueError('when self.columns is None specify either X and y as 1D arrays or X as a matrix with 2 columns')
            if y is None:
                self.draw(X[:, 0], X[:, 1], xlabel='0', ylabel='1')
                return self
            self.draw(X, y, xlabel='x', ylabel='y')
            return self
        if isinstance(self.columns, (int, str)):
            if y is None:
                raise YellowbrickValueError('when self.columns is a single index, y must be specified')
            x = self._index_into(self.columns, X)
            self.draw(x, y, xlabel=str(self.columns), ylabel='target')
            return self
        columns = tuple(self.columns)
        if len(columns) != 2:
            raise YellowbrickValueError("'{}' contains too many indices or is invalid for joint plot".format(columns))
        x = self._index_into(columns[0], X)
        y = self._index_into(columns[1], X)
        self.draw(x, y, xlabel=str(columns[0]), ylabel=str(columns[1]))
        return self

    def draw(self, x, y, xlabel=None, ylabel=None):
        if False:
            return 10
        '\n        Draw the joint plot for the data in x and y.\n\n        Parameters\n        ----------\n        x, y : 1D array-like\n            The data to plot for the x axis and the y axis\n\n        xlabel, ylabel : str\n            The labels for the x and y axes.\n        '
        self.corr_ = self.correlation_methods[self.correlation](x, y)
        joint_kws = self.joint_kws or {}
        joint_kws.setdefault('alpha', self.alpha)
        joint_kws.setdefault('label', '{}={:0.3f}'.format(self.correlation, self.corr_))
        if self.kind == 'scatter':
            self.ax.scatter(x, y, **joint_kws)
        elif self.kind in ('hex', 'hexbin'):
            joint_kws.setdefault('mincnt', 1)
            joint_kws.setdefault('gridsize', 50)
            joint_kws.setdefault('cmap', 'Blues')
            self.ax.hexbin(x, y, **joint_kws)
        else:
            raise ValueError("unknown joint plot kind '{}'".format(self.kind))
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if not self.hist:
            plt.sca(self.ax)
            return self.ax
        hist_kws = self.hist_kws or {}
        hist_kws.setdefault('bins', 50)
        if self.hist == 'density':
            hist_kws.setdefault('density', True)
        self.xhax.hist(x, **hist_kws)
        self.yhax.hist(y, orientation='horizontal', **hist_kws)
        plt.sca(self.ax)
        return self.ax

    def finalize(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finalize executes any remaining image modifications making it ready to show.\n        '
        self.set_title('')
        if self.kind == 'scatter':
            self.ax.legend(loc='best', frameon=True)
        if self.hist:
            plt.setp(self.xhax.get_xticklabels(), visible=False)
            plt.setp(self.yhax.get_yticklabels(), visible=False)
            plt.sca(self.ax)
        self.fig.tight_layout()

    def _index_into(self, idx, data):
        if False:
            i = 10
            return i + 15
        '\n        Attempts to get the column from the data using the specified index, raises an\n        exception if this is not possible from this point in the stack.\n        '
        try:
            if is_dataframe(data):
                return data[idx]
            return data[:, idx]
        except Exception as e:
            raise IndexError("could not index column '{}' into type {}: {}".format(self.columns, data.__class__.__name__, e))
JointPlotVisualizer = JointPlot

def joint_plot(X, y, ax=None, columns=None, correlation='pearson', kind='scatter', hist=True, alpha=0.65, joint_kws=None, hist_kws=None, show=True, **kwargs):
    if False:
        return 10
    '\n    Joint plots are useful for machine learning on multi-dimensional data, allowing for\n    the visualization of complex interactions between different data dimensions, their\n    varying distributions, and even their relationships to the target variable for\n    prediction.\n\n    The Yellowbrick ``JointPlot`` can be used both for pairwise feature analysis and\n    feature-to-target plots. For pairwise feature analysis, the ``columns`` argument can\n    be used to specify the index of the two desired columns in ``X``. If ``y`` is also\n    specified, the plot can be colored with a heatmap or by class. For feature-to-target\n    plots, the user can provide either ``X`` and ``y`` as 1D vectors, or a ``columns``\n    argument with an index to a single feature in ``X`` to be plotted against ``y``.\n\n    Histograms can be included by setting the ``hist`` argument to ``True`` for a\n    frequency distribution, or to ``"density"`` for a probability density function. Note\n    that histograms requires matplotlib 2.0.2 or greater.\n\n    Parameters\n    ----------\n    X : array-like\n        An array-like object of either 1 or 2 dimensions depending on self.columns.\n        Usually this is a 2D table with shape (n, m)\n\n    y : array-like, default: None\n        An vector or 1D array that has the same length as X. May be used to either\n        directly plot data or to color data points.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes will be\n        used (or generated if required). This is considered the base axes where the\n        the primary joint plot is drawn. It will be shifted and two additional axes\n        added above (xhax) and to the right (yhax) if hist=True.\n\n    columns : int, str, [int, int], [str, str], default: None\n        Determines what data is plotted in the joint plot and acts as a selection index\n        into the data passed to ``fit(X, y)``. This data therefore must be indexable by\n        the column type (e.g. an int for a numpy array or a string for a DataFrame).\n\n        If None is specified then either both X and y must be 1D vectors and they will\n        be plotted against each other or X must be a 2D array with only 2 columns. If a\n        single index is specified then the data is indexed as ``X[columns]`` and plotted\n        jointly with the target variable, y. If two indices are specified then they are\n        both selected from X, additionally in this case, if y is specified, then it is\n        used to plot the color of points.\n\n        Note that these names are also used as the x and y axes labels if they aren\'t\n        specified in the joint_kws argument.\n\n    correlation : str, default: \'pearson\'\n        The algorithm used to compute the relationship between the variables in the\n        joint plot, one of: \'pearson\', \'covariance\', \'spearman\', \'kendalltau\'.\n\n    kind : str in {\'scatter\', \'hex\'}, default: \'scatter\'\n        The type of plot to render in the joint axes. Note that when kind=\'hex\' the\n        target cannot be plotted by color.\n\n    hist : {True, False, None, \'density\', \'frequency\'}, default: True\n        Draw histograms showing the distribution of the variables plotted jointly.\n        If set to \'density\', the probability density function will be plotted.\n        If set to True or \'frequency\' then the frequency will be plotted.\n        Requires Matplotlib >= 2.0.2.\n\n    alpha : float, default: 0.65\n        Specify a transparency where 1 is completely opaque and 0 is completely\n        transparent. This property makes densely clustered points more visible.\n\n    {joint, hist}_kws : dict, default: None\n        Additional keyword arguments for the plot components.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Attributes\n    ----------\n    corr_ : float\n        The correlation or relationship of the data in the joint plot, specified by the\n        correlation algorithm.\n    '
    visualizer = JointPlot(ax=ax, columns=columns, correlation=correlation, kind=kind, hist=hist, alpha=alpha, joint_kws=joint_kws, hist_kws=hist_kws, **kwargs)
    visualizer.fit(X, y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer