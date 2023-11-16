"""
Implements radviz for feature analysis.
"""
import numpy as np
import matplotlib.patches as patches
from yellowbrick.draw import manual_legend
from yellowbrick.utils import is_dataframe
from yellowbrick.utils import nan_warnings
from yellowbrick.features.base import DataVisualizer

class RadialVisualizer(DataVisualizer):
    """
    RadViz is a multivariate data visualization algorithm that plots each
    axis uniformely around the circumference of a circle then plots points on
    the interior of the circle such that the point normalizes its values on
    the axes from the center to each arc.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features : list, default: None
        a list of feature names to use
        The names of the features specified by the columns of the input dataset.
        This length of this list must match the number of columns in X, otherwise
        an exception will be raised on ``fit()``.

    classes : list, default: None
        a list of class names for the legend
        The class labels for each class in y, ordered by sorted class index. These
        names act as a label encoder for the legend, identifying integer classes
        or renaming string labels. If omitted, the class labels will be taken from
        the unique values in y.

        Note that the length of this list must match the number of unique values in
        y, otherwise an exception is raised. This parameter is only used in the
        discrete target type case and is ignored otherwise.

    colors : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        A single color to plot all instances as or a list of colors to color each
        instance according to its class. If not enough colors per class are
        specified then the colors are treated as a cycle.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        The colormap used to create the individual colors. If classes are
        specified the colormap is used to evenly space colors across each class.

    alpha : float, default: 1.0
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> visualizer = RadViz()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    Attributes
    ----------
    features_ : ndarray, shape (n_features,)
        The names of the features discovered or used in the visualizer that
        can be used as an index to access or modify data in X. If a user passes
        feature names in, those features are used. Otherwise the columns of a
        DataFrame are used or just simply the indices of the data array.

    classes_ : ndarray, shape (n_classes,)
        The class labels that define the discrete values in the target. Only
        available if the target type is discrete. This is guaranteed to be
        strings even if the classes are a different type.
    """

    def __init__(self, ax=None, features=None, classes=None, colors=None, colormap=None, alpha=1.0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'target_type' not in kwargs:
            kwargs['target_type'] = 'discrete'
        super(RadialVisualizer, self).__init__(ax=ax, features=features, classes=classes, colors=colors, colormap=colormap, **kwargs)
        self.alpha = alpha

    @staticmethod
    def normalize(X):
        if False:
            i = 10
            return i + 15
        '\n        MinMax normalization to fit a matrix in the space [0,1] by column.\n        '
        a = X.min(axis=0)
        b = X.max(axis=0)
        return (X - a[np.newaxis, :]) / (b - a)[np.newaxis, :]

    def fit(self, X, y=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        The fit method is the primary drawing input for the\n        visualization since it has both the X and y data required for the\n        viz and the transform method does not.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target or class values\n\n        kwargs : dict\n            Pass generic arguments to the drawing method\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the transformer/visualizer\n        '
        super(RadialVisualizer, self).fit(X, y)
        self.draw(X, y, **kwargs)
        return self

    def draw(self, X, y, **kwargs):
        if False:
            print('Hello World!')
        '\n        Called from the fit method, this method creates the radviz canvas and\n        draws each instance as a class or target colored point, whose location\n        is determined by the feature data set.\n        '
        if is_dataframe(X):
            X = X.values
        nan_warnings.warn_if_nans_exist(X)
        (X, y) = nan_warnings.filter_missing(X, y)
        (nrows, ncols) = X.shape
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        to_plot = {label: [[], []] for label in self.classes_}
        s = np.array([(np.cos(t), np.sin(t)) for t in [2.0 * np.pi * (i / float(ncols)) for i in range(ncols)]])
        for (i, row) in enumerate(self.normalize(X)):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            xy = (s * row_).sum(axis=0) / row.sum()
            label = self._label_encoder[y[i]]
            to_plot[label][0].append(xy[0])
            to_plot[label][1].append(xy[1])
        for label in self.classes_:
            color = self.get_colors([label])[0]
            self.ax.scatter(to_plot[label][0], to_plot[label][1], color=color, label=label, alpha=self.alpha, **kwargs)
        self.ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none', edgecolor='grey', linewidth=0.5))
        for (xy, name) in zip(s, self.features_):
            self.ax.add_patch(patches.Circle(xy, radius=0.025, facecolor='#777777'))
            if xy[0] < 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] - 0.025, xy[1] - 0.025, name, ha='right', va='top', size='small')
            elif xy[0] < 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] - 0.025, xy[1] + 0.025, name, ha='right', va='bottom', size='small')
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] + 0.025, xy[1] - 0.025, name, ha='left', va='top', size='small')
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] + 0.025, xy[1] + 0.025, name, ha='left', va='bottom', size='small')
        self.ax.axis('equal')
        return self.ax

    def finalize(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the title and adds a legend. Removes the ticks from the graph to\n        make a cleaner visualization.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        self.set_title('RadViz for {} Features'.format(len(self.features_)))
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        colors = self.get_colors(self.classes_)
        manual_legend(self, self.classes_, colors, loc='best')

def radviz(X, y=None, ax=None, features=None, classes=None, colors=None, colormap=None, alpha=1.0, show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Displays each feature as an axis around a circle surrounding a scatter\n    plot whose points are each individual instance.\n\n    This helper function is a quick wrapper to utilize the RadialVisualizer\n    (Transformer) for one-off analysis.\n\n    Parameters\n    ----------\n\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    y : ndarray or Series of length n, default:None\n        An array or series of target or class values\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on.\n\n    features : list of strings, default: None\n        The names of the features or columns\n\n    classes : list of strings, default: None\n        The names of the classes in the target\n\n    colors : list or tuple of colors, default: None\n        Specify the colors for each individual class\n\n    colormap : string or matplotlib cmap, default: None\n        Sequential colormap for continuous target\n\n    alpha : float, default: 1.0\n        Specify a transparency where 1 is completely opaque and 0 is completely\n        transparent. This property makes densely clustered points more visible.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n        \n    kwargs : dict\n        Keyword arguments passed to the visualizer base classes.\n\n    Returns\n    -------\n    viz : RadViz\n        Returns the fitted, finalized visualizer\n    '
    visualizer = RadialVisualizer(ax=ax, features=features, classes=classes, colors=colors, colormap=colormap, alpha=alpha, **kwargs)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer
RadViz = RadialVisualizer