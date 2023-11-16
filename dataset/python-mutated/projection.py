"""
Base class for all projection (decomposition) high dimensional data visualizers.
"""
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d
from yellowbrick.draw import manual_legend
from yellowbrick.features.base import DataVisualizer, TargetType
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning, NotFitted

class ProjectionVisualizer(DataVisualizer):
    """
    The ProjectionVisualizer provides functionality for projecting a multi-dimensional
    dataset into either 2 or 3 components so they can be plotted as a scatter plot on
    2d or 3d axes. The visualizer acts as a transformer, and draws the transformed data
    on behalf of the user. Because it is a DataVisualizer, the ProjectionVisualizer
    can plot continuous scatter plots with a colormap or discrete scatter plots with
    a legend.

    This visualizer is a base class and is not intended to be uses directly.
    Subclasses should implement a ``transform()`` method that calls ``draw()`` using
    the transformed data and the optional target as input.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes.
        will be used (or generated if required).

    features : list, default: None
        The names of the features specified by the columns of the input dataset.
        This length of this list must match the number of columns in X, otherwise
        an exception will be raised on ``fit()``.

    classes : list, default: None
        The class labels for each class in y, ordered by sorted class index. These
        names act as a label encoder for the legend, identifying integer classes
        or renaming string labels. If omitted, the class labels will be taken from
        the unique values in y.

        Note that the length of this list must match the number of unique values in
        y, otherwise an exception is raised. This parameter is only used in the
        discrete target type case and is ignored otherwise.

    colors : list or tuple, default: None
        A single color to plot all instances as or a list of colors to color each
        instance according to its class in the discrete case or as an ordered
        colormap in the sequential case. If not enough colors per class are
        specified then the colors are treated as a cycle.

    colormap : string or cmap, default: None
        The colormap used to create the individual colors. In the discrete case
        it is used to compute the number of colors needed for each class and
        in the continuous case it is used to create a sequential color map based
        on the range of the target.

    target_type : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", then it will
        attempt to determine the type by counting the number of unique values.

        If the target is discrete, the colors are returned as a dict with classes
        being the keys. If continuous the colors will be list having value of
        color for each point. In either case, if no target is specified, then
        color will be specified as the first color in the color cycle.

    projection : int or string, default: 2
        The number of axes to project into, either 2d or 3d. To plot 3d plots
        with matplotlib, please ensure a 3d axes is passed to the visualizer,
        otherwise one will be created using the current figure.

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
    """

    def __init__(self, ax=None, features=None, classes=None, colors=None, colormap=None, target_type='auto', projection=2, alpha=0.75, colorbar=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ProjectionVisualizer, self).__init__(ax=ax, features=features, classes=classes, colors=colors, colormap=colormap, target_type=target_type, **kwargs)
        if isinstance(projection, str):
            if projection in {'2D', '2d'}:
                projection = 2
            if projection in {'3D', '3d'}:
                projection = 3
        if projection not in {2, 3}:
            raise YellowbrickValueError('Projection dimensions must be either 2 or 3')
        self.projection = projection
        if self.ax.name != '3d' and self.projection == 3:
            warnings.warn('data projection to 3 dimensions requires a 3d axes to draw on.', YellowbrickWarning)
        self.alpha = alpha
        self.colorbar = colorbar
        self._cax = None

    @property
    def cax(self):
        if False:
            return 10
        '\n        The axes of the colorbar, right of the scatterplot.\n        '
        if self._cax is None:
            raise AttributeError('This visualizer does not have an axes for colorbar')
        return self._cax

    @property
    def ax(self):
        if False:
            print('Hello World!')
        '\n        Overloads the axes property from base class. If no axes is specified then\n        creates an axes for users. A 3d axes is created for 3 dimensional plots.\n        '
        if not hasattr(self, '_ax') or self._ax is None:
            if self.projection == 3:
                fig = plt.gcf()
                self._ax = fig.add_subplot(111, projection='3d')
            else:
                self._ax = plt.gca()
        return self._ax

    @ax.setter
    def ax(self, ax):
        if False:
            for i in range(10):
                print('nop')
        self._ax = ax

    def layout(self, divider=None):
        if False:
            while True:
                i = 10
        '\n        Creates the layout for colorbar when target type is continuous.\n        The colorbar is added to the right of the scatterplot.\n\n        Subclasses can override this method to add other axes or layouts.\n\n        Parameters\n        ----------\n        divider: AxesDivider\n            An AxesDivider to be passed among all layout calls.\n        '
        if self._target_color_type == TargetType.CONTINUOUS and self.projection == 2 and self.colorbar and (self._cax is None):
            if make_axes_locatable is None:
                raise YellowbrickValueError('Colorbar requires matplotlib 2.0.2 or greater please upgrade matplotlib')
            if divider is None:
                divider = make_axes_locatable(self.ax)
            self._cax = divider.append_axes('right', size='5%', pad=0.3)
            self._cax.set_yticks([])
            self._cax.set_xticks([])

    def fit_transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits the visualizer on the input data, and returns transformed X.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix or data frame of n instances with m features where m>2.\n\n        y : array-like of shape (n,), optional\n            A vector or series with target values for each instance in X. This\n            vector is used to determine the color of the points in X.\n\n        Returns\n        -------\n        Xprime : array-like of shape (n, 2)\n            Returns the 2-dimensional embedding of the instances.\n        '
        return self.fit(X, y).transform(X, y)

    def draw(self, Xp, y=None):
        if False:
            return 10
        '\n        Draws the points described by Xp and colored by the points in y. Can be\n        called multiple times before finalize to add more scatter plots to the\n        axes, however ``fit()`` must be called before use.\n\n        Parameters\n        ----------\n        Xp : array-like of shape (n, 2) or (n, 3)\n            The matrix produced by the ``transform()`` method.\n\n        y : array-like of shape (n,), optional\n            The target, used to specify the colors of the points.\n\n        Returns\n        -------\n        self.ax : matplotlib Axes object\n            Returns the axes that the scatter plot was drawn on.\n        '
        scatter_kwargs = self._determine_scatter_kwargs(y)
        self.layout()
        if self.projection == 2:
            self.ax.scatter(Xp[:, 0], Xp[:, 1], **scatter_kwargs)
        if self.projection == 3:
            self.ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], **scatter_kwargs)
        return self.ax

    def finalize(self):
        if False:
            while True:
                i = 10
        '\n        Draws legends and colorbar for scatter plots.\n        '
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        if self.projection == 3:
            self.ax.set_zticklabels([])
        if self._target_color_type == TargetType.DISCRETE:
            manual_legend(self, self.classes_, list(self._colors.values()), frameon=True)
        elif self._target_color_type == TargetType.CONTINUOUS:
            if self.colorbar:
                if self.projection == 3:
                    sm = plt.cm.ScalarMappable(cmap=self._colors, norm=self._norm)
                    sm.set_array([])
                    self.cbar = plt.colorbar(sm, ax=self.ax)
                else:
                    self.cbar = mpl.colorbar.ColorbarBase(self.cax, cmap=self._colors, norm=self._norm)

    def _determine_scatter_kwargs(self, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Determines scatter argumnets to pass into ``plt.scatter()``. If y is\n        discrete or single then determine colors. If continuous then determine\n        colors and colormap.Also normalize to range\n\n        Parameters\n        ----------\n        y : array-like of shape (n,), optional\n            The target, used to specify the colors of the points for continuous\n            target.\n        '
        scatter_kwargs = {'alpha': self.alpha}
        if self._target_color_type == TargetType.SINGLE:
            scatter_kwargs['c'] = self._colors
        elif self._target_color_type == TargetType.DISCRETE:
            if y is None:
                raise YellowbrickValueError('y is required for discrete target')
            try:
                scatter_kwargs['c'] = [self._colors[self.classes_[yi]] for yi in y]
            except IndexError:
                raise YellowbrickValueError('Target needs to be label encoded.')
        elif self._target_color_type == TargetType.CONTINUOUS:
            if y is None:
                raise YellowbrickValueError('y is required for continuous target')
            scatter_kwargs['c'] = y
            scatter_kwargs['cmap'] = self._colors
            self._norm = mpl.colors.Normalize(vmin=self.range_[0], vmax=self.range_[1])
        else:
            raise NotFitted('could not determine target color type')
        return scatter_kwargs