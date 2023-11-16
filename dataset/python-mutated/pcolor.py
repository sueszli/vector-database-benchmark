"""
Colorplot visualizer for gridsearch results.
"""
import numpy as np
from .base import GridSearchVisualizer
__all__ = ['GridSearchColorPlot', 'gridsearch_color_plot']

def gridsearch_color_plot(estimator, x_param, y_param, X=None, y=None, ax=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Quick method:\n    Create a color plot showing the best grid search scores across two\n    parameters.\n\n    This helper function is a quick wrapper to utilize GridSearchColorPlot\n    for one-off analysis.\n\n    If no `X` data is passed, the model is assumed to be fit already. This\n    allows quick exploration without waiting for the grid search to re-run.\n\n    Parameters\n    ----------\n    estimator : Scikit-Learn grid search object\n        Should be an instance of GridSearchCV. If not, an exception is raised.\n        The model may be fit or unfit.\n\n    x_param : string\n        The name of the parameter to be visualized on the horizontal axis.\n\n    y_param : string\n        The name of the parameter to be visualized on the vertical axis.\n\n    metric : string (default 'mean_test_score')\n        The field from the grid search's `cv_results` that we want to display.\n\n    X  : ndarray or DataFrame of shape n x m or None (default None)\n        A matrix of n instances with m features. If not None, forces the\n        GridSearchCV object to be fit.\n\n    y  : ndarray or Series of length n or None (default None)\n        An array or series of target or class values.\n\n    ax : matplotlib axes\n        The axes to plot the figure on.\n\n    classes : list of strings\n        The names of the classes in the target\n\n    Returns\n    -------\n    ax : matplotlib axes\n        Returns the axes that the classification report was drawn on.\n    "
    visualizer = GridSearchColorPlot(estimator, x_param, y_param, ax=ax, **kwargs)
    if X is not None:
        visualizer.fit(X, y)
    else:
        visualizer.draw()
    return visualizer.ax

class GridSearchColorPlot(GridSearchVisualizer):
    """
    Create a color plot showing the best grid search scores across two
    parameters.

    Parameters
    ----------
    estimator : Scikit-Learn grid search object
        Should be an instance of GridSearchCV. If not, an exception is raised.

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    metric : string (default 'mean_test_score')
        The field from the grid search's `cv_results` that we want to display.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> from yellowbrick.gridsearch import GridSearchColorPlot
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import SVC
    >>> gridsearch = GridSearchCV(SVC(),
                                  {'kernel': ['rbf', 'linear'], 'C': [1, 10]})
    >>> model = GridSearchColorPlot(gridsearch, x_param='kernel', y_param='C')
    >>> model.fit(X)
    >>> model.show()
    """

    def __init__(self, estimator, x_param, y_param, metric='mean_test_score', colormap='RdBu_r', ax=None, **kwargs):
        if False:
            while True:
                i = 10
        super(GridSearchColorPlot, self).__init__(estimator, ax=ax, **kwargs)
        self.x_param = x_param
        self.y_param = y_param
        self.metric = metric
        self.colormap = colormap

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        (x_vals, y_vals, best_scores) = self.param_projection(self.x_param, self.y_param, metric=self.metric)
        data = np.ma.masked_invalid(best_scores)
        mesh = self.ax.pcolor(data, cmap=self.colormap, vmin=np.nanmin(data), vmax=np.nanmax(data))
        self.ax.patch.set(hatch='x', edgecolor='black')
        self.ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        self.ax.set_yticks(np.arange(len(y_vals)) + 0.5)
        self.ax.set_xticklabels(x_vals, rotation=45)
        self.ax.set_yticklabels(y_vals, rotation=45)
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)
        self.ax.set_aspect('equal')

    def finalize(self):
        if False:
            i = 10
            return i + 15
        self.set_title('Grid Search Scores')
        self.ax.set_xlabel(self.x_param)
        self.ax.set_ylabel(self.y_param)