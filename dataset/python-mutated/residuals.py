"""
Visualize the residuals between predicted and actual data for regression problems
"""
import matplotlib.pyplot as plt
from scipy.stats import probplot
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None
from yellowbrick.draw import manual_legend
from yellowbrick.utils.decorators import memoized
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.regressor.base import RegressionScoreVisualizer
__all__ = ['ResidualsPlot', 'residuals_plot']

class ResidualsPlot(RegressionScoreVisualizer):
    """
    A residual plot shows the residuals on the vertical axis and the
    independent variable on the horizontal axis.

    If the points are randomly dispersed around the horizontal axis, a linear
    regression model is appropriate for the data; otherwise, a non-linear
    model is more appropriate.

    Parameters
    ----------
    estimator : a Scikit-Learn regressor
        Should be an instance of a regressor, otherwise will raise a
        YellowbrickTypeError exception on instantiation.
        If the estimator is not fitted, it is fit when the visualizer is fitted,
        unless otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    hist : {True, False, None, 'density', 'frequency'}, default: True
        Draw a histogram showing the distribution of the residuals on the
        right side of the figure. Requires Matplotlib >= 2.0.2.
        If set to 'density', the probability density function will be plotted.
        If set to True or 'frequency' then the frequency will be plotted.

    qqplot : {True, False}, default: False
        Draw a Q-Q plot on the right side of the figure, comparing the quantiles
        of the residuals against quantiles of a standard normal distribution.
        Q-Q plot and histogram of residuals can not be plotted simultaneously,
        either `hist` or `qqplot` has to be set to False.

    train_color : color, default: 'b'
        Residuals for training data are ploted with this color but also
        given an opacity of 0.5 to ensure that the test data residuals
        are more visible. Can be any matplotlib color.

    test_color : color, default: 'g'
        Residuals for test data are plotted with this color. In order to
        create generalizable models, reserved test data residuals are of
        the most analytical interest, so these points are highlighted by
        having full opacity. Can be any matplotlib color.

    line_color : color, default: dark grey
        Defines the color of the zero error line, can be any matplotlib color.

    train_alpha : float, default: 0.75
        Specify a transparency for traininig data, where 1 is completely opaque
        and 0 is completely transparent. This property makes densely clustered
        points more visible.

    test_alpha : float, default: 0.75
        Specify a transparency for test data, where 1 is completely opaque
        and 0 is completely transparent. This property makes densely clustered
        points more visible.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If 'auto' (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------

    train_score_ : float
        The R^2 score that specifies the goodness of fit of the underlying
        regression model to the training data.

    test_score_ : float
        The R^2 score that specifies the goodness of fit of the underlying
        regression model to the test data.

    Examples
    --------

    >>> from yellowbrick.regressor import ResidualsPlot
    >>> from sklearn.linear_model import Ridge
    >>> model = ResidualsPlot(Ridge())
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    >>> model.show()

    Notes
    -----
    ResidualsPlot is a ScoreVisualizer, meaning that it wraps a model and
    its primary entry point is the ``score()`` method.

    The residuals histogram feature requires matplotlib 2.0.2 or greater.
    """

    def __init__(self, estimator, ax=None, hist=True, qqplot=False, train_color='b', test_color='g', line_color=LINE_COLOR, train_alpha=0.75, test_alpha=0.75, is_fitted='auto', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ResidualsPlot, self).__init__(estimator, ax=ax, is_fitted=is_fitted, **kwargs)
        self.colors = {'train_point': train_color, 'test_point': test_color, 'line': line_color}
        self.hist = hist
        if self.hist not in {True, 'density', 'frequency', None, False}:
            raise YellowbrickValueError("'{}' is an invalid argument for hist, use None, True, False, 'density', or 'frequency'".format(hist))
        self.qqplot = qqplot
        if self.qqplot not in {True, False}:
            raise YellowbrickValueError("'{}' is an invalid argument for qqplot, use True,  or False".format(hist))
        if self.hist in {True, 'density', 'frequency'} and self.qqplot in {True}:
            raise YellowbrickValueError('Set either hist or qqplot to False, can not plot both of them simultaneously.')
        if self.hist in {True, 'density', 'frequency'}:
            self.hax
        if self.qqplot in {True}:
            self.qqax
        (self._labels, self._colors) = ([], [])
        self.alphas = {'train_point': train_alpha, 'test_point': test_alpha}

    @memoized
    def hax(self):
        if False:
            print('Hello World!')
        '\n        Returns the histogram axes, creating it only on demand.\n        '
        if make_axes_locatable is None:
            raise YellowbrickValueError('residuals histogram requires matplotlib 2.0.2 or greater please upgrade matplotlib or set hist=False on the visualizer')
        divider = make_axes_locatable(self.ax)
        hax = divider.append_axes('right', size=1, pad=0.1, sharey=self.ax)
        hax.yaxis.tick_right()
        hax.grid(False, axis='x')
        return hax

    @memoized
    def qqax(self):
        if False:
            return 10
        '\n        Returns the Q-Q plot axes, creating it only on demand.\n        '
        if make_axes_locatable is None:
            raise YellowbrickValueError('residuals histogram requires matplotlib 2.0.2 or greater please upgrade matplotlib or set qqplot=False on the visualizer')
        divider = make_axes_locatable(self.ax)
        qqax = divider.append_axes('right', size=2, pad=0.25, sharey=self.ax)
        qqax.yaxis.tick_right()
        return qqax

    def fit(self, X, y, **kwargs):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target values\n\n        kwargs: keyword arguments passed to Scikit-Learn API.\n\n        Returns\n        -------\n        self : ResidualsPlot\n            The visualizer instance\n        '
        super(ResidualsPlot, self).fit(X, y, **kwargs)
        self.score(X, y, train=True)
        return self

    def score(self, X, y=None, train=False, **kwargs):
        if False:
            return 10
        '\n        Generates predicted target values using the Scikit-Learn\n        estimator.\n\n        Parameters\n        ----------\n        X : array-like\n            X (also X_test) are the dependent variables of test set to predict\n\n        y : array-like\n            y (also y_test) is the independent actual variables to score against\n\n        train : boolean\n            If False, `score` assumes that the residual points being plotted\n            are from the test data; if True, `score` assumes the residuals\n            are the train data.\n\n        Returns\n        -------\n        score : float\n            The score of the underlying estimator, usually the R-squared score\n            for regression estimators.\n        '
        score = self.estimator.score(X, y, **kwargs)
        if train:
            self.train_score_ = score
        else:
            self.test_score_ = score
        y_pred = self.predict(X)
        residuals = y_pred - y
        self.draw(y_pred, residuals, train=train)
        return score

    def draw(self, y_pred, residuals, train=False, **kwargs):
        if False:
            print('Hello World!')
        '\n        Draw the residuals against the predicted value for the specified split.\n        It is best to draw the training split first, then the test split so\n        that the test split (usually smaller) is above the training split;\n        particularly if the histogram is turned on.\n\n        Parameters\n        ----------\n        y_pred : ndarray or Series of length n\n            An array or series of predicted target values\n\n        residuals : ndarray or Series of length n\n            An array or series of the difference between the predicted and the\n            target values\n\n        train : boolean, default: False\n            If False, `draw` assumes that the residual points being plotted\n            are from the test data; if True, `draw` assumes the residuals\n            are the train data.\n\n        Returns\n        -------\n        ax : matplotlib Axes\n            The axis with the plotted figure\n        '
        if train:
            color = self.colors['train_point']
            label = 'Train $R^2 = {:0.3f}$'.format(self.train_score_)
            alpha = self.alphas['train_point']
        else:
            color = self.colors['test_point']
            label = 'Test $R^2 = {:0.3f}$'.format(self.test_score_)
            alpha = self.alphas['test_point']
        self._labels.append(label)
        self._colors.append(color)
        self.ax.scatter(y_pred, residuals, c=color, alpha=alpha, label=label)
        if self.hist in {True, 'frequency'}:
            self.hax.hist(residuals, bins=50, orientation='horizontal', color=color)
        elif self.hist == 'density':
            self.hax.hist(residuals, bins=50, orientation='horizontal', density=True, color=color)
        if self.qqplot in {True}:
            (osm, osr) = probplot(residuals, dist='norm', fit=False)
            self.qqax.scatter(osm, osr, c=color, alpha=alpha, label=label)
        plt.sca(self.ax)
        return self.ax

    def finalize(self, **kwargs):
        if False:
            return 10
        '\n        Prepares the plot for rendering by adding a title, legend, and axis labels.\n        Also draws a line at the zero residuals to show the baseline.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        self.set_title('Residuals for {} Model'.format(self.name))
        manual_legend(self, self._labels, self._colors, loc='best', frameon=True)
        self.ax.axhline(y=0, c=self.colors['line'])
        self.ax.set_ylabel('Residuals')
        self.ax.set_xlabel('Predicted Value')
        if self.hist:
            self.hax.axhline(y=0, c=self.colors['line'])
            self.hax.set_xlabel('Distribution')
        if self.qqplot:
            self.qqax.set_title('Q-Q plot')
            self.qqax.set_xlabel('Theoretical quantiles')
            self.qqax.set_ylabel('Observed quantiles')

def residuals_plot(estimator, X_train, y_train, X_test=None, y_test=None, ax=None, hist=True, qqplot=False, train_color='b', test_color='g', line_color=LINE_COLOR, train_alpha=0.75, test_alpha=0.75, is_fitted='auto', show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "ResidualsPlot quick method:\n\n    A residual plot shows the residuals on the vertical axis and the\n    independent variable on the horizontal axis.\n\n    If the points are randomly dispersed around the horizontal axis, a linear\n    regression model is appropriate for the data; otherwise, a non-linear\n    model is more appropriate.\n\n    Parameters\n    ----------\n    estimator : a Scikit-Learn regressor\n        Should be an instance of a regressor, otherwise will raise a\n        YellowbrickTypeError exception on instantiation.\n        If the estimator is not fitted, it is fit when the visualizer is fitted,\n        unless otherwise specified by ``is_fitted``.\n\n    X_train : ndarray or DataFrame of shape n x m\n        A feature array of n instances with m features the model is trained on.\n        Used to fit the visualizer and also to score the visualizer if test splits are\n        not directly specified.\n\n    y_train : ndarray or Series of length n\n        An array or series of target or class values. Used to fit the visualizer and\n        also to score the visualizer if test splits are not specified.\n\n    X_test : ndarray or DataFrame of shape n x m, default: None\n        An optional feature array of n instances with m features that the model\n        is scored on if specified, using X_train as the training data.\n\n    y_test : ndarray or Series of length n, default: None\n        An optional array or series of target or class values that serve as actual\n        labels for X_test for scoring purposes.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    hist : {True, False, None, 'density', 'frequency'}, default: True\n        Draw a histogram showing the distribution of the residuals on the\n        right side of the figure. Requires Matplotlib >= 2.0.2.\n        If set to 'density', the probability density function will be plotted.\n        If set to True or 'frequency' then the frequency will be plotted.\n\n    qqplot : {True, False}, default: False\n        Draw a Q-Q plot on the right side of the figure, comparing the quantiles\n        of the residuals against quantiles of a standard normal distribution.\n        Q-Q plot and histogram of residuals can not be plotted simultaneously,\n        either `hist` or `qqplot` has to be set to False.\n\n    train_color : color, default: 'b'\n        Residuals for training data are ploted with this color but also\n        given an opacity of 0.5 to ensure that the test data residuals\n        are more visible. Can be any matplotlib color.\n\n    test_color : color, default: 'g'\n        Residuals for test data are plotted with this color. In order to\n        create generalizable models, reserved test data residuals are of\n        the most analytical interest, so these points are highlighted by\n        having full opacity. Can be any matplotlib color.\n\n    line_color : color, default: dark grey\n        Defines the color of the zero error line, can be any matplotlib color.\n\n    train_alpha : float, default: 0.75\n        Specify a transparency for traininig data, where 1 is completely opaque\n        and 0 is completely transparent. This property makes densely clustered\n        points more visible.\n\n    test_alpha : float, default: 0.75\n        Specify a transparency for test data, where 1 is completely opaque\n        and 0 is completely transparent. This property makes densely clustered\n        points more visible.\n\n    is_fitted : bool or str, default='auto'\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If 'auto' (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    viz : ResidualsPlot\n        Returns the fitted ResidualsPlot that created the figure.\n    "
    viz = ResidualsPlot(estimator=estimator, ax=ax, hist=hist, qqplot=qqplot, train_color=train_color, test_color=test_color, line_color=line_color, train_alpha=train_alpha, test_alpha=test_alpha, is_fitted=is_fitted, **kwargs)
    viz.fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        viz.score(X_test, y_test)
    elif X_test is not None or y_test is not None:
        raise YellowbrickValueError('both X_test and y_test are required if one is specified')
    else:
        viz.score(X_train, y_train)
    if show:
        viz.show()
    else:
        viz.finalize()
    return viz