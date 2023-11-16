"""
Visualize the influence and leverage of individual instances on a regression model.
"""
import numpy as np
import scipy as sp
from yellowbrick.base import Visualizer
from sklearn.linear_model import LinearRegression

class CooksDistance(Visualizer):
    """
    Cook's Distance is a measure of how influential an instance is to the computation of
    a regression, e.g. if the instance is removed would the estimated coeficients of the
    underlying model be substantially changed? Because of this, Cook's Distance is
    generally used to detect outliers in standard, OLS regression. In fact, a general
    rule of thumb is that D(i) > 4/n is a good threshold for determining highly
    influential points as outliers and this visualizer can report the percentage of data
    that is above that threshold.

    This implementation of Cook's Distance assumes Ordinary Least Squares regression,
    and therefore embeds a ``sklearn.linear_model.LinearRegression`` under the hood.
    Distance is computed via the non-whitened leverage of the projection matrix,
    computed inside of ``fit()``. The results of this visualizer are therefore similar
    to, but not as advanced, as a similar computation using statsmodels. Computing the
    influence for other regression models requires leave one out validation and can be
    expensive to compute.

    .. seealso::
        For a longer discussion on detecting outliers in regression and computing
        leverage and influence, see `linear regression in python, outliers/leverage
        detect <http://bit.ly/2If2fga>`_ by Huiming Song.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    draw_threshold : bool, default: True
        Draw a horizontal line at D(i) == 4/n to easily identify the most influential
        points on the final regression. This will also draw a legend that specifies the
        percentage of data points that are above the threshold.

    linefmt : str, default: 'C0-'
        A string defining the properties of the vertical lines of the stem plot, usually
        this will be a color or a color and a line style. The default is simply a solid
        line with the first color of the color cycle.

    markerfmt : str, default: ','
        A string defining the properties of the markers at the stem plot heads. The
        default is "pixel", e.g. basically no marker head at the top of the stem plot.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence the final
        visualization (e.g. size or title parameters).

    Attributes
    ----------
    distance_ : array, 1D
        The Cook's distance value for each instance specified in ``X``, e.g. an 1D array
        with shape ``(X.shape[0],)``.

    p_values_ : array, 1D
        The p values associated with the F-test of Cook's distance distribution. A 1D
        array whose shape matches ``distance_``.

    influence_threshold_ : float
        A rule of thumb influence threshold to determine outliers in the regression
        model, defined as It=4/n.

    outlier_percentage_ : float
        The percentage of instances whose Cook's distance is greater than the influnce
        threshold, the percentage is 0.0 <= p  <= 100.0.

    Notes
    -----
    Cook's Distance is very similar to DFFITS, another diagnostic that is meant to show
    how influential a point is in a statistical regression. Although the computed values
    of Cook's and DFFITS are different, they are conceptually identical and there even
    exists a closed-form formula to convert one value to another. Because of this, we
    have chosen to implement Cook's distance rather than or in addition to DFFITS.
    """

    def __init__(self, ax=None, draw_threshold=True, linefmt='C0-', markerfmt=',', **kwargs):
        if False:
            return 10
        super(CooksDistance, self).__init__(ax=ax, **kwargs)
        self.draw_threshold = draw_threshold
        self.linefmt = linefmt
        self.markerfmt = markerfmt
        self._model = LinearRegression()

    def fit(self, X, y):
        if False:
            return 10
        "\n        Computes the leverage of X and uses the residuals of a\n        ``sklearn.linear_model.LinearRegression`` to compute the Cook's Distance of each\n        observation in X, their p-values and the number of outliers defined by the\n        number of observations supplied.\n\n        Parameters\n        ----------\n        X : array-like, 2D\n            The exogenous design matrix, e.g. training data.\n\n        y : array-like, 1D\n            The endogenous response variable, e.g. target data.\n\n        Returns\n        -------\n        self : CooksDistance\n            Fit returns the visualizer instance.\n        "
        self._model.fit(X, y)
        leverage = (X * np.linalg.pinv(X).T).sum(1)
        rank = np.linalg.matrix_rank(X)
        df = X.shape[0] - rank
        residuals = y - self._model.predict(X)
        mse = np.dot(residuals, residuals) / df
        residuals_studentized = residuals / np.sqrt(mse) / np.sqrt(1 - leverage)
        self.distance_ = residuals_studentized ** 2 / X.shape[1]
        self.distance_ *= leverage / (1 - leverage)
        self.p_values_ = sp.stats.f.sf(self.distance_, X.shape[1], df)
        self.influence_threshold_ = 4 / X.shape[0]
        self.outlier_percentage_ = sum(self.distance_ > self.influence_threshold_) / X.shape[0]
        self.outlier_percentage_ *= 100.0
        self.draw()
        return self

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Draws a stem plot where each stem is the Cook's Distance of the instance at the\n        index specified by the x axis. Optionaly draws a threshold line.\n        "
        (_, _, baseline) = self.ax.stem(self.distance_, linefmt=self.linefmt, markerfmt=self.markerfmt, use_line_collection=True)
        self.ax.set_xlim(0, len(self.distance_))
        if self.draw_threshold:
            label = '{:0.2f}% > $I_t$ ($I_t=\\frac {{4}} {{n}}$)'.format(self.outlier_percentage_)
            self.ax.axhline(self.influence_threshold_, ls='--', label=label, c=baseline.get_color(), lw=baseline.get_linewidth())
        return self.ax

    def finalize(self):
        if False:
            i = 10
            return i + 15
        '\n        Prepares the visualization for presentation and reporting.\n        '
        self.set_title("Cook's Distance Outlier Detection")
        self.ax.set_xlabel('instance index')
        self.ax.set_ylabel('influence (I)')
        if self.draw_threshold:
            self.ax.legend(loc='best', frameon=True)

def cooks_distance(X, y, ax=None, draw_threshold=True, linefmt='C0-', markerfmt=',', show=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Cook\'s Distance is a measure of how influential an instance is to the computation of\n    a regression, e.g. if the instance is removed would the estimated coeficients of the\n    underlying model be substantially changed? Because of this, Cook\'s Distance is\n    generally used to detect outliers in standard, OLS regression. In fact, a general\n    rule of thumb is that D(i) > 4/n is a good threshold for determining highly\n    influential points as outliers and this visualizer can report the percentage of data\n    that is above that threshold.\n\n    This implementation of Cook\'s Distance assumes Ordinary Least Squares regression,\n    and therefore embeds a ``sklearn.linear_model.LinearRegression`` under the hood.\n    Distance is computed via the non-whitened leverage of the projection matrix,\n    computed inside of ``fit()``. The results of this visualizer are therefore similar\n    to, but not as advanced, as a similar computation using statsmodels. Computing the\n    influence for other regression models requires leave one out validation and can be\n    expensive to compute.\n\n    .. seealso::\n        For a longer discussion on detecting outliers in regression and computing\n        leverage and influence, see `linear regression in python, outliers/leverage\n        detect <http://bit.ly/2If2fga>`_ by Huiming Song.\n\n    Parameters\n    ----------\n    X : array-like, 2D\n        The exogenous design matrix, e.g. training data.\n\n    y : array-like, 1D\n        The endogenous response variable, e.g. target data.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    draw_threshold : bool, default: True\n        Draw a horizontal line at D(i) == 4/n to easily identify the most influential\n        points on the final regression. This will also draw a legend that specifies the\n        percentage of data points that are above the threshold.\n\n    linefmt : str, default: \'C0-\'\n        A string defining the properties of the vertical lines of the stem plot, usually\n        this will be a color or a color and a line style. The default is simply a solid\n        line with the first color of the color cycle.\n\n    markerfmt: str, default: \',\'\n        A string defining the properties of the markers at the stem plot heads. The\n        default is "pixel", e.g. basically no marker head at the top of the stem plot.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence the final\n        visualization (e.g. size or title parameters).\n    '
    viz = CooksDistance(ax=ax, draw_threshold=draw_threshold, linefmt=linefmt, markerfmt=markerfmt, **kwargs)
    viz.fit(X, y)
    if show:
        viz.show()
    else:
        viz.finalize()
    return viz