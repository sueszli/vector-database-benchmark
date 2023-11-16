"""
Uses Scikit-Learn to compute a best fit function, then draws it in the plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse
from operator import itemgetter
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.exceptions import YellowbrickValueError
LINEAR = 'linear'
QUADRATIC = 'quadratic'
EXPONENTIAL = 'exponential'
LOG = 'log'
SELECT_BEST = 'select_best'

def draw_best_fit(X, y, ax, estimator='linear', **kwargs):
    if False:
        return 10
    "\n    Uses Scikit-Learn to fit a model to X and y then uses the resulting model\n    to predict the curve based on the X values. This curve is drawn to the ax\n    (matplotlib axis) which must be passed as the third variable.\n\n    The estimator function can be one of the following:\n\n    - ``'linear'``:      Uses OLS to fit the regression\n    - ``'quadratic'``:   Uses OLS with Polynomial order 2\n    - ``'exponential'``: Not implemented yet\n    - ``'log'``:         Not implemented yet\n    - ``'select_best'``: Selects the best fit via MSE\n\n    The remaining keyword arguments are passed to ax.plot to define and\n    describe the line of best fit.\n\n    Parameters\n    ----------\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    y : ndarray or Series of length n\n        An array or series of target or class values\n\n    ax : matplotlib Axes, default: None\n        The axis to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    estimator : string, default: 'linear'\n        The name of the estimator function used to draw the best fit line.\n        The estimator can currently be one of linear, quadratic, exponential,\n        log, or select_best. The select best method uses the minimum MSE to\n        select the best fit line.\n\n    kwargs : dict\n        Keyword arguments to pass to the matplotlib plot function to style and\n        label the line of best fit. By default, the standard line color is\n        used unless the color keyword argument is passed in.\n\n    Returns\n    -------\n\n    ax : matplotlib Axes\n        The axes with the line drawn on it.\n    "
    estimators = {LINEAR: fit_linear, QUADRATIC: fit_quadratic, EXPONENTIAL: fit_exponential, LOG: fit_log, SELECT_BEST: fit_select_best}
    if estimator not in estimators:
        raise YellowbrickValueError("'{}' not a valid type of estimator; choose from {}".format(estimator, ', '.join(estimators.keys())))
    estimator = estimators[estimator]
    if len(X) != len(y):
        raise YellowbrickValueError("X and y must have same length: X len {} doesn't match y len {}!".format(len(X), len(y)))
    X = np.array(X)
    y = np.array(y)
    if X.ndim < 2:
        X = X[:, np.newaxis]
    if X.ndim > 2:
        raise YellowbrickValueError('X must be a (1,) or (n,1) dimensional array not {}'.format(X.shape))
    if y.ndim > 1:
        raise YellowbrickValueError('y must be a (1,) dimensional array not {}'.format(y.shape))
    model = estimator(X, y)
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['color'] = LINE_COLOR
    ax = ax or plt.gca()
    xr = np.linspace(*ax.get_xlim(), num=100)
    ax.plot(xr, model.predict(xr[:, np.newaxis]), **kwargs)
    return ax

def fit_select_best(X, y):
    if False:
        print('Hello World!')
    '\n    Selects the best fit of the estimators already implemented by choosing the\n    model with the smallest mean square error metric for the trained values.\n    '
    models = [fit(X, y) for fit in [fit_linear, fit_quadratic]]
    errors = map(lambda model: mse(y, model.predict(X)), models)
    return min(zip(models, errors), key=itemgetter(1))[0]

def fit_linear(X, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Uses OLS to fit the regression.\n    '
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def fit_quadratic(X, y):
    if False:
        print('Hello World!')
    '\n    Uses OLS with Polynomial order 2.\n    '
    model = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    model.fit(X, y)
    return model

def fit_exponential(X, y):
    if False:
        return 10
    '\n    Fits an exponential curve to the data.\n    '
    raise NotImplementedError('Exponential best fit lines are not implemented')

def fit_log(X, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fit a logrithmic curve to the data.\n    '
    raise NotImplementedError('Logrithmic best fit lines are not implemented')

def draw_identity_line(ax=None, dynamic=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draws a 45 degree identity line such that y=x for all points within the\n    given axes x and y limits. This function also registeres a callback so\n    that as the figure is modified, the axes are updated and the line remains\n    drawn correctly.\n\n    Parameters\n    ----------\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    dynamic : bool, default : True\n        If the plot is dynamic, callbacks will be registered to update the\n        identiy line as axes are changed.\n\n    kwargs : dict\n        Keyword arguments to pass to the matplotlib plot function to style the\n        identity line.\n\n\n    Returns\n    -------\n\n    ax : matplotlib Axes\n        The axes with the line drawn on it.\n\n    Notes\n    -----\n\n    .. seealso:: `StackOverflow discussion: Does matplotlib have a function for drawing diagonal lines in axis coordinates? <https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates>`_\n    '
    ax = ax or plt.gca()
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['color'] = LINE_COLOR
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5
    (identity,) = ax.plot([], [], **kwargs)

    def callback(ax):
        if False:
            for i in range(10):
                print('nop')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data = (max(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
        identity.set_data(data, data)
    callback(ax)
    if dynamic:
        ax.callbacks.connect('xlim_changed', callback)
        ax.callbacks.connect('ylim_changed', callback)
    return ax
if __name__ == '__main__':
    import os
    import pandas as pd
    path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'data', 'concrete.xls')
    if not os.path.exists(path):
        raise Exception('Could not find path for testing')
    xkey = 'Fine Aggregate (component 7)(kg in a m^3 mixture)'
    ykey = 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'
    data = pd.read_excel(path)
    (fig, axe) = plt.subplots()
    axe.scatter(data[xkey], data[ykey])
    draw_best_fit(data[xkey], data[ykey], axe, 'select_best')
    plt.show()