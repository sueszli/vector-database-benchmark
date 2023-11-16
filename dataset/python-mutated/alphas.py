"""
Implements alpha selection visualizers for regularization
"""
import numpy as np
from functools import partial
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.regressor.base import RegressionScoreVisualizer
from sklearn.model_selection import cross_val_score
__all__ = ['AlphaSelection', 'ManualAlphaSelection']

class AlphaSelection(RegressionScoreVisualizer):
    """
    The Alpha Selection Visualizer demonstrates how different values of alpha
    influence model selection during the regularization of linear models.
    Generally speaking, alpha increases the affect of regularization, e.g. if
    alpha is zero there is no regularization and the higher the alpha, the
    more the regularization parameter influences the final model.

    Regularization is designed to penalize model complexity, therefore the
    higher the alpha, the less complex the model, decreasing the error due to
    variance (overfit). Alphas that are too high on the other hand increase
    the error due to bias (underfit). It is important, therefore to choose an
    optimal Alpha such that the error is minimized in both directions.

    To do this, typically you would you use one of the "RegressionCV" models
    in Scikit-Learn. E.g. instead of using the ``Ridge`` (L2) regularizer, you
    can use ``RidgeCV`` and pass a list of alphas, which will be selected
    based on the cross-validation score of each alpha. This visualizer wraps
    a "RegressionCV" model and visualizes the alpha/error curve. Use this
    visualization to detect if the model is responding to regularization, e.g.
    as you increase or decrease alpha, the model responds and error is
    decreased. If the visualization shows a jagged or random plot, then
    potentially the model is not sensitive to that type of regularization and
    another is required (e.g. L1 or ``Lasso`` regularization).

    Parameters
    ----------

    estimator : a Scikit-Learn regressor
        Should be an instance of a regressor, and specifically one whose name
        ends with "CV" otherwise a will raise a YellowbrickTypeError exception
        on instantiation. To use non-CV regressors see: ``ManualAlphaSelection``.
        If the estimator is not fitted, it is fit when the visualizer is fitted,
        unless otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If 'auto' (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.regressor import AlphaSelection
    >>> from sklearn.linear_model import LassoCV
    >>> model = AlphaSelection(LassoCV())
    >>> model.fit(X, y)
    >>> model.show()

    Notes
    -----

    This class expects an estimator whose name ends with "CV". If you wish to
    use some other estimator, please see the ``ManualAlphaSelection``
    Visualizer for manually iterating through all alphas and selecting the
    best one.

    This Visualizer hooks into the Scikit-Learn API during ``fit()``. In
    order to pass a fitted model to the Visualizer, call the ``draw()`` method
    directly after instantiating the visualizer with the fitted model.

    Note, each "RegressorCV" module has many different methods for storing
    alphas and error. This visualizer attempts to get them all and is known
    to work for RidgeCV, LassoCV, LassoLarsCV, and ElasticNetCV. If your
    favorite regularization method doesn't work, please submit a bug report.

    For RidgeCV, make sure ``store_cv_values=True``.
    """

    def __init__(self, estimator, ax=None, is_fitted='auto', **kwargs):
        if False:
            i = 10
            return i + 15
        name = estimator.__class__.__name__
        if not name.endswith('CV'):
            raise YellowbrickTypeError("'{}' is not a CV regularization model; try ManualAlphaSelection instead.".format(name))
        if 'store_cv_values' in estimator.get_params().keys():
            estimator.set_params(store_cv_values=True)
        super(AlphaSelection, self).__init__(estimator, ax=ax, **kwargs)

    def fit(self, X, y, **kwargs):
        if False:
            print('Hello World!')
        '\n        A simple pass-through method; calls fit on the estimator and then\n        draws the alpha-error plot.\n        '
        super(AlphaSelection, self).fit(X, y, **kwargs)
        self.draw()
        return self

    def draw(self):
        if False:
            print('Hello World!')
        '\n        Draws the alpha plot based on the values on the estimator.\n        '
        alphas = self._find_alphas_param()
        errors = self._find_errors_param()
        alpha = self.estimator.alpha_
        name = self.name[:-2].lower()
        self.ax.plot(alphas, errors, label=name)
        label = '$\\alpha={:0.3f}$'.format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)
        return self.ax

    def finalize(self):
        if False:
            return 10
        '\n        Prepare the figure for rendering by setting the title as well as the\n        X and Y axis labels and adding the legend.\n        '
        self.set_title('{} Alpha Error'.format(self.name))
        self.ax.set_xlabel('alpha')
        self.ax.set_ylabel('error (or score)')
        self.ax.legend(loc='best', frameon=True)

    def _find_alphas_param(self):
        if False:
            return 10
        '\n        Searches for the parameter on the estimator that contains the array of\n        alphas that was used to produce the error selection. If it cannot find\n        the parameter then a YellowbrickValueError is raised.\n        '
        for attr in ('cv_alphas_', 'alphas_', 'alphas'):
            try:
                return getattr(self.estimator, attr)
            except AttributeError:
                continue
        raise YellowbrickValueError('could not find alphas param on {} estimator'.format(self.estimator.__class__.__name__))

    def _find_errors_param(self):
        if False:
            print('Hello World!')
        '\n        Searches for the parameter on the estimator that contains the array of\n        errors that was used to determine the optimal alpha. If it cannot find\n        the parameter then a YellowbrickValueError is raised.\n        '
        if hasattr(self.estimator, 'mse_path_'):
            return self.estimator.mse_path_.mean(1)
        if hasattr(self.estimator, 'cv_values_'):
            return self.estimator.cv_values_.mean(0)
        raise YellowbrickValueError('could not find errors param on {} estimator'.format(self.estimator.__class__.__name__))

class ManualAlphaSelection(AlphaSelection):
    """
    The ``AlphaSelection`` visualizer requires a "RegressorCV", that is a
    specialized class that performs cross-validated alpha-selection on behalf
    of the model. If the regressor you wish to use doesn't have an associated
    "CV" estimator, or for some reason you would like to specify more control
    over the alpha selection process, then you can use this manual alpha
    selection visualizer, which is essentially a wrapper for
    ``cross_val_score``, fitting a model for each alpha specified.

    Parameters
    ----------

    estimator : an unfitted Scikit-Learn regressor
        Should be an instance of an unfitted regressor, and specifically one
        whose name doesn't end with "CV". The regressor must support a call to
        ``set_params(alpha=alpha)`` and be fit multiple times. If the
        regressor name ends with "CV" a ``YellowbrickValueError`` is raised.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    alphas : ndarray or Series, default: np.logspace(-10, 2, 200)
        An array of alphas to fit each model with

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        This argument is passed to the
        ``sklearn.model_selection.cross_val_score`` method to produce the
        cross validated score for each alpha.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

        This argument is passed to the
        ``sklearn.model_selection.cross_val_score`` method to produce the
        cross validated score for each alpha.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.regressor import ManualAlphaSelection
    >>> from sklearn.linear_model import Ridge
    >>> model = ManualAlphaSelection(
    ...     Ridge(), cv=12, scoring='neg_mean_squared_error'
    ... )
    ...
    >>> model.fit(X, y)
    >>> model.show()

    Notes
    -----

    This class does not take advantage of estimator-specific searching and is
    therefore less optimal and more time consuming than the regular
    "RegressorCV" estimators.
    """

    def __init__(self, estimator, ax=None, alphas=None, cv=None, scoring=None, **kwargs):
        if False:
            while True:
                i = 10
        name = estimator.__class__.__name__
        if name.endswith('CV'):
            raise YellowbrickTypeError("'{}' is a CV regularization model; try AlphaSelection instead.".format(name))
        super(AlphaSelection, self).__init__(estimator, ax=ax, **kwargs)
        if alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = np.logspace(-10, -2, 200)
        self.errors = None
        self.score_method = partial(cross_val_score, cv=cv, scoring=scoring)

    def fit(self, X, y, **args):
        if False:
            i = 10
            return i + 15
        '\n        The fit method is the primary entry point for the manual alpha\n        selection visualizer. It sets the alpha param for each alpha in the\n        alphas list on the wrapped estimator, then scores the model using the\n        passed in X and y data set. Those scores are then aggregated and\n        drawn using matplotlib.\n        '
        self.errors = []
        for alpha in self.alphas:
            self.estimator.set_params(alpha=alpha)
            scores = self.score_method(self.estimator, X, y)
            self.errors.append(scores.mean())
        self.errors = np.array(self.errors)
        self.draw()
        return self

    def draw(self):
        if False:
            print('Hello World!')
        '\n        Draws the alphas values against their associated error in a similar\n        fashion to the AlphaSelection visualizer.\n        '
        self.ax.plot(self.alphas, self.errors, label=self.name.lower())
        alpha = self.alphas[np.where(self.errors == self.errors.max())][0]
        label = '$\\alpha_{{max}}={:0.3f}$'.format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)
        alpha = self.alphas[np.where(self.errors == self.errors.min())][0]
        label = '$\\alpha_{{min}}={:0.3f}$'.format(alpha)
        self.ax.axvline(alpha, color='k', linestyle='dashed', label=label)
        return self.ax

def alphas(estimator, X, y=None, ax=None, is_fitted='auto', show=True, **kwargs):
    if False:
        i = 10
        return i + 15
    'Quick Method:\n    The Alpha Selection Visualizer demonstrates how different values of alpha\n    influence model selection during the regularization of linear models.\n    Generally speaking, alpha increases the affect of regularization, e.g. if\n    alpha is zero there is no regularization and the higher the alpha, the\n    more the regularization parameter influences the final model.\n\n    Parameters\n    ----------\n\n    estimator : a Scikit-Learn regressor\n        Should be an instance of a regressor, and specifically one whose name\n        ends with "CV" otherwise a will raise a YellowbrickTypeError exception\n        on instantiation. To use non-CV regressors see: ``ManualAlphaSelection``.\n        If the estimator is not fitted, it is fit when the visualizer is fitted,\n        unless otherwise specified by ``is_fitted``.\n\n    X  : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features.\n\n    y  : ndarray or Series of length n\n        An array or series of target values.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    is_fitted : bool or str, default=\'auto\'\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If \'auto\' (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    visualizer : AlphaSelection\n        Returns the alpha selection visualizer\n    '
    visualizer = AlphaSelection(estimator, ax, is_fitted=is_fitted, **kwargs)
    visualizer.fit(X, y)
    visualizer.score(X, y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer

def manual_alphas(estimator, X, y=None, ax=None, alphas=None, cv=None, scoring=None, show=True, **kwargs):
    if False:
        return 10
    'Quick Method:\n    The Manual Alpha Selection Visualizer demonstrates how different values of alpha\n    influence model selection during the regularization of linear models.\n    Generally speaking, alpha increases the affect of regularization, e.g. if\n    alpha is zero there is no regularization and the higher the alpha, the\n    more the regularization parameter influences the final model.\n\n    Parameters\n    ----------\n\n    estimator : an unfitted Scikit-Learn regressor\n        Should be an instance of an unfitted regressor, and specifically one\n        whose name doesn\'t end with "CV". The regressor must support a call to\n        ``set_params(alpha=alpha)`` and be fit multiple times. If the\n        regressor name ends with "CV" a ``YellowbrickValueError`` is raised.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    alphas : ndarray or Series, default: np.logspace(-10, 2, 200)\n        An array of alphas to fit each model with\n\n    cv : int, cross-validation generator or an iterable, optional\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - None, to use the default 3-fold cross validation,\n        - integer, to specify the number of folds in a `(Stratified)KFold`,\n        - An object to be used as a cross-validation generator.\n        - An iterable yielding train, test splits.\n\n        This argument is passed to the\n        ``sklearn.model_selection.cross_val_score`` method to produce the\n        cross validated score for each alpha.\n\n    scoring : string, callable or None, optional, default: None\n        A string (see model evaluation documentation) or\n        a scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n\n        This argument is passed to the\n        ``sklearn.model_selection.cross_val_score`` method to produce the\n        cross validated score for each alpha.\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    visualizer : AlphaSelection\n        Returns the alpha selection visualizer\n    '
    visualizer = ManualAlphaSelection(estimator, ax, alphas=alphas, scoring=scoring, cv=cv, **kwargs)
    visualizer.fit(X, y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer