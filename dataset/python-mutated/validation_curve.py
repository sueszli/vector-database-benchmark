"""
Implements a visual validation curve for a hyperparameter.
"""
import numpy as np
from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.model_selection import validation_curve as sk_validation_curve

class ValidationCurve(ModelVisualizer):
    """
    Visualizes the validation curve for both test and training data for a
    range of values for a single hyperparameter of the model. Adjusting the
    value of a hyperparameter adjusts the complexity of a model. Less complex
    models suffer from increased error due to bias, while more complex models
    suffer from increased error due to variance. By inspecting the training
    and cross-validated test score error, it is possible to estimate a good
    value for a hyperparameter that balances the bias/variance trade-off.

    The visualizer evaluates cross-validated training and test scores for the
    different hyperparameters supplied. The curve is plotted so that the
    x-axis is the value of the hyperparameter and the y-axis is the model
    score. This is similar to a grid search with a single hyperparameter.

    The cross-validation generator splits the dataset k times, and scores are
    averaged over all k runs for the training and test subsets. The curve
    plots the mean score, and the filled in area suggests the variability of
    cross-validation by plotting one standard deviation above and below the
    mean for each split.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    logx : boolean, optional
        If True, plots the x-axis with a logarithmic scale.

    groups : array-like, with shape (n_samples,)
        Optional group labels for the samples used while splitting the dataset
        into train/test sets.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        see the scikit-learn
        `cross-validation guide <https://bit.ly/2MMQAI7>`_
        for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``. See scikit-learn model evaluation
        documentation for names of possible metrics.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    markers : string, default: '-d'
        Matplotlib style markers for points on the plot points
        Options: '-,', '-+', '-o', '-*', '-v', '-h', '-d'

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    train_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    train_scores_mean_ : array, shape (n_ticks,)
        Mean training data scores for each training split

    train_scores_std_ : array, shape (n_ticks,)
        Standard deviation of training data scores for each training split

    test_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    test_scores_mean_ : array, shape (n_ticks,)
        Mean test data scores for each test split

    test_scores_std_ : array, shape (n_ticks,)
        Standard deviation of test data scores for each test split

    Examples
    --------

    >>> import numpy as np
    >>> from yellowbrick.model_selection import ValidationCurve
    >>> from sklearn.svm import SVC
    >>> pr = np.logspace(-6,-1,5)
    >>> model = ValidationCurve(SVC(), param_name="gamma", param_range=pr)
    >>> model.fit(X, y)
    >>> model.show()

    Notes
    -----
    This visualizer is essentially a wrapper for the
    ``sklearn.model_selection.learning_curve utility``, discussed in the
    `validation curves <https://bit.ly/2KlumeB>`__
    documentation.

    .. seealso:: The documentation for the
        `learning_curve <https://bit.ly/2Yz9sBB>`__
        function, which this visualizer wraps.
    """

    def __init__(self, estimator, param_name, param_range, ax=None, logx=False, groups=None, cv=None, scoring=None, n_jobs=1, pre_dispatch='all', markers='-d', **kwargs):
        if False:
            print('Hello World!')
        super(ValidationCurve, self).__init__(estimator, ax=ax, **kwargs)
        param_range = np.asarray(param_range)
        if param_range.ndim != 1:
            raise YellowbrickValueError("must specify array of param values, '{}' is not valid".format(repr(param_range)))
        self.param_name = param_name
        self.param_range = param_range
        self.logx = logx
        self.groups = groups
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.markers = markers

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits the validation curve with the wrapped estimator and parameter\n        array to the specified data. Draws training and test score curves and\n        saves the scores to the visualizer.\n\n        Parameters\n        ----------\n        X : array-like, shape (n_samples, n_features)\n            Training vector, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        y : array-like, shape (n_samples) or (n_samples, n_features), optional\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the validation curve visualizer for use in\n            pipelines and other sequential transformers.\n        '
        skvc_kwargs = {key: self.get_params()[key] for key in ('param_name', 'param_range', 'groups', 'cv', 'scoring', 'n_jobs', 'pre_dispatch')}
        curve = sk_validation_curve(self.estimator, X, y, **skvc_kwargs)
        (self.train_scores_, self.test_scores_) = curve
        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)
        self.draw()
        return self

    def draw(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Renders the training and test curves.\n        '
        labels = ('Training Score', 'Cross Validation Score')
        curves = ((self.train_scores_mean_, self.train_scores_std_), (self.test_scores_mean_, self.test_scores_std_))
        colors = resolve_colors(n_colors=2)
        for (idx, (mean, std)) in enumerate(curves):
            self.ax.fill_between(self.param_range, mean - std, mean + std, alpha=0.25, color=colors[idx])
        for (idx, (mean, _)) in enumerate(curves):
            self.ax.plot(self.param_range, mean, self.markers, color=colors[idx], label=labels[idx])
        if self.logx:
            self.ax.set_xscale('log')
        return self.ax

    def finalize(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Add the title, legend, and other visual final touches to the plot.\n        '
        self.set_title('Validation Curve for {}'.format(self.name))
        self.ax.legend(frameon=True, loc='best')
        self.ax.set_xlabel(self.param_name)
        self.ax.set_ylabel('score')

def validation_curve(estimator, X, y, param_name, param_range, ax=None, logx=False, groups=None, cv=None, scoring=None, n_jobs=1, pre_dispatch='all', show=True, markers='-d', **kwargs):
    if False:
        print('Hello World!')
    "\n    Displays a validation curve for the specified param and values, plotting\n    both the train and cross-validated test scores. The validation curve is a\n    visual, single-parameter grid search used to tune a model to find the best\n    balance between error due to bias and error due to variance.\n\n    This helper function is a wrapper to use the ValidationCurve in a fast,\n    visual analysis.\n\n    Parameters\n    ----------\n    estimator : a scikit-learn estimator\n        An object that implements ``fit`` and ``predict``, can be a\n        classifier, regressor, or clusterer so long as there is also a valid\n        associated scoring metric.\n\n        Note that the object is cloned for each validation.\n\n    X : array-like, shape (n_samples, n_features)\n        Training vector, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape (n_samples) or (n_samples, n_features), optional\n        Target relative to X for classification or regression;\n        None for unsupervised learning.\n\n    param_name : string\n        Name of the parameter that will be varied.\n\n    param_range : array-like, shape (n_values,)\n        The values of the parameter that will be evaluated.\n\n    ax : matplotlib.Axes object, optional\n        The axes object to plot the figure on.\n\n    logx : boolean, optional\n        If True, plots the x-axis with a logarithmic scale.\n\n    groups : array-like, with shape (n_samples,)\n        Optional group labels for the samples used while splitting the dataset\n        into train/test sets.\n\n    cv : int, cross-validation generator or an iterable, optional\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n          - None, to use the default 3-fold cross-validation,\n          - integer, to specify the number of folds.\n          - An object to be used as a cross-validation generator.\n          - An iterable yielding train/test splits.\n\n        see the scikit-learn\n        `cross-validation guide <https://bit.ly/2MMQAI7>`_\n        for more information on the possible strategies that can be used here.\n\n    scoring : string, callable or None, optional, default: None\n        A string or scorer callable object / function with signature\n        ``scorer(estimator, X, y)``. See scikit-learn model evaluation\n        documentation for names of possible metrics.\n\n    n_jobs : integer, optional\n        Number of jobs to run in parallel (default 1).\n\n    pre_dispatch : integer or string, optional\n        Number of predispatched jobs for parallel execution (default is\n        all). The option can reduce the allocated memory. The string can\n        be an expression like '2*n_jobs'.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    markers : string, default: '-d'\n        Matplotlib style markers for points on the plot points\n        Options: '-,', '-+', '-o', '-*', '-v', '-h', '-d'\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers. These arguments are\n        also passed to the ``show()`` method, e.g. can pass a path to save the\n        figure to.\n\n    Returns\n    -------\n    visualizer : ValidationCurve\n        The fitted visualizer\n    "
    oz = ValidationCurve(estimator, param_name, param_range, ax=ax, logx=logx, groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch, markers=markers)
    oz.fit(X, y)
    if show:
        oz.show(**kwargs)
    else:
        oz.finalize()
    return oz