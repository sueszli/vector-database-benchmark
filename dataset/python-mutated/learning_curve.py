"""
Implements a learning curve visualization for model selection.
"""
import numpy as np
from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.model_selection import learning_curve as sk_learning_curve
DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 5)

class LearningCurve(ModelVisualizer):
    """
    Visualizes the learning curve for both test and training data for
    different training set sizes. These curves can act as a proxy to
    demonstrate the implied learning rate with experience (e.g. how much data
    is required to make an adequate model). They also demonstrate if the model
    is more sensitive to error due to bias vs. error due to variance and can
    be used to quickly check if a model is overfitting.

    The visualizer evaluates cross-validated training and test scores for
    different training set sizes. These curves are plotted so that the x-axis
    is the training set size and the y-axis is the score.

    The cross-validation generator splits the whole dataset k times, scores
    are averaged over all k runs for the training subset. The curve plots the
    mean score for the k splits, and the filled in area suggests the
    variability of the cross-validation by plotting one standard deviation
    above and below the mean for each split.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    groups : array-like, with shape (n_samples,)
        Optional group labels for the samples used while splitting the dataset
        into train/test sets.

    train_sizes : array-like, shape (n_ticks,)
        default: ``np.linspace(0.1,1.0,5)``

        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as
        a fraction of the maximum size of the training set, otherwise it is
        interpreted as absolute sizes of the training sets.

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

    exploit_incremental_learning : boolean, default: False
        If the estimator supports incremental learning, this will be used to
        speed up fitting for different training set sizes.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    shuffle : boolean, optional
        Whether to shuffle training data before taking prefixes of it
        based on``train_sizes``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` is True.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    train_sizes_ : array, shape = (n_unique_ticks,), dtype int
        Numbers of training examples that has been used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

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

    >>> from yellowbrick.model_selection import LearningCurve
    >>> from sklearn.naive_bayes import GaussianNB
    >>> model = LearningCurve(GaussianNB())
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

    def __init__(self, estimator, ax=None, groups=None, train_sizes=DEFAULT_TRAIN_SIZES, cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=1, pre_dispatch='all', shuffle=False, random_state=None, **kwargs):
        if False:
            return 10
        super(LearningCurve, self).__init__(estimator, ax=ax, **kwargs)
        train_sizes = np.asarray(train_sizes)
        if train_sizes.ndim != 1:
            raise YellowbrickValueError("must specify array of train sizes, '{}' is not valid".format(repr(train_sizes)))
        self.groups = groups
        self.train_sizes = train_sizes
        self.cv = cv
        self.scoring = scoring
        self.exploit_incremental_learning = exploit_incremental_learning
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Fits the learning curve with the wrapped model to the specified data.\n        Draws training and test score curves and saves the scores to the\n        estimator.\n\n        Parameters\n        ----------\n        X : array-like, shape (n_samples, n_features)\n            Training vector, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        y : array-like, shape (n_samples) or (n_samples, n_features), optional\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the learning curve visualizer for use in\n            pipelines and other sequential transformers.\n        '
        sklc_kwargs = {key: self.get_params()[key] for key in ('groups', 'train_sizes', 'cv', 'scoring', 'exploit_incremental_learning', 'n_jobs', 'pre_dispatch', 'shuffle', 'random_state')}
        curve = sk_learning_curve(self.estimator, X, y, **sklc_kwargs)
        (self.train_sizes_, self.train_scores_, self.test_scores_) = curve
        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)
        self.draw()
        return self

    def draw(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Renders the training and test learning curves.\n        '
        labels = ('Training Score', 'Cross Validation Score')
        curves = ((self.train_scores_mean_, self.train_scores_std_), (self.test_scores_mean_, self.test_scores_std_))
        colors = resolve_colors(n_colors=2)
        for (idx, (mean, std)) in enumerate(curves):
            self.ax.fill_between(self.train_sizes_, mean - std, mean + std, alpha=0.25, color=colors[idx])
        for (idx, (mean, _)) in enumerate(curves):
            self.ax.plot(self.train_sizes_, mean, 'o-', color=colors[idx], label=labels[idx])
        return self.ax

    def finalize(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Add the title, legend, and other visual final touches to the plot.\n        '
        self.set_title('Learning Curve for {}'.format(self.name))
        self.ax.legend(frameon=True, loc='best')
        self.ax.set_xlabel('Training Instances')
        self.ax.set_ylabel('Score')

def learning_curve(estimator, X, y, ax=None, groups=None, train_sizes=DEFAULT_TRAIN_SIZES, cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=1, pre_dispatch='all', shuffle=False, random_state=None, show=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Displays a learning curve based on number of samples vs training and\n    cross validation scores. The learning curve aims to show how a model\n    learns and improves with experience.\n\n    This helper function is a quick wrapper to utilize the LearningCurve\n    for one-off analysis.\n\n    Parameters\n    ----------\n    estimator : a scikit-learn estimator\n        An object that implements ``fit`` and ``predict``, can be a\n        classifier, regressor, or clusterer so long as there is also a valid\n        associated scoring metric.\n\n        Note that the object is cloned for each validation.\n\n    X : array-like, shape (n_samples, n_features)\n        Training vector, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape (n_samples) or (n_samples, n_features), optional\n        Target relative to X for classification or regression;\n        None for unsupervised learning.\n\n    ax : matplotlib.Axes object, optional\n        The axes object to plot the figure on.\n\n    groups : array-like, with shape (n_samples,)\n        Optional group labels for the samples used while splitting the dataset\n        into train/test sets.\n\n    train_sizes : array-like, shape (n_ticks,)\n        default: ``np.linspace(0.1,1.0,5)``\n\n        Relative or absolute numbers of training examples that will be used to\n        generate the learning curve. If the dtype is float, it is regarded as\n        a fraction of the maximum size of the training set, otherwise it is\n        interpreted as absolute sizes of the training sets.\n\n    cv : int, cross-validation generator or an iterable, optional\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n          - None, to use the default 3-fold cross-validation,\n          - integer, to specify the number of folds.\n          - An object to be used as a cross-validation generator.\n          - An iterable yielding train/test splits.\n\n        see the scikit-learn\n        `cross-validation guide <https://bit.ly/2MMQAI7>`_\n        for more information on the possible strategies that can be used here.\n\n    scoring : string, callable or None, optional, default: None\n        A string or scorer callable object / function with signature\n        ``scorer(estimator, X, y)``. See scikit-learn model evaluation\n        documentation for names of possible metrics.\n\n    exploit_incremental_learning : boolean, default: False\n        If the estimator supports incremental learning, this will be used to\n        speed up fitting for different training set sizes.\n\n    n_jobs : integer, optional\n        Number of jobs to run in parallel (default 1).\n\n    pre_dispatch : integer or string, optional\n        Number of predispatched jobs for parallel execution (default is\n        all). The option can reduce the allocated memory. The string can\n        be an expression like '2*n_jobs'.\n\n    shuffle : boolean, optional\n        Whether to shuffle training data before taking prefixes of it\n        based on``train_sizes``.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`. Used when ``shuffle`` is True.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers. These arguments are\n        also passed to the `show()` method, e.g. can pass a path to save the\n        figure to.\n\n    Returns\n    -------\n    visualizer : LearningCurve\n        Returns the fitted visualizer.\n    "
    oz = LearningCurve(estimator, ax=ax, groups=groups, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch, shuffle=shuffle, random_state=random_state, exploit_incremental_learning=exploit_incremental_learning, **kwargs)
    oz.fit(X, y)
    if show:
        oz.show()
    else:
        oz.finalize()
    return oz