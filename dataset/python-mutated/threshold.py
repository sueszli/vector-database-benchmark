"""
DiscriminationThreshold visualizer for probabilistic classifiers.
"""
import bisect
import numpy as np
from scipy.stats import mstats
from collections import defaultdict
from sklearn.base import clone
from sklearn.utils import indexable
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import type_of_target
try:
    from sklearn.utils import safe_indexing as _safe_indexing
except ImportError:
    from sklearn.utils import _safe_indexing
from yellowbrick.base import ModelVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.utils import is_classifier, is_monotonic, is_probabilistic
from yellowbrick.exceptions import YellowbrickTypeError, YellowbrickValueError
QUANTILES_MEDIAN_80 = np.array([0.1, 0.5, 0.9])
METRICS = ['precision', 'recall', 'fscore', 'queue_rate']

class DiscriminationThreshold(ModelVisualizer):
    """
    Visualizes how precision, recall, f1 score, and queue rate change as the
    discrimination threshold increases. For probabilistic, binary classifiers,
    the discrimination threshold is the probability at which you choose the
    positive class over the negative. Generally this is set to 50%, but
    adjusting the discrimination threshold will adjust sensitivity to false
    positives which is described by the inverse relationship of precision and
    recall with respect to the threshold.

    The visualizer also accounts for variability in the model by running
    multiple trials with different train and test splits of the data. The
    variability is visualized using a band such that the curve is drawn as the
    median score of each trial and the band is from the 10th to 90th
    percentile.

    The visualizer is intended to help users determine an appropriate
    threshold for decision making (e.g. at what threshold do we have a human
    review the data), given a tolerance for precision and recall or limiting
    the number of records to check (the queue rate).

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    n_trials : integer, default: 50
        Number of times to shuffle and split the dataset to account for noise
        in the threshold metrics curves. Note if cv provides > 1 splits,
        the number of trials will be n_trials * cv.get_n_splits()

    cv : float or cross-validation generator, default: 0.1
        Determines the splitting strategy for each trial. Possible inputs are:

        - float, to specify the percent of the test split
        - object to be used as cross-validation generator

        This attribute is meant to give flexibility with stratified splitting
        but if a splitter is provided, it should only return one split and
        have shuffle set to True.

    fbeta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    argmax : str or None, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None or passed to exclude,
        will not annotate the graph.

    exclude : str or list, optional
        Specify metrics to omit from the graph, can include:

        - ``"precision"``
        - ``"recall"``
        - ``"queue_rate"``
        - ``"fscore"``

        Excluded metrics will not be displayed in the graph, nor will they
        be available in ``thresholds_``; however, they will be computed on fit.

    quantiles : sequence, default: np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of
        trials. Must be monotonic and have three elements such that the first
        element is the lower bound, the second is the drawn curve, and the
        third is the upper bound. By default the curve is drawn at the median,
        and the bounds from the 10th percentile to the 90th percentile.

    random_state : int, optional
        Used to seed the random state for shuffling the data while composing
        different train and test splits. If supplied, the random state is
        incremented in a deterministic fashion for each split.

        Note that if a splitter is provided, it's random state will also be
        updated with this random state, even if it was previously set.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Attributes
    ----------
    thresholds_ : array
        The uniform thresholds identified by each of the trial runs.

    cv_scores_ : dict of arrays of ``len(thresholds_)``
        The values for all included metrics including the upper and lower
        bounds of the metrics defined by quantiles.

    Notes
    -----
    The term "discrimination threshold" is rare in the literature. Here, we
    use it to mean the probability at which the positive class is selected
    over the negative class in binary classification.

    Classification models must implement either a ``decision_function`` or
    ``predict_proba`` method in order to be used with this class. A
    ``YellowbrickTypeError`` is raised otherwise.

    .. caution:: This method only works for binary, probabilistic classifiers.

    .. seealso::
        For a thorough explanation of discrimination thresholds, see:
        `Visualizing Machine Learning Thresholds to Make Better Business
        Decisions
        <http://blog.insightdatalabs.com/visualizing-classifier-thresholds/>`_
        by Insight Data.
    """

    def __init__(self, estimator, ax=None, n_trials=50, cv=0.1, fbeta=1.0, argmax='fscore', exclude=None, quantiles=QUANTILES_MEDIAN_80, random_state=None, is_fitted='auto', force_model=False, **kwargs):
        if False:
            i = 10
            return i + 15
        if not force_model and (not is_classifier(estimator) or not is_probabilistic(estimator)):
            raise YellowbrickTypeError('{} requires a probabilistic binary classifier'.format(self.__class__.__name__))
        self._check_quantiles(quantiles)
        self._check_cv(cv)
        self._check_exclude(exclude)
        self._check_argmax(argmax, exclude)
        super(DiscriminationThreshold, self).__init__(estimator, ax=ax, is_fitted=is_fitted, **kwargs)
        self.n_trials = n_trials
        self.cv = cv
        self.fbeta = fbeta
        self.argmax = argmax
        self.exclude = exclude
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Fit is the entry point for the visualizer. Given instances described\n        by X and binary classes described in the target y, fit performs n\n        trials by shuffling and splitting the dataset then computing the\n        precision, recall, f1, and queue rate scores for each trial. The\n        scores are aggregated by the quantiles expressed then drawn.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target or class values. The target y must\n            be a binary classification target.\n\n        kwargs: dict\n            keyword arguments passed to Scikit-Learn API.\n\n        Returns\n        -------\n        self : instance\n            Returns the instance of the visualizer\n\n        raises: YellowbrickValueError\n            If the target y is not a binary classification target.\n        '
        if type_of_target(y) != 'binary':
            raise YellowbrickValueError('multiclass format is not supported')
        (X, y) = indexable(X, y)
        trials = [metric for idx in range(self.n_trials) for metric in self._split_fit_score_trial(X, y, idx)]
        n_thresholds = np.array([len(t['thresholds']) for t in trials]).min()
        self.thresholds_ = np.linspace(0.0, 1.0, num=n_thresholds)
        metrics = frozenset(METRICS) - self._check_exclude(self.exclude)
        uniform_metrics = defaultdict(list)
        for trial in trials:
            rows = defaultdict(list)
            for t in self.thresholds_:
                idx = bisect.bisect_left(trial['thresholds'], t)
                for metric in metrics:
                    rows[metric].append(trial[metric][idx])
            for (metric, row) in rows.items():
                uniform_metrics[metric].append(row)
        uniform_metrics = {metric: np.array(values) for (metric, values) in uniform_metrics.items()}
        quantiles = self._check_quantiles(self.quantiles)
        self.cv_scores_ = {}
        for (metric, values) in uniform_metrics.items():
            (lower, median, upper) = mstats.mquantiles(values, prob=quantiles, axis=0)
            self.cv_scores_[metric] = median
            self.cv_scores_['{}_lower'.format(metric)] = lower
            self.cv_scores_['{}_upper'.format(metric)] = upper
        super(DiscriminationThreshold, self).fit(X, y)
        self.draw()
        return self

    def _split_fit_score_trial(self, X, y, idx=0):
        if False:
            print('Hello World!')
        '\n        Splits the dataset, fits a clone of the estimator, then scores it\n        according to the required metrics.\n\n        The index of the split is added to the random_state if the\n        random_state is not None; this ensures that every split is shuffled\n        differently but in a deterministic fashion for testing purposes.\n        '
        random_state = self.random_state
        if random_state is not None:
            random_state += idx
        splitter = self._check_cv(self.cv, random_state)
        for (train_index, test_index) in splitter.split(X, y):
            X_train = _safe_indexing(X, train_index)
            y_train = _safe_indexing(y, train_index)
            X_test = _safe_indexing(X, test_index)
            y_test = _safe_indexing(y, test_index)
            model = clone(self.estimator)
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.decision_function(X_test)
            curve_metrics = precision_recall_curve(y_test, y_scores)
            (precision, recall, thresholds) = curve_metrics
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = self.fbeta ** 2
                f_score = (1 + beta) * precision * recall / (beta * precision + recall)
            thresholds = np.append(thresholds, 1)
            queue_rate = np.array([(y_scores >= threshold).mean() for threshold in thresholds])
            yield {'thresholds': thresholds, 'precision': precision, 'recall': recall, 'fscore': f_score, 'queue_rate': queue_rate}

    def draw(self):
        if False:
            i = 10
            return i + 15
        '\n        Draws the cv scores as a line chart on the current axes.\n        '
        color_values = resolve_colors(n_colors=4, colors=self.color)
        argmax = self._check_argmax(self.argmax, self.exclude)
        for (idx, metric) in enumerate(METRICS):
            if metric not in self.cv_scores_:
                continue
            color = color_values[idx]
            if metric == 'fscore':
                if self.fbeta == 1.0:
                    label = '$f_1$'
                else:
                    label = '$f_{{\x08eta={:0.1f}}}'.format(self.fbeta)
            else:
                label = metric.replace('_', ' ')
            self.ax.plot(self.thresholds_, self.cv_scores_[metric], color=color, label=label)
            lower = self.cv_scores_['{}_lower'.format(metric)]
            upper = self.cv_scores_['{}_upper'.format(metric)]
            self.ax.fill_between(self.thresholds_, upper, lower, alpha=0.35, linewidth=0, color=color)
            if argmax and argmax == metric:
                argmax = self.cv_scores_[metric].argmax()
                threshold = self.thresholds_[argmax]
                self.ax.axvline(threshold, ls='--', c='k', lw=1, label='$t_{}={:0.2f}$'.format(metric[0], threshold))
        return self.ax

    def finalize(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Sets a title and axis labels on the visualizer and ensures that the\n        axis limits are scaled to valid threshold values.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        super(DiscriminationThreshold, self).finalize(**kwargs)
        self.set_title('Threshold Plot for {}'.format(self.name))
        self.ax.legend(frameon=True, loc='best')
        self.ax.set_xlabel('discrimination threshold')
        self.ax.set_ylabel('score')
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

    def _check_quantiles(self, val):
        if False:
            print('Hello World!')
        '\n        Validate the quantiles passed in. Returns the np array if valid.\n        '
        if len(val) != 3 or not is_monotonic(val) or (not np.all(val < 1)):
            raise YellowbrickValueError('quantiles must be a sequence of three monotonically increasing values less than 1')
        return np.asarray(val)

    def _check_cv(self, val, random_state=None):
        if False:
            while True:
                i = 10
        '\n        Validate the cv method passed in. Returns the split strategy if no\n        validation exception is raised.\n        '
        if val is None:
            val = 0.1
        if isinstance(val, float) and val <= 1.0:
            return ShuffleSplit(n_splits=1, test_size=val, random_state=random_state)
        if hasattr(val, 'split') and hasattr(val, 'get_n_splits'):
            if random_state is not None and hasattr(val, 'random_state'):
                val.random_state = random_state
            return val
        raise YellowbrickValueError("'{}' is not a valid cv splitter".format(type(val)))

    def _check_exclude(self, val):
        if False:
            print('Hello World!')
        '\n        Validate the excluded metrics. Returns the set of excluded params.\n        '
        if val is None:
            exclude = frozenset()
        elif isinstance(val, str):
            exclude = frozenset([val.lower()])
        else:
            exclude = frozenset(map(lambda s: s.lower(), val))
        if len(exclude - frozenset(METRICS)) > 0:
            raise YellowbrickValueError("'{}' is not a valid metric to exclude".format(repr(val)))
        return exclude

    def _check_argmax(self, val, exclude=None):
        if False:
            print('Hello World!')
        '\n        Validate the argmax metric. Returns the metric used to annotate the graph.\n        '
        if val is None:
            return None
        argmax = val.lower()
        if argmax not in METRICS:
            raise YellowbrickValueError("'{}' is not a valid metric to use".format(repr(val)))
        exclude = self._check_exclude(exclude)
        if argmax in exclude:
            argmax = None
        return argmax

def discrimination_threshold(estimator, X, y, ax=None, n_trials=50, cv=0.1, fbeta=1.0, argmax='fscore', exclude=None, quantiles=QUANTILES_MEDIAN_80, random_state=None, is_fitted='auto', force_model=False, show=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Discrimination Threshold\n\n    Visualizes how precision, recall, f1 score, and queue rate change as the\n    discrimination threshold increases. For probabilistic, binary classifiers,\n    the discrimination threshold is the probability at which you choose the\n    positive class over the negative. Generally this is set to 50%, but\n    adjusting the discrimination threshold will adjust sensitivity to false\n    positives which is described by the inverse relationship of precision and\n    recall with respect to the threshold.\n\n    .. seealso:: See DiscriminationThreshold for more details.\n\n    Parameters\n    ----------\n    estimator : estimator\n        A scikit-learn estimator that should be a classifier. If the model is\n        not a classifier, an exception is raised. If the internal model is not\n        fitted, it is fit when the visualizer is fitted, unless otherwise specified\n        by ``is_fitted``.\n\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    y : ndarray or Series of length n\n        An array or series of target or class values. The target y must\n        be a binary classification target.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If not specified the current axes will be\n        used (or generated if required).\n\n    n_trials : integer, default: 50\n        Number of times to shuffle and split the dataset to account for noise\n        in the threshold metrics curves. Note if cv provides > 1 splits,\n        the number of trials will be n_trials * cv.get_n_splits()\n\n    cv : float or cross-validation generator, default: 0.1\n        Determines the splitting strategy for each trial. Possible inputs are:\n\n        - float, to specify the percent of the test split\n        - object to be used as cross-validation generator\n\n        This attribute is meant to give flexibility with stratified splitting\n        but if a splitter is provided, it should only return one split and\n        have shuffle set to True.\n\n    fbeta : float, 1.0 by default\n        The strength of recall versus precision in the F-score.\n\n    argmax : str or None, default: \'fscore\'\n        Annotate the threshold maximized by the supplied metric (see exclude\n        for the possible metrics to use). If None or passed to exclude,\n        will not annotate the graph.\n\n    exclude : str or list, optional\n        Specify metrics to omit from the graph, can include:\n\n        - ``"precision"``\n        - ``"recall"``\n        - ``"queue_rate"``\n        - ``"fscore"``\n\n        Excluded metrics will not be displayed in the graph, nor will they\n        be available in ``thresholds_``; however, they will be computed on fit.\n\n    quantiles : sequence, default: np.array([0.1, 0.5, 0.9])\n        Specify the quantiles to view model variability across a number of\n        trials. Must be monotonic and have three elements such that the first\n        element is the lower bound, the second is the drawn curve, and the\n        third is the upper bound. By default the curve is drawn at the median,\n        and the bounds from the 10th percentile to the 90th percentile.\n\n    random_state : int, optional\n        Used to seed the random state for shuffling the data while composing\n        different train and test splits. If supplied, the random state is\n        incremented in a deterministic fashion for each split.\n\n        Note that if a splitter is provided, it\'s random state will also be\n        updated with this random state, even if it was previously set.\n\n    is_fitted : bool or str, default="auto"\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If "auto" (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    force_model : bool, default: False\n        Do not check to ensure that the underlying estimator is a classifier. This\n        will prevent an exception when the visualizer is initialized but may result\n        in unexpected or unintended behavior.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments passed to the visualizer base classes.\n\n    Notes\n    -----\n    The term "discrimination threshold" is rare in the literature. Here, we\n    use it to mean the probability at which the positive class is selected\n    over the negative class in binary classification.\n\n    Classification models must implement either a ``decision_function`` or\n    ``predict_proba`` method in order to be used with this class. A\n    ``YellowbrickTypeError`` is raised otherwise.\n\n    .. seealso::\n        For a thorough explanation of discrimination thresholds, see:\n        `Visualizing Machine Learning Thresholds to Make Better Business\n        Decisions\n        <http://blog.insightdatalabs.com/visualizing-classifier-thresholds/>`_\n        by Insight Data.\n\n    Examples\n    --------\n    >>> from yellowbrick.classifier.threshold import discrimination_threshold\n    >>> from sklearn.linear_model import LogisticRegression\n    >>> from yellowbrick.datasets import load_occupancy\n    >>> X, y = load_occupancy()\n    >>> model = LogisticRegression(multi_class="auto", solver="liblinear")\n    >>> discrimination_threshold(model, X, y)\n\n    Returns\n    -------\n    viz : DiscriminationThreshold\n        Returns the fitted and finalized visualizer object.\n    '
    visualizer = DiscriminationThreshold(estimator, ax=ax, n_trials=n_trials, cv=cv, fbeta=fbeta, argmax=argmax, exclude=exclude, quantiles=quantiles, random_state=random_state, is_fitted=is_fitted, force_model=force_model, **kwargs)
    visualizer.fit(X, y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer