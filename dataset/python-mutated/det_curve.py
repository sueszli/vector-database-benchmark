import scipy as sp
from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import det_curve

class DetCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    """DET curve visualization.

    It is recommend to use :func:`~sklearn.metrics.DetCurveDisplay.from_estimator`
    or :func:`~sklearn.metrics.DetCurveDisplay.from_predictions` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    fnr : ndarray
        False negative rate.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

    Attributes
    ----------
    line_ : matplotlib Artist
        DET Curve.

    ax_ : matplotlib Axes
        Axes with DET Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    det_curve : Compute error rates for different probability thresholds.
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import det_curve, DetCurveDisplay
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=0)
    >>> clf = SVC(random_state=0).fit(X_train, y_train)
    >>> y_pred = clf.decision_function(X_test)
    >>> fpr, fnr, _ = det_curve(y_test, y_pred)
    >>> display = DetCurveDisplay(
    ...     fpr=fpr, fnr=fnr, estimator_name="SVC"
    ... )
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, fpr, fnr, estimator_name=None, pos_label=None):
        if False:
            return 10
        self.fpr = fpr
        self.fnr = fnr
        self.estimator_name = estimator_name
        self.pos_label = pos_label

    @classmethod
    def from_estimator(cls, estimator, X, y, *, sample_weight=None, response_method='auto', pos_label=None, name=None, ax=None, **kwargs):
        if False:
            print('Hello World!')
        "Plot DET curve given an estimator and data.\n\n        Read more in the :ref:`User Guide <visualizations>`.\n\n        .. versionadded:: 1.0\n\n        Parameters\n        ----------\n        estimator : estimator instance\n            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`\n            in which the last estimator is a classifier.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input values.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        response_method : {'predict_proba', 'decision_function', 'auto'}                 default='auto'\n            Specifies whether to use :term:`predict_proba` or\n            :term:`decision_function` as the predicted target response. If set\n            to 'auto', :term:`predict_proba` is tried first and if it does not\n            exist :term:`decision_function` is tried next.\n\n        pos_label : int, float, bool or str, default=None\n            The label of the positive class. When `pos_label=None`, if `y_true`\n            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an\n            error will be raised.\n\n        name : str, default=None\n            Name of DET curve for labeling. If `None`, use the name of the\n            estimator.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        **kwargs : dict\n            Additional keywords arguments passed to matplotlib `plot` function.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.DetCurveDisplay`\n            Object that stores computed values.\n\n        See Also\n        --------\n        det_curve : Compute error rates for different probability thresholds.\n        DetCurveDisplay.from_predictions : Plot DET curve given the true and\n            predicted labels.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import DetCurveDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.svm import SVC\n        >>> X, y = make_classification(n_samples=1000, random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...     X, y, test_size=0.4, random_state=0)\n        >>> clf = SVC(random_state=0).fit(X_train, y_train)\n        >>> DetCurveDisplay.from_estimator(\n        ...    clf, X_test, y_test)\n        <...>\n        >>> plt.show()\n        "
        (y_pred, pos_label, name) = cls._validate_and_get_response_values(estimator, X, y, response_method=response_method, pos_label=pos_label, name=name)
        return cls.from_predictions(y_true=y, y_pred=y_pred, sample_weight=sample_weight, name=name, ax=ax, pos_label=pos_label, **kwargs)

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, sample_weight=None, pos_label=None, name=None, ax=None, **kwargs):
        if False:
            while True:
                i = 10
        'Plot the DET curve given the true and predicted labels.\n\n        Read more in the :ref:`User Guide <visualizations>`.\n\n        .. versionadded:: 1.0\n\n        Parameters\n        ----------\n        y_true : array-like of shape (n_samples,)\n            True labels.\n\n        y_pred : array-like of shape (n_samples,)\n            Target scores, can either be probability estimates of the positive\n            class, confidence values, or non-thresholded measure of decisions\n            (as returned by `decision_function` on some classifiers).\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        pos_label : int, float, bool or str, default=None\n            The label of the positive class. When `pos_label=None`, if `y_true`\n            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an\n            error will be raised.\n\n        name : str, default=None\n            Name of DET curve for labeling. If `None`, name will be set to\n            `"Classifier"`.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        **kwargs : dict\n            Additional keywords arguments passed to matplotlib `plot` function.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.DetCurveDisplay`\n            Object that stores computed values.\n\n        See Also\n        --------\n        det_curve : Compute error rates for different probability thresholds.\n        DetCurveDisplay.from_estimator : Plot DET curve given an estimator and\n            some data.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import DetCurveDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.svm import SVC\n        >>> X, y = make_classification(n_samples=1000, random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...     X, y, test_size=0.4, random_state=0)\n        >>> clf = SVC(random_state=0).fit(X_train, y_train)\n        >>> y_pred = clf.decision_function(X_test)\n        >>> DetCurveDisplay.from_predictions(\n        ...    y_test, y_pred)\n        <...>\n        >>> plt.show()\n        '
        (pos_label_validated, name) = cls._validate_from_predictions_params(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label, name=name)
        (fpr, fnr, _) = det_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
        viz = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name, pos_label=pos_label_validated)
        return viz.plot(ax=ax, name=name, **kwargs)

    def plot(self, ax=None, *, name=None, **kwargs):
        if False:
            return 10
        'Plot visualization.\n\n        Parameters\n        ----------\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        name : str, default=None\n            Name of DET curve for labeling. If `None`, use `estimator_name` if\n            it is not `None`, otherwise no labeling is shown.\n\n        **kwargs : dict\n            Additional keywords arguments passed to matplotlib `plot` function.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.DetCurveDisplay`\n            Object that stores computed values.\n        '
        (self.ax_, self.figure_, name) = self._validate_plot_params(ax=ax, name=name)
        line_kwargs = {} if name is None else {'label': name}
        line_kwargs.update(**kwargs)
        (self.line_,) = self.ax_.plot(sp.stats.norm.ppf(self.fpr), sp.stats.norm.ppf(self.fnr), **line_kwargs)
        info_pos_label = f' (Positive label: {self.pos_label})' if self.pos_label is not None else ''
        xlabel = 'False Positive Rate' + info_pos_label
        ylabel = 'False Negative Rate' + info_pos_label
        self.ax_.set(xlabel=xlabel, ylabel=ylabel)
        if 'label' in line_kwargs:
            self.ax_.legend(loc='lower right')
        ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999]
        tick_locations = sp.stats.norm.ppf(ticks)
        tick_labels = ['{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s) for s in ticks]
        self.ax_.set_xticks(tick_locations)
        self.ax_.set_xticklabels(tick_labels)
        self.ax_.set_xlim(-3, 3)
        self.ax_.set_yticks(tick_locations)
        self.ax_.set_yticklabels(tick_labels)
        self.ax_.set_ylim(-3, 3)
        return self