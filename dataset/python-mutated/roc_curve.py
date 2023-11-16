from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import auc, roc_curve

class RocCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    """ROC Curve visualization.

    It is recommend to use
    :func:`~sklearn.metrics.RocCurveDisplay.from_estimator` or
    :func:`~sklearn.metrics.RocCurveDisplay.from_predictions` to create
    a :class:`~sklearn.metrics.RocCurveDisplay`. All parameters are
    stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    tpr : ndarray
        True positive rate.

    roc_auc : float, default=None
        Area under ROC curve. If None, the roc_auc score is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing the roc auc
        metrics. By default, `estimators.classes_[1]` is considered
        as the positive class.

        .. versionadded:: 0.24

    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.

    chance_level_ : matplotlib Artist or None
        The chance level line. It is `None` if the chance level is not plotted.

        .. versionadded:: 1.3

    ax_ : matplotlib Axes
        Axes with ROC Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.
    roc_auc_score : Compute the area under the ROC curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    >>> roc_auc = metrics.auc(fpr, tpr)
    >>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    ...                                   estimator_name='example estimator')
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, fpr, tpr, roc_auc=None, estimator_name=None, pos_label=None):
        if False:
            i = 10
            return i + 15
        self.estimator_name = estimator_name
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label

    def plot(self, ax=None, *, name=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Plot visualization.\n\n        Extra keyword arguments will be passed to matplotlib's ``plot``.\n\n        Parameters\n        ----------\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        name : str, default=None\n            Name of ROC Curve for labeling. If `None`, use `estimator_name` if\n            not `None`, otherwise no labeling is shown.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib's `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Keyword arguments to be passed to matplotlib's `plot`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.RocCurveDisplay`\n            Object that stores computed values.\n        "
        (self.ax_, self.figure_, name) = self._validate_plot_params(ax=ax, name=name)
        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs['label'] = f'{name} (AUC = {self.roc_auc:0.2f})'
        elif self.roc_auc is not None:
            line_kwargs['label'] = f'AUC = {self.roc_auc:0.2f}'
        elif name is not None:
            line_kwargs['label'] = name
        line_kwargs.update(**kwargs)
        chance_level_line_kw = {'label': 'Chance level (AUC = 0.5)', 'color': 'k', 'linestyle': '--'}
        if chance_level_kw is not None:
            chance_level_line_kw.update(**chance_level_kw)
        (self.line_,) = self.ax_.plot(self.fpr, self.tpr, **line_kwargs)
        info_pos_label = f' (Positive label: {self.pos_label})' if self.pos_label is not None else ''
        xlabel = 'False Positive Rate' + info_pos_label
        ylabel = 'True Positive Rate' + info_pos_label
        self.ax_.set(xlabel=xlabel, xlim=(-0.01, 1.01), ylabel=ylabel, ylim=(-0.01, 1.01), aspect='equal')
        if plot_chance_level:
            (self.chance_level_,) = self.ax_.plot((0, 1), (0, 1), **chance_level_line_kw)
        else:
            self.chance_level_ = None
        if 'label' in line_kwargs or 'label' in chance_level_line_kw:
            self.ax_.legend(loc='lower right')
        return self

    @classmethod
    def from_estimator(cls, estimator, X, y, *, sample_weight=None, drop_intermediate=True, response_method='auto', pos_label=None, name=None, ax=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Create a ROC Curve display from an estimator.\n\n        Parameters\n        ----------\n        estimator : estimator instance\n            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`\n            in which the last estimator is a classifier.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input values.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        drop_intermediate : bool, default=True\n            Whether to drop some suboptimal thresholds which would not appear\n            on a plotted ROC curve. This is useful in order to create lighter\n            ROC curves.\n\n        response_method : {'predict_proba', 'decision_function', 'auto'}                 default='auto'\n            Specifies whether to use :term:`predict_proba` or\n            :term:`decision_function` as the target response. If set to 'auto',\n            :term:`predict_proba` is tried first and if it does not exist\n            :term:`decision_function` is tried next.\n\n        pos_label : int, float, bool or str, default=None\n            The class considered as the positive class when computing the roc auc\n            metrics. By default, `estimators.classes_[1]` is considered\n            as the positive class.\n\n        name : str, default=None\n            Name of ROC Curve for labeling. If `None`, use the name of the\n            estimator.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is created.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib's `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Keyword arguments to be passed to matplotlib's `plot`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.RocCurveDisplay`\n            The ROC Curve display.\n\n        See Also\n        --------\n        roc_curve : Compute Receiver operating characteristic (ROC) curve.\n        RocCurveDisplay.from_predictions : ROC Curve visualization given the\n            probabilities of scores of a classifier.\n        roc_auc_score : Compute the area under the ROC curve.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import RocCurveDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.svm import SVC\n        >>> X, y = make_classification(random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...     X, y, random_state=0)\n        >>> clf = SVC(random_state=0).fit(X_train, y_train)\n        >>> RocCurveDisplay.from_estimator(\n        ...    clf, X_test, y_test)\n        <...>\n        >>> plt.show()\n        "
        (y_pred, pos_label, name) = cls._validate_and_get_response_values(estimator, X, y, response_method=response_method, pos_label=pos_label, name=name)
        return cls.from_predictions(y_true=y, y_pred=y_pred, sample_weight=sample_weight, drop_intermediate=drop_intermediate, name=name, ax=ax, pos_label=pos_label, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw, **kwargs)

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, sample_weight=None, drop_intermediate=True, pos_label=None, name=None, ax=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot ROC curve given the true and predicted values.\n\n        Read more in the :ref:`User Guide <visualizations>`.\n\n        .. versionadded:: 1.0\n\n        Parameters\n        ----------\n        y_true : array-like of shape (n_samples,)\n            True labels.\n\n        y_pred : array-like of shape (n_samples,)\n            Target scores, can either be probability estimates of the positive\n            class, confidence values, or non-thresholded measure of decisions\n            (as returned by “decision_function” on some classifiers).\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        drop_intermediate : bool, default=True\n            Whether to drop some suboptimal thresholds which would not appear\n            on a plotted ROC curve. This is useful in order to create lighter\n            ROC curves.\n\n        pos_label : int, float, bool or str, default=None\n            The label of the positive class. When `pos_label=None`, if `y_true`\n            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an\n            error will be raised.\n\n        name : str, default=None\n            Name of ROC curve for labeling. If `None`, name will be set to\n            `"Classifier"`.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib\'s `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Additional keywords arguments passed to matplotlib `plot` function.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.RocCurveDisplay`\n            Object that stores computed values.\n\n        See Also\n        --------\n        roc_curve : Compute Receiver operating characteristic (ROC) curve.\n        RocCurveDisplay.from_estimator : ROC Curve visualization given an\n            estimator and some data.\n        roc_auc_score : Compute the area under the ROC curve.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import RocCurveDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.svm import SVC\n        >>> X, y = make_classification(random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...     X, y, random_state=0)\n        >>> clf = SVC(random_state=0).fit(X_train, y_train)\n        >>> y_pred = clf.decision_function(X_test)\n        >>> RocCurveDisplay.from_predictions(\n        ...    y_test, y_pred)\n        <...>\n        >>> plt.show()\n        '
        (pos_label_validated, name) = cls._validate_from_predictions_params(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label, name=name)
        (fpr, tpr, _) = roc_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, drop_intermediate=drop_intermediate)
        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name, pos_label=pos_label_validated)
        return viz.plot(ax=ax, name=name, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw, **kwargs)