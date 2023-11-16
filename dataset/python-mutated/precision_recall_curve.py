from collections import Counter
from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import average_precision_score, precision_recall_curve

class PrecisionRecallDisplay(_BinaryClassifierCurveDisplayMixin):
    """Precision Recall visualization.

    It is recommend to use
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_estimator` or
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_predictions` to create
    a :class:`~sklearn.metrics.PrecisionRecallDisplay`. All parameters are
    stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    precision : ndarray
        Precision values.

    recall : ndarray
        Recall values.

    average_precision : float, default=None
        Average precision. If None, the average precision is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, then the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

        .. versionadded:: 0.24

    prevalence_pos_label : float, default=None
        The prevalence of the positive label. It is used for plotting the
        chance level line. If None, the chance level line will not be plotted
        even if `plot_chance_level` is set to True when plotting.

        .. versionadded:: 1.3

    Attributes
    ----------
    line_ : matplotlib Artist
        Precision recall curve.

    chance_level_ : matplotlib Artist or None
        The chance level line. It is `None` if the chance level is not plotted.

        .. versionadded:: 1.3

    ax_ : matplotlib Axes
        Axes with precision recall curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.
    PrecisionRecallDisplay.from_estimator : Plot Precision Recall Curve given
        a binary classifier.
    PrecisionRecallDisplay.from_predictions : Plot Precision Recall Curve
        using predictions from a binary classifier.

    Notes
    -----
    The average precision (cf. :func:`~sklearn.metrics.average_precision_score`) in
    scikit-learn is computed without any interpolation. To be consistent with
    this metric, the precision-recall curve is plotted without any
    interpolation as well (step-wise style).

    You can change this style by passing the keyword argument
    `drawstyle="default"` in :meth:`plot`, :meth:`from_estimator`, or
    :meth:`from_predictions`. However, the curve will not be strictly
    consistent with the reported average precision.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import (precision_recall_curve,
    ...                              PrecisionRecallDisplay)
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> clf = SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> predictions = clf.predict(X_test)
    >>> precision, recall, _ = precision_recall_curve(y_test, predictions)
    >>> disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    >>> disp.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, precision, recall, *, average_precision=None, estimator_name=None, pos_label=None, prevalence_pos_label=None):
        if False:
            while True:
                i = 10
        self.estimator_name = estimator_name
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.pos_label = pos_label
        self.prevalence_pos_label = prevalence_pos_label

    def plot(self, ax=None, *, name=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            while True:
                i = 10
        'Plot visualization.\n\n        Extra keyword arguments will be passed to matplotlib\'s `plot`.\n\n        Parameters\n        ----------\n        ax : Matplotlib Axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        name : str, default=None\n            Name of precision recall curve for labeling. If `None`, use\n            `estimator_name` if not `None`, otherwise no labeling is shown.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level. The chance level is the prevalence\n            of the positive label computed from the data passed during\n            :meth:`from_estimator` or :meth:`from_predictions` call.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib\'s `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Keyword arguments to be passed to matplotlib\'s `plot`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`\n            Object that stores computed values.\n\n        Notes\n        -----\n        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)\n        in scikit-learn is computed without any interpolation. To be consistent\n        with this metric, the precision-recall curve is plotted without any\n        interpolation as well (step-wise style).\n\n        You can change this style by passing the keyword argument\n        `drawstyle="default"`. However, the curve will not be strictly\n        consistent with the reported average precision.\n        '
        (self.ax_, self.figure_, name) = self._validate_plot_params(ax=ax, name=name)
        line_kwargs = {'drawstyle': 'steps-post'}
        if self.average_precision is not None and name is not None:
            line_kwargs['label'] = f'{name} (AP = {self.average_precision:0.2f})'
        elif self.average_precision is not None:
            line_kwargs['label'] = f'AP = {self.average_precision:0.2f}'
        elif name is not None:
            line_kwargs['label'] = name
        line_kwargs.update(**kwargs)
        (self.line_,) = self.ax_.plot(self.recall, self.precision, **line_kwargs)
        info_pos_label = f' (Positive label: {self.pos_label})' if self.pos_label is not None else ''
        xlabel = 'Recall' + info_pos_label
        ylabel = 'Precision' + info_pos_label
        self.ax_.set(xlabel=xlabel, xlim=(-0.01, 1.01), ylabel=ylabel, ylim=(-0.01, 1.01), aspect='equal')
        if plot_chance_level:
            if self.prevalence_pos_label is None:
                raise ValueError('You must provide prevalence_pos_label when constructing the PrecisionRecallDisplay object in order to plot the chance level line. Alternatively, you may use PrecisionRecallDisplay.from_estimator or PrecisionRecallDisplay.from_predictions to automatically set prevalence_pos_label')
            chance_level_line_kw = {'label': f'Chance level (AP = {self.prevalence_pos_label:0.2f})', 'color': 'k', 'linestyle': '--'}
            if chance_level_kw is not None:
                chance_level_line_kw.update(chance_level_kw)
            (self.chance_level_,) = self.ax_.plot((0, 1), (self.prevalence_pos_label, self.prevalence_pos_label), **chance_level_line_kw)
        else:
            self.chance_level_ = None
        if 'label' in line_kwargs or plot_chance_level:
            self.ax_.legend(loc='lower left')
        return self

    @classmethod
    def from_estimator(cls, estimator, X, y, *, sample_weight=None, pos_label=None, drop_intermediate=False, response_method='auto', name=None, ax=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot precision-recall curve given an estimator and some data.\n\n        Parameters\n        ----------\n        estimator : estimator instance\n            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`\n            in which the last estimator is a classifier.\n\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Input values.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        pos_label : int, float, bool or str, default=None\n            The class considered as the positive class when computing the\n            precision and recall metrics. By default, `estimators.classes_[1]`\n            is considered as the positive class.\n\n        drop_intermediate : bool, default=False\n            Whether to drop some suboptimal thresholds which would not appear\n            on a plotted precision-recall curve. This is useful in order to\n            create lighter precision-recall curves.\n\n            .. versionadded:: 1.3\n\n        response_method : {\'predict_proba\', \'decision_function\', \'auto\'},             default=\'auto\'\n            Specifies whether to use :term:`predict_proba` or\n            :term:`decision_function` as the target response. If set to \'auto\',\n            :term:`predict_proba` is tried first and if it does not exist\n            :term:`decision_function` is tried next.\n\n        name : str, default=None\n            Name for labeling curve. If `None`, no name is used.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is created.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level. The chance level is the prevalence\n            of the positive label computed from the data passed during\n            :meth:`from_estimator` or :meth:`from_predictions` call.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib\'s `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Keyword arguments to be passed to matplotlib\'s `plot`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`\n\n        See Also\n        --------\n        PrecisionRecallDisplay.from_predictions : Plot precision-recall curve\n            using estimated probabilities or output of decision function.\n\n        Notes\n        -----\n        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)\n        in scikit-learn is computed without any interpolation. To be consistent\n        with this metric, the precision-recall curve is plotted without any\n        interpolation as well (step-wise style).\n\n        You can change this style by passing the keyword argument\n        `drawstyle="default"`. However, the curve will not be strictly\n        consistent with the reported average precision.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import PrecisionRecallDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.linear_model import LogisticRegression\n        >>> X, y = make_classification(random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...         X, y, random_state=0)\n        >>> clf = LogisticRegression()\n        >>> clf.fit(X_train, y_train)\n        LogisticRegression()\n        >>> PrecisionRecallDisplay.from_estimator(\n        ...    clf, X_test, y_test)\n        <...>\n        >>> plt.show()\n        '
        (y_pred, pos_label, name) = cls._validate_and_get_response_values(estimator, X, y, response_method=response_method, pos_label=pos_label, name=name)
        return cls.from_predictions(y, y_pred, sample_weight=sample_weight, name=name, pos_label=pos_label, drop_intermediate=drop_intermediate, ax=ax, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw, **kwargs)

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, sample_weight=None, pos_label=None, drop_intermediate=False, name=None, ax=None, plot_chance_level=False, chance_level_kw=None, **kwargs):
        if False:
            while True:
                i = 10
        'Plot precision-recall curve given binary class predictions.\n\n        Parameters\n        ----------\n        y_true : array-like of shape (n_samples,)\n            True binary labels.\n\n        y_pred : array-like of shape (n_samples,)\n            Estimated probabilities or output of decision function.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        pos_label : int, float, bool or str, default=None\n            The class considered as the positive class when computing the\n            precision and recall metrics.\n\n        drop_intermediate : bool, default=False\n            Whether to drop some suboptimal thresholds which would not appear\n            on a plotted precision-recall curve. This is useful in order to\n            create lighter precision-recall curves.\n\n            .. versionadded:: 1.3\n\n        name : str, default=None\n            Name for labeling curve. If `None`, name will be set to\n            `"Classifier"`.\n\n        ax : matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is created.\n\n        plot_chance_level : bool, default=False\n            Whether to plot the chance level. The chance level is the prevalence\n            of the positive label computed from the data passed during\n            :meth:`from_estimator` or :meth:`from_predictions` call.\n\n            .. versionadded:: 1.3\n\n        chance_level_kw : dict, default=None\n            Keyword arguments to be passed to matplotlib\'s `plot` for rendering\n            the chance level line.\n\n            .. versionadded:: 1.3\n\n        **kwargs : dict\n            Keyword arguments to be passed to matplotlib\'s `plot`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.metrics.PrecisionRecallDisplay`\n\n        See Also\n        --------\n        PrecisionRecallDisplay.from_estimator : Plot precision-recall curve\n            using an estimator.\n\n        Notes\n        -----\n        The average precision (cf. :func:`~sklearn.metrics.average_precision_score`)\n        in scikit-learn is computed without any interpolation. To be consistent\n        with this metric, the precision-recall curve is plotted without any\n        interpolation as well (step-wise style).\n\n        You can change this style by passing the keyword argument\n        `drawstyle="default"`. However, the curve will not be strictly\n        consistent with the reported average precision.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_classification\n        >>> from sklearn.metrics import PrecisionRecallDisplay\n        >>> from sklearn.model_selection import train_test_split\n        >>> from sklearn.linear_model import LogisticRegression\n        >>> X, y = make_classification(random_state=0)\n        >>> X_train, X_test, y_train, y_test = train_test_split(\n        ...         X, y, random_state=0)\n        >>> clf = LogisticRegression()\n        >>> clf.fit(X_train, y_train)\n        LogisticRegression()\n        >>> y_pred = clf.predict_proba(X_test)[:, 1]\n        >>> PrecisionRecallDisplay.from_predictions(\n        ...    y_test, y_pred)\n        <...>\n        >>> plt.show()\n        '
        (pos_label, name) = cls._validate_from_predictions_params(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label, name=name)
        (precision, recall, _) = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, drop_intermediate=drop_intermediate)
        average_precision = average_precision_score(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
        class_count = Counter(y_true)
        prevalence_pos_label = class_count[pos_label] / sum(class_count.values())
        viz = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision, estimator_name=name, pos_label=pos_label, prevalence_pos_label=prevalence_pos_label)
        return viz.plot(ax=ax, name=name, plot_chance_level=plot_chance_level, chance_level_kw=chance_level_kw, **kwargs)