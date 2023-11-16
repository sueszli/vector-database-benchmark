"""
Visual classification report for classifier scoring.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from yellowbrick.style import find_text_color
from yellowbrick.style.palettes import color_sequence
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer
PERCENT = 'percent'
CMAP_UNDERCOLOR = 'w'
CMAP_OVERCOLOR = '#2a7d4f'
SCORES_KEYS = ('precision', 'recall', 'f1', 'support')

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, F1, and support scores
    for the model. Integrates numerical scores as well as a color-coded heatmap.

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

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    cmap : string, default: ``'YlOrRd'``
        Specify a colormap to define the heatmap of the predicted class
        against the actual class in the classification report.

    support: {True, False, None, 'percent', 'count'}, default: None
        Specify if support will be displayed. It can be further defined by
        whether support should be reported as a raw count or percentage.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    colorbar : bool, default: True
        Specify if the color bar should be present

    fontsize : int or None, default: None
        Specify the font size of the x and y labels

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Examples
    --------
    >>> from yellowbrick.classifier import ClassificationReport
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = ClassificationReport(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.show()

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels observed while fitting.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.

    score_ : float
        An evaluation metric of the classifier on test data produced when
        ``score()`` is called. This metric is between 0 and 1 -- higher scores are
        generally better. For classifiers, this score is usually accuracy, but
        ensure you check the underlying model for more details about the score.

    scores_ : dict of dicts
        Outer dictionary composed of precision, recall, f1, and support scores with
        inner dictionaries specifiying the values for each class listed.
    """

    def __init__(self, estimator, ax=None, classes=None, cmap='YlOrRd', support=None, encoder=None, is_fitted='auto', force_model=False, colorbar=True, fontsize=None, **kwargs):
        if False:
            print('Hello World!')
        super(ClassificationReport, self).__init__(estimator, ax=ax, classes=classes, encoder=encoder, is_fitted=is_fitted, force_model=force_model, **kwargs)
        self.colorbar = colorbar
        self.support = support
        self.cmap = color_sequence(cmap)
        self.cmap.set_over(color=CMAP_OVERCOLOR)
        self.cmap.set_under(color=CMAP_UNDERCOLOR)
        self._displayed_scores = [key for key in SCORES_KEYS]
        self.fontsize = fontsize
        if support not in {None, True, False, 'percent', 'count'}:
            raise YellowbrickValueError("'{}' is an invalid argument for support, use None, True, False, 'percent', or 'count'".format(support))
        if not support:
            self._displayed_scores.remove('support')

    def score(self, X, y):
        if False:
            print('Hello World!')
        '\n        Generates the Scikit-Learn classification report.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target or class values\n\n        Returns\n        -------\n\n        score_ : float\n            Global accuracy score\n        '
        super(ClassificationReport, self).score(X, y)
        labels = range(len(self.classes_))
        y_pred = self.predict(X)
        scores = precision_recall_fscore_support(y, y_pred, labels=labels)
        self.support_score_ = scores[-1]
        scores = list(scores)
        scores[-1] = scores[-1] / scores[-1].sum()
        scores = map(lambda s: dict(zip(self.classes_, s)), scores)
        self.scores_ = dict(zip(SCORES_KEYS, scores))
        if not self.support:
            self.scores_.pop('support')
        self.draw()
        return self.score_

    def draw(self):
        if False:
            while True:
                i = 10
        '\n        Renders the classification report across each axis.\n        '
        cr_display = np.zeros((len(self.classes_), len(self._displayed_scores)))
        for (idx, cls) in enumerate(self.classes_):
            for (jdx, metric) in enumerate(self._displayed_scores):
                cr_display[idx, jdx] = self.scores_[metric][cls]
        (X, Y) = (np.arange(len(self.classes_) + 1), np.arange(len(self._displayed_scores) + 1))
        self.ax.set_ylim(bottom=0, top=cr_display.shape[0])
        self.ax.set_xlim(left=0, right=cr_display.shape[1])
        labels = self._labels()
        if labels is None:
            labels = self.classes_
        xticklabels = self._displayed_scores
        yticklabels = labels[::-1]
        yticks = np.arange(len(labels)) + 0.5
        xticks = np.arange(len(self._displayed_scores)) + 0.5
        self.ax.set(yticks=yticks, xticks=xticks)
        self.ax.set_xticklabels(xticklabels, rotation=45, fontsize=self.fontsize)
        self.ax.set_yticklabels(yticklabels, fontsize=self.fontsize)
        for x in X[:-1]:
            for y in Y[:-1]:
                value = cr_display[x, y]
                svalue = '{:0.3f}'.format(value)
                if y == 3:
                    if self.support != PERCENT:
                        svalue = self.support_score_[x]
                base_color = self.cmap(value)
                text_color = find_text_color(base_color)
                (cx, cy) = (x + 0.5, y + 0.5)
                self.ax.text(cy, cx, svalue, va='center', ha='center', color=text_color)
        g = self.ax.pcolormesh(Y, X, cr_display, vmin=0, vmax=1, cmap=self.cmap, edgecolor='w')
        if self.colorbar:
            plt.colorbar(g, ax=self.ax)
        else:
            pass
        return self.ax

    def finalize(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Adds a title and sets the axis labels correctly. Also calls tight layout\n        to ensure that no parts of the figure are cut off in the final visualization.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        self.set_title('{} Classification Report'.format(self.name))
        self.ax.set_xticks(np.arange(len(self._displayed_scores)) + 0.5)
        self.ax.set_yticks(np.arange(len(self.classes_)) + 0.5)
        self.ax.set_xticklabels(self._displayed_scores, rotation=45)
        self.ax.set_yticklabels(self.classes_)
        self.fig.tight_layout()

def classification_report(estimator, X_train, y_train, X_test=None, y_test=None, ax=None, classes=None, cmap='YlOrRd', support=None, encoder=None, is_fitted='auto', force_model=False, show=True, colorbar=True, fontsize=None, **kwargs):
    if False:
        while True:
            i = 10
    "Classification Report\n\n    Displays precision, recall, F1, and support scores for the model.\n    Integrates numerical scores as well as color-coded heatmap.\n\n    Parameters\n    ----------\n    estimator : estimator\n        A scikit-learn estimator that should be a classifier. If the model is\n        not a classifier, an exception is raised. If the internal model is not\n        fitted, it is fit when the visualizer is fitted, unless otherwise specified\n        by ``is_fitted``.\n\n    X_train : ndarray or DataFrame of shape n x m\n        A feature array of n instances with m features the model is trained on.\n        Used to fit the visualizer and also to score the visualizer if test splits are\n        not directly specified.\n\n    y_train : ndarray or Series of length n\n        An array or series of target or class values. Used to fit the visualizer and\n        also to score the visualizer if test splits are not specified.\n\n    X_test : ndarray or DataFrame of shape n x m, default: None\n        An optional feature array of n instances with m features that the model\n        is scored on if specified, using X_train as the training data.\n\n    y_test : ndarray or Series of length n, default: None\n        An optional array or series of target or class values that serve as actual\n        labels for X_test for scoring purposes.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If not specified the current axes will be\n        used (or generated if required).\n\n    classes : list of str, defult: None\n        The class labels to use for the legend ordered by the index of the sorted\n        classes discovered in the ``fit()`` method. Specifying classes in this\n        manner is used to change the class names to a more specific format or\n        to label encoded integer classes. Some visualizers may also use this\n        field to filter the visualization for specific classes. For more advanced\n        usage specify an encoder rather than class labels.\n\n    cmap : string, default: ``'YlOrRd'``\n        Specify a colormap to define the heatmap of the predicted class\n        against the actual class in the classification report.\n\n    support: {True, False, None, 'percent', 'count'}, default: None\n        Specify if support will be displayed. It can be further defined by\n        whether support should be reported as a raw count or percentage.\n\n    encoder : dict or LabelEncoder, default: None\n        A mapping of classes to human readable labels. Often there is a mismatch\n        between desired class labels and those contained in the target variable\n        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch\n        ensuring that classes are labeled correctly in the visualization.\n\n    is_fitted : bool or str, default='auto'\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If 'auto' (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    force_model : bool, default: False\n        Do not check to ensure that the underlying estimator is a classifier. This\n        will prevent an exception when the visualizer is initialized but may result\n        in unexpected or unintended behavior.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n    \n    colorbar : bool, default: True\n        Specify if the color bar should be present\n\n    fontsize : int or None, default: None\n        Specify the font size of the x and y labels\n\n    kwargs : dict\n        Keyword arguments passed to the visualizer base classes.\n\n    Returns\n    -------\n    viz : ClassificationReport\n        Returns the fitted, finalized visualizer\n    "
    visualizer = ClassificationReport(estimator=estimator, ax=ax, classes=classes, cmap=cmap, support=support, encoder=encoder, is_fitted=is_fitted, force_model=force_model, colorbar=colorbar, fontsize=fontsize, **kwargs)
    visualizer.fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        visualizer.score(X_test, y_test)
    elif X_test is not None or y_test is not None:
        raise YellowbrickValueError('both X_test and y_test are required if one is specified')
    else:
        visualizer.score(X_train, y_train)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer