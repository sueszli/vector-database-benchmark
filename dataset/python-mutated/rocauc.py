"""
Implements visual ROC/AUC curves for classification evaluation.
"""
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
from yellowbrick.exceptions import ModelError
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer
MACRO = 'macro'
MICRO = 'micro'
BINARY = 'binary'
MULTICLASS = 'multiclass'

class ROCAUC(ClassificationScoreVisualizer):
    """
    Receiver Operating Characteristic (ROC) curves are a measure of a
    classifier's predictive quality that compares and visualizes the tradeoff
    between the models' sensitivity and specificity. The ROC curve displays
    the true positive rate on the Y axis and the false positive rate on the
    X axis on both a global average and per-class basis. The ideal point is
    therefore the top-left corner of the plot: false positives are zero and
    true positives are one.

    This leads to another metric, area under the curve (AUC), a computation
    of the relationship between false positives and true positives. The higher
    the AUC, the better the model generally is. However, it is also important
    to inspect the "steepness" of the curve, as this describes the
    maximization of the true positive rate while minimizing the false positive
    rate. Generalizing "steepness" usually leads to discussions about
    convexity, which we do not get into here.

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

    micro : bool, default: True
        Plot the micro-averages ROC curve, computed from the sum of all true
        positives and false positives across all classes. Micro is not defined
        for binary classification problems with estimators with only a
        decision_function method.

    macro : bool, default: True
        Plot the macro-averages ROC curve, which simply takes the average of
        curves across all classes. Macro is not defined for binary
        classification problems with estimators with only a decision_function
        method.

    per_class : bool, default: True
        Plot the ROC curves for each individual class. This should be set
        to false if only the macro or micro average curves are required. For true
        binary classifiers, setting per_class=False will plot the positive class
        ROC curve, and per_class=True will use ``1-P(1)`` to compute the curve of
        the negative class if only a decision_function method exists on the estimator.

    binary : bool, default: False
        This argument quickly resets the visualizer for true binary classification
        by updating the micro, macro, and per_class arguments to False (do not use
        in conjunction with those other arguments). Note that this is not a true
        hyperparameter to the visualizer, it just collects other parameters into
        a single, simpler argument.

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

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

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

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
        if micro or macro is specified this returns an F1 score.

    target_type_ : string
        Specifies if the detected classification target was binary or multiclass.

    Notes
    -----
    ROC curves are typically used in binary classification, and in fact the
    Scikit-Learn ``roc_curve`` metric is only able to perform metrics for
    binary classifiers. As a result it is necessary to binarize the output or
    to use one-vs-rest or one-vs-all strategies of classification. The
    visualizer does its best to handle multiple situations, but exceptions can
    arise from unexpected models or outputs.

    Another important point is the relationship of class labels specified on
    initialization to those drawn on the curves. The classes are not used to
    constrain ordering or filter curves; the ROC computation happens on the
    unique values specified in the target vector to the ``score`` method. To
    ensure the best quality visualization, do not use a LabelEncoder for this
    and do not pass in class labels.

    .. seealso::
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    .. todo:: Allow the class list to filter the curves on the visualization.

    Examples
    --------
    >>> from yellowbrick.classifier import ROCAUC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("occupancy")
    >>> features = ["temp", "relative humidity", "light", "C02", "humidity"]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> oz = ROCAUC(LogisticRegression())
    >>> oz.fit(X_train, y_train)
    >>> oz.score(X_test, y_test)
    >>> oz.show()
    """

    def __init__(self, estimator, ax=None, micro=True, macro=True, per_class=True, binary=False, classes=None, encoder=None, is_fitted='auto', force_model=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ROCAUC, self).__init__(estimator, ax=ax, classes=classes, encoder=encoder, is_fitted=is_fitted, force_model=force_model, **kwargs)
        self.binary = binary
        if self.binary:
            self.micro = False
            self.macro = False
            self.per_class = False
        else:
            self.micro = micro
            self.macro = macro
            self.per_class = per_class

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        '\n        Fit the classification model.\n        '
        ttype = type_of_target(y)
        if ttype.startswith(MULTICLASS):
            self.target_type_ = MULTICLASS
        elif ttype.startswith(BINARY):
            self.target_type_ = BINARY
        else:
            raise YellowbrickValueError("{} does not support target type '{}', please provide a binary or multiclass single-output target".format(self.__class__.__name__, ttype))
        return super(ROCAUC, self).fit(X, y)

    def score(self, X, y=None):
        if False:
            return 10
        '\n        Generates the predicted target values using the Scikit-Learn\n        estimator.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target or class values\n\n        Returns\n        -------\n        score_ : float\n            Global accuracy unless micro or macro scores are requested.\n        '
        super(ROCAUC, self).score(X, y)
        y_pred = self._get_y_scores(X)
        if self.target_type_ == BINARY:
            if (self.micro or self.macro) and (not self.per_class):
                raise ModelError('no curves will be drawn; ', 'set per_class=True or micro=False and macro=False.')
            if (self.micro or self.macro) and len(y_pred.shape) == 1:
                raise ModelError('no curves will be drawn; set binary=True.')
        if self.target_type_ == MULTICLASS:
            if not self.micro and (not self.macro) and (not self.per_class):
                raise YellowbrickValueError('no curves will be drawn; specify micro, macro, or per_class')
        classes = np.unique(y)
        n_classes = len(classes)
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        if self.target_type_ is BINARY and (not self.per_class):
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                (self.fpr[BINARY], self.tpr[BINARY], _) = roc_curve(y, y_pred[:, 1])
            else:
                (self.fpr[BINARY], self.tpr[BINARY], _) = roc_curve(y, y_pred)
            self.roc_auc[BINARY] = auc(self.fpr[BINARY], self.tpr[BINARY])
        elif self.target_type_ is BINARY and self.per_class:
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                (self.fpr[1], self.tpr[1], _) = roc_curve(y, y_pred[:, 1])
            else:
                (self.fpr[1], self.tpr[1], _) = roc_curve(y, y_pred)
            self.roc_auc[1] = auc(self.fpr[1], self.tpr[1])
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                (self.fpr[0], self.tpr[0], _) = roc_curve(1 - y, y_pred[:, 0])
            else:
                (self.fpr[0], self.tpr[0], _) = roc_curve(1 - y, -y_pred)
            self.roc_auc[0] = auc(self.fpr[0], self.tpr[0])
        else:
            for (i, c) in enumerate(classes):
                (self.fpr[i], self.tpr[i], _) = roc_curve(y, y_pred[:, i], pos_label=c)
                self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        if self.micro:
            self._score_micro_average(y, y_pred, classes, n_classes)
        if self.macro:
            self._score_macro_average(n_classes)
        self.draw()
        if self.micro:
            self.score_ = self.roc_auc[MICRO]
        if self.macro:
            self.score_ = self.roc_auc[MACRO]
        return self.score_

    def draw(self):
        if False:
            return 10
        '\n        Renders ROC-AUC plot.\n        Called internally by score, possibly more than once\n\n        Returns\n        -------\n        ax : the axis with the plotted figure\n        '
        colors = self.class_colors_[0:len(self.classes_)]
        n_classes = len(colors)
        if self.target_type_ == BINARY and (not self.per_class):
            self.ax.plot(self.fpr[BINARY], self.tpr[BINARY], label='ROC for binary decision, AUC = {:0.2f}'.format(self.roc_auc[BINARY]))
        if self.per_class:
            for (i, color) in zip(range(n_classes), colors):
                self.ax.plot(self.fpr[i], self.tpr[i], color=color, label='ROC of class {}, AUC = {:0.2f}'.format(self.classes_[i], self.roc_auc[i]))
        if self.micro:
            self.ax.plot(self.fpr[MICRO], self.tpr[MICRO], linestyle='--', color=self.class_colors_[len(self.classes_) - 1], label='micro-average ROC curve, AUC = {:0.2f}'.format(self.roc_auc['micro']))
        if self.macro:
            self.ax.plot(self.fpr[MACRO], self.tpr[MACRO], linestyle='--', color=self.class_colors_[len(self.classes_) - 1], label='macro-average ROC curve, AUC = {:0.2f}'.format(self.roc_auc['macro']))
        self.ax.plot([0, 1], [0, 1], linestyle=':', c=LINE_COLOR)
        return self.ax

    def finalize(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Sets a title and axis labels of the figures and ensures the axis limits\n        are scaled between the valid ROCAUC score values.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        self.set_title('ROC Curves for {}'.format(self.name))
        self.ax.legend(loc='lower right', frameon=True)
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])
        self.ax.set_ylabel('True Positive Rate')
        self.ax.set_xlabel('False Positive Rate')

    def _get_y_scores(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        The ``roc_curve`` metric requires target scores that can either be the\n        probability estimates of the positive class, confidence values or non-\n        thresholded measure of decisions (as returned by "decision_function").\n\n        This method computes the scores by resolving the estimator methods\n        that retreive these values.\n\n        .. todo:: implement confidence values metric.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features -- generally the test data\n            that is associated with y_true values.\n        '
        attrs = ('predict_proba', 'decision_function')
        for attr in attrs:
            try:
                method = getattr(self.estimator, attr, None)
                if method:
                    return method(X)
            except AttributeError:
                continue
        raise ModelError('ROCAUC requires estimators with predict_proba or decision_function methods.')

    def _score_micro_average(self, y, y_pred, classes, n_classes):
        if False:
            return 10
        '\n        Compute the micro average scores for the ROCAUC curves.\n        '
        y = label_binarize(y, classes=classes)
        if n_classes == 2:
            y = np.hstack((1 - y, y))
        (self.fpr[MICRO], self.tpr[MICRO], _) = roc_curve(y.ravel(), y_pred.ravel())
        self.roc_auc[MICRO] = auc(self.fpr[MICRO], self.tpr[MICRO])

    def _score_macro_average(self, n_classes):
        if False:
            while True:
                i = 10
        '\n        Compute the macro average scores for the ROCAUC curves.\n        '
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(n_classes)]))
        avg_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            avg_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])
        avg_tpr /= n_classes
        self.fpr[MACRO] = all_fpr
        self.tpr[MACRO] = avg_tpr
        self.roc_auc[MACRO] = auc(self.fpr[MACRO], self.tpr[MACRO])

def roc_auc(estimator, X_train, y_train, X_test=None, y_test=None, ax=None, micro=True, macro=True, per_class=True, binary=False, classes=None, encoder=None, is_fitted='auto', force_model=False, show=True, **kwargs):
    if False:
        while True:
            i = 10
    'ROCAUC\n\n    Receiver Operating Characteristic (ROC) curves are a measure of a\n    classifier\'s predictive quality that compares and visualizes the tradeoff\n    between the models\' sensitivity and specificity. The ROC curve displays\n    the true positive rate on the Y axis and the false positive rate on the\n    X axis on both a global average and per-class basis. The ideal point is\n    therefore the top-left corner of the plot: false positives are zero and\n    true positives are one.\n\n    This leads to another metric, area under the curve  (AUC), a computation\n    of the relationship between false positives and true positives. The higher\n    the AUC, the better the model generally is. However, it is also important\n    to inspect the "steepness" of the curve, as this describes the\n    maximization of the true positive rate while minimizing the false positive\n    rate. Generalizing "steepness" usually leads to discussions about\n    convexity, which we do not get into here.\n\n    Parameters\n    ----------\n    estimator : estimator\n        A scikit-learn estimator that should be a classifier. If the model is\n        not a classifier, an exception is raised. If the internal model is not\n        fitted, it is fit when the visualizer is fitted, unless otherwise specified\n        by ``is_fitted``.\n\n    X_train : array-like, 2D\n        The table of instance data or independent variables that describe the outcome of\n        the dependent variable, y. Used to fit the visualizer and also to score the\n        visualizer if test splits are not specified.\n\n    y_train : array-like, 2D\n        The vector of target data or the dependent variable predicted by X. Used to fit\n        the visualizer and also to score the visualizer if test splits not specified.\n\n    X_test: array-like, 2D, default: None\n        The table of instance data or independent variables that describe the outcome of\n        the dependent variable, y. Used to score the visualizer if specified.\n\n    y_test: array-like, 1D, default: None\n        The vector of target data or the dependent variable predicted by X.\n        Used to score the visualizer if specified.\n\n    ax : matplotlib Axes, default: None\n        The axes to plot the figure on. If not specified the current axes will be\n        used (or generated if required).\n\n    test_size : float, default=0.2\n        The percentage of the data to reserve as test data.\n\n    random_state : int or None, default=None\n        The value to seed the random number generator for shuffling data.\n\n    micro : bool, default: True\n        Plot the micro-averages ROC curve, computed from the sum of all true\n        positives and false positives across all classes. Micro is not defined\n        for binary classification problems with estimators with only a\n        decision_function method.\n\n    macro : bool, default: True\n        Plot the macro-averages ROC curve, which simply takes the average of\n        curves across all classes. Macro is not defined for binary\n        classification problems with estimators with only a decision_function\n        method.\n\n    per_class : bool, default: True\n        Plot the ROC curves for each individual class. This should be set\n        to false if only the macro or micro average curves are required. For true\n        binary classifiers, setting per_class=False will plot the positive class\n        ROC curve, and per_class=True will use ``1-P(1)`` to compute the curve of\n        the negative class if only a decision_function method exists on the estimator.\n\n    binary : bool, default: False\n        This argument quickly resets the visualizer for true binary classification\n        by updating the micro, macro, and per_class arguments to False (do not use\n        in conjunction with those other arguments). Note that this is not a true\n        hyperparameter to the visualizer, it just collects other parameters into\n        a single, simpler argument.\n\n    classes : list of str, defult: None\n        The class labels to use for the legend ordered by the index of the sorted\n        classes discovered in the ``fit()`` method. Specifying classes in this\n        manner is used to change the class names to a more specific format or\n        to label encoded integer classes. Some visualizers may also use this\n        field to filter the visualization for specific classes. For more advanced\n        usage specify an encoder rather than class labels.\n\n    encoder : dict or LabelEncoder, default: None\n        A mapping of classes to human readable labels. Often there is a mismatch\n        between desired class labels and those contained in the target variable\n        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch\n        ensuring that classes are labeled correctly in the visualization.\n\n    is_fitted : bool or str, default="auto"\n        Specify if the wrapped estimator is already fitted. If False, the estimator\n        will be fit when the visualizer is fit, otherwise, the estimator will not be\n        modified. If "auto" (default), a helper method will check if the estimator\n        is fitted before fitting it again.\n\n    force_model : bool, default: False\n        Do not check to ensure that the underlying estimator is a classifier. This\n        will prevent an exception when the visualizer is initialized but may result\n        in unexpected or unintended behavior.\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments passed to the visualizer base classes.\n\n    Notes\n    -----\n    ROC curves are typically used in binary classification, and in fact the\n    Scikit-Learn ``roc_curve`` metric is only able to perform metrics for\n    binary classifiers. As a result it is necessary to binarize the output or\n    to use one-vs-rest or one-vs-all strategies of classification. The\n    visualizer does its best to handle multiple situations, but exceptions can\n    arise from unexpected models or outputs.\n\n    Another important point is the relationship of class labels specified on\n    initialization to those drawn on the curves. The classes are not used to\n    constrain ordering or filter curves; the ROC computation happens on the\n    unique values specified in the target vector to the ``score`` method. To\n    ensure the best quality visualization, do not use a LabelEncoder for this\n    and do not pass in class labels.\n\n    .. seealso:: https://bit.ly/2IORWO2\n    .. todo:: Allow the class list to filter the curves on the visualization.\n\n    Examples\n    --------\n    >>> from yellowbrick.classifier import ROCAUC\n    >>> from sklearn.linear_model import LogisticRegression\n    >>> data = load_data("occupancy")\n    >>> features = ["temp", "relative humidity", "light", "C02", "humidity"]\n    >>> X = data[features].values\n    >>> y = data.occupancy.values\n    >>> roc_auc(LogisticRegression(), X, y)\n\n    Returns\n    -------\n    viz : ROCAUC\n        Returns the fitted, finalized visualizer object\n    '
    visualizer = ROCAUC(estimator=estimator, ax=ax, micro=micro, macro=macro, per_class=per_class, binary=binary, classes=classes, encoder=encoder, is_fitted=is_fitted, force_model=force_model, **kwargs)
    visualizer.fit(X_train, y_train, **kwargs)
    if X_test is not None and y_test is not None:
        visualizer.score(X_test, y_test)
    else:
        visualizer.score(X_train, y_train)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer