"""
Class balance visualizer for showing per-class support.
"""
import numpy as np
from yellowbrick.style.colors import resolve_colors
from yellowbrick.target.base import TargetVisualizer
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.utils.multiclass import unique_labels, type_of_target
BALANCE = 'balance'
COMPARE = 'compare'

class ClassBalance(TargetVisualizer):
    """
    One of the biggest challenges for classification models is an imbalance of
    classes in the training data. The ClassBalance visualizer shows the
    relationship of the support for each class in both the training and test
    data by displaying how frequently each class occurs as a bar graph.

    The ClassBalance visualizer can be displayed in two modes:

    1. Balance mode: show the frequency of each class in the dataset.
    2. Compare mode: show the relationship of support in train and test data.

    These modes are determined by what is passed to the ``fit()`` method.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    labels: list, optional
        A list of class names for the x-axis if the target is already encoded.
        Ensure that the labels are ordered lexicographically with respect to
        the values in the target. A common use case is to pass
        ``LabelEncoder.classes_`` as this parameter. If not specified, the labels
        in the data will be used.

    colors: list of strings
        Specify colors for the barchart (will override colormap if both are provided).

    colormap : string or matplotlib cmap
        Specify a colormap to color the classes.

    kwargs: dict, optional
        Keyword arguments passed to the super class. Here, used
        to colorize the bars in the histogram.

    Attributes
    ----------
    classes_ : array-like
        The actual unique classes discovered in the target.

    support_ : array of shape (n_classes,) or (2, n_classes)
        A table representing the support of each class in the target. It is a
        vector when in balance mode, or a table with two rows in compare mode.

    Examples
    --------
    To simply observe the balance of classes in the target:

    >>> viz = ClassBalance().fit(y)
    >>> viz.show()

    To compare the relationship between training and test data:

    >>> _, _, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> viz = ClassBalance()
    >>> viz.fit(y_train, y_test)
    >>> viz.show()
    """

    def __init__(self, ax=None, labels=None, colors=None, colormap=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.labels = labels
        self.colors = colors
        self.colormap = colormap
        super(ClassBalance, self).__init__(ax, **kwargs)

    def fit(self, y_train, y_test=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the visualizer to the the target variables, which must be 1D\n        vectors containing discrete (classification) data. Fit has two modes:\n\n        1. Balance mode: if only y_train is specified\n        2. Compare mode: if both train and test are specified\n\n        In balance mode, the bar chart is displayed with each class as its own\n        color. In compare mode, a side-by-side bar chart is displayed colored\n        by train or test respectively.\n\n        Parameters\n        ----------\n        y_train : array-like\n            Array or list of shape (n,) that contains discrete data.\n\n        y_test : array-like, optional\n            Array or list of shape (m,) that contains discrete data. If\n            specified, the bar chart will be drawn in compare mode.\n        '
        if y_train.ndim == 2:
            raise YellowbrickValueError('fit has changed to only require a 1D array, y since version 0.9; please see the docs for more info')
        self._validate_target(y_train)
        self._validate_target(y_test)
        targets = (y_train,) if y_test is None else (y_train, y_test)
        self.classes_ = unique_labels(*targets)
        if self.labels is not None:
            if len(self.labels) != len(self.classes_):
                raise YellowbrickValueError('discovered {} classes in the data, does not match the {} labels specified.'.format(len(self.classes_), len(self.labels)))
        self._mode = BALANCE if y_test is None else COMPARE
        if self._mode == BALANCE:
            self.support_ = np.array([(y_train == idx).sum() for idx in self.classes_])
        else:
            self.support_ = np.array([[(y == idx).sum() for idx in self.classes_] for y in targets])
        self.draw()
        return self

    def draw(self):
        if False:
            return 10
        '\n        Renders the class balance chart on the specified axes from support.\n        '
        colors = resolve_colors(len(self.support_), colormap=self.colormap, colors=self.colors)
        if self._mode == BALANCE:
            self.ax.bar(np.arange(len(self.support_)), self.support_, color=colors, align='center', width=0.5)
        else:
            bar_width = 0.35
            labels = ['train', 'test']
            for (idx, support) in enumerate(self.support_):
                index = np.arange(len(self.classes_))
                if idx > 0:
                    index = index + bar_width
                self.ax.bar(index, support, bar_width, color=colors[idx], label=labels[idx])
        return self.ax

    def finalize(self, **kwargs):
        if False:
            return 10
        '\n        Finalizes the figure for drawing by setting a title, the legend, and axis\n        labels, removing the grid, and making sure the figure is correctly zoomed\n        into the bar chart.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        '
        self.set_title('Class Balance for {:,} Instances'.format(self.support_.sum()))
        labels = self.labels if self.labels is not None else self.classes_
        xticks = np.arange(len(labels))
        if self._mode == COMPARE:
            xticks = xticks + 0.35 / 2
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(labels)
        cmax = self.support_.max()
        self.ax.set_ylim(0, cmax + cmax * 0.1)
        self.ax.set_ylabel('support')
        self.ax.grid(False, axis='x')
        if self._mode == COMPARE:
            self.ax.legend(frameon=True)

    def _validate_target(self, y):
        if False:
            while True:
                i = 10
        '\n        Raises a value error if the target is not a classification target.\n        '
        if y is None:
            return
        y_type = type_of_target(y)
        if y_type not in ('binary', 'multiclass'):
            raise YellowbrickValueError("'{}' target type not supported, only binary and multiclass".format(y_type))

def class_balance(y_train, y_test=None, ax=None, labels=None, color=None, colormap=None, show=True, **kwargs):
    if False:
        while True:
            i = 10
    'Quick method:\n\n    One of the biggest challenges for classification models is an imbalance of\n    classes in the training data. This function vizualizes the relationship of\n    the support for each class in both the training and test data by\n    displaying how frequently each class occurs as a bar graph.\n\n    The figure can be displayed in two modes:\n\n    1. Balance mode: show the frequency of each class in the dataset.\n    2. Compare mode: show the relationship of support in train and test data.\n\n    Balance mode is the default if only y_train is specified. Compare mode\n    happens when both y_train and y_test are specified.\n\n    Parameters\n    ----------\n    y_train : array-like\n        Array or list of shape (n,) that containes discrete data.\n\n    y_test : array-like, optional\n        Array or list of shape (m,) that contains discrete data. If\n        specified, the bar chart will be drawn in compare mode.\n\n    ax : matplotlib Axes, default: None\n        The axis to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    labels: list, optional\n        A list of class names for the x-axis if the target is already encoded.\n        Ensure that the labels are ordered lexicographically with respect to\n        the values in the target. A common use case is to pass\n        ``LabelEncoder.classes_`` as this parameter. If not specified, the labels\n        in the data will be used.\n\n    colors: list of strings\n        Specify colors for the barchart (will override colormap if both are provided).\n\n    colormap : string or matplotlib cmap\n        Specify a colormap to color the classes.\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs: dict, optional\n        Keyword arguments passed to the super class. Here, used\n        to colorize the bars in the histogram.\n\n    Returns\n    -------\n    visualizer : ClassBalance\n        Returns the fitted visualizer\n    '
    visualizer = ClassBalance(ax=ax, labels=labels, color=None, colormap=None, **kwargs)
    visualizer.fit(y_train, y_test)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer