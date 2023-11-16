"""
Feature Correlation to Dependent Variable Visualizer.
"""
import numpy as np
from yellowbrick.utils import is_dataframe
from yellowbrick.target.base import TargetVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
CORRELATION_LABELS = {'pearson': 'Pearson Correlation', 'mutual_info-regression': 'Mutual Information', 'mutual_info-classification': 'Mutual Information'}
CORRELATION_METHODS = {'mutual_info-regression': mutual_info_regression, 'mutual_info-classification': mutual_info_classif}

class FeatureCorrelation(TargetVisualizer):
    """
    Displays the correlation between features and dependent variables.

    This visualizer can be used side-by-side with
    ``yellowbrick.features.JointPlotVisualizer`` that plots a feature
    against the target and shows the distribution of each via a
    histogram on each axis.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    method : str, default: 'pearson'
        The method to calculate correlation between features and target.
        Options include:

            - 'pearson', which uses ``scipy.stats.pearsonr``
            - 'mutual_info-regression', which uses ``mutual_info-regression``
              from ``sklearn.feature_selection``
            - 'mutual_info-classification', which uses ``mutual_info_classif``
              from ``sklearn.feature_selection``

    labels : list, default: None
        A list of feature names to use. If a DataFrame is passed to fit and
        features is None, feature names are selected as the column names.

    sort : boolean, default: False
        If false, the features are are not sorted in the plot; otherwise
        features are sorted in ascending order of correlation.

    feature_index : list,
        A list of feature index to include in the plot.

    feature_names : list of feature names
        A list of feature names to include in the plot.
        Must have labels or the fitted data is a DataFrame with column names.
        If feature_index is provided, feature_names will be ignored.

    color: string
        Specify color for barchart

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : np.array
        The feature labels

    scores_ : np.array
        Correlation between features and dependent variable.

    Examples
    --------

    >>> viz = FeatureCorrelation()
    >>> viz.fit(X, y)
    >>> viz.show()
    """

    def __init__(self, ax=None, method='pearson', labels=None, sort=False, feature_index=None, feature_names=None, color=None, **kwargs):
        if False:
            print('Hello World!')
        super(FeatureCorrelation, self).__init__(ax, **kwargs)
        self.correlation_labels = CORRELATION_LABELS
        self.correlation_methods = CORRELATION_METHODS
        if method not in self.correlation_labels:
            raise YellowbrickValueError('Method {} not implement; choose from {}'.format(method, ', '.join(self.correlation_labels)))
        self.sort = sort
        self.color = color
        self.method = method
        self.labels = labels
        self.feature_index = feature_index
        self.feature_names = feature_names

    def fit(self, X, y, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits the estimator to calculate feature correlation to\n        dependent variable.\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features\n\n        y : ndarray or Series of length n\n            An array or series of target or class values\n\n        kwargs : dict\n            Keyword arguments passed to the fit method of the estimator.\n\n        Returns\n        -------\n        self : visualizer\n            The fit method must always return self to support pipelines.\n        '
        self._create_labels_for_features(X)
        self._select_features_to_plot(X)
        if self.method == 'pearson':
            self.scores_ = np.array([pearsonr(x, y, **kwargs)[0] for x in np.asarray(X).T])
        else:
            self.scores_ = np.array(self.correlation_methods[self.method](X, y, **kwargs))
        if self.feature_index:
            self.scores_ = self.scores_[self.feature_index]
            self.features_ = self.features_[self.feature_index]
        if self.sort:
            sort_idx = np.argsort(self.scores_)
            self.scores_ = self.scores_[sort_idx]
            self.features_ = self.features_[sort_idx]
        self.draw()
        return self

    def draw(self):
        if False:
            while True:
                i = 10
        '\n        Draws the feature correlation to dependent variable, called from fit.\n        '
        pos = np.arange(self.scores_.shape[0]) + 0.5
        self.ax.barh(pos, self.scores_, color=self.color)
        self.ax.set_yticks(pos)
        self.ax.set_yticklabels(self.features_)
        return self.ax

    def finalize(self):
        if False:
            i = 10
            return i + 15
        '\n        Finalize the drawing setting labels and title.\n        '
        self.set_title('Features correlation with dependent variable')
        self.ax.set_xlabel(self.correlation_labels[self.method])
        self.ax.grid(False, axis='y')

    def _create_labels_for_features(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Create labels for the features\n\n        NOTE: this code is duplicated from MultiFeatureVisualizer\n        '
        if self.labels is None:
            if is_dataframe(X):
                self.features_ = np.array(X.columns)
            else:
                (_, ncols) = X.shape
                self.features_ = np.arange(0, ncols)
        else:
            self.features_ = np.array(self.labels)

    def _select_features_to_plot(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Select features to plot.\n\n        feature_index is always used as the filter and\n        if filter_names is supplied, a new feature_index\n        is computed from those names.\n        '
        if self.feature_index:
            if self.feature_names:
                raise YellowbrickWarning('Both feature_index and feature_names are specified. feature_names is ignored')
            if min(self.feature_index) < 0 or max(self.feature_index) >= X.shape[1]:
                raise YellowbrickValueError('Feature index is out of range')
        elif self.feature_names:
            self.feature_index = []
            features_list = self.features_.tolist()
            for feature_name in self.feature_names:
                try:
                    self.feature_index.append(features_list.index(feature_name))
                except ValueError:
                    raise YellowbrickValueError('{} not in labels'.format(feature_name))

def feature_correlation(X, y, ax=None, method='pearson', labels=None, sort=False, feature_index=None, feature_names=None, color=None, show=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Displays the correlation between features and dependent variables.\n\n    This visualizer can be used side-by-side with\n    yellowbrick.features.JointPlotVisualizer that plots a feature\n    against the target and shows the distribution of each via a\n    histogram on each axis.\n\n    Parameters\n    ----------\n    X : ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features\n\n    y : ndarray or Series of length n\n        An array or series of target or class values\n\n    ax : matplotlib Axes, default: None\n        The axis to plot the figure on. If None is passed in the current axes\n        will be used (or generated if required).\n\n    method : str, default: 'pearson'\n        The method to calculate correlation between features and target.\n        Options include:\n\n            - 'pearson', which uses ``scipy.stats.pearsonr``\n            - 'mutual_info-regression', which uses ``mutual_info-regression``\n              from ``sklearn.feature_selection``\n            - 'mutual_info-classification', which uses ``mutual_info_classif``\n              from ``sklearn.feature_selection``\n\n    labels : list, default: None\n        A list of feature names to use. If a DataFrame is passed to fit and\n        features is None, feature names are selected as the column names.\n\n    sort : boolean, default: False\n        If false, the features are are not sorted in the plot; otherwise\n        features are sorted in ascending order of correlation.\n\n    feature_index : list,\n        A list of feature index to include in the plot.\n\n    feature_names : list of feature names\n        A list of feature names to include in the plot.\n        Must have labels or the fitted data is a DataFrame with column names.\n        If feature_index is provided, feature_names will be ignored.\n\n    color: string\n        Specify color for barchart\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot\n        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply\n        calls ``finalize()``\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    visualizer : FeatureCorrelation\n        Returns the fitted visualizer.\n    "
    visualizer = FeatureCorrelation(ax=ax, method=method, labels=labels, sort=sort, color=color, feature_index=feature_index, feature_names=feature_names, **kwargs)
    visualizer.fit(X, y, **kwargs)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer