import numpy as np
from ...base import is_regressor
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_matplotlib_support
from ...utils._response import _get_response_values
from ...utils.validation import _is_arraylike_not_scalar, _num_features, check_is_fitted

def _check_boundary_response_method(estimator, response_method, class_of_interest):
    if False:
        while True:
            i = 10
    "Validate the response methods to be used with the fitted estimator.\n\n    Parameters\n    ----------\n    estimator : object\n        Fitted estimator to check.\n\n    response_method : {'auto', 'predict_proba', 'decision_function', 'predict'}\n        Specifies whether to use :term:`predict_proba`,\n        :term:`decision_function`, :term:`predict` as the target response.\n        If set to 'auto', the response method is tried in the following order:\n        :term:`decision_function`, :term:`predict_proba`, :term:`predict`.\n\n    class_of_interest : int, float, bool, str or None\n        The class considered when plotting the decision. If the label is specified, it\n        is then possible to plot the decision boundary in multiclass settings.\n\n        .. versionadded:: 1.4\n\n    Returns\n    -------\n    prediction_method : list of str or str\n        The name or list of names of the response methods to use.\n    "
    has_classes = hasattr(estimator, 'classes_')
    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        msg = 'Multi-label and multi-output multi-class classifiers are not supported'
        raise ValueError(msg)
    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {'auto', 'predict'} and class_of_interest is None:
            msg = "Multiclass classifiers are only supported when `response_method` is 'predict' or 'auto'. Else you must provide `class_of_interest` to plot the decision boundary of a specific class."
            raise ValueError(msg)
        prediction_method = 'predict' if response_method == 'auto' else response_method
    elif response_method == 'auto':
        if is_regressor(estimator):
            prediction_method = 'predict'
        else:
            prediction_method = ['decision_function', 'predict_proba', 'predict']
    else:
        prediction_method = response_method
    return prediction_method

class DecisionBoundaryDisplay:
    """Decisions boundary visualization.

    It is recommended to use
    :func:`~sklearn.inspection.DecisionBoundaryDisplay.from_estimator`
    to create a :class:`DecisionBoundaryDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    xx0 : ndarray of shape (grid_resolution, grid_resolution)
        First output of :func:`meshgrid <numpy.meshgrid>`.

    xx1 : ndarray of shape (grid_resolution, grid_resolution)
        Second output of :func:`meshgrid <numpy.meshgrid>`.

    response : ndarray of shape (grid_resolution, grid_resolution)
        Values of the response function.

    xlabel : str, default=None
        Default label to place on x axis.

    ylabel : str, default=None
        Default label to place on y axis.

    Attributes
    ----------
    surface_ : matplotlib `QuadContourSet` or `QuadMesh`
        If `plot_method` is 'contour' or 'contourf', `surface_` is a
        :class:`QuadContourSet <matplotlib.contour.QuadContourSet>`. If
        `plot_method` is 'pcolormesh', `surface_` is a
        :class:`QuadMesh <matplotlib.collections.QuadMesh>`.

    ax_ : matplotlib Axes
        Axes with decision boundary.

    figure_ : matplotlib Figure
        Figure containing the decision boundary.

    See Also
    --------
    DecisionBoundaryDisplay.from_estimator : Plot decision boundary given an estimator.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.inspection import DecisionBoundaryDisplay
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> iris = load_iris()
    >>> feature_1, feature_2 = np.meshgrid(
    ...     np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
    ...     np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
    ... )
    >>> grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    >>> tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
    >>> y_pred = np.reshape(tree.predict(grid), feature_1.shape)
    >>> display = DecisionBoundaryDisplay(
    ...     xx0=feature_1, xx1=feature_2, response=y_pred
    ... )
    >>> display.plot()
    <...>
    >>> display.ax_.scatter(
    ...     iris.data[:, 0], iris.data[:, 1], c=iris.target, edgecolor="black"
    ... )
    <...>
    >>> plt.show()
    """

    def __init__(self, *, xx0, xx1, response, xlabel=None, ylabel=None):
        if False:
            for i in range(10):
                print('nop')
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, plot_method='contourf', ax=None, xlabel=None, ylabel=None, **kwargs):
        if False:
            print('Hello World!')
        "Plot visualization.\n\n        Parameters\n        ----------\n        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'\n            Plotting method to call when plotting the response. Please refer\n            to the following matplotlib documentation for details:\n            :func:`contourf <matplotlib.pyplot.contourf>`,\n            :func:`contour <matplotlib.pyplot.contour>`,\n            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.\n\n        ax : Matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        xlabel : str, default=None\n            Overwrite the x-axis label.\n\n        ylabel : str, default=None\n            Overwrite the y-axis label.\n\n        **kwargs : dict\n            Additional keyword arguments to be passed to the `plot_method`.\n\n        Returns\n        -------\n        display: :class:`~sklearn.inspection.DecisionBoundaryDisplay`\n            Object that stores computed values.\n        "
        check_matplotlib_support('DecisionBoundaryDisplay.plot')
        import matplotlib.pyplot as plt
        if plot_method not in ('contourf', 'contour', 'pcolormesh'):
            raise ValueError("plot_method must be 'contourf', 'contour', or 'pcolormesh'")
        if ax is None:
            (_, ax) = plt.subplots()
        plot_func = getattr(ax, plot_method)
        self.surface_ = plot_func(self.xx0, self.xx1, self.response, **kwargs)
        if xlabel is not None or not ax.get_xlabel():
            xlabel = self.xlabel if xlabel is None else xlabel
            ax.set_xlabel(xlabel)
        if ylabel is not None or not ax.get_ylabel():
            ylabel = self.ylabel if ylabel is None else ylabel
            ax.set_ylabel(ylabel)
        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def from_estimator(cls, estimator, X, *, grid_resolution=100, eps=1.0, plot_method='contourf', response_method='auto', class_of_interest=None, xlabel=None, ylabel=None, ax=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Plot decision boundary given an estimator.\n\n        Read more in the :ref:`User Guide <visualizations>`.\n\n        Parameters\n        ----------\n        estimator : object\n            Trained estimator used to plot the decision boundary.\n\n        X : {array-like, sparse matrix, dataframe} of shape (n_samples, 2)\n            Input data that should be only 2-dimensional.\n\n        grid_resolution : int, default=100\n            Number of grid points to use for plotting decision boundary.\n            Higher values will make the plot look nicer but be slower to\n            render.\n\n        eps : float, default=1.0\n            Extends the minimum and maximum values of X for evaluating the\n            response function.\n\n        plot_method : {\'contourf\', \'contour\', \'pcolormesh\'}, default=\'contourf\'\n            Plotting method to call when plotting the response. Please refer\n            to the following matplotlib documentation for details:\n            :func:`contourf <matplotlib.pyplot.contourf>`,\n            :func:`contour <matplotlib.pyplot.contour>`,\n            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.\n\n        response_method : {\'auto\', \'predict_proba\', \'decision_function\',                 \'predict\'}, default=\'auto\'\n            Specifies whether to use :term:`predict_proba`,\n            :term:`decision_function`, :term:`predict` as the target response.\n            If set to \'auto\', the response method is tried in the following order:\n            :term:`decision_function`, :term:`predict_proba`, :term:`predict`.\n            For multiclass problems, :term:`predict` is selected when\n            `response_method="auto"`.\n\n        class_of_interest : int, float, bool or str, default=None\n            The class considered when plotting the decision. If None,\n            `estimator.classes_[1]` is considered as the positive class\n            for binary classifiers. For multiclass classifiers, passing\n            an explicit value for `class_of_interest` is mandatory.\n\n            .. versionadded:: 1.4\n\n        xlabel : str, default=None\n            The label used for the x-axis. If `None`, an attempt is made to\n            extract a label from `X` if it is a dataframe, otherwise an empty\n            string is used.\n\n        ylabel : str, default=None\n            The label used for the y-axis. If `None`, an attempt is made to\n            extract a label from `X` if it is a dataframe, otherwise an empty\n            string is used.\n\n        ax : Matplotlib axes, default=None\n            Axes object to plot on. If `None`, a new figure and axes is\n            created.\n\n        **kwargs : dict\n            Additional keyword arguments to be passed to the\n            `plot_method`.\n\n        Returns\n        -------\n        display : :class:`~sklearn.inspection.DecisionBoundaryDisplay`\n            Object that stores the result.\n\n        See Also\n        --------\n        DecisionBoundaryDisplay : Decision boundary visualization.\n        sklearn.metrics.ConfusionMatrixDisplay.from_estimator : Plot the\n            confusion matrix given an estimator, the data, and the label.\n        sklearn.metrics.ConfusionMatrixDisplay.from_predictions : Plot the\n            confusion matrix given the true and predicted labels.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import load_iris\n        >>> from sklearn.linear_model import LogisticRegression\n        >>> from sklearn.inspection import DecisionBoundaryDisplay\n        >>> iris = load_iris()\n        >>> X = iris.data[:, :2]\n        >>> classifier = LogisticRegression().fit(X, iris.target)\n        >>> disp = DecisionBoundaryDisplay.from_estimator(\n        ...     classifier, X, response_method="predict",\n        ...     xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],\n        ...     alpha=0.5,\n        ... )\n        >>> disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")\n        <...>\n        >>> plt.show()\n        '
        check_matplotlib_support(f'{cls.__name__}.from_estimator')
        check_is_fitted(estimator)
        if not grid_resolution > 1:
            raise ValueError(f'grid_resolution must be greater than 1. Got {grid_resolution} instead.')
        if not eps >= 0:
            raise ValueError(f'eps must be greater than or equal to 0. Got {eps} instead.')
        possible_plot_methods = ('contourf', 'contour', 'pcolormesh')
        if plot_method not in possible_plot_methods:
            available_methods = ', '.join(possible_plot_methods)
            raise ValueError(f'plot_method must be one of {available_methods}. Got {plot_method} instead.')
        num_features = _num_features(X)
        if num_features != 2:
            raise ValueError(f'n_features must be equal to 2. Got {num_features} instead.')
        (x0, x1) = (_safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1))
        (x0_min, x0_max) = (x0.min() - eps, x0.max() + eps)
        (x1_min, x1_max) = (x1.min() - eps, x1.max() + eps)
        (xx0, xx1) = np.meshgrid(np.linspace(x0_min, x0_max, grid_resolution), np.linspace(x1_min, x1_max, grid_resolution))
        if hasattr(X, 'iloc'):
            X_grid = X.iloc[[], :].copy()
            X_grid.iloc[:, 0] = xx0.ravel()
            X_grid.iloc[:, 1] = xx1.ravel()
        else:
            X_grid = np.c_[xx0.ravel(), xx1.ravel()]
        prediction_method = _check_boundary_response_method(estimator, response_method, class_of_interest)
        try:
            (response, _, response_method_used) = _get_response_values(estimator, X_grid, response_method=prediction_method, pos_label=class_of_interest, return_response_method_used=True)
        except ValueError as exc:
            if 'is not a valid label' in str(exc):
                raise ValueError(f'class_of_interest={class_of_interest} is not a valid label: It should be one of {estimator.classes_}') from exc
            raise
        if response_method_used == 'predict' and hasattr(estimator, 'classes_'):
            encoder = LabelEncoder()
            encoder.classes_ = estimator.classes_
            response = encoder.transform(response)
        if response.ndim != 1:
            if is_regressor(estimator):
                raise ValueError('Multi-output regressors are not supported')
            col_idx = np.flatnonzero(estimator.classes_ == class_of_interest)[0]
            response = response[:, col_idx]
        if xlabel is None:
            xlabel = X.columns[0] if hasattr(X, 'columns') else ''
        if ylabel is None:
            ylabel = X.columns[1] if hasattr(X, 'columns') else ''
        display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=response.reshape(xx0.shape), xlabel=xlabel, ylabel=ylabel)
        return display.plot(ax=ax, plot_method=plot_method, **kwargs)