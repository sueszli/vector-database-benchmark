import numbers
from itertools import chain
from math import ceil
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ...base import is_regressor
from ...utils import Bunch, _safe_indexing, check_array, check_matplotlib_support, check_random_state
from ...utils._encode import _unique
from ...utils.parallel import Parallel, delayed
from .. import partial_dependence
from .._pd_utils import _check_feature_names, _get_feature_index

class PartialDependenceDisplay:
    """Partial Dependence Plot (PDP).

    This can also display individual partial dependencies which are often
    referred to as: Individual Condition Expectation (ICE).

    It is recommended to use
    :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` to create a
    :class:`~sklearn.inspection.PartialDependenceDisplay`. All parameters are
    stored as attributes.

    Read more in
    :ref:`sphx_glr_auto_examples_miscellaneous_plot_partial_dependence_visualization_api.py`
    and the :ref:`User Guide <partial_dependence>`.

        .. versionadded:: 0.22

    Parameters
    ----------
    pd_results : list of Bunch
        Results of :func:`~sklearn.inspection.partial_dependence` for
        ``features``.

    features : list of (int,) or list of (int, int)
        Indices of features for a given plot. A tuple of one integer will plot
        a partial dependence curve of one feature. A tuple of two integers will
        plot a two-way partial dependence curve as a contour plot.

    feature_names : list of str
        Feature names corresponding to the indices in ``features``.

    target_idx : int

        - In a multiclass setting, specifies the class for which the PDPs
          should be computed. Note that for binary classification, the
          positive class (index 1) is always used.
        - In a multioutput setting, specifies the task for which the PDPs
          should be computed.

        Ignored in binary classification or classical regression settings.

    deciles : dict
        Deciles for feature indices in ``features``.

    kind : {'average', 'individual', 'both'} or list of such str,             default='average'
        Whether to plot the partial dependence averaged across all the samples
        in the dataset or one line per sample or both.

        - ``kind='average'`` results in the traditional PD plot;
        - ``kind='individual'`` results in the ICE plot;
        - ``kind='both'`` results in plotting both the ICE and PD on the same
          plot.

        A list of such strings can be provided to specify `kind` on a per-plot
        basis. The length of the list should be the same as the number of
        interaction requested in `features`.

        .. note::
           ICE ('individual' or 'both') is not a valid option for 2-ways
           interactions plot. As a result, an error will be raised.
           2-ways interaction plots should always be configured to
           use the 'average' kind instead.

        .. note::
           The fast ``method='recursion'`` option is only available for
           `kind='average'` and `sample_weights=None`. Computing individual
           dependencies and doing weighted averages requires using the slower
           `method='brute'`.

        .. versionadded:: 0.24
           Add `kind` parameter with `'average'`, `'individual'`, and `'both'`
           options.

        .. versionadded:: 1.1
           Add the possibility to pass a list of string specifying `kind`
           for each plot.

    subsample : float, int or None, default=1000
        Sampling for ICE curves when `kind` is 'individual' or 'both'.
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to be used to plot ICE curves. If int, represents the
        maximum absolute number of samples to use.

        Note that the full dataset is still used to calculate partial
        dependence when `kind='both'`.

        .. versionadded:: 0.24

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the selected samples when subsamples is not
        `None`. See :term:`Glossary <random_state>` for details.

        .. versionadded:: 0.24

    is_categorical : list of (bool,) or list of (bool, bool), default=None
        Whether each target feature in `features` is categorical or not.
        The list should be same size as `features`. If `None`, all features
        are assumed to be continuous.

        .. versionadded:: 1.2

    Attributes
    ----------
    bounding_ax_ : matplotlib Axes or None
        If `ax` is an axes or None, the `bounding_ax_` is the axes where the
        grid of partial dependence plots are drawn. If `ax` is a list of axes
        or a numpy array of axes, `bounding_ax_` is None.

    axes_ : ndarray of matplotlib Axes
        If `ax` is an axes or None, `axes_[i, j]` is the axes on the i-th row
        and j-th column. If `ax` is a list of axes, `axes_[i]` is the i-th item
        in `ax`. Elements that are None correspond to a nonexisting axes in
        that position.

    lines_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `lines_[i, j]` is the partial dependence
        curve on the i-th row and j-th column. If `ax` is a list of axes,
        `lines_[i]` is the partial dependence curve corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a line plot.

    deciles_vlines_ : ndarray of matplotlib LineCollection
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the x axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a PDP plot.

        .. versionadded:: 0.23

    deciles_hlines_ : ndarray of matplotlib LineCollection
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the y axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a 2-way plot.

        .. versionadded:: 0.23

    contours_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `contours_[i, j]` is the partial dependence
        plot on the i-th row and j-th column. If `ax` is a list of axes,
        `contours_[i]` is the partial dependence plot corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a contour plot.

    bars_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `bars_[i, j]` is the partial dependence bar
        plot on the i-th row and j-th column (for a categorical feature).
        If `ax` is a list of axes, `bars_[i]` is the partial dependence bar
        plot corresponding to the i-th item in `ax`. Elements that are None
        correspond to a nonexisting axes or an axes that does not include a
        bar plot.

        .. versionadded:: 1.2

    heatmaps_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `heatmaps_[i, j]` is the partial dependence
        heatmap on the i-th row and j-th column (for a pair of categorical
        features) . If `ax` is a list of axes, `heatmaps_[i]` is the partial
        dependence heatmap corresponding to the i-th item in `ax`. Elements
        that are None correspond to a nonexisting axes or an axes that does not
        include a heatmap.

        .. versionadded:: 1.2

    figure_ : matplotlib Figure
        Figure containing partial dependence plots.

    See Also
    --------
    partial_dependence : Compute Partial Dependence values.
    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.inspection import PartialDependenceDisplay
    >>> from sklearn.inspection import partial_dependence
    >>> X, y = make_friedman1()
    >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
    >>> features, feature_names = [(0,)], [f"Features #{i}" for i in range(X.shape[1])]
    >>> deciles = {0: np.linspace(0, 1, num=5)}
    >>> pd_results = partial_dependence(
    ...     clf, X, features=0, kind="average", grid_resolution=5)
    >>> display = PartialDependenceDisplay(
    ...     [pd_results], features=features, feature_names=feature_names,
    ...     target_idx=0, deciles=deciles
    ... )
    >>> display.plot(pdp_lim={1: (-1.38, 0.66)})
    <...>
    >>> plt.show()
    """

    def __init__(self, pd_results, *, features, feature_names, target_idx, deciles, kind='average', subsample=1000, random_state=None, is_categorical=None):
        if False:
            print('Hello World!')
        self.pd_results = pd_results
        self.features = features
        self.feature_names = feature_names
        self.target_idx = target_idx
        self.deciles = deciles
        self.kind = kind
        self.subsample = subsample
        self.random_state = random_state
        self.is_categorical = is_categorical

    @classmethod
    def from_estimator(cls, estimator, X, features, *, sample_weight=None, categorical_features=None, feature_names=None, target=None, response_method='auto', n_cols=3, grid_resolution=100, percentiles=(0.05, 0.95), method='auto', n_jobs=None, verbose=0, line_kw=None, ice_lines_kw=None, pd_line_kw=None, contour_kw=None, ax=None, kind='average', centered=False, subsample=1000, random_state=None):
        if False:
            print('Hello World!')
        "Partial dependence (PD) and individual conditional expectation (ICE) plots.\n\n        Partial dependence plots, individual conditional expectation plots or an\n        overlay of both of them can be plotted by setting the ``kind``\n        parameter. The ``len(features)`` plots are arranged in a grid with\n        ``n_cols`` columns. Two-way partial dependence plots are plotted as\n        contour plots. The deciles of the feature values will be shown with tick\n        marks on the x-axes for one-way plots, and on both axes for two-way\n        plots.\n\n        Read more in the :ref:`User Guide <partial_dependence>`.\n\n        .. note::\n\n            :func:`PartialDependenceDisplay.from_estimator` does not support using the\n            same axes with multiple calls. To plot the partial dependence for\n            multiple estimators, please pass the axes created by the first call to the\n            second call::\n\n               >>> from sklearn.inspection import PartialDependenceDisplay\n               >>> from sklearn.datasets import make_friedman1\n               >>> from sklearn.linear_model import LinearRegression\n               >>> from sklearn.ensemble import RandomForestRegressor\n               >>> X, y = make_friedman1()\n               >>> est1 = LinearRegression().fit(X, y)\n               >>> est2 = RandomForestRegressor().fit(X, y)\n               >>> disp1 = PartialDependenceDisplay.from_estimator(est1, X,\n               ...                                                 [1, 2])\n               >>> disp2 = PartialDependenceDisplay.from_estimator(est2, X, [1, 2],\n               ...                                                 ax=disp1.axes_)\n\n        .. warning::\n\n            For :class:`~sklearn.ensemble.GradientBoostingClassifier` and\n            :class:`~sklearn.ensemble.GradientBoostingRegressor`, the\n            `'recursion'` method (used by default) will not account for the `init`\n            predictor of the boosting process. In practice, this will produce\n            the same values as `'brute'` up to a constant offset in the target\n            response, provided that `init` is a constant estimator (which is the\n            default). However, if `init` is not a constant estimator, the\n            partial dependence values are incorrect for `'recursion'` because the\n            offset will be sample-dependent. It is preferable to use the `'brute'`\n            method. Note that this only applies to\n            :class:`~sklearn.ensemble.GradientBoostingClassifier` and\n            :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to\n            :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and\n            :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.\n\n        .. versionadded:: 1.0\n\n        Parameters\n        ----------\n        estimator : BaseEstimator\n            A fitted estimator object implementing :term:`predict`,\n            :term:`predict_proba`, or :term:`decision_function`.\n            Multioutput-multiclass classifiers are not supported.\n\n        X : {array-like, dataframe} of shape (n_samples, n_features)\n            ``X`` is used to generate a grid of values for the target\n            ``features`` (where the partial dependence will be evaluated), and\n            also to generate values for the complement features when the\n            `method` is `'brute'`.\n\n        features : list of {int, str, pair of int, pair of str}\n            The target features for which to create the PDPs.\n            If `features[i]` is an integer or a string, a one-way PDP is created;\n            if `features[i]` is a tuple, a two-way PDP is created (only supported\n            with `kind='average'`). Each tuple must be of size 2.\n            If any entry is a string, then it must be in ``feature_names``.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights are used to calculate weighted means when averaging the\n            model output. If `None`, then samples are equally weighted. If\n            `sample_weight` is not `None`, then `method` will be set to `'brute'`.\n            Note that `sample_weight` is ignored for `kind='individual'`.\n\n            .. versionadded:: 1.3\n\n        categorical_features : array-like of shape (n_features,) or shape                 (n_categorical_features,), dtype={bool, int, str}, default=None\n            Indicates the categorical features.\n\n            - `None`: no feature will be considered categorical;\n            - boolean array-like: boolean mask of shape `(n_features,)`\n              indicating which features are categorical. Thus, this array has\n              the same shape has `X.shape[1]`;\n            - integer or string array-like: integer indices or strings\n              indicating categorical features.\n\n            .. versionadded:: 1.2\n\n        feature_names : array-like of shape (n_features,), dtype=str, default=None\n            Name of each feature; `feature_names[i]` holds the name of the feature\n            with index `i`.\n            By default, the name of the feature corresponds to their numerical\n            index for NumPy array and their column name for pandas dataframe.\n\n        target : int, default=None\n            - In a multiclass setting, specifies the class for which the PDPs\n              should be computed. Note that for binary classification, the\n              positive class (index 1) is always used.\n            - In a multioutput setting, specifies the task for which the PDPs\n              should be computed.\n\n            Ignored in binary classification or classical regression settings.\n\n        response_method : {'auto', 'predict_proba', 'decision_function'},                 default='auto'\n            Specifies whether to use :term:`predict_proba` or\n            :term:`decision_function` as the target response. For regressors\n            this parameter is ignored and the response is always the output of\n            :term:`predict`. By default, :term:`predict_proba` is tried first\n            and we revert to :term:`decision_function` if it doesn't exist. If\n            ``method`` is `'recursion'`, the response is always the output of\n            :term:`decision_function`.\n\n        n_cols : int, default=3\n            The maximum number of columns in the grid plot. Only active when `ax`\n            is a single axis or `None`.\n\n        grid_resolution : int, default=100\n            The number of equally spaced points on the axes of the plots, for each\n            target feature.\n\n        percentiles : tuple of float, default=(0.05, 0.95)\n            The lower and upper percentile used to create the extreme values\n            for the PDP axes. Must be in [0, 1].\n\n        method : str, default='auto'\n            The method used to calculate the averaged predictions:\n\n            - `'recursion'` is only supported for some tree-based estimators\n              (namely\n              :class:`~sklearn.ensemble.GradientBoostingClassifier`,\n              :class:`~sklearn.ensemble.GradientBoostingRegressor`,\n              :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,\n              :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,\n              :class:`~sklearn.tree.DecisionTreeRegressor`,\n              :class:`~sklearn.ensemble.RandomForestRegressor`\n              but is more efficient in terms of speed.\n              With this method, the target response of a\n              classifier is always the decision function, not the predicted\n              probabilities. Since the `'recursion'` method implicitly computes\n              the average of the ICEs by design, it is not compatible with ICE and\n              thus `kind` must be `'average'`.\n\n            - `'brute'` is supported for any estimator, but is more\n              computationally intensive.\n\n            - `'auto'`: the `'recursion'` is used for estimators that support it,\n              and `'brute'` is used otherwise. If `sample_weight` is not `None`,\n              then `'brute'` is used regardless of the estimator.\n\n            Please see :ref:`this note <pdp_method_differences>` for\n            differences between the `'brute'` and `'recursion'` method.\n\n        n_jobs : int, default=None\n            The number of CPUs to use to compute the partial dependences.\n            Computation is parallelized over features specified by the `features`\n            parameter.\n\n            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n            for more details.\n\n        verbose : int, default=0\n            Verbose output during PD computations.\n\n        line_kw : dict, default=None\n            Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.\n            For one-way partial dependence plots. It can be used to define common\n            properties for both `ice_lines_kw` and `pdp_line_kw`.\n\n        ice_lines_kw : dict, default=None\n            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.\n            For ICE lines in the one-way partial dependence plots.\n            The key value pairs defined in `ice_lines_kw` takes priority over\n            `line_kw`.\n\n        pd_line_kw : dict, default=None\n            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.\n            For partial dependence in one-way partial dependence plots.\n            The key value pairs defined in `pd_line_kw` takes priority over\n            `line_kw`.\n\n        contour_kw : dict, default=None\n            Dict with keywords passed to the ``matplotlib.pyplot.contourf`` call.\n            For two-way partial dependence plots.\n\n        ax : Matplotlib axes or array-like of Matplotlib axes, default=None\n            - If a single axis is passed in, it is treated as a bounding axes\n              and a grid of partial dependence plots will be drawn within\n              these bounds. The `n_cols` parameter controls the number of\n              columns in the grid.\n            - If an array-like of axes are passed in, the partial dependence\n              plots will be drawn directly into these axes.\n            - If `None`, a figure and a bounding axes is created and treated\n              as the single axes case.\n\n        kind : {'average', 'individual', 'both'}, default='average'\n            Whether to plot the partial dependence averaged across all the samples\n            in the dataset or one line per sample or both.\n\n            - ``kind='average'`` results in the traditional PD plot;\n            - ``kind='individual'`` results in the ICE plot.\n\n           Note that the fast `method='recursion'` option is only available for\n           `kind='average'` and `sample_weights=None`. Computing individual\n           dependencies and doing weighted averages requires using the slower\n           `method='brute'`.\n\n        centered : bool, default=False\n            If `True`, the ICE and PD lines will start at the origin of the\n            y-axis. By default, no centering is done.\n\n            .. versionadded:: 1.1\n\n        subsample : float, int or None, default=1000\n            Sampling for ICE curves when `kind` is 'individual' or 'both'.\n            If `float`, should be between 0.0 and 1.0 and represent the proportion\n            of the dataset to be used to plot ICE curves. If `int`, represents the\n            absolute number samples to use.\n\n            Note that the full dataset is still used to calculate averaged partial\n            dependence when `kind='both'`.\n\n        random_state : int, RandomState instance or None, default=None\n            Controls the randomness of the selected samples when subsamples is not\n            `None` and `kind` is either `'both'` or `'individual'`.\n            See :term:`Glossary <random_state>` for details.\n\n        Returns\n        -------\n        display : :class:`~sklearn.inspection.PartialDependenceDisplay`\n\n        See Also\n        --------\n        partial_dependence : Compute Partial Dependence values.\n\n        Examples\n        --------\n        >>> import matplotlib.pyplot as plt\n        >>> from sklearn.datasets import make_friedman1\n        >>> from sklearn.ensemble import GradientBoostingRegressor\n        >>> from sklearn.inspection import PartialDependenceDisplay\n        >>> X, y = make_friedman1()\n        >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)\n        >>> PartialDependenceDisplay.from_estimator(clf, X, [0, (0, 1)])\n        <...>\n        >>> plt.show()\n        "
        check_matplotlib_support(f'{cls.__name__}.from_estimator')
        import matplotlib.pyplot as plt
        if hasattr(estimator, 'classes_') and np.size(estimator.classes_) > 2:
            if target is None:
                raise ValueError('target must be specified for multi-class')
            target_idx = np.searchsorted(estimator.classes_, target)
            if not 0 <= target_idx < len(estimator.classes_) or estimator.classes_[target_idx] != target:
                raise ValueError('target not in est.classes_, got {}'.format(target))
        else:
            target_idx = 0
        if not (hasattr(X, '__array__') or sparse.issparse(X)):
            X = check_array(X, force_all_finite='allow-nan', dtype=object)
        n_features = X.shape[1]
        feature_names = _check_feature_names(X, feature_names)
        kind_ = [kind] * len(features) if isinstance(kind, str) else kind
        if len(kind_) != len(features):
            raise ValueError(f'When `kind` is provided as a list of strings, it should contain as many elements as `features`. `kind` contains {len(kind_)} element(s) and `features` contains {len(features)} element(s).')
        (tmp_features, ice_for_two_way_pd) = ([], [])
        for (kind_plot, fxs) in zip(kind_, features):
            if isinstance(fxs, (numbers.Integral, str)):
                fxs = (fxs,)
            try:
                fxs = tuple((_get_feature_index(fx, feature_names=feature_names) for fx in fxs))
            except TypeError as e:
                raise ValueError('Each entry in features must be either an int, a string, or an iterable of size at most 2.') from e
            if not 1 <= np.size(fxs) <= 2:
                raise ValueError('Each entry in features must be either an int, a string, or an iterable of size at most 2.')
            ice_for_two_way_pd.append(kind_plot != 'average' and np.size(fxs) > 1)
            tmp_features.append(fxs)
        if any(ice_for_two_way_pd):
            kind_ = ['average' if forcing_average else kind_plot for (forcing_average, kind_plot) in zip(ice_for_two_way_pd, kind_)]
            raise ValueError(f"ICE plot cannot be rendered for 2-way feature interactions. 2-way feature interactions mandates PD plots using the 'average' kind: features={features!r} should be configured to use kind={kind_!r} explicitly.")
        features = tmp_features
        if categorical_features is None:
            is_categorical = [(False,) if len(fxs) == 1 else (False, False) for fxs in features]
        else:
            categorical_features = np.array(categorical_features, copy=False)
            if categorical_features.dtype.kind == 'b':
                if categorical_features.size != n_features:
                    raise ValueError(f'When `categorical_features` is a boolean array-like, the array should be of shape (n_features,). Got {categorical_features.size} elements while `X` contains {n_features} features.')
                is_categorical = [tuple((categorical_features[fx] for fx in fxs)) for fxs in features]
            elif categorical_features.dtype.kind in ('i', 'O', 'U'):
                categorical_features_idx = [_get_feature_index(cat, feature_names=feature_names) for cat in categorical_features]
                is_categorical = [tuple([idx in categorical_features_idx for idx in fxs]) for fxs in features]
            else:
                raise ValueError(f'Expected `categorical_features` to be an array-like of boolean, integer, or string. Got {categorical_features.dtype} instead.')
            for cats in is_categorical:
                if np.size(cats) == 2 and cats[0] != cats[1]:
                    raise ValueError('Two-way partial dependence plots are not supported for pairs of continuous and categorical features.')
            categorical_features_targeted = set([fx for (fxs, cats) in zip(features, is_categorical) for fx in fxs if any(cats)])
            if categorical_features_targeted:
                min_n_cats = min([len(_unique(_safe_indexing(X, idx, axis=1))) for idx in categorical_features_targeted])
                if grid_resolution < min_n_cats:
                    raise ValueError(f'The resolution of the computed grid is less than the minimum number of categories in the targeted categorical features. Expect the `grid_resolution` to be greater than {min_n_cats}. Got {grid_resolution} instead.')
            for (is_cat, kind_plot) in zip(is_categorical, kind_):
                if any(is_cat) and kind_plot != 'average':
                    raise ValueError('It is not possible to display individual effects for categorical features.')
        if ax is not None and (not isinstance(ax, plt.Axes)):
            axes = np.asarray(ax, dtype=object)
            if axes.size != len(features):
                raise ValueError('Expected ax to have {} axes, got {}'.format(len(features), axes.size))
        for i in chain.from_iterable(features):
            if i >= len(feature_names):
                raise ValueError('All entries of features must be less than len(feature_names) = {0}, got {1}.'.format(len(feature_names), i))
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(f'When an integer, subsample={subsample} should be positive.')
        elif isinstance(subsample, numbers.Real):
            if subsample <= 0 or subsample >= 1:
                raise ValueError(f'When a floating-point, subsample={subsample} should be in the (0, 1) range.')
        pd_results = Parallel(n_jobs=n_jobs, verbose=verbose)((delayed(partial_dependence)(estimator, X, fxs, sample_weight=sample_weight, feature_names=feature_names, categorical_features=categorical_features, response_method=response_method, method=method, grid_resolution=grid_resolution, percentiles=percentiles, kind=kind_plot) for (kind_plot, fxs) in zip(kind_, features)))
        pd_result = pd_results[0]
        n_tasks = pd_result.average.shape[0] if kind_[0] == 'average' else pd_result.individual.shape[0]
        if is_regressor(estimator) and n_tasks > 1:
            if target is None:
                raise ValueError('target must be specified for multi-output regressors')
            if not 0 <= target <= n_tasks:
                raise ValueError('target must be in [0, n_tasks], got {}.'.format(target))
            target_idx = target
        deciles = {}
        for (fxs, cats) in zip(features, is_categorical):
            for (fx, cat) in zip(fxs, cats):
                if not cat and fx not in deciles:
                    X_col = _safe_indexing(X, fx, axis=1)
                    deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))
        display = PartialDependenceDisplay(pd_results=pd_results, features=features, feature_names=feature_names, target_idx=target_idx, deciles=deciles, kind=kind, subsample=subsample, random_state=random_state, is_categorical=is_categorical)
        return display.plot(ax=ax, n_cols=n_cols, line_kw=line_kw, ice_lines_kw=ice_lines_kw, pd_line_kw=pd_line_kw, contour_kw=contour_kw, centered=centered)

    def _get_sample_count(self, n_samples):
        if False:
            while True:
                i = 10
        'Compute the number of samples as an integer.'
        if isinstance(self.subsample, numbers.Integral):
            if self.subsample < n_samples:
                return self.subsample
            return n_samples
        elif isinstance(self.subsample, numbers.Real):
            return ceil(n_samples * self.subsample)
        return n_samples

    def _plot_ice_lines(self, preds, feature_values, n_ice_to_plot, ax, pd_plot_idx, n_total_lines_by_plot, individual_line_kw):
        if False:
            return 10
        'Plot the ICE lines.\n\n        Parameters\n        ----------\n        preds : ndarray of shape                 (n_instances, n_grid_points)\n            The predictions computed for all points of `feature_values` for a\n            given feature for all samples in `X`.\n        feature_values : ndarray of shape (n_grid_points,)\n            The feature values for which the predictions have been computed.\n        n_ice_to_plot : int\n            The number of ICE lines to plot.\n        ax : Matplotlib axes\n            The axis on which to plot the ICE lines.\n        pd_plot_idx : int\n            The sequential index of the plot. It will be unraveled to find the\n            matching 2D position in the grid layout.\n        n_total_lines_by_plot : int\n            The total number of lines expected to be plot on the axis.\n        individual_line_kw : dict\n            Dict with keywords passed when plotting the ICE lines.\n        '
        rng = check_random_state(self.random_state)
        ice_lines_idx = rng.choice(preds.shape[0], n_ice_to_plot, replace=False)
        ice_lines_subsampled = preds[ice_lines_idx, :]
        for (ice_idx, ice) in enumerate(ice_lines_subsampled):
            line_idx = np.unravel_index(pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape)
            self.lines_[line_idx] = ax.plot(feature_values, ice.ravel(), **individual_line_kw)[0]

    def _plot_average_dependence(self, avg_preds, feature_values, ax, pd_line_idx, line_kw, categorical, bar_kw):
        if False:
            print('Hello World!')
        'Plot the average partial dependence.\n\n        Parameters\n        ----------\n        avg_preds : ndarray of shape (n_grid_points,)\n            The average predictions for all points of `feature_values` for a\n            given feature for all samples in `X`.\n        feature_values : ndarray of shape (n_grid_points,)\n            The feature values for which the predictions have been computed.\n        ax : Matplotlib axes\n            The axis on which to plot the average PD.\n        pd_line_idx : int\n            The sequential index of the plot. It will be unraveled to find the\n            matching 2D position in the grid layout.\n        line_kw : dict\n            Dict with keywords passed when plotting the PD plot.\n        categorical : bool\n            Whether feature is categorical.\n        bar_kw: dict\n            Dict with keywords passed when plotting the PD bars (categorical).\n        '
        if categorical:
            bar_idx = np.unravel_index(pd_line_idx, self.bars_.shape)
            self.bars_[bar_idx] = ax.bar(feature_values, avg_preds, **bar_kw)[0]
            ax.tick_params(axis='x', rotation=90)
        else:
            line_idx = np.unravel_index(pd_line_idx, self.lines_.shape)
            self.lines_[line_idx] = ax.plot(feature_values, avg_preds, **line_kw)[0]

    def _plot_one_way_partial_dependence(self, kind, preds, avg_preds, feature_values, feature_idx, n_ice_lines, ax, n_cols, pd_plot_idx, n_lines, ice_lines_kw, pd_line_kw, categorical, bar_kw, pdp_lim):
        if False:
            while True:
                i = 10
        'Plot 1-way partial dependence: ICE and PDP.\n\n        Parameters\n        ----------\n        kind : str\n            The kind of partial plot to draw.\n        preds : ndarray of shape                 (n_instances, n_grid_points) or None\n            The predictions computed for all points of `feature_values` for a\n            given feature for all samples in `X`.\n        avg_preds : ndarray of shape (n_grid_points,)\n            The average predictions for all points of `feature_values` for a\n            given feature for all samples in `X`.\n        feature_values : ndarray of shape (n_grid_points,)\n            The feature values for which the predictions have been computed.\n        feature_idx : int\n            The index corresponding to the target feature.\n        n_ice_lines : int\n            The number of ICE lines to plot.\n        ax : Matplotlib axes\n            The axis on which to plot the ICE and PDP lines.\n        n_cols : int or None\n            The number of column in the axis.\n        pd_plot_idx : int\n            The sequential index of the plot. It will be unraveled to find the\n            matching 2D position in the grid layout.\n        n_lines : int\n            The total number of lines expected to be plot on the axis.\n        ice_lines_kw : dict\n            Dict with keywords passed when plotting the ICE lines.\n        pd_line_kw : dict\n            Dict with keywords passed when plotting the PD plot.\n        categorical : bool\n            Whether feature is categorical.\n        bar_kw: dict\n            Dict with keywords passed when plotting the PD bars (categorical).\n        pdp_lim : dict\n            Global min and max average predictions, such that all plots will\n            have the same scale and y limits. `pdp_lim[1]` is the global min\n            and max for single partial dependence curves.\n        '
        from matplotlib import transforms
        if kind in ('individual', 'both'):
            self._plot_ice_lines(preds[self.target_idx], feature_values, n_ice_lines, ax, pd_plot_idx, n_lines, ice_lines_kw)
        if kind in ('average', 'both'):
            if kind == 'average':
                pd_line_idx = pd_plot_idx
            else:
                pd_line_idx = pd_plot_idx * n_lines + n_ice_lines
            self._plot_average_dependence(avg_preds[self.target_idx].ravel(), feature_values, ax, pd_line_idx, pd_line_kw, categorical, bar_kw)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
        if self.deciles.get(feature_idx[0], None) is not None:
            self.deciles_vlines_[vlines_idx] = ax.vlines(self.deciles[feature_idx[0]], 0, 0.05, transform=trans, color='k')
        min_val = min((val[0] for val in pdp_lim.values()))
        max_val = max((val[1] for val in pdp_lim.values()))
        ax.set_ylim([min_val, max_val])
        if not ax.get_xlabel():
            ax.set_xlabel(self.feature_names[feature_idx[0]])
        if n_cols is None or pd_plot_idx % n_cols == 0:
            if not ax.get_ylabel():
                ax.set_ylabel('Partial dependence')
        else:
            ax.set_yticklabels([])
        if pd_line_kw.get('label', None) and kind != 'individual' and (not categorical):
            ax.legend()

    def _plot_two_way_partial_dependence(self, avg_preds, feature_values, feature_idx, ax, pd_plot_idx, Z_level, contour_kw, categorical, heatmap_kw):
        if False:
            for i in range(10):
                print('nop')
        'Plot 2-way partial dependence.\n\n        Parameters\n        ----------\n        avg_preds : ndarray of shape                 (n_instances, n_grid_points, n_grid_points)\n            The average predictions for all points of `feature_values[0]` and\n            `feature_values[1]` for some given features for all samples in `X`.\n        feature_values : seq of 1d array\n            A sequence of array of the feature values for which the predictions\n            have been computed.\n        feature_idx : tuple of int\n            The indices of the target features\n        ax : Matplotlib axes\n            The axis on which to plot the ICE and PDP lines.\n        pd_plot_idx : int\n            The sequential index of the plot. It will be unraveled to find the\n            matching 2D position in the grid layout.\n        Z_level : ndarray of shape (8, 8)\n            The Z-level used to encode the average predictions.\n        contour_kw : dict\n            Dict with keywords passed when plotting the contours.\n        categorical : bool\n            Whether features are categorical.\n        heatmap_kw: dict\n            Dict with keywords passed when plotting the PD heatmap\n            (categorical).\n        '
        if categorical:
            import matplotlib.pyplot as plt
            default_im_kw = dict(interpolation='nearest', cmap='viridis')
            im_kw = {**default_im_kw, **heatmap_kw}
            data = avg_preds[self.target_idx]
            im = ax.imshow(data, **im_kw)
            text = None
            (cmap_min, cmap_max) = (im.cmap(0), im.cmap(1.0))
            text = np.empty_like(data, dtype=object)
            thresh = (data.max() + data.min()) / 2.0
            for flat_index in range(data.size):
                (row, col) = np.unravel_index(flat_index, data.shape)
                color = cmap_max if data[row, col] < thresh else cmap_min
                values_format = '.2f'
                text_data = format(data[row, col], values_format)
                text_kwargs = dict(ha='center', va='center', color=color)
                text[row, col] = ax.text(col, row, text_data, **text_kwargs)
            fig = ax.figure
            fig.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(len(feature_values[1])), yticks=np.arange(len(feature_values[0])), xticklabels=feature_values[1], yticklabels=feature_values[0], xlabel=self.feature_names[feature_idx[1]], ylabel=self.feature_names[feature_idx[0]])
            plt.setp(ax.get_xticklabels(), rotation='vertical')
            heatmap_idx = np.unravel_index(pd_plot_idx, self.heatmaps_.shape)
            self.heatmaps_[heatmap_idx] = im
        else:
            from matplotlib import transforms
            (XX, YY) = np.meshgrid(feature_values[0], feature_values[1])
            Z = avg_preds[self.target_idx].T
            CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5, colors='k')
            contour_idx = np.unravel_index(pd_plot_idx, self.contours_.shape)
            self.contours_[contour_idx] = ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1], vmin=Z_level[0], **contour_kw)
            ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            (xlim, ylim) = (ax.get_xlim(), ax.get_ylim())
            vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
            self.deciles_vlines_[vlines_idx] = ax.vlines(self.deciles[feature_idx[0]], 0, 0.05, transform=trans, color='k')
            hlines_idx = np.unravel_index(pd_plot_idx, self.deciles_hlines_.shape)
            self.deciles_hlines_[hlines_idx] = ax.hlines(self.deciles[feature_idx[1]], 0, 0.05, transform=trans, color='k')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if not ax.get_xlabel():
                ax.set_xlabel(self.feature_names[feature_idx[0]])
            ax.set_ylabel(self.feature_names[feature_idx[1]])

    def plot(self, *, ax=None, n_cols=3, line_kw=None, ice_lines_kw=None, pd_line_kw=None, contour_kw=None, bar_kw=None, heatmap_kw=None, pdp_lim=None, centered=False):
        if False:
            print('Hello World!')
        'Plot partial dependence plots.\n\n        Parameters\n        ----------\n        ax : Matplotlib axes or array-like of Matplotlib axes, default=None\n            - If a single axis is passed in, it is treated as a bounding axes\n                and a grid of partial dependence plots will be drawn within\n                these bounds. The `n_cols` parameter controls the number of\n                columns in the grid.\n            - If an array-like of axes are passed in, the partial dependence\n                plots will be drawn directly into these axes.\n            - If `None`, a figure and a bounding axes is created and treated\n                as the single axes case.\n\n        n_cols : int, default=3\n            The maximum number of columns in the grid plot. Only active when\n            `ax` is a single axes or `None`.\n\n        line_kw : dict, default=None\n            Dict with keywords passed to the `matplotlib.pyplot.plot` call.\n            For one-way partial dependence plots.\n\n        ice_lines_kw : dict, default=None\n            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.\n            For ICE lines in the one-way partial dependence plots.\n            The key value pairs defined in `ice_lines_kw` takes priority over\n            `line_kw`.\n\n            .. versionadded:: 1.0\n\n        pd_line_kw : dict, default=None\n            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.\n            For partial dependence in one-way partial dependence plots.\n            The key value pairs defined in `pd_line_kw` takes priority over\n            `line_kw`.\n\n            .. versionadded:: 1.0\n\n        contour_kw : dict, default=None\n            Dict with keywords passed to the `matplotlib.pyplot.contourf`\n            call for two-way partial dependence plots.\n\n        bar_kw : dict, default=None\n            Dict with keywords passed to the `matplotlib.pyplot.bar`\n            call for one-way categorical partial dependence plots.\n\n            .. versionadded:: 1.2\n\n        heatmap_kw : dict, default=None\n            Dict with keywords passed to the `matplotlib.pyplot.imshow`\n            call for two-way categorical partial dependence plots.\n\n            .. versionadded:: 1.2\n\n        pdp_lim : dict, default=None\n            Global min and max average predictions, such that all plots will have the\n            same scale and y limits. `pdp_lim[1]` is the global min and max for single\n            partial dependence curves. `pdp_lim[2]` is the global min and max for\n            two-way partial dependence curves. If `None` (default), the limit will be\n            inferred from the global minimum and maximum of all predictions.\n\n            .. versionadded:: 1.1\n\n        centered : bool, default=False\n            If `True`, the ICE and PD lines will start at the origin of the\n            y-axis. By default, no centering is done.\n\n            .. versionadded:: 1.1\n\n        Returns\n        -------\n        display : :class:`~sklearn.inspection.PartialDependenceDisplay`\n            Returns a :class:`~sklearn.inspection.PartialDependenceDisplay`\n            object that contains the partial dependence plots.\n        '
        check_matplotlib_support('plot_partial_dependence')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        if isinstance(self.kind, str):
            kind = [self.kind] * len(self.features)
        else:
            kind = self.kind
        if self.is_categorical is None:
            is_categorical = [(False,) if len(fx) == 1 else (False, False) for fx in self.features]
        else:
            is_categorical = self.is_categorical
        if len(kind) != len(self.features):
            raise ValueError(f'When `kind` is provided as a list of strings, it should contain as many elements as `features`. `kind` contains {len(kind)} element(s) and `features` contains {len(self.features)} element(s).')
        valid_kinds = {'average', 'individual', 'both'}
        if any([k not in valid_kinds for k in kind]):
            raise ValueError(f'Values provided to `kind` must be one of: {valid_kinds!r} or a list of such values. Currently, kind={self.kind!r}')
        if not centered:
            pd_results_ = self.pd_results
        else:
            pd_results_ = []
            for (kind_plot, pd_result) in zip(kind, self.pd_results):
                current_results = {'grid_values': pd_result['grid_values']}
                if kind_plot in ('individual', 'both'):
                    preds = pd_result.individual
                    preds = preds - preds[self.target_idx, :, 0, None]
                    current_results['individual'] = preds
                if kind_plot in ('average', 'both'):
                    avg_preds = pd_result.average
                    avg_preds = avg_preds - avg_preds[self.target_idx, 0, None]
                    current_results['average'] = avg_preds
                pd_results_.append(Bunch(**current_results))
        if pdp_lim is None:
            pdp_lim = {}
            for (kind_plot, pdp) in zip(kind, pd_results_):
                values = pdp['grid_values']
                preds = pdp.average if kind_plot == 'average' else pdp.individual
                min_pd = preds[self.target_idx].min()
                max_pd = preds[self.target_idx].max()
                span = max_pd - min_pd
                min_pd -= 0.05 * span
                max_pd += 0.05 * span
                n_fx = len(values)
                (old_min_pd, old_max_pd) = pdp_lim.get(n_fx, (min_pd, max_pd))
                min_pd = min(min_pd, old_min_pd)
                max_pd = max(max_pd, old_max_pd)
                pdp_lim[n_fx] = (min_pd, max_pd)
        if line_kw is None:
            line_kw = {}
        if ice_lines_kw is None:
            ice_lines_kw = {}
        if pd_line_kw is None:
            pd_line_kw = {}
        if bar_kw is None:
            bar_kw = {}
        if heatmap_kw is None:
            heatmap_kw = {}
        if ax is None:
            (_, ax) = plt.subplots()
        if contour_kw is None:
            contour_kw = {}
        default_contour_kws = {'alpha': 0.75}
        contour_kw = {**default_contour_kws, **contour_kw}
        n_features = len(self.features)
        is_average_plot = [kind_plot == 'average' for kind_plot in kind]
        if all(is_average_plot):
            n_ice_lines = 0
            n_lines = 1
        else:
            ice_plot_idx = is_average_plot.index(False)
            n_ice_lines = self._get_sample_count(len(pd_results_[ice_plot_idx].individual[0]))
            if any([kind_plot == 'both' for kind_plot in kind]):
                n_lines = n_ice_lines + 1
            else:
                n_lines = n_ice_lines
        if isinstance(ax, plt.Axes):
            if not ax.axison:
                raise ValueError('The ax was already used in another plot function, please set ax=display.axes_ instead')
            ax.set_axis_off()
            self.bounding_ax_ = ax
            self.figure_ = ax.figure
            n_cols = min(n_cols, n_features)
            n_rows = int(np.ceil(n_features / float(n_cols)))
            self.axes_ = np.empty((n_rows, n_cols), dtype=object)
            if all(is_average_plot):
                self.lines_ = np.empty((n_rows, n_cols), dtype=object)
            else:
                self.lines_ = np.empty((n_rows, n_cols, n_lines), dtype=object)
            self.contours_ = np.empty((n_rows, n_cols), dtype=object)
            self.bars_ = np.empty((n_rows, n_cols), dtype=object)
            self.heatmaps_ = np.empty((n_rows, n_cols), dtype=object)
            axes_ravel = self.axes_.ravel()
            gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_subplotspec())
            for (i, spec) in zip(range(n_features), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)
        else:
            ax = np.asarray(ax, dtype=object)
            if ax.size != n_features:
                raise ValueError('Expected ax to have {} axes, got {}'.format(n_features, ax.size))
            if ax.ndim == 2:
                n_cols = ax.shape[1]
            else:
                n_cols = None
            self.bounding_ax_ = None
            self.figure_ = ax.ravel()[0].figure
            self.axes_ = ax
            if all(is_average_plot):
                self.lines_ = np.empty_like(ax, dtype=object)
            else:
                self.lines_ = np.empty(ax.shape + (n_lines,), dtype=object)
            self.contours_ = np.empty_like(ax, dtype=object)
            self.bars_ = np.empty_like(ax, dtype=object)
            self.heatmaps_ = np.empty_like(ax, dtype=object)
        if 2 in pdp_lim:
            Z_level = np.linspace(*pdp_lim[2], num=8)
        self.deciles_vlines_ = np.empty_like(self.axes_, dtype=object)
        self.deciles_hlines_ = np.empty_like(self.axes_, dtype=object)
        for (pd_plot_idx, (axi, feature_idx, cat, pd_result, kind_plot)) in enumerate(zip(self.axes_.ravel(), self.features, is_categorical, pd_results_, kind)):
            avg_preds = None
            preds = None
            feature_values = pd_result['grid_values']
            if kind_plot == 'individual':
                preds = pd_result.individual
            elif kind_plot == 'average':
                avg_preds = pd_result.average
            else:
                avg_preds = pd_result.average
                preds = pd_result.individual
            if len(feature_values) == 1:
                default_line_kws = {'color': 'C0', 'label': 'average' if kind_plot == 'both' else None}
                if kind_plot == 'individual':
                    default_ice_lines_kws = {'alpha': 0.3, 'linewidth': 0.5}
                    default_pd_lines_kws = {}
                elif kind_plot == 'both':
                    default_ice_lines_kws = {'alpha': 0.3, 'linewidth': 0.5, 'color': 'tab:blue'}
                    default_pd_lines_kws = {'color': 'tab:orange', 'linestyle': '--'}
                else:
                    default_ice_lines_kws = {}
                    default_pd_lines_kws = {}
                ice_lines_kw = {**default_line_kws, **default_ice_lines_kws, **line_kw, **ice_lines_kw}
                del ice_lines_kw['label']
                pd_line_kw = {**default_line_kws, **default_pd_lines_kws, **line_kw, **pd_line_kw}
                default_bar_kws = {'color': 'C0'}
                bar_kw = {**default_bar_kws, **bar_kw}
                default_heatmap_kw = {}
                heatmap_kw = {**default_heatmap_kw, **heatmap_kw}
                self._plot_one_way_partial_dependence(kind_plot, preds, avg_preds, feature_values[0], feature_idx, n_ice_lines, axi, n_cols, pd_plot_idx, n_lines, ice_lines_kw, pd_line_kw, cat[0], bar_kw, pdp_lim)
            else:
                self._plot_two_way_partial_dependence(avg_preds, feature_values, feature_idx, axi, pd_plot_idx, Z_level, contour_kw, cat[0] and cat[1], heatmap_kw)
        return self