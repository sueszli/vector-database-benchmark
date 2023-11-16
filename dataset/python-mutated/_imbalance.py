"""Transform a dataset into an imbalanced dataset."""
from collections import Counter
from collections.abc import Mapping
from ..under_sampling import RandomUnderSampler
from ..utils import check_sampling_strategy
from ..utils._param_validation import validate_params

@validate_params({'X': ['array-like'], 'y': ['array-like'], 'sampling_strategy': [Mapping, callable, None], 'random_state': ['random_state'], 'verbose': ['boolean']}, prefer_skip_nested_validation=True)
def make_imbalance(X, y, *, sampling_strategy=None, random_state=None, verbose=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Turn a dataset into an imbalanced dataset with a specific sampling strategy.\n\n    A simple toy dataset to visualize clustering and classification\n    algorithms.\n\n    Read more in the :ref:`User Guide <make_imbalanced>`.\n\n    Parameters\n    ----------\n    X : {array-like, dataframe} of shape (n_samples, n_features)\n        Matrix containing the data to be imbalanced.\n\n    y : array-like of shape (n_samples,)\n        Corresponding label for each sample in X.\n\n    sampling_strategy : dict or callable,\n        Ratio to use for resampling the data set.\n\n        - When ``dict``, the keys correspond to the targeted classes. The\n          values correspond to the desired number of samples for each targeted\n          class.\n\n        - When callable, function taking ``y`` and returns a ``dict``. The keys\n          correspond to the targeted classes. The values correspond to the\n          desired number of samples for each class.\n\n    random_state : int, RandomState instance or None, default=None\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by np.random.\n\n    verbose : bool, default=False\n        Show information regarding the sampling.\n\n    **kwargs : dict\n        Dictionary of additional keyword arguments to pass to\n        ``sampling_strategy``.\n\n    Returns\n    -------\n    X_resampled : {ndarray, dataframe} of shape (n_samples_new, n_features)\n        The array containing the imbalanced data.\n\n    y_resampled : ndarray of shape (n_samples_new)\n        The corresponding label of `X_resampled`.\n\n    Notes\n    -----\n    See\n    :ref:`sphx_glr_auto_examples_applications_plot_multi_class_under_sampling.py`,\n    :ref:`sphx_glr_auto_examples_datasets_plot_make_imbalance.py`, and\n    :ref:`sphx_glr_auto_examples_api_plot_sampling_strategy_usage.py`.\n\n    Examples\n    --------\n    >>> from collections import Counter\n    >>> from sklearn.datasets import load_iris\n    >>> from imblearn.datasets import make_imbalance\n\n    >>> data = load_iris()\n    >>> X, y = data.data, data.target\n    >>> print(f'Distribution before imbalancing: {Counter(y)}')\n    Distribution before imbalancing: Counter({0: 50, 1: 50, 2: 50})\n    >>> X_res, y_res = make_imbalance(X, y,\n    ...                               sampling_strategy={0: 10, 1: 20, 2: 30},\n    ...                               random_state=42)\n    >>> print(f'Distribution after imbalancing: {Counter(y_res)}')\n    Distribution after imbalancing: Counter({2: 30, 1: 20, 0: 10})\n    "
    target_stats = Counter(y)
    if isinstance(sampling_strategy, Mapping) or callable(sampling_strategy):
        sampling_strategy_ = check_sampling_strategy(sampling_strategy, y, 'under-sampling', **kwargs)
    if verbose:
        print(f'The original target distribution in the dataset is: {target_stats}')
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy_, replacement=False, random_state=random_state)
    (X_resampled, y_resampled) = rus.fit_resample(X, y)
    if verbose:
        print(f'Make the dataset imbalanced: {Counter(y_resampled)}')
    return (X_resampled, y_resampled)