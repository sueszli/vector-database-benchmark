"""Common utilities for distribution checks."""
from numbers import Number
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, wasserstein_distance
from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.distribution.plot import CategoriesSortingKind, drift_score_bar_traces, feature_distribution_traces
from deepchecks.utils.distribution.preprocessing import preprocess_2_cat_cols_to_same_bins
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES
from deepchecks.utils.strings import format_number
__all__ = ['calc_drift_and_plot', 'get_drift_method', 'SUPPORTED_CATEGORICAL_METHODS', 'SUPPORTED_NUMERIC_METHODS', 'drift_condition', 'get_drift_plot_sidenote', 'cramers_v', 'psi']
PSI_MIN_PERCENTAGE = 0.01
SUPPORTED_CATEGORICAL_METHODS = ["Cramer's V", 'PSI']
SUPPORTED_NUMERIC_METHODS = ["Earth Mover's Distance", 'Kolmogorov-Smirnov']

def filter_margins_by_quantile(dist: Union[np.ndarray, pd.Series], margin_quantile_filter: float) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Filter the margins of the distribution by a quantile.'
    (qt_min, qt_max) = np.quantile(dist, [margin_quantile_filter, 1 - margin_quantile_filter])
    return dist[(qt_max >= dist) & (dist >= qt_min)]

def get_drift_method(result_dict: Dict):
    if False:
        return 10
    'Return which drift scoring methods were in use.\n\n    Parameters\n    ----------\n    result_dict : Dict\n        the result dict of the drift check.\n    Returns\n    -------\n    Tuple(str, str)\n        the categorical scoring method and then the numeric scoring method.\n\n    '
    result_df = pd.DataFrame(result_dict).T
    cat_mthod_arr = result_df[result_df['Method'].isin(SUPPORTED_CATEGORICAL_METHODS)]['Method']
    cat_method = cat_mthod_arr.iloc[0] if len(cat_mthod_arr) else None
    num_mthod_arr = result_df[result_df['Method'].isin(SUPPORTED_NUMERIC_METHODS)]['Method']
    num_method = num_mthod_arr.iloc[0] if len(num_mthod_arr) else None
    return (cat_method, num_method)

def rebalance_distributions(dist1_counts: np.array, dist2_counts: np.array):
    if False:
        for i in range(10):
            print('nop')
    'Rebalance the distributions as if dist1 was a even distribution.\n\n    This is a util function for an unbalanced version categorical drift scoring methods. The rebalancing is done\n    so in practice all categories of the distributions are treated with the same weight.\n\n    The function redefines the dist1_counts to have equal counts for all categories, and then redefines the\n    dist2_counts to have the same "change" it had from dist1_counts, but relative to the new dist1_counts. In\n    addition, we add 1 to all counts. This is inspired from the properties of Beta distribution for a bernoulli\n    distribution parameter estimation, see\n    http://www.ece.virginia.edu/~ffh8x/docs/teaching/esl/2020-04/farnoud-slgm-chap03.pdf for additional details.\n\n    Example:\n        if dist1_counts was [9000, 1000] and dist2_counts was [8000, 2000]. This means that if we treat dist1 as a\n        even distribution, the new counts should be dist1_counts = [5000, 5000].\n        The relative change of dist2 from dist1 was a decrease of ~11% in the first category and an increase of\n        ~200% in the second category. The new dist2_counts should be [4445, 9995].\n        # When re-adjusting to the original total num_samples of dist2, the new dist2_counts should be [3078, 6922]\n    '
    (dist1_counts, dist2_counts) = _balance_sizes_downsizing(dist1_counts, dist2_counts, round_to_int=False)
    multipliers = [x2 / x1 for (x1, x2) in zip(dist1_counts + 1, dist2_counts + 1)]
    dist1_counts = np.array([int(np.sum(dist2_counts + 1) / len(dist2_counts))] * len(dist2_counts))
    dist2_counts = np.round(dist1_counts * multipliers)
    dist2_counts = np.round(dist2_counts * (sum(dist1_counts) / sum(dist2_counts)))
    return (dist1_counts, dist2_counts)

def cramers_v(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series], balance_classes: bool=False, min_category_size_ratio: float=0, max_num_categories: int=None, sort_by: str='dist1', from_freqs: bool=False) -> float:
    if False:
        i = 10
        return i + 15
    'Calculate the Cramer\'s V statistic.\n\n    For more on Cramer\'s V, see https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V\n    Uses the Cramer\'s V bias correction, see http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf\n\n    Function is for categorical data only.\n\n    Parameters\n    ----------\n    dist1 : Union[np.ndarray, pd.Series]\n        array of numerical values.\n    dist2 : Union[np.ndarray, pd.Series]\n        array of numerical values to compare dist1 to.\n    balance_classes : bool, default False\n        whether to balance the classes of the distributions. Use this in case of extremely unbalanced classes.\n    min_category_size_ratio: float, default 0.01\n        minimum size ratio for categories. Categories with size ratio lower than this number are binned\n        into an "Other" category.\n    max_num_categories: int, default: None\n        max number of allowed categories. If there are more categories than this number, categories are ordered by\n        magnitude and all the smaller categories are binned into an "Other" category.\n        If max_num_categories=None, there is no limit.\n        > Note that if this parameter is used, the ordering of categories (and by extension, the decision which\n        categories are kept by name and which are binned to the "Other" category) is done by default according to the\n        values of dist1, which is treated as the "expected" distribution. This behavior can be changed by using the\n        sort_by parameter.\n    sort_by: str, default: \'dist1\'\n        Specify how categories should be sorted, affecting which categories will get into the "Other" category.\n        Possible values:\n        - \'dist1\': Sort by the largest dist1 categories.\n        - \'dist2\': Sort by the largest dist2 categories.\n        - \'difference\': Sort by the largest difference between categories.\n        > Note that this parameter has no effect if max_num_categories = None or there are not enough unique categories.\n    from_freqs: bool, default: False\n        Whether the data is already in the form of frequencies.\n    Returns\n    -------\n    float\n        the bias-corrected Cramer\'s V value of the 2 distributions.\n\n    '
    min_category_size_ratio = min_category_size_ratio if balance_classes is False else 0
    if from_freqs:
        (dist1_counts, dist2_counts) = (dist1, dist2)
    else:
        (dist1_counts, dist2_counts, cat_list) = preprocess_2_cat_cols_to_same_bins(dist1, dist2, min_category_size_ratio, max_num_categories, sort_by)
        if len(cat_list) == 1:
            return 0
        if balance_classes is True:
            (dist1_counts, dist2_counts) = rebalance_distributions(dist1_counts, dist2_counts)
        else:
            (dist1_counts, dist2_counts) = _balance_sizes_downsizing(dist1_counts, dist2_counts)
    contingency_matrix = pd.DataFrame([dist1_counts, dist2_counts], dtype=int)
    contingency_matrix = contingency_matrix.loc[:, (contingency_matrix != 0).any(axis=0)]
    chi2 = chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()
    phi2 = chi2 / n
    (r, k) = contingency_matrix.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))

def _balance_sizes_downsizing(dist1_counts, dist2_counts, round_to_int: bool=True):
    if False:
        print('Hello World!')
    'Balance the sizes of the distributions by multiplying the larger one by a constant.'
    (dist1_sum, dist2_sum) = (sum(dist1_counts), sum(dist2_counts))
    if dist1_sum > dist2_sum:
        dist1_counts = dist1_counts * (dist2_sum / dist1_sum)
    elif dist1_sum < dist2_sum:
        dist2_counts = dist2_counts * (dist1_sum / dist2_sum)
    if round_to_int is True:
        (dist1_counts, dist2_counts) = (np.round(dist1_counts), np.round(dist2_counts))
    return (dist1_counts, dist2_counts)

def psi(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series], min_category_size_ratio: float=0, max_num_categories: int=None, sort_by: str='dist1', from_freqs: bool=False) -> float:
    if False:
        return 10
    '\n    Calculate the PSI (Population Stability Index).\n\n    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf\n\n    Parameters\n    ----------\n    dist1 : Union[np.ndarray, pd.Series]\n        array of numerical values.\n    dist2 : Union[np.ndarray, pd.Series]\n        array of numerical values to compare dist1 to.\n    min_category_size_ratio: float, default 0.01\n        minimum size ratio for categories. Categories with size ratio lower than this number are binned\n        into an "Other" category.\n    max_num_categories: int, default: None\n        max number of allowed categories. If there are more categories than this number, categories are ordered by\n        magnitude and all the smaller categories are binned into an "Other" category.\n        If max_num_categories=None, there is no limit.\n        > Note that if this parameter is used, the ordering of categories (and by extension, the decision which\n        categories are kept by name and which are binned to the "Other" category) is done by default according to the\n        values of dist1, which is treated as the "expected" distribution. This behavior can be changed by using the\n        sort_by parameter.\n    sort_by: str, default: \'dist1\'\n        Specify how categories should be sorted, affecting which categories will get into the "Other" category.\n        Possible values:\n        - \'dist1\': Sort by the largest dist1 categories.\n        - \'dist2\': Sort by the largest dist2 categories.\n        - \'difference\': Sort by the largest difference between categories.\n        > Note that this parameter has no effect if max_num_categories = None or there are not enough unique categories.\n    from_freqs: bool, default: False\n        Whether the data is already in the form of frequencies.\n    Returns\n    -------\n    psi\n        The PSI score\n    '
    if from_freqs:
        (expected_counts, actual_counts) = (dist1, dist2)
    else:
        (expected_counts, actual_counts, _) = preprocess_2_cat_cols_to_same_bins(dist1, dist2, min_category_size_ratio, max_num_categories, sort_by)
    (size_expected, size_actual) = (sum(expected_counts), sum(actual_counts))
    psi_value = 0
    for i in range(len(expected_counts)):
        e_perc = max(expected_counts[i] / size_expected, PSI_MIN_PERCENTAGE)
        a_perc = max(actual_counts[i] / size_actual, PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value
    return psi_value

def kolmogorov_smirnov(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series]) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform the two-sample Kolmogorov-Smirnov test for goodness of fit.\n\n    This test compares the underlying continuous distributions F(x) and G(x)\n    of two independent samples.\n\n    This function is based on the ks_2samp function from scipy.stats, but it only calculates\n    the test statistic. This is useful for large datasets, where the p-value is not needed.\n\n    Also, this function assumes the alternative hypothesis is two-sided (F(x)!= G(x)).\n\n    Parameters\n    ----------\n    dist1, dist2 : array_like, 1-Dimensional\n        Two arrays of sample observations assumed to be drawn from a continuous\n        distribution, sample sizes can be different.\n\n    Returns\n    -------\n    statistic : float\n        KS statistic.\n\n\n    License\n    ----------\n    This is a modified version of the ks_2samp function from scipy.stats. The original license is as follows:\n\n    Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.\n    All rights reserved.\n\n    Redistribution and use in source and binary forms, with or without\n    modification, are permitted provided that the following conditions\n    are met:\n\n    1. Redistributions of source code must retain the above copyright\n       notice, this list of conditions and the following disclaimer.\n\n    2. Redistributions in binary form must reproduce the above\n       copyright notice, this list of conditions and the following\n       disclaimer in the documentation and/or other materials provided\n       with the distribution.\n\n    3. Neither the name of the copyright holder nor the names of its\n       contributors may be used to endorse or promote products derived\n       from this software without specific prior written permission.\n\n    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\n    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR\n    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT\n    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,\n    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT\n    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\n    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\n    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\n    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n    '
    if np.ma.is_masked(dist1):
        dist1 = dist1.compressed()
    if np.ma.is_masked(dist2):
        dist2 = dist2.compressed()
    dist1 = np.sort(dist1)
    dist2 = np.sort(dist2)
    n1 = dist1.shape[0]
    n2 = dist2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data must not be empty')
    data_all = np.concatenate([dist1, dist2])
    cdf1 = np.searchsorted(dist1, data_all, side='right') / n1
    cdf2 = np.searchsorted(dist2, data_all, side='right') / n2
    cddiffs = np.abs(cdf1 - cdf2)
    return np.max(cddiffs)

def earth_movers_distance(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series], margin_quantile_filter: float) -> float:
    if False:
        return 10
    '\n    Calculate the Earth Movers Distance (Wasserstein distance).\n\n    See https://en.wikipedia.org/wiki/Wasserstein_metric\n\n    Function is for numerical data only.\n\n    Parameters\n    ----------\n    dist1: Union[np.ndarray, pd.Series]\n        array of numerical values.\n    dist2: Union[np.ndarray, pd.Series]\n        array of numerical values to compare dist1 to.\n    margin_quantile_filter: float\n        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered\n        out of the EMD calculation. This is done in order for extreme values not to affect the calculation\n        disproportionally. This filter is applied to both distributions, in both margins.\n    Returns\n    -------\n    Any\n        the Wasserstein distance between the two distributions.\n\n    Raises\n    -------\n    DeepchecksValueError\n        if the value of margin_quantile_filter is not in range [0, 0.5)\n\n    '
    if not isinstance(margin_quantile_filter, Number) or margin_quantile_filter < 0 or margin_quantile_filter >= 0.5:
        raise DeepchecksValueError(f'margin_quantile_filter expected a value in range [0, 0.5), instead got {margin_quantile_filter}')
    if margin_quantile_filter != 0:
        dist1 = filter_margins_by_quantile(dist1, margin_quantile_filter)
        dist2 = filter_margins_by_quantile(dist2, margin_quantile_filter)
    val_max = np.max([np.max(dist1), np.max(dist2)])
    val_min = np.min([np.min(dist1), np.min(dist2)])
    if val_max == val_min:
        return 0
    dist1 = (dist1 - val_min) / (val_max - val_min)
    dist2 = (dist2 - val_min) / (val_max - val_min)
    return wasserstein_distance(dist1, dist2)

def calc_drift_and_plot(train_column: pd.Series, test_column: pd.Series, value_name: str, column_type: str, plot_title: Optional[str]=None, margin_quantile_filter: float=0.025, max_num_categories_for_drift: Optional[int]=None, min_category_size_ratio: float=0.01, max_num_categories_for_display: int=10, show_categories_by: CategoriesSortingKind='largest_difference', numerical_drift_method: str='KS', categorical_drift_method: str='cramers_v', balance_classes: bool=False, ignore_na: bool=True, min_samples: int=10, raise_min_samples_error: bool=False, with_display: bool=True, dataset_names: Tuple[str, str]=DEFAULT_DATASET_NAMES) -> Tuple[float, str, Optional[Figure]]:
    if False:
        return 10
    '\n    Calculate drift score per column.\n\n    Parameters\n    ----------\n    train_column: pd.Series\n        column from train dataset\n    test_column: pd.Series\n        same column from test dataset\n    value_name: str\n        title of the x axis, if plot_title is None then also the title of the whole plot.\n    column_type: str\n        type of column (either "numerical" or "categorical")\n    plot_title: str or None\n        if None use value_name as title otherwise use this.\n    margin_quantile_filter: float, default: 0.025\n        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered\n        out of the EMD calculation. This is done in order for extreme values not to affect the calculation\n        disproportionally. This filter is applied to both distributions, in both margins.\n    min_category_size_ratio: float, default 0.01\n        minimum size ratio for categories. Categories with size ratio lower than this number are binned\n        into an "Other" category.\n    max_num_categories_for_drift: int, default: None\n        Max number of allowed categories. If there are more, they are binned into an "Other" category.\n    max_num_categories_for_display: int, default: 10\n        Max number of categories to show in plot.\n    show_categories_by: str, default: \'largest_difference\'\n        Specify which categories to show for categorical features\' graphs, as the number of shown categories is limited\n        by max_num_categories_for_display. Possible values:\n        - \'train_largest\': Show the largest train categories.\n        - \'test_largest\': Show the largest test categories.\n        - \'largest_difference\': Show the largest difference between categories.\n    numerical_drift_method: str, default: "KS"\n        decides which method to use on numerical variables. Possible values are:\n        "EMD" for Earth Mover\'s Distance (EMD), "KS" for Kolmogorov-Smirnov (KS).\n    categorical_drift_method: str, default: "cramers_v"\n        decides which method to use on categorical variables. Possible values are:\n        "cramers_v" for Cramer\'s V, "PSI" for Population Stability Index (PSI).\n    balance_classes: bool, default: False\n        If True, all categories will have an equal weight in the Cramer\'s V score. This is useful when the categorical\n        variable is highly imbalanced, and we want to be alerted on changes in proportion to the category size,\n        and not only to the entire dataset. Must have categorical_drift_method = "cramers_v".\n    ignore_na: bool, default True\n        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a\n        separate category. For numerical columns we always ignore nones.\n    min_samples : int , default: 10\n        Minimum number of samples required to calculate the drift score. If any of the distributions have less than\n        min_samples, the function will either raise an error or return an invalid output (depends on\n        ``raise_min_sample_error``)\n    raise_min_samples_error : bool , default: False\n        Determines whether to raise an error if the number of samples is less than min_samples. If False, returns the\n        output \'not_enough_samples\', None, None.\n    with_display: bool, default: True\n        flag that determines if function will calculate display.\n    dataset_names: tuple, default: DEFAULT_DATASET_NAMES\n        The names to show in the display for the first and second datasets.\n    Returns\n    -------\n    Tuple[float, str, Callable]\n        - drift score of the difference between the two columns\' distributions\n        - method name\n        - graph comparing the two distributions (density for numerical, stack bar for categorical)\n    '
    if min_category_size_ratio < 0 or min_category_size_ratio > 1:
        raise DeepchecksValueError(f'min_category_size_ratio expected a value in range [0, 1], instead got {min_category_size_ratio}.')
    if column_type == 'categorical' and ignore_na is False:
        train_dist = np.array(train_column.values).reshape(-1)
        test_dist = np.array(test_column.values).reshape(-1)
    else:
        train_dist = np.array(train_column.dropna().values).reshape(-1)
        test_dist = np.array(test_column.dropna().values).reshape(-1)
    if len(train_dist) < min_samples or len(test_dist) < min_samples:
        if raise_min_samples_error is True:
            raise NotEnoughSamplesError(f"Not enough samples to calculate drift score. Minimum {min_samples} samples required. Note that for numerical labels, None values do not count as samples.Use the 'min_samples' parameter to change this requirement.")
        else:
            return ('not_enough_samples', None, None)
    if column_type == 'numerical':
        train_dist = train_dist.astype('float')
        test_dist = test_dist.astype('float')
        if numerical_drift_method.lower() == 'emd':
            scorer_name = "Earth Mover's Distance"
            score = earth_movers_distance(dist1=train_dist, dist2=test_dist, margin_quantile_filter=margin_quantile_filter)
        elif numerical_drift_method.lower() in ['ks', 'kolmogorov-smirnov']:
            scorer_name = 'Kolmogorov-Smirnov'
            score = kolmogorov_smirnov(dist1=train_dist, dist2=test_dist)
        else:
            raise DeepchecksValueError(f'Expected numerical_drift_method to be one of ["EMD", "KS"], received: {numerical_drift_method}')
        if not with_display:
            return (score, scorer_name, None)
        (bar_traces, bar_x_axis, bar_y_axis) = drift_score_bar_traces(score)
        (dist_traces, dist_x_axis, dist_y_axis) = feature_distribution_traces(train_dist, test_dist, value_name, dataset_names=dataset_names)
    elif column_type == 'categorical':
        if balance_classes is True and categorical_drift_method.lower() not in ['cramer_v', 'cramers_v']:
            raise DeepchecksValueError("balance_classes is only supported for Cramer's V. please set balance_classes=False or use 'cramers_v' as categorical_drift_method")
        sort_by = 'difference' if show_categories_by == 'largest_difference' else 'dist1' if show_categories_by == 'train_largest' else 'dist2'
        if categorical_drift_method.lower() in ['cramer_v', 'cramers_v']:
            scorer_name = "Cramer's V"
            score = cramers_v(dist1=train_dist, dist2=test_dist, balance_classes=balance_classes, min_category_size_ratio=min_category_size_ratio, max_num_categories=max_num_categories_for_drift, sort_by=sort_by)
        elif categorical_drift_method.lower() == 'psi':
            scorer_name = 'PSI'
            score = psi(dist1=train_dist, dist2=test_dist, min_category_size_ratio=min_category_size_ratio, max_num_categories=max_num_categories_for_drift, sort_by=sort_by)
        else:
            raise DeepchecksValueError(f'Expected categorical_drift_method to be one of ["cramers_v", "PSI"], received: {categorical_drift_method}')
        if not with_display:
            return (score, scorer_name, None)
        (bar_traces, bar_x_axis, bar_y_axis) = drift_score_bar_traces(score, bar_max=1)
        (dist_traces, dist_x_axis, dist_y_axis) = feature_distribution_traces(train_dist, test_dist, value_name, is_categorical=True, max_num_categories=max_num_categories_for_display, show_categories_by=show_categories_by, dataset_names=dataset_names)
    else:
        raise DeepchecksValueError(f'Unsupported column type for drift: {column_type}')
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, shared_yaxes=False, shared_xaxes=False, row_heights=[0.1, 0.9], subplot_titles=[f'Drift Score ({scorer_name})', 'Distribution Plot'])
    fig.add_traces(bar_traces, rows=1, cols=1)
    fig.update_xaxes(bar_x_axis, row=1, col=1)
    fig.update_yaxes(bar_y_axis, row=1, col=1)
    fig.add_traces(dist_traces, rows=2, cols=1)
    fig.update_xaxes(dist_x_axis, row=2, col=1)
    if balance_classes is True:
        dist_y_axis['title'] += ' (Log Scale)'
        fig.update_yaxes(dist_y_axis, row=2, col=1, type='log')
    else:
        fig.update_yaxes(dist_y_axis, row=2, col=1)
    fig.update_layout(legend=dict(title='Legend', yanchor='top', y=0.6), height=400, title=dict(text=plot_title or value_name, x=0.5, xanchor='center'), bargroupgap=0)
    return (score, scorer_name, fig)

def get_drift_plot_sidenote(max_num_categories_for_display: int, show_categories_by: str) -> str:
    if False:
        i = 10
        return i + 15
    "\n    Return a sidenote for the drift score plots regarding the number of categories shown in discrete distributions.\n\n    Parameters\n    ----------\n    max_num_categories_for_display: int, default: 10\n        Max number of categories to show in plot.\n    show_categories_by: str, default: 'largest_difference'\n        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited\n        by max_num_categories_for_display. Possible values:\n        - 'train_largest': Show the largest train categories.\n        - 'test_largest': Show the largest test categories.\n        - 'largest_difference': Show the largest difference between categories.\n    Returns\n    -------\n    str\n        sidenote for the drift score plots regarding the number of categories shown in discrete distributions.\n    "
    param_to_print_dict = {'train_largest': 'largest categories (by train)', 'test_largest': 'largest categories (by test)', 'largest_difference': 'categories with largest difference between train and test'}
    return f'For discrete distribution plots, showing the top {max_num_categories_for_display} {param_to_print_dict[show_categories_by]}.'

def drift_condition(max_allowed_categorical_score: float, max_allowed_numeric_score: float, subject_single: str, subject_multi: str, allowed_num_subjects_exceeding_threshold: int=0):
    if False:
        return 10
    "Create a condition function to be used in drift check's conditions.\n\n    Parameters\n    ----------\n    max_allowed_categorical_score: float\n        Max value allowed for categorical drift\n    max_allowed_numeric_score: float\n        Max value allowed for numerical drift\n    subject_single: str\n        String that represents the subject being tested as single (feature, column, property)\n    subject_multi: str\n        String that represents the subject being tested as multiple (features, columns, properties)\n    allowed_num_subjects_exceeding_threshold: int, default: 0\n        Determines the number of properties with drift score above threshold needed to fail the condition.\n    "

    def condition(result: dict):
        if False:
            print('Hello World!')
        (cat_method, num_method) = get_drift_method(result)
        cat_drift_props = {prop: d['Drift score'] for (prop, d) in result.items() if d['Method'] in SUPPORTED_CATEGORICAL_METHODS}
        not_passing_categorical_props = {props: format_number(d) for (props, d) in cat_drift_props.items() if d >= max_allowed_categorical_score}
        num_drift_props = {prop: d['Drift score'] for (prop, d) in result.items() if d['Method'] in SUPPORTED_NUMERIC_METHODS}
        not_passing_numeric_props = {prop: format_number(d) for (prop, d) in num_drift_props.items() if d >= max_allowed_numeric_score}
        num_failed = len(not_passing_categorical_props) + len(not_passing_numeric_props)
        if num_failed > allowed_num_subjects_exceeding_threshold:
            details = f'Failed for {num_failed} out of {len(result)} {subject_multi}.'
            if not_passing_categorical_props:
                details += f'\nFound {len(not_passing_categorical_props)} categorical {subject_multi} with {cat_method} above threshold: {not_passing_categorical_props}'
            if not_passing_numeric_props:
                details += f'\nFound {len(not_passing_numeric_props)} numeric {subject_multi} with {num_method} above threshold: {not_passing_numeric_props}'
            return ConditionResult(ConditionCategory.FAIL, details)
        else:
            details = f'Passed for {len(result) - num_failed} {subject_multi} out of {len(result)} {subject_multi}.'
            if cat_drift_props:
                (prop, score) = get_dict_entry_by_value(cat_drift_props)
                details += f'\nFound {subject_single} "{prop}" has the highest categorical drift score: {format_number(score)}'
            if num_drift_props:
                (prop, score) = get_dict_entry_by_value(num_drift_props)
                details += f'\nFound {subject_single} "{prop}" has the highest numerical drift score: {format_number(score)}'
            return ConditionResult(ConditionCategory.PASS, details)
    return condition