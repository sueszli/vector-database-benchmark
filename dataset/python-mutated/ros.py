"""
Implementation of Regression on Order Statistics for imputing left-
censored (non-detect data)

Method described in *Nondetects and Data Analysis* by Dennis R.
Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
values of a dataset.

Author: Paul M. Hobson
Company: Geosyntec Consultants (Portland, OR)
Date: 2016-06-14

"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats

def _ros_sort(df, observations, censorship, warn=False):
    if False:
        print('Hello World!')
    '\n    This function prepares a dataframe for ROS.\n\n    It sorts ascending with\n    left-censored observations first. Censored observations larger than\n    the maximum uncensored observations are removed from the dataframe.\n\n    Parameters\n    ----------\n    df : DataFrame\n\n    observations : str\n        Name of the column in the dataframe that contains observed\n        values. Censored values should be set to the detection (upper)\n        limit.\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    Returns\n    ------\n    sorted_df : DataFrame\n        The sorted dataframe with all columns dropped except the\n        observation and censorship columns.\n    '
    censored = df[df[censorship]].sort_values(observations, axis=0)
    uncensored = df[~df[censorship]].sort_values(observations, axis=0)
    if censored[observations].max() > uncensored[observations].max():
        censored = censored[censored[observations] <= uncensored[observations].max()]
        if warn:
            msg = 'Dropping censored observations greater than the max uncensored observation.'
            warnings.warn(msg)
    combined = pd.concat([censored, uncensored], axis=0)
    return combined[[observations, censorship]].reset_index(drop=True)

def cohn_numbers(df, observations, censorship):
    if False:
        while True:
            i = 10
    '\n    Computes the Cohn numbers for the detection limits in the dataset.\n\n    The Cohn Numbers are:\n\n        - :math:`A_j =` the number of uncensored obs above the jth\n          threshold.\n        - :math:`B_j =` the number of observations (cen & uncen) below\n          the jth threshold.\n        - :math:`C_j =` the number of censored observations at the jth\n          threshold.\n        - :math:`\\mathrm{PE}_j =` the probability of exceeding the jth\n          threshold\n        - :math:`\\mathrm{DL}_j =` the unique, sorted detection limits\n        - :math:`\\mathrm{DL}_{j+1} = \\mathrm{DL}_j` shifted down a\n          single index (row)\n\n    Parameters\n    ----------\n    dataframe : DataFrame\n\n    observations : str\n        Name of the column in the dataframe that contains observed\n        values. Censored values should be set to the detection (upper)\n        limit.\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    Returns\n    -------\n    cohn : DataFrame\n    '

    def nuncen_above(row):
        if False:
            i = 10
            return i + 15
        ' A, the number of uncensored obs above the given threshold.\n        '
        above = df[observations] >= row['lower_dl']
        below = df[observations] < row['upper_dl']
        detect = ~df[censorship]
        return df[above & below & detect].shape[0]

    def nobs_below(row):
        if False:
            i = 10
            return i + 15
        ' B, the number of observations (cen & uncen) below the given\n        threshold\n        '
        less_than = df[observations] < row['lower_dl']
        less_thanequal = df[observations] <= row['lower_dl']
        uncensored = ~df[censorship]
        censored = df[censorship]
        LTE_censored = df[less_thanequal & censored].shape[0]
        LT_uncensored = df[less_than & uncensored].shape[0]
        return LTE_censored + LT_uncensored

    def ncen_equal(row):
        if False:
            i = 10
            return i + 15
        ' C, the number of censored observations at the given\n        threshold.\n        '
        censored_index = df[censorship]
        censored_data = df[observations][censored_index]
        censored_below = censored_data == row['lower_dl']
        return censored_below.sum()

    def set_upper_limit(cohn):
        if False:
            i = 10
            return i + 15
        ' Sets the upper_dl DL for each row of the Cohn dataframe. '
        if cohn.shape[0] > 1:
            return cohn['lower_dl'].shift(-1).fillna(value=np.inf)
        else:
            return [np.inf]

    def compute_PE(A, B):
        if False:
            for i in range(10):
                print('nop')
        ' Computes the probability of excedance for each row of the\n        Cohn dataframe. '
        N = len(A)
        PE = np.empty(N, dtype='float64')
        PE[-1] = 0.0
        for j in range(N - 2, -1, -1):
            PE[j] = PE[j + 1] + (1 - PE[j + 1]) * A[j] / (A[j] + B[j])
        return PE
    censored_data = df[censorship]
    DLs = pd.unique(df.loc[censored_data, observations])
    DLs.sort()
    if DLs.shape[0] > 0:
        if df[observations].min() < DLs.min():
            DLs = np.hstack([df[observations].min(), DLs])
        cohn = pd.DataFrame(DLs, columns=['lower_dl'])
        cohn.loc[:, 'upper_dl'] = set_upper_limit(cohn)
        cohn.loc[:, 'nuncen_above'] = cohn.apply(nuncen_above, axis=1)
        cohn.loc[:, 'nobs_below'] = cohn.apply(nobs_below, axis=1)
        cohn.loc[:, 'ncen_equal'] = cohn.apply(ncen_equal, axis=1)
        cohn = cohn.reindex(range(DLs.shape[0] + 1))
        cohn.loc[:, 'prob_exceedance'] = compute_PE(cohn['nuncen_above'], cohn['nobs_below'])
    else:
        dl_cols = ['lower_dl', 'upper_dl', 'nuncen_above', 'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = pd.DataFrame(np.empty((0, len(dl_cols))), columns=dl_cols)
    return cohn

def _detection_limit_index(obs, cohn):
    if False:
        while True:
            i = 10
    '\n    Locates the corresponding detection limit for each observation.\n\n    Basically, creates an array of indices for the detection limits\n    (Cohn numbers) corresponding to each data point.\n\n    Parameters\n    ----------\n    obs : float\n        A single observation from the larger dataset.\n\n    cohn : DataFrame\n        DataFrame of Cohn numbers.\n\n    Returns\n    -------\n    det_limit_index : int\n        The index of the corresponding detection limit in `cohn`\n\n    See Also\n    --------\n    cohn_numbers\n    '
    if cohn.shape[0] > 0:
        (index,) = np.where(cohn['lower_dl'] <= obs)
        det_limit_index = index[-1]
    else:
        det_limit_index = 0
    return det_limit_index

def _ros_group_rank(df, dl_idx, censorship):
    if False:
        while True:
            i = 10
    "\n    Ranks each observation within the data groups.\n\n    In this case, the groups are defined by the record's detection\n    limit index and censorship status.\n\n    Parameters\n    ----------\n    df : DataFrame\n\n    dl_idx : str\n        Name of the column in the dataframe the index of the\n        observations' corresponding detection limit in the `cohn`\n        dataframe.\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    Returns\n    -------\n    ranks : ndarray\n        Array of ranks for the dataset.\n    "
    ranks = df.copy()
    ranks.loc[:, 'rank'] = 1
    ranks = ranks.groupby(by=[dl_idx, censorship])['rank'].transform(lambda g: g.cumsum())
    return ranks

def _ros_plot_pos(row, censorship, cohn):
    if False:
        return 10
    "\n    ROS-specific plotting positions.\n\n    Computes the plotting position for an observation based on its rank,\n    censorship status, and detection limit index.\n\n    Parameters\n    ----------\n    row : {Series, dict}\n        Full observation (row) from a censored dataset. Requires a\n        'rank', 'detection_limit', and `censorship` column.\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    cohn : DataFrame\n        DataFrame of Cohn numbers.\n\n    Returns\n    -------\n    plotting_position : float\n\n    See Also\n    --------\n    cohn_numbers\n    "
    DL_index = row['det_limit_index']
    rank = row['rank']
    censored = row[censorship]
    dl_1 = cohn.iloc[DL_index]
    dl_2 = cohn.iloc[DL_index + 1]
    if censored:
        return (1 - dl_1['prob_exceedance']) * rank / (dl_1['ncen_equal'] + 1)
    else:
        return 1 - dl_1['prob_exceedance'] + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * rank / (dl_1['nuncen_above'] + 1)

def _norm_plot_pos(observations):
    if False:
        while True:
            i = 10
    '\n    Computes standard normal (Gaussian) plotting positions using scipy.\n\n    Parameters\n    ----------\n    observations : array_like\n        Sequence of observed quantities.\n\n    Returns\n    -------\n    plotting_position : array of floats\n    '
    (ppos, sorted_res) = stats.probplot(observations, fit=False)
    return stats.norm.cdf(ppos)

def plotting_positions(df, censorship, cohn):
    if False:
        print('Hello World!')
    "\n    Compute the plotting positions for the observations.\n\n    The ROS-specific plotting postions are based on the observations'\n    rank, censorship status, and corresponding detection limit.\n\n    Parameters\n    ----------\n    df : DataFrame\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    cohn : DataFrame\n        DataFrame of Cohn numbers.\n\n    Returns\n    -------\n    plotting_position : array of float\n\n    See Also\n    --------\n    cohn_numbers\n    "
    plot_pos = df.apply(lambda r: _ros_plot_pos(r, censorship, cohn), axis=1)
    ND_plotpos = plot_pos[df[censorship]]
    ND_plotpos_arr = np.require(ND_plotpos, requirements='W')
    ND_plotpos_arr.sort()
    plot_pos.loc[df[censorship].index[df[censorship]]] = ND_plotpos_arr
    return plot_pos

def _impute(df, observations, censorship, transform_in, transform_out):
    if False:
        print('Hello World!')
    '\n    Executes the basic regression on order stat (ROS) proceedure.\n\n    Uses ROS to impute censored from the best-fit line of a\n    probability plot of the uncensored values.\n\n    Parameters\n    ----------\n    df : DataFrame\n    observations : str\n        Name of the column in the dataframe that contains observed\n        values. Censored values should be set to the detection (upper)\n        limit.\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n    transform_in, transform_out : callable\n        Transformations to be applied to the data prior to fitting\n        the line and after estimated values from that line. Typically,\n        `np.log` and `np.exp` are used, respectively.\n\n    Returns\n    -------\n    estimated : DataFrame\n        A new dataframe with two new columns: "estimated" and "final".\n        The "estimated" column contains of the values inferred from the\n        best-fit line. The "final" column contains the estimated values\n        only where the original observations were censored, and the original\n        observations everwhere else.\n    '
    uncensored_mask = ~df[censorship]
    censored_mask = df[censorship]
    fit_params = stats.linregress(df['Zprelim'][uncensored_mask], transform_in(df[observations][uncensored_mask]))
    (slope, intercept) = fit_params[:2]
    df.loc[:, 'estimated'] = transform_out(slope * df['Zprelim'][censored_mask] + intercept)
    df.loc[:, 'final'] = np.where(df[censorship], df['estimated'], df[observations])
    return df

def _do_ros(df, observations, censorship, transform_in, transform_out):
    if False:
        print('Hello World!')
    '\n    DataFrame-centric function to impute censored valies with ROS.\n\n    Prepares a dataframe for, and then esimates the values of a censored\n    dataset using Regression on Order Statistics\n\n    Parameters\n    ----------\n    df : DataFrame\n\n    observations : str\n        Name of the column in the dataframe that contains observed\n        values. Censored values should be set to the detection (upper)\n        limit.\n\n    censorship : str\n        Name of the column in the dataframe that indicates that a\n        observation is left-censored. (i.e., True -> censored,\n        False -> uncensored)\n\n    transform_in, transform_out : callable\n        Transformations to be applied to the data prior to fitting\n        the line and after estimated values from that line. Typically,\n        `np.log` and `np.exp` are used, respectively.\n\n    Returns\n    -------\n    estimated : DataFrame\n        A new dataframe with two new columns: "estimated" and "final".\n        The "estimated" column contains of the values inferred from the\n        best-fit line. The "final" column contains the estimated values\n        only where the original observations were censored, and the original\n        observations everwhere else.\n    '
    cohn = cohn_numbers(df, observations=observations, censorship=censorship)
    modeled = _ros_sort(df, observations=observations, censorship=censorship)
    modeled.loc[:, 'det_limit_index'] = modeled[observations].apply(_detection_limit_index, args=(cohn,))
    modeled.loc[:, 'rank'] = _ros_group_rank(modeled, 'det_limit_index', censorship)
    modeled.loc[:, 'plot_pos'] = plotting_positions(modeled, censorship, cohn)
    modeled.loc[:, 'Zprelim'] = stats.norm.ppf(modeled['plot_pos'])
    return _impute(modeled, observations, censorship, transform_in, transform_out)

def impute_ros(observations, censorship, df=None, min_uncensored=2, max_fraction_censored=0.8, substitution_fraction=0.5, transform_in=np.log, transform_out=np.exp, as_array=True):
    if False:
        print('Hello World!')
    '\n    Impute censored dataset using Regression on Order Statistics (ROS).\n\n    Method described in *Nondetects and Data Analysis* by Dennis R.\n    Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)\n    values of a dataset. When there is insufficient non-censorded data,\n    simple substitution is used.\n\n    Parameters\n    ----------\n    observations : str or array-like\n        Label of the column or the float array of censored observations\n\n    censorship : str\n        Label of the column or the bool array of the censorship\n        status of the observations.\n\n          * True if censored,\n          * False if uncensored\n\n    df : DataFrame, optional\n        If `observations` and `censorship` are labels, this is the\n        DataFrame that contains those columns.\n\n    min_uncensored : int (default is 2)\n        The minimum number of uncensored values required before ROS\n        can be used to impute the censored observations. When this\n        criterion is not met, simple substituion is used instead.\n\n    max_fraction_censored : float (default is 0.8)\n        The maximum fraction of censored data below which ROS can be\n        used to impute the censored observations. When this fraction is\n        exceeded, simple substituion is used instead.\n\n    substitution_fraction : float (default is 0.5)\n        The fraction of the detection limit to be used during simple\n        substitution of the censored values.\n\n    transform_in : callable (default is np.log)\n        Transformation to be applied to the values prior to fitting a\n        line to the plotting positions vs. uncensored values.\n\n    transform_out : callable (default is np.exp)\n        Transformation to be applied to the imputed censored values\n        estimated from the previously computed best-fit line.\n\n    as_array : bool (default is True)\n        When True, a numpy array of the imputed observations is\n        returned. Otherwise, a modified copy of the original dataframe\n        with all of the intermediate calculations is returned.\n\n    Returns\n    -------\n    imputed : {ndarray, DataFrame}\n        The final observations where the censored values have either been\n        imputed through ROS or substituted as a fraction of the\n        detection limit.\n\n    Notes\n    -----\n    This function requires pandas 0.14 or more recent.\n    '
    if df is None:
        df = pd.DataFrame({'obs': observations, 'cen': censorship})
        observations = 'obs'
        censorship = 'cen'
    N_observations = df.shape[0]
    N_censored = df[censorship].astype(int).sum()
    N_uncensored = N_observations - N_censored
    fraction_censored = N_censored / N_observations
    if N_censored == 0:
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]
    elif N_uncensored < min_uncensored or fraction_censored > max_fraction_censored:
        output = df[[observations, censorship]].copy()
        output.loc[:, 'final'] = df[observations]
        output.loc[df[censorship], 'final'] *= substitution_fraction
    else:
        output = _do_ros(df, observations, censorship, transform_in, transform_out)
    if as_array:
        output = output['final'].values
    return output