from __future__ import absolute_import, division, print_function
import logging
from tqdm.auto import tqdm
from copy import deepcopy
import concurrent.futures
import numpy as np
import pandas as pd
logger = logging.getLogger('prophet')

def generate_cutoffs(df, horizon, initial, period):
    if False:
        i = 10
        return i + 15
    'Generate cutoff dates\n\n    Parameters\n    ----------\n    df: pd.DataFrame with historical data.\n    horizon: pd.Timedelta forecast horizon.\n    initial: pd.Timedelta window of the initial forecast period.\n    period: pd.Timedelta simulated forecasts are done with this period.\n\n    Returns\n    -------\n    list of pd.Timestamp\n    '
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]
    while result[-1] >= min(df['ds']) + initial:
        cutoff -= period
        if not ((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any():
            if cutoff > df['ds'].min():
                closest_date = df[df['ds'] <= cutoff].max()['ds']
                cutoff = closest_date - horizon
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError('Less data than horizon after initial window. Make horizon or initial shorter.')
    logger.info('Making {} forecasts with cutoffs between {} and {}'.format(len(result), result[-1], result[0]))
    return list(reversed(result))

def cross_validation(model, horizon, period=None, initial=None, parallel=None, cutoffs=None, disable_tqdm=False, extra_output_columns=None):
    if False:
        print('Hello World!')
    "Cross-Validation for time series.\n\n    Computes forecasts from historical cutoff points, which user can input.\n    If not provided, begins from (end - horizon) and works backwards, making\n    cutoffs with a spacing of period until initial is reached.\n\n    When period is equal to the time interval of the data, this is the\n    technique described in https://robjhyndman.com/hyndsight/tscv/ .\n\n    Parameters\n    ----------\n    model: Prophet class object. Fitted Prophet model.\n    horizon: string with pd.Timedelta compatible style, e.g., '5 days',\n        '3 hours', '10 seconds'.\n    period: string with pd.Timedelta compatible style. Simulated forecast will\n        be done at every this period. If not provided, 0.5 * horizon is used.\n    initial: string with pd.Timedelta compatible style. The first training\n        period will include at least this much data. If not provided,\n        3 * horizon is used.\n    cutoffs: list of pd.Timestamp specifying cutoffs to be used during\n        cross validation. If not provided, they are generated as described\n        above.\n    parallel : {None, 'processes', 'threads', 'dask', object}\n        How to parallelize the forecast computation. By default no parallelism\n        is used.\n\n        * None : No parallelism.\n        * 'processes' : Parallelize with concurrent.futures.ProcessPoolExectuor.\n        * 'threads' : Parallelize with concurrent.futures.ThreadPoolExecutor.\n            Note that some operations currently hold Python's Global Interpreter\n            Lock, so parallelizing with threads may be slower than training\n            sequentially.\n        * 'dask': Parallelize with Dask.\n           This requires that a dask.distributed Client be created.\n        * object : Any instance with a `.map` method. This method will\n          be called with :func:`single_cutoff_forecast` and a sequence of\n          iterables where each element is the tuple of arguments to pass to\n          :func:`single_cutoff_forecast`\n\n          .. code-block::\n\n             class MyBackend:\n                 def map(self, func, *iterables):\n                     results = [\n                        func(*args)\n                        for args in zip(*iterables)\n                     ]\n                     return results\n                     \n    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None\n    extra_output_columns: A String or List of Strings e.g. 'trend' or ['trend'].\n         Additional columns to 'yhat' and 'ds' to be returned in output.\n\n    Returns\n    -------\n    A pd.DataFrame with the forecast, actual value and cutoff.\n    "
    if model.history is None:
        raise Exception('Model has not been fit. Fitting the model provides contextual parameters for cross validation.')
    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)
    predict_columns = ['ds', 'yhat']
    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])
    if extra_output_columns is not None:
        if isinstance(extra_output_columns, str):
            extra_output_columns = [extra_output_columns]
        predict_columns.extend([c for c in extra_output_columns if c not in predict_columns])
    period_max = 0.0
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = pd.Timedelta(str(period_max) + ' days')
    if cutoffs is None:
        period = 0.5 * horizon if period is None else pd.Timedelta(period)
        initial = max(3 * horizon, seasonality_dt) if initial is None else pd.Timedelta(initial)
        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else:
        if min(cutoffs) <= df['ds'].min():
            raise ValueError('Minimum cutoff value is not strictly greater than min date in history')
        end_date_minus_horizon = df['ds'].max() - horizon
        if max(cutoffs) > end_date_minus_horizon:
            raise ValueError('Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining')
        initial = cutoffs[0] - df['ds'].min()
    if initial < seasonality_dt:
        msg = 'Seasonality has period of {} days '.format(period_max)
        msg += 'which is larger than initial window. '
        msg += 'Consider increasing initial.'
        logger.warning(msg)
    if parallel:
        valid = {'threads', 'processes', 'dask'}
        if parallel == 'threads':
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == 'processes':
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == 'dask':
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requires the optional dependency dask.") from e
            pool = get_client()
            (df, model) = pool.scatter([df, model])
        elif hasattr(parallel, 'map'):
            pool = parallel
        else:
            msg = "'parallel' should be one of {} for an instance with a 'map' method".format(', '.join(valid))
            raise ValueError(msg)
        iterables = ((df, model, cutoff, horizon, predict_columns) for cutoff in cutoffs)
        iterables = zip(*iterables)
        logger.info('Applying in parallel with %s', pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == 'dask':
            predicts = pool.gather(predicts)
    else:
        predicts = [single_cutoff_forecast(df, model, cutoff, horizon, predict_columns) for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)]
    return pd.concat(predicts, axis=0).reset_index(drop=True)

def single_cutoff_forecast(df, model, cutoff, horizon, predict_columns):
    if False:
        i = 10
        return i + 15
    "Forecast for single cutoff. Used in cross validation function\n    when evaluating for multiple cutoffs either sequentially or in parallel .\n\n    Parameters\n    ----------\n    df: pd.DataFrame.\n        DataFrame with history to be used for single\n        cutoff forecast.\n    model: Prophet model object.\n    cutoff: pd.Timestamp cutoff date.\n        Simulated Forecast will start from this date.\n    horizon: pd.Timedelta forecast horizon.\n    predict_columns: List of strings e.g. ['ds', 'yhat'].\n        Columns with date and forecast to be returned in output.\n\n    Returns\n    -------\n    A pd.DataFrame with the forecast, actual value and cutoff.\n\n    "
    m = prophet_copy(model, cutoff)
    history_c = df[df['ds'] <= cutoff]
    if history_c.shape[0] < 2:
        raise Exception('Less than two datapoints before cutoff. Increase initial window.')
    m.fit(history_c, **model.fit_kwargs)
    index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
    columns = ['ds']
    if m.growth == 'logistic':
        columns.append('cap')
        if m.logistic_floor:
            columns.append('floor')
    columns.extend(m.extra_regressors.keys())
    columns.extend([props['condition_name'] for props in m.seasonalities.values() if props['condition_name'] is not None])
    yhat = m.predict(df[index_predicted][columns])
    return pd.concat([yhat[predict_columns], df[index_predicted][['y']].reset_index(drop=True), pd.DataFrame({'cutoff': [cutoff] * len(yhat)})], axis=1)

def prophet_copy(m, cutoff=None):
    if False:
        while True:
            i = 10
    "Copy Prophet object\n\n    Parameters\n    ----------\n    m: Prophet model.\n    cutoff: pd.Timestamp or None, default None.\n        cuttoff Timestamp for changepoints member variable.\n        changepoints are only retained if 'changepoints <= cutoff'\n\n    Returns\n    -------\n    Prophet class object with the same parameter with model variable\n    "
    if m.history is None:
        raise Exception('This is for copying a fitted Prophet object.')
    if m.specified_changepoints:
        changepoints = m.changepoints
        if cutoff is not None:
            last_history_date = max(m.history['ds'][m.history['ds'] <= cutoff])
            changepoints = changepoints[changepoints < last_history_date]
    else:
        changepoints = None
    m2 = m.__class__(growth=m.growth, n_changepoints=m.n_changepoints, changepoint_range=m.changepoint_range, changepoints=changepoints, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, holidays=m.holidays, holidays_mode=m.holidays_mode, seasonality_mode=m.seasonality_mode, seasonality_prior_scale=m.seasonality_prior_scale, changepoint_prior_scale=m.changepoint_prior_scale, holidays_prior_scale=m.holidays_prior_scale, mcmc_samples=m.mcmc_samples, interval_width=m.interval_width, uncertainty_samples=m.uncertainty_samples, stan_backend=m.stan_backend.get_type() if m.stan_backend is not None else None)
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.country_holidays = deepcopy(m.country_holidays)
    return m2

def performance_metrics(df, metrics=None, rolling_window=0.1, monthly=False):
    if False:
        print('Hello World!')
    "Compute performance metrics from cross-validation results.\n\n    Computes a suite of performance metrics on the output of cross-validation.\n    By default the following metrics are included:\n    'mse': mean squared error\n    'rmse': root mean squared error\n    'mae': mean absolute error\n    'mape': mean absolute percent error\n    'mdape': median absolute percent error\n    'smape': symmetric mean absolute percentage error\n    'coverage': coverage of the upper and lower intervals\n\n    A subset of these can be specified by passing a list of names as the\n    `metrics` argument.\n\n    Metrics are calculated over a rolling window of cross validation\n    predictions, after sorting by horizon. Averaging is first done within each\n    value of horizon, and then across horizons as needed to reach the window\n    size. The size of that window (number of simulated forecast points) is\n    determined by the rolling_window argument, which specifies a proportion of\n    simulated forecast points to include in each window. rolling_window=0 will\n    compute it separately for each horizon. The default of rolling_window=0.1\n    will use 10% of the rows in df in each window. rolling_window=1 will\n    compute the metric across all simulated forecast points. The results are\n    set to the right edge of the window.\n\n    If rolling_window < 0, then metrics are computed at each datapoint with no\n    averaging (i.e., 'mse' will actually be squared error with no mean).\n\n    The output is a dataframe containing column 'horizon' along with columns\n    for each of the metrics computed.\n\n    Parameters\n    ----------\n    df: The dataframe returned by cross_validation.\n    metrics: A list of performance metrics to compute. If not provided, will\n        use ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'].\n    rolling_window: Proportion of data to use in each rolling window for\n        computing the metrics. Should be in [0, 1] to average.\n    monthly: monthly=True will compute horizons as numbers of calendar months \n        from the cutoff date, starting from 0 for the cutoff month.\n\n    Returns\n    -------\n    Dataframe with a column for each metric, and column 'horizon'\n    "
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if ('yhat_lower' not in df or 'yhat_upper' not in df) and 'coverage' in metrics:
        metrics.remove('coverage')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(valid_metrics)):
        raise ValueError('Valid values for metrics are: {}'.format(valid_metrics))
    df_m = df.copy()
    if monthly:
        df_m['horizon'] = df_m['ds'].dt.to_period('M').astype(int) - df_m['cutoff'].dt.to_period('M').astype(int)
    else:
        df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    df_m.sort_values('horizon', inplace=True)
    if 'mape' in metrics and df_m['y'].abs().min() < 1e-08:
        logger.info('Skipping MAPE because y close to 0')
        metrics.remove('mape')
    if len(metrics) == 0:
        return None
    w = int(rolling_window * df_m.shape[0])
    if w >= 0:
        w = max(w, 1)
        w = min(w, df_m.shape[0])
    dfs = {}
    for metric in metrics:
        dfs[metric] = eval(metric)(df_m, w)
    res = dfs[metrics[0]]
    for i in range(1, len(metrics)):
        res_m = dfs[metrics[i]]
        assert np.array_equal(res['horizon'].values, res_m['horizon'].values)
        res[metrics[i]] = res_m[metrics[i]]
    return res

def rolling_mean_by_h(x, h, w, name):
    if False:
        print('Hello World!')
    'Compute a rolling mean of x, after first aggregating by h.\n\n    Right-aligned. Computes a single mean for each unique value of h. Each\n    mean is over at least w samples.\n\n    Parameters\n    ----------\n    x: Array.\n    h: Array of horizon for each value in x.\n    w: Integer window size (number of elements).\n    name: Name for metric in result dataframe\n\n    Returns\n    -------\n    Dataframe with columns horizon and name, the rolling mean of x.\n    '
    df = pd.DataFrame({'x': x, 'h': h})
    df2 = df.groupby('h').agg(['sum', 'count']).reset_index().sort_values('h')
    xs = df2['x']['sum'].values
    ns = df2['x']['count'].values
    hs = df2.h.values
    trailing_i = len(df2) - 1
    x_sum = 0
    n_sum = 0
    res_x = np.empty(len(df2))
    for i in range(len(df2) - 1, -1, -1):
        x_sum += xs[i]
        n_sum += ns[i]
        while n_sum >= w:
            excess_n = n_sum - w
            excess_x = excess_n * xs[i] / ns[i]
            res_x[trailing_i] = (x_sum - excess_x) / w
            x_sum -= xs[trailing_i]
            n_sum -= ns[trailing_i]
            trailing_i -= 1
    res_h = hs[trailing_i + 1:]
    res_x = res_x[trailing_i + 1:]
    return pd.DataFrame({'horizon': res_h, name: res_x})

def rolling_median_by_h(x, h, w, name):
    if False:
        return 10
    "Compute a rolling median of x, after first aggregating by h.\n\n    Right-aligned. Computes a single median for each unique value of h. Each\n    median is over at least w samples.\n\n    For each h where there are fewer than w samples, we take samples from the previous h,\n    moving backwards. (In other words, we ~ assume that the x's are shuffled within each h.)\n\n    Parameters\n    ----------\n    x: Array.\n    h: Array of horizon for each value in x.\n    w: Integer window size (number of elements).\n    name: Name for metric in result dataframe\n\n    Returns\n    -------\n    Dataframe with columns horizon and name, the rolling median of x.\n    "
    df = pd.DataFrame({'x': x, 'h': h})
    grouped = df.groupby('h')
    df2 = grouped.size().reset_index().sort_values('h')
    hs = df2['h']
    res_h = []
    res_x = []
    i = len(hs) - 1
    while i >= 0:
        h_i = hs[i]
        xs = grouped.get_group(h_i).x.tolist()
        next_idx_to_add = np.array(h == h_i).argmax() - 1
        while len(xs) < w and next_idx_to_add >= 0:
            xs.append(x[next_idx_to_add])
            next_idx_to_add -= 1
        if len(xs) < w:
            break
        res_h.append(hs[i])
        res_x.append(np.median(xs))
        i -= 1
    res_h.reverse()
    res_x.reverse()
    return pd.DataFrame({'horizon': res_h, name: res_x})

def mse(df, w):
    if False:
        while True:
            i = 10
    'Mean squared error\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and mse.\n    '
    se = (df['y'] - df['yhat']) ** 2
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mse': se})
    return rolling_mean_by_h(x=se.values, h=df['horizon'].values, w=w, name='mse')

def rmse(df, w):
    if False:
        return 10
    'Root mean squared error\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and rmse.\n    '
    res = mse(df, w)
    res['mse'] = np.sqrt(res['mse'])
    res.rename({'mse': 'rmse'}, axis='columns', inplace=True)
    return res

def mae(df, w):
    if False:
        return 10
    'Mean absolute error\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and mae.\n    '
    ae = np.abs(df['y'] - df['yhat'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mae': ae})
    return rolling_mean_by_h(x=ae.values, h=df['horizon'].values, w=w, name='mae')

def mape(df, w):
    if False:
        i = 10
        return i + 15
    'Mean absolute percent error\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and mape.\n    '
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mape': ape})
    return rolling_mean_by_h(x=ape.values, h=df['horizon'].values, w=w, name='mape')

def mdape(df, w):
    if False:
        for i in range(10):
            print('nop')
    'Median absolute percent error\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and mdape.\n    '
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mdape': ape})
    return rolling_median_by_h(x=ape.values, h=df['horizon'], w=w, name='mdape')

def smape(df, w):
    if False:
        while True:
            i = 10
    'Symmetric mean absolute percentage error\n    based on Chen and Yang (2004) formula\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and smape.\n    '
    sape = np.abs(df['y'] - df['yhat']) / ((np.abs(df['y']) + np.abs(df['yhat'])) / 2)
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'smape': sape})
    return rolling_mean_by_h(x=sape.values, h=df['horizon'].values, w=w, name='smape')

def coverage(df, w):
    if False:
        return 10
    'Coverage\n\n    Parameters\n    ----------\n    df: Cross-validation results dataframe.\n    w: Aggregation window size.\n\n    Returns\n    -------\n    Dataframe with columns horizon and coverage.\n    '
    is_covered = (df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'coverage': is_covered})
    return rolling_mean_by_h(x=is_covered.values, h=df['horizon'].values, w=w, name='coverage')