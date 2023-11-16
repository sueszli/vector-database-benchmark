from numpy import ndarray
import numpy as np
from bigdl.nano.utils.common import invalidInputError
from timeit import repeat
import random
EPSILON = 1e-10

def mae(y_label, y_predict):
    if False:
        while True:
            i = 10
    '\n    Calculate the mean absolute error (MAE).\n\n    .. math::\n\n        \\text{MAE} = \\frac{1}{n}\\sum_{t=1}^n |y_t-\\hat{y_t}|\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 0.0).\n    '
    result = np.mean(np.abs(y_label - y_predict))
    return result

def mse(y_label, y_predict):
    if False:
        return 10
    '\n    Calculate the mean squared error (MSE).\n\n    .. math::\n\n        \\text{MSE} = \\frac{1}{n}\\sum_{t=1}^n (y_t-\\hat{y_t})^2\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 0.0).\n    '
    result = np.mean((y_label - y_predict) ** 2)
    return result

def rmse(y_label, y_predict):
    if False:
        print('Hello World!')
    '\n    Calculate square root of the mean squared error (RMSE).\n\n    .. math::\n\n        \\text{RMSE} = \\sqrt{(\\frac{1}{n}\\sum_{t=1}^n (y_t-\\hat{y_t})^2)}\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 0.0).\n    '
    return np.sqrt(mse(y_label, y_predict))

def mape(y_label, y_predict):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate mean absolute percentage error (MAPE).\n\n    .. math::\n\n        \\text{MAPE} = \\frac{100\\%}{n}\\sum_{t=1}^n  |\\frac{y_t-\\hat{y_t}}{y_t}|\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 0.0).\n    '
    return np.mean(np.abs((y_label - y_predict) / (y_label + EPSILON)))

def smape(y_label, y_predict):
    if False:
        print('Hello World!')
    '\n    Calculate Symmetric mean absolute percentage error (sMAPE).\n\n    .. math::\n\n        \\text{sMAPE} = \\frac{100\\%}{n} \\sum_{t=1}^n \\frac{|y_t-\\hat{y_t}|}{|y_t|+|\\hat{y_t}|}\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 0.0).\n    '
    abs_diff = np.abs(y_predict - y_label)
    abs_per_error = abs_diff / (np.abs(y_predict) + np.abs(y_label) + EPSILON)
    sum_abs_per_error = np.mean(abs_per_error)
    return sum_abs_per_error * 100

def r2(y_label, y_predict):
    if False:
        print('Hello World!')
    '\n    Calculate the r2 score.\n\n    .. math::\n\n        R^2 = 1-\\frac{\\sum_{t=1}^n (y_t-\\hat{y_t})^2}{\\sum_{t=1}^n (y_t-\\bar{y})^2}\n\n    :param y_label: Array-like of shape = (n_samples, \\*).\n           Ground truth (correct) target values.\n    :param y_predict: Array-like of shape = (n_samples, \\*).\n           Estimated target values.\n    :return: Ndarray of floats.\n             An array of non-negative floating point values (the best value is 1.0).\n    '
    return 1 - np.sum((y_label - y_predict) ** 2) / np.sum((y_label - np.mean(y_label)) ** 2)
REGRESSION_MAP = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'smape': smape, 'r2': r2}

def _standard_input(metrics, y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    '\n    Standardize input functions. Format metrics,\n    check the ndim of y_pred and y_true,\n    converting 1-3 dim y_true and y_pred to 2 dim.\n    '
    if not isinstance(metrics, list):
        metrics = [metrics]
    if isinstance(metrics[0], str):
        metrics = list(map(lambda x: x.lower(), metrics))
        invalidInputError(all((metric in REGRESSION_MAP.keys() for metric in metrics)), f'metric should be one of {REGRESSION_MAP.keys()}, but get {metrics}.')
        invalidInputError(type(y_true) is type(y_pred) and isinstance(y_pred, ndarray), f'y_pred and y_true type must be numpy.ndarray, but found {type(y_pred)} and {type(y_true)}.')
    invalidInputError(y_true.shape == y_pred.shape, f'y_true and y_pred should have the same shape, but get {y_true.shape} and {y_pred.shape}.')
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        original_shape = y_true.shape
    elif y_true.ndim == 3:
        original_shape = y_true.shape
        y_true = y_true.reshape(y_true.shape[0], y_true.shape[1] * y_true.shape[2])
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1] * y_pred.shape[2])
    else:
        original_shape = y_true.shape
    return (metrics, y_true, y_pred, original_shape)

def _check_shape(input1, input2, input_name1, input_name2):
    if False:
        print('Hello World!')
    invalidInputError(input1.shape == input2.shape, f'{input_name1} does not have same input as {input_name2}, {input_name1} has a shape as {input1.shape} while {input_name2} has a shape as {input2.shape}.')

class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred.
    """

    @staticmethod
    def evaluate(metrics, y_true, y_pred, aggregate='mean'):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate a specific metrics for y_true and y_pred.\n        :param metrics: String or list in [\'mae\', \'mse\', \'rmse\', \'r2\', \'mape\', \'smape\'] for built-in\n               metrics. If callable function, it signature should be func(y_true, y_pred), where\n               y_true and y_pred are numpy ndarray.\n        :param y_true: Array-like of shape = (n_samples, \\*). Ground truth (correct) target values.\n        :param y_pred: Array-like of shape = (n_samples, \\*). Estimated target values.\n        :param aggregate: aggregation method. Currently, "mean" and None are supported,\n               \'mean\' represents aggregating by mean, while None will return the element-wise\n               result. The value defaults to \'mean\'.\n        :return: Float or ndarray of floats.\n                 A floating point value, or an\n                 array of floating point values, one for each individual target.\n        '
        (metrics, y_true, y_pred, original_shape) = _standard_input(metrics, y_true, y_pred)
        res_list = []
        for metric in metrics:
            if callable(metric):
                metric_func = metric
            else:
                metric_func = REGRESSION_MAP[metric]
            if len(original_shape) in [2, 3] and aggregate is None:
                res = np.zeros(y_true.shape[-1])
                for i in range(y_true.shape[-1]):
                    res[i] = metric_func(y_true[..., i], y_pred[..., i])
                res = res.reshape(original_shape[1:])
                res_list.append(res)
            else:
                res = metric_func(y_true, y_pred)
                res_list.append(res)
        return res_list

    @staticmethod
    def get_latency(func, *args, num_running=100, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Return the time cost in milliseconds of a specific function by running multiple times.\n\n        :param func: The function to be tested for latency.\n        :param args: arguments for the tested function.\n        :param num_running: Int and the value is positive. Specify the running number of\n               the function and the value defaults to 100.\n        :param kwargs: other arguments for the tested function.\n\n        :return: Dictionary of str:float.\n                 Show the information of the time cost in milliseconds.\n\n        Example:\n            >>> # to get the inferencing performance of a trained TCNForecaster\n            >>> x = next(iter(test_loader))[0]\n            >>> # run forecaster.predict(x.numpy()) for len(tsdata_test.df) times\n            >>> # to evaluate the time cost\n            >>> latency = Evaluator.get_latency(forecaster.predict, x.numpy(),                num_running = len(tsdata_test.df))\n            >>> # an example output:\n            >>> # {"p50": 3.853, "p90": 3.881, "p95": 3.933, "p99": 4.107}\n        '
        invalidInputError(isinstance(num_running, int), f'num_running type must be int, but found {type(num_running)}.')
        if num_running < 0:
            invalidInputError(False, f'num_running value must be positive, but found {num_running}.')
        time_list = repeat(lambda : func(*args, **kwargs), number=1, repeat=num_running)
        sorted_time = np.sort(time_list)
        latency_list = {'p50': round(1000 * np.median(time_list), 3), 'p90': round(1000 * sorted_time[int(0.9 * num_running)], 3), 'p95': round(1000 * sorted_time[int(0.95 * num_running)], 3), 'p99': round(1000 * sorted_time[int(0.99 * num_running)], 3)}
        return latency_list

    @staticmethod
    def plot(y, std=None, ground_truth=None, x=None, feature_index=0, instance_index=None, layout=(1, 1), prediction_interval=0.95, figsize=(16, 16), output_file=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        `Evaluator.plot` function helps users to visualize their forecasting result.\n\n        :param y: predict result, a 3-dim numpy ndarray with shape represented as\n               (batch_size, predict_length, output_feature_dim).\n        :param std: standard deviation, a 3-dim numpy ndarray with shape represented\n               as (batch_size, predict_length, output_feature_dim). Same shape as `y`.\n        :param ground_truth: ground truth, a 3-dim numpy ndarray with shape represented as\n               (batch_size, predict_length, output_feature_dim). Same shape as `y`.\n        :param x: input numpy array, a 3-dim numpy ndarray with shape represented\n               as (batch_size, lookback_length, input_feature_dim).\n        :param feature_index: int, the feature index (along last dim) to plot.\n               Default to the first feature.\n        :param instance_index: int/tuple/list, the instance index to show. Default to None\n               which represents random number.\n        :param layout: a 2-dim tuple, indicate the row_num and col_num to plot.\n        :param prediction_internval: a float, indicates the confidence percentile. Default to\n               0.95 refer to 95% confidence. This only effective when `std` is not None.\n        :param figsize: figure size to be inputed to pyplot. Default to (16,16).\n        :param output_file: a path, indicates the save path of the output plot. Default to\n               None, indicates no output file is needed.\n        :param **kwargs: other paramters will be passed to matplotlib.pyplot.\n        '
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            invalidInputError(False, 'To enable visualization, you need to install matplotlib by:\n\t\t pip install matplotlib\n')
        if std is not None:
            _check_shape(y, std, 'y', 'std')
        if ground_truth is not None:
            _check_shape(y, ground_truth, 'y', 'ground_truth')
        invalidInputError(len(layout) == 2, f'len of `layout` should be 2 while get {len(layout)}')
        batch_num = y.shape[0]
        horizon = y.shape[1]
        lookback = 0 if x is None else x.shape[1]
        row_num = 1 if layout is None else layout[0]
        col_num = 1 if layout is None else layout[1]
        y_index = list(range(lookback, horizon + lookback))
        x_index = list(range(0, lookback))
        iter_num = 1
        instance_index_iter = iter(instance_index) if instance_index is not None else None
        plt.figure(figsize=figsize, **kwargs)
        for row_iter in range(1, row_num + 1):
            for col_iter in range(1, col_num + 1):
                if instance_index_iter is None:
                    instance_index = random.randint(0, y.shape[0] - 1)
                else:
                    try:
                        instance_index = next(instance_index_iter)
                    except e:
                        continue
                ax = plt.subplot(row_num, col_num, iter_num)
                ax.plot(y_index, y[instance_index, :, feature_index], color='royalblue')
                if ground_truth is not None:
                    ax.plot(y_index, ground_truth[instance_index, :, feature_index], color='limegreen')
                if x is not None:
                    ax.plot(x_index, x[instance_index, :, feature_index], color='black')
                    ax.plot([x_index[-1], y_index[0]], np.array([x[instance_index, -1, feature_index], y[instance_index, 0, feature_index]]), color='royalblue')
                    if ground_truth is not None:
                        ax.plot([x_index[-1], y_index[0]], np.array([x[instance_index, -1, feature_index], ground_truth[instance_index, 0, feature_index]]), color='limegreen')
                if std is not None:
                    import scipy.stats
                    ppf_value = scipy.stats.norm.ppf(prediction_interval)
                    ax.fill_between(y_index, y[instance_index, :, feature_index] - std[instance_index, :, feature_index] * ppf_value, y[instance_index, :, feature_index] + std[instance_index, :, feature_index] * ppf_value, alpha=0.2)
                if ground_truth is not None:
                    ax.legend(['prediction', 'ground truth'])
                else:
                    ax.legend(['prediction'])
                ax.set_title(f'index {instance_index}')
                iter_num += 1
        if output_file:
            plt.savefig(output_file)