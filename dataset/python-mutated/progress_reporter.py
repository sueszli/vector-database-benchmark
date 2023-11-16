from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import AUTO_RESULT_KEYS, DEFAULT_METRIC, DONE, EPISODE_REWARD_MEAN, EXPERIMENT_TAG, MEAN_ACCURACY, MEAN_LOSS, NODE_IP, PID, TIME_TOTAL_S, TIMESTEPS_TOTAL, TRIAL_ID
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
try:
    from collections.abc import Mapping, MutableMapping
except ImportError:
    from collections import Mapping, MutableMapping
IS_NOTEBOOK = ray.widgets.util.in_notebook()
SKIP_RESULTS_IN_REPORT = {'config', TRIAL_ID, EXPERIMENT_TAG, DONE}

@PublicAPI
class ProgressReporter:
    """Abstract class for experiment progress reporting.

    `should_report()` is called to determine whether or not `report()` should
    be called. Tune will call these functions after trial state transitions,
    receiving training results, and so on.
    """

    def setup(self, start_time: Optional[float]=None, total_samples: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, **kwargs):
        if False:
            while True:
                i = 10
        'Setup progress reporter for a new Ray Tune run.\n\n        This function is used to initialize parameters that are set on runtime.\n        It will be called before any of the other methods.\n\n        Defaults to no-op.\n\n        Args:\n            start_time: Timestamp when the Ray Tune run is started.\n            total_samples: Number of samples the Ray Tune run will run.\n            metric: Metric to optimize.\n            mode: Must be one of [min, max]. Determines whether objective is\n                minimizing or maximizing the metric attribute.\n            **kwargs: Keyword arguments for forward-compatibility.\n        '
        pass

    def should_report(self, trials: List[Trial], done: bool=False):
        if False:
            print('Hello World!')
        'Returns whether or not progress should be reported.\n\n        Args:\n            trials: Trials to report on.\n            done: Whether this is the last progress report attempt.\n        '
        raise NotImplementedError

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        if False:
            return 10
        'Reports progress across trials.\n\n        Args:\n            trials: Trials to report on.\n            done: Whether this is the last progress report attempt.\n            sys_info: System info.\n        '
        raise NotImplementedError

@DeveloperAPI
class TuneReporterBase(ProgressReporter):
    """Abstract base class for the default Tune reporters.

    If metric_columns is not overridden, Tune will attempt to automatically
    infer the metrics being outputted, up to 'infer_limit' number of
    metrics.

    Args:
        metric_columns: Names of metrics to
            include in progress table. If this is a dict, the keys should
            be metric names and the values should be the displayed names.
            If this is a list, the metric name is used directly.
        parameter_columns: Names of parameters to
            include in progress table. If this is a dict, the keys should
            be parameter names and the values should be the displayed names.
            If this is a list, the parameter name is used directly. If empty,
            defaults to all available parameters.
        max_progress_rows: Maximum number of rows to print
            in the progress table. The progress table describes the
            progress of each trial. Defaults to 20.
        max_error_rows: Maximum number of rows to print in the
            error table. The error table lists the error file, if any,
            corresponding to each trial. Defaults to 20.
        max_column_length: Maximum column length (in characters). Column
            headers and values longer than this will be abbreviated.
        max_report_frequency: Maximum report frequency in seconds.
            Defaults to 5s.
        infer_limit: Maximum number of metrics to automatically infer
            from tune results.
        print_intermediate_tables: Print intermediate result
            tables. If None (default), will be set to True for verbosity
            levels above 3, otherwise False. If True, intermediate tables
            will be printed with experiment progress. If False, tables
            will only be printed at then end of the tuning run for verbosity
            levels greater than 2.
        metric: Metric used to determine best current trial.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """
    DEFAULT_COLUMNS = collections.OrderedDict({MEAN_ACCURACY: 'acc', MEAN_LOSS: 'loss', TRAINING_ITERATION: 'iter', TIME_TOTAL_S: 'total time (s)', TIMESTEPS_TOTAL: 'ts', EPISODE_REWARD_MEAN: 'reward'})
    VALID_SUMMARY_TYPES = {int, float, np.float32, np.float64, np.int32, np.int64, type(None)}

    def __init__(self, *, metric_columns: Optional[Union[List[str], Dict[str, str]]]=None, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: Optional[int]=None, max_progress_rows: int=20, max_error_rows: int=20, max_column_length: int=20, max_report_frequency: int=5, infer_limit: int=3, print_intermediate_tables: Optional[bool]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
        if False:
            i = 10
            return i + 15
        self._total_samples = total_samples
        self._metrics_override = metric_columns is not None
        self._inferred_metrics = {}
        self._metric_columns = metric_columns or self.DEFAULT_COLUMNS.copy()
        self._parameter_columns = parameter_columns or []
        self._max_progress_rows = max_progress_rows
        self._max_error_rows = max_error_rows
        self._max_column_length = max_column_length
        self._infer_limit = infer_limit
        if print_intermediate_tables is None:
            self._print_intermediate_tables = has_verbosity(Verbosity.V3_TRIAL_DETAILS)
        else:
            self._print_intermediate_tables = print_intermediate_tables
        self._max_report_freqency = max_report_frequency
        self._last_report_time = 0
        self._start_time = time.time()
        self._metric = metric
        self._mode = mode
        self._sort_by_metric = sort_by_metric

    def setup(self, start_time: Optional[float]=None, total_samples: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, **kwargs):
        if False:
            print('Hello World!')
        self.set_start_time(start_time)
        self.set_total_samples(total_samples)
        self.set_search_properties(metric=metric, mode=mode)

    def set_search_properties(self, metric: Optional[str], mode: Optional[str]):
        if False:
            while True:
                i = 10
        if self._metric and metric or (self._mode and mode):
            raise ValueError('You passed a `metric` or `mode` argument to `tune.TuneConfig()`, but the reporter you are using was already instantiated with their own `metric` and `mode` parameters. Either remove the arguments from your reporter or from your call to `tune.TuneConfig()`')
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode
        if self._metric is None and self._mode:
            self._metric = DEFAULT_METRIC
        return True

    def set_total_samples(self, total_samples: int):
        if False:
            print('Hello World!')
        self._total_samples = total_samples

    def set_start_time(self, timestamp: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        if timestamp is not None:
            self._start_time = time.time()
        else:
            self._start_time = timestamp

    def should_report(self, trials: List[Trial], done: bool=False):
        if False:
            i = 10
            return i + 15
        if time.time() - self._last_report_time > self._max_report_freqency:
            self._last_report_time = time.time()
            return True
        return done

    def add_metric_column(self, metric: str, representation: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        'Adds a metric to the existing columns.\n\n        Args:\n            metric: Metric to add. This must be a metric being returned\n                in training step results.\n            representation: Representation to use in table. Defaults to\n                `metric`.\n        '
        self._metrics_override = True
        if metric in self._metric_columns:
            raise ValueError('Column {} already exists.'.format(metric))
        if isinstance(self._metric_columns, MutableMapping):
            representation = representation or metric
            self._metric_columns[metric] = representation
        else:
            if representation is not None and representation != metric:
                raise ValueError('`representation` cannot differ from `metric` if this reporter was initialized with a list of metric columns.')
            self._metric_columns.append(metric)

    def add_parameter_column(self, parameter: str, representation: Optional[str]=None):
        if False:
            while True:
                i = 10
        'Adds a parameter to the existing columns.\n\n        Args:\n            parameter: Parameter to add. This must be a parameter\n                specified in the configuration.\n            representation: Representation to use in table. Defaults to\n                `parameter`.\n        '
        if parameter in self._parameter_columns:
            raise ValueError('Column {} already exists.'.format(parameter))
        if isinstance(self._parameter_columns, MutableMapping):
            representation = representation or parameter
            self._parameter_columns[parameter] = representation
        else:
            if representation is not None and representation != parameter:
                raise ValueError('`representation` cannot differ from `parameter` if this reporter was initialized with a list of metric columns.')
            self._parameter_columns.append(parameter)

    def _progress_str(self, trials: List[Trial], done: bool, *sys_info: Dict, fmt: str='psql', delim: str='\n'):
        if False:
            return 10
        'Returns full progress string.\n\n        This string contains a progress table and error table. The progress\n        table describes the progress of each trial. The error table lists\n        the error file, if any, corresponding to each trial. The latter only\n        exists if errors have occurred.\n\n        Args:\n            trials: Trials to report on.\n            done: Whether this is the last progress report attempt.\n            fmt: Table format. See `tablefmt` in tabulate API.\n            delim: Delimiter between messages.\n        '
        if self._sort_by_metric and (self._metric is None or self._mode is None):
            self._sort_by_metric = False
            warnings.warn("Both 'metric' and 'mode' must be set to be able to sort by metric. No sorting is performed.")
        if not self._metrics_override:
            user_metrics = self._infer_user_metrics(trials, self._infer_limit)
            self._metric_columns.update(user_metrics)
        messages = ['== Status ==', _time_passed_str(self._start_time, time.time()), *sys_info]
        if done:
            max_progress = None
            max_error = None
        else:
            max_progress = self._max_progress_rows
            max_error = self._max_error_rows
        (current_best_trial, metric) = self._current_best_trial(trials)
        if current_best_trial:
            messages.append(_best_trial_str(current_best_trial, metric, self._parameter_columns))
        if has_verbosity(Verbosity.V1_EXPERIMENT):
            messages.append(_trial_progress_str(trials, metric_columns=self._metric_columns, parameter_columns=self._parameter_columns, total_samples=self._total_samples, force_table=self._print_intermediate_tables, fmt=fmt, max_rows=max_progress, max_column_length=self._max_column_length, done=done, metric=self._metric, mode=self._mode, sort_by_metric=self._sort_by_metric))
            messages.append(_trial_errors_str(trials, fmt=fmt, max_rows=max_error))
        return delim.join(messages) + delim

    def _infer_user_metrics(self, trials: List[Trial], limit: int=4):
        if False:
            while True:
                i = 10
        'Try to infer the metrics to print out.'
        if len(self._inferred_metrics) >= limit:
            return self._inferred_metrics
        self._inferred_metrics = {}
        for t in trials:
            if not t.last_result:
                continue
            for (metric, value) in t.last_result.items():
                if metric not in self.DEFAULT_COLUMNS:
                    if metric not in AUTO_RESULT_KEYS:
                        if type(value) in self.VALID_SUMMARY_TYPES:
                            self._inferred_metrics[metric] = metric
                if len(self._inferred_metrics) >= limit:
                    return self._inferred_metrics
        return self._inferred_metrics

    def _current_best_trial(self, trials: List[Trial]):
        if False:
            print('Hello World!')
        if not trials:
            return (None, None)
        (metric, mode) = (self._metric, self._mode)
        if not metric:
            if len(self._inferred_metrics) == 1:
                metric = list(self._inferred_metrics.keys())[0]
        if not metric or not mode:
            return (None, metric)
        metric_op = 1.0 if mode == 'max' else -1.0
        best_metric = float('-inf')
        best_trial = None
        for t in trials:
            if not t.last_result:
                continue
            metric_value = unflattened_lookup(metric, t.last_result, default=None)
            if pd.isnull(metric_value):
                continue
            if not best_trial or metric_value * metric_op > best_metric:
                best_metric = metric_value * metric_op
                best_trial = t
        return (best_trial, metric)

@DeveloperAPI
class RemoteReporterMixin:
    """Remote reporter abstract mixin class.

    Subclasses of this class will use a Ray Queue to display output
    on the driver side when running Ray Client."""

    @property
    def output_queue(self) -> Queue:
        if False:
            print('Hello World!')
        return getattr(self, '_output_queue', None)

    @output_queue.setter
    def output_queue(self, value: Queue):
        if False:
            while True:
                i = 10
        self._output_queue = value

    def display(self, string: str) -> None:
        if False:
            i = 10
            return i + 15
        'Display the progress string.\n\n        Args:\n            string: String to display.\n        '
        raise NotImplementedError

@PublicAPI
class JupyterNotebookReporter(TuneReporterBase, RemoteReporterMixin):
    """Jupyter notebook-friendly Reporter that can update display in-place.

    Args:
        overwrite: Flag for overwriting the cell contents before initialization.
        metric_columns: Names of metrics to
            include in progress table. If this is a dict, the keys should
            be metric names and the values should be the displayed names.
            If this is a list, the metric name is used directly.
        parameter_columns: Names of parameters to
            include in progress table. If this is a dict, the keys should
            be parameter names and the values should be the displayed names.
            If this is a list, the parameter name is used directly. If empty,
            defaults to all available parameters.
        max_progress_rows: Maximum number of rows to print
            in the progress table. The progress table describes the
            progress of each trial. Defaults to 20.
        max_error_rows: Maximum number of rows to print in the
            error table. The error table lists the error file, if any,
            corresponding to each trial. Defaults to 20.
        max_column_length: Maximum column length (in characters). Column
            headers and values longer than this will be abbreviated.
        max_report_frequency: Maximum report frequency in seconds.
            Defaults to 5s.
        infer_limit: Maximum number of metrics to automatically infer
            from tune results.
        print_intermediate_tables: Print intermediate result
            tables. If None (default), will be set to True for verbosity
            levels above 3, otherwise False. If True, intermediate tables
            will be printed with experiment progress. If False, tables
            will only be printed at then end of the tuning run for verbosity
            levels greater than 2.
        metric: Metric used to determine best current trial.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """

    def __init__(self, *, overwrite: bool=True, metric_columns: Optional[Union[List[str], Dict[str, str]]]=None, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: Optional[int]=None, max_progress_rows: int=20, max_error_rows: int=20, max_column_length: int=20, max_report_frequency: int=5, infer_limit: int=3, print_intermediate_tables: Optional[bool]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
        if False:
            print('Hello World!')
        super(JupyterNotebookReporter, self).__init__(metric_columns=metric_columns, parameter_columns=parameter_columns, total_samples=total_samples, max_progress_rows=max_progress_rows, max_error_rows=max_error_rows, max_column_length=max_column_length, max_report_frequency=max_report_frequency, infer_limit=infer_limit, print_intermediate_tables=print_intermediate_tables, metric=metric, mode=mode, sort_by_metric=sort_by_metric)
        if not IS_NOTEBOOK:
            warnings.warn('You are using the `JupyterNotebookReporter`, but not IPython/Jupyter-compatible environment was detected. If this leads to unformatted output (e.g. like <IPython.core.display.HTML object>), consider passing a `CLIReporter` as the `progress_reporter` argument to `train.RunConfig()` instead.')
        self._overwrite = overwrite
        self._display_handle = None
        self.display('')

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        if False:
            return 10
        progress = self._progress_html(trials, done, *sys_info)
        if self.output_queue is not None:
            self.output_queue.put(progress)
        else:
            self.display(progress)

    def display(self, string: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        from IPython.display import HTML, clear_output, display
        if not self._display_handle:
            if self._overwrite:
                clear_output(wait=True)
            self._display_handle = display(HTML(string), display_id=True)
        else:
            self._display_handle.update(HTML(string))

    def _progress_html(self, trials: List[Trial], done: bool, *sys_info) -> str:
        if False:
            i = 10
            return i + 15
        'Generate an HTML-formatted progress update.\n\n        Args:\n            trials: List of trials for which progress should be\n                displayed\n            done: True if the trials are finished, False otherwise\n            *sys_info: System information to be displayed\n\n        Returns:\n            Progress update to be rendered in a notebook, including HTML\n                tables and formatted error messages. Includes\n                - Duration of the tune job\n                - Memory consumption\n                - Trial progress table, with information about each experiment\n        '
        if not self._metrics_override:
            user_metrics = self._infer_user_metrics(trials, self._infer_limit)
            self._metric_columns.update(user_metrics)
        (current_time, running_for) = _get_time_str(self._start_time, time.time())
        (used_gb, total_gb, memory_message) = _get_memory_usage()
        status_table = tabulate([('Current time:', current_time), ('Running for:', running_for), ('Memory:', f'{used_gb}/{total_gb} GiB')], tablefmt='html')
        trial_progress_data = _trial_progress_table(trials=trials, metric_columns=self._metric_columns, parameter_columns=self._parameter_columns, fmt='html', max_rows=None if done else self._max_progress_rows, metric=self._metric, mode=self._mode, sort_by_metric=self._sort_by_metric, max_column_length=self._max_column_length)
        trial_progress = trial_progress_data[0]
        trial_progress_messages = trial_progress_data[1:]
        trial_errors = _trial_errors_str(trials, fmt='html', max_rows=None if done else self._max_error_rows)
        if any([memory_message, trial_progress_messages, trial_errors]):
            msg = Template('tune_status_messages.html.j2').render(memory_message=memory_message, trial_progress_messages=trial_progress_messages, trial_errors=trial_errors)
        else:
            msg = None
        return Template('tune_status.html.j2').render(status_table=status_table, sys_info_message=_generate_sys_info_str(*sys_info), trial_progress=trial_progress, messages=msg)

@PublicAPI
class CLIReporter(TuneReporterBase):
    """Command-line reporter

    Args:
        metric_columns: Names of metrics to
            include in progress table. If this is a dict, the keys should
            be metric names and the values should be the displayed names.
            If this is a list, the metric name is used directly.
        parameter_columns: Names of parameters to
            include in progress table. If this is a dict, the keys should
            be parameter names and the values should be the displayed names.
            If this is a list, the parameter name is used directly. If empty,
            defaults to all available parameters.
        max_progress_rows: Maximum number of rows to print
            in the progress table. The progress table describes the
            progress of each trial. Defaults to 20.
        max_error_rows: Maximum number of rows to print in the
            error table. The error table lists the error file, if any,
            corresponding to each trial. Defaults to 20.
        max_column_length: Maximum column length (in characters). Column
            headers and values longer than this will be abbreviated.
        max_report_frequency: Maximum report frequency in seconds.
            Defaults to 5s.
        infer_limit: Maximum number of metrics to automatically infer
            from tune results.
        print_intermediate_tables: Print intermediate result
            tables. If None (default), will be set to True for verbosity
            levels above 3, otherwise False. If True, intermediate tables
            will be printed with experiment progress. If False, tables
            will only be printed at then end of the tuning run for verbosity
            levels greater than 2.
        metric: Metric used to determine best current trial.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """

    def __init__(self, *, metric_columns: Optional[Union[List[str], Dict[str, str]]]=None, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: Optional[int]=None, max_progress_rows: int=20, max_error_rows: int=20, max_column_length: int=20, max_report_frequency: int=5, infer_limit: int=3, print_intermediate_tables: Optional[bool]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
        if False:
            print('Hello World!')
        super(CLIReporter, self).__init__(metric_columns=metric_columns, parameter_columns=parameter_columns, total_samples=total_samples, max_progress_rows=max_progress_rows, max_error_rows=max_error_rows, max_column_length=max_column_length, max_report_frequency=max_report_frequency, infer_limit=infer_limit, print_intermediate_tables=print_intermediate_tables, metric=metric, mode=mode, sort_by_metric=sort_by_metric)

    def _print(self, msg: str):
        if False:
            while True:
                i = 10
        safe_print(msg)

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        if False:
            return 10
        self._print(self._progress_str(trials, done, *sys_info))

def _get_memory_usage() -> Tuple[float, float, Optional[str]]:
    if False:
        print('Hello World!')
    'Get the current memory consumption.\n\n    Returns:\n        Memory used, memory available, and optionally a warning\n            message to be shown to the user when memory consumption is higher\n            than 90% or if `psutil` is not installed\n    '
    try:
        import ray
        import psutil
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        used_gb = total_gb - psutil.virtual_memory().available / 1024 ** 3
        if used_gb > total_gb * 0.9:
            message = ': ***LOW MEMORY*** less than 10% of the memory on this node is available for use. This can cause unexpected crashes. Consider reducing the memory used by your application or reducing the Ray object store size by setting `object_store_memory` when calling `ray.init`.'
        else:
            message = None
        return (round(used_gb, 1), round(total_gb, 1), message)
    except ImportError:
        return (np.nan, np.nan, 'Unknown memory usage. Please run `pip install psutil` to resolve')

def _get_time_str(start_time: float, current_time: float) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    'Get strings representing the current and elapsed time.\n\n    Args:\n        start_time: POSIX timestamp of the start of the tune run\n        current_time: POSIX timestamp giving the current time\n\n    Returns:\n        Current time and elapsed time for the current run\n    '
    current_time_dt = datetime.datetime.fromtimestamp(current_time)
    start_time_dt = datetime.datetime.fromtimestamp(start_time)
    delta: datetime.timedelta = current_time_dt - start_time_dt
    rest = delta.total_seconds()
    days = rest // (60 * 60 * 24)
    rest -= days * (60 * 60 * 24)
    hours = rest // (60 * 60)
    rest -= hours * (60 * 60)
    minutes = rest // 60
    seconds = rest - minutes * 60
    if days > 0:
        running_for_str = f'{days:.0f} days, '
    else:
        running_for_str = ''
    running_for_str += f'{hours:02.0f}:{minutes:02.0f}:{seconds:05.2f}'
    return (f'{current_time_dt:%Y-%m-%d %H:%M:%S}', running_for_str)

def _time_passed_str(start_time: float, current_time: float) -> str:
    if False:
        return 10
    'Generate a message describing the current and elapsed time in the run.\n\n    Args:\n        start_time: POSIX timestamp of the start of the tune run\n        current_time: POSIX timestamp giving the current time\n\n    Returns:\n        Message with the current and elapsed time for the current tune run,\n            formatted to be displayed to the user\n    '
    (current_time_str, running_for_str) = _get_time_str(start_time, current_time)
    return f'Current time: {current_time_str} (running for {running_for_str})'

def _get_trials_by_state(trials: List[Trial]):
    if False:
        return 10
    trials_by_state = collections.defaultdict(list)
    for t in trials:
        trials_by_state[t.status].append(t)
    return trials_by_state

def _trial_progress_str(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: int=0, force_table: bool=False, fmt: str='psql', max_rows: Optional[int]=None, max_column_length: int=20, done: bool=False, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a human readable message for printing to the console.\n\n    This contains a table where each row represents a trial, its parameters\n    and the current values of its metrics.\n\n    Args:\n        trials: List of trials to get progress string for.\n        metric_columns: Names of metrics to include.\n            If this is a dict, the keys are metric names and the values are\n            the names to use in the message. If this is a list, the metric\n            name is used in the message directly.\n        parameter_columns: Names of parameters to\n            include. If this is a dict, the keys are parameter names and the\n            values are the names to use in the message. If this is a list,\n            the parameter name is used in the message directly. If this is\n            empty, all parameters are used in the message.\n        total_samples: Total number of trials that will be generated.\n        force_table: Force printing a table. If False, a table will\n            be printed only at the end of the training for verbosity levels\n            above `Verbosity.V2_TRIAL_NORM`.\n        fmt: Output format (see tablefmt in tabulate API).\n        max_rows: Maximum number of rows in the trial table. Defaults to\n            unlimited.\n        max_column_length: Maximum column length (in characters).\n        done: True indicates that the tuning run finished.\n        metric: Metric used to sort trials.\n        mode: One of [min, max]. Determines whether objective is\n            minimizing or maximizing the metric attribute.\n        sort_by_metric: Sort terminated trials by metric in the\n            intermediate table. Defaults to False.\n    '
    messages = []
    delim = '<br>' if fmt == 'html' else '\n'
    if len(trials) < 1:
        return delim.join(messages)
    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)
    for local_dir in sorted({t.local_experiment_path for t in trials}):
        messages.append('Result logdir: {}'.format(local_dir))
    num_trials_strs = ['{} {}'.format(len(trials_by_state[state]), state) for state in sorted(trials_by_state)]
    if total_samples and total_samples >= sys.maxsize:
        total_samples = 'infinite'
    messages.append('Number of trials: {}{} ({})'.format(num_trials, f'/{total_samples}' if total_samples else '', ', '.join(num_trials_strs)))
    if force_table or (has_verbosity(Verbosity.V2_TRIAL_NORM) and done):
        messages += _trial_progress_table(trials=trials, metric_columns=metric_columns, parameter_columns=parameter_columns, fmt=fmt, max_rows=max_rows, metric=metric, mode=mode, sort_by_metric=sort_by_metric, max_column_length=max_column_length)
    return delim.join(messages)

def _max_len(value: Any, max_len: int=20, add_addr: bool=False, wrap: bool=False) -> Any:
    if False:
        i = 10
        return i + 15
    'Abbreviate a string representation of an object to `max_len` characters.\n\n    For numbers, booleans and None, the original value will be returned for\n    correct rendering in the table formatting tool.\n\n    Args:\n        value: Object to be represented as a string.\n        max_len: Maximum return string length.\n        add_addr: If True, will add part of the object address to the end of the\n            string, e.g. to identify different instances of the same class. If\n            False, three dots (``...``) will be used instead.\n    '
    if value is None or isinstance(value, (int, float, numbers.Number, bool)):
        return value
    string = str(value)
    if len(string) <= max_len:
        return string
    if wrap:
        if len(value) > max_len * 2:
            value = '...' + string[3 - max_len * 2:]
        wrapped = textwrap.wrap(value, width=max_len)
        return '\n'.join(wrapped)
    if add_addr and (not isinstance(value, (int, float, bool))):
        result = f'{string[:max_len - 5]}_{hex(id(value))[-4:]}'
        return result
    result = '...' + string[3 - max_len:]
    return result

def _get_progress_table_data(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, max_rows: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False, max_column_length: int=20) -> Tuple[List, List[str], Tuple[bool, str]]:
    if False:
        i = 10
        return i + 15
    'Generate a table showing the current progress of tuning trials.\n\n    Args:\n        trials: List of trials for which progress is to be shown.\n        metric_columns: Metrics to be displayed in the table.\n        parameter_columns: List of parameters to be included in the data\n        max_rows: Maximum number of rows to show. If there\'s overflow, a\n            message will be shown to the user indicating that some rows\n            are not displayed\n        metric: Metric which is being tuned\n        mode: Sort the table in descending order if mode is "max";\n            ascending otherwise\n        sort_by_metric: If true, the table will be sorted by the metric\n        max_column_length: Max number of characters in each column\n\n    Returns:\n        - Trial data\n        - List of column names\n        - Overflow tuple:\n            - boolean indicating whether the table has rows which are hidden\n            - string with info about the overflowing rows\n    '
    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)
    if sort_by_metric:
        trials_by_state[Trial.TERMINATED] = sorted(trials_by_state[Trial.TERMINATED], reverse=mode == 'max', key=lambda t: unflattened_lookup(metric, t.last_result, default=None))
    state_tbl_order = [Trial.RUNNING, Trial.PAUSED, Trial.PENDING, Trial.TERMINATED, Trial.ERROR]
    max_rows = max_rows or float('inf')
    if num_trials > max_rows:
        trials_by_state_trunc = _fair_filter_trials(trials_by_state, max_rows, sort_by_metric)
        trials = []
        overflow_strs = []
        for state in state_tbl_order:
            if state not in trials_by_state:
                continue
            trials += trials_by_state_trunc[state]
            num = len(trials_by_state[state]) - len(trials_by_state_trunc[state])
            if num > 0:
                overflow_strs.append('{} {}'.format(num, state))
        overflow = num_trials - max_rows
        overflow_str = ', '.join(overflow_strs)
    else:
        overflow = False
        overflow_str = ''
        trials = []
        for state in state_tbl_order:
            if state not in trials_by_state:
                continue
            trials += trials_by_state[state]
    if isinstance(metric_columns, Mapping):
        metric_keys = list(metric_columns.keys())
    else:
        metric_keys = metric_columns
    metric_keys = [k for k in metric_keys if any((unflattened_lookup(k, t.last_result, default=None) is not None for t in trials))]
    if not parameter_columns:
        parameter_keys = sorted(set().union(*[t.evaluated_params for t in trials]))
    elif isinstance(parameter_columns, Mapping):
        parameter_keys = list(parameter_columns.keys())
    else:
        parameter_keys = parameter_columns
    trial_table = [_get_trial_info(trial, parameter_keys, metric_keys, max_column_length=max_column_length) for trial in trials]
    if isinstance(metric_columns, Mapping):
        formatted_metric_columns = [_max_len(metric_columns[k], max_len=max_column_length, add_addr=False, wrap=True) for k in metric_keys]
    else:
        formatted_metric_columns = [_max_len(k, max_len=max_column_length, add_addr=False, wrap=True) for k in metric_keys]
    if isinstance(parameter_columns, Mapping):
        formatted_parameter_columns = [_max_len(parameter_columns[k], max_len=max_column_length, add_addr=False, wrap=True) for k in parameter_keys]
    else:
        formatted_parameter_columns = [_max_len(k, max_len=max_column_length, add_addr=False, wrap=True) for k in parameter_keys]
    columns = ['Trial name', 'status', 'loc'] + formatted_parameter_columns + formatted_metric_columns
    return (trial_table, columns, (overflow, overflow_str))

def _trial_progress_table(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, fmt: str='psql', max_rows: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False, max_column_length: int=20) -> List[str]:
    if False:
        return 10
    'Generate a list of trial progress table messages.\n\n    Args:\n        trials: List of trials for which progress is to be shown.\n        metric_columns: Metrics to be displayed in the table.\n        parameter_columns: List of parameters to be included in the data\n        fmt: Format of the table; passed to tabulate as the fmtstr argument\n        max_rows: Maximum number of rows to show. If there\'s overflow, a\n            message will be shown to the user indicating that some rows\n            are not displayed\n        metric: Metric which is being tuned\n        mode: Sort the table in descenting order if mode is "max";\n            ascending otherwise\n        sort_by_metric: If true, the table will be sorted by the metric\n        max_column_length: Max number of characters in each column\n\n    Returns:\n        Messages to be shown to the user containing progress tables\n    '
    (data, columns, (overflow, overflow_str)) = _get_progress_table_data(trials, metric_columns, parameter_columns, max_rows, metric, mode, sort_by_metric, max_column_length)
    messages = [tabulate(data, headers=columns, tablefmt=fmt, showindex=False)]
    if overflow:
        messages.append(f'... {overflow} more trials not shown ({overflow_str})')
    return messages

def _generate_sys_info_str(*sys_info) -> str:
    if False:
        while True:
            i = 10
    'Format system info into a string.\n        *sys_info: System info strings to be included.\n\n    Returns:\n        Formatted string containing system information.\n    '
    if sys_info:
        return '<br>'.join(sys_info).replace('\n', '<br>')
    return ''

def _trial_errors_str(trials: List[Trial], fmt: str='psql', max_rows: Optional[int]=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a readable message regarding trial errors.\n\n    Args:\n        trials: List of trials to get progress string for.\n        fmt: Output format (see tablefmt in tabulate API).\n        max_rows: Maximum number of rows in the error table. Defaults to\n            unlimited.\n    '
    messages = []
    failed = [t for t in trials if t.error_file]
    num_failed = len(failed)
    if num_failed > 0:
        messages.append('Number of errored trials: {}'.format(num_failed))
        if num_failed > (max_rows or float('inf')):
            messages.append('Table truncated to {} rows ({} overflow)'.format(max_rows, num_failed - max_rows))
        fail_header = ['Trial name', '# failures', 'error file']
        fail_table_data = [[str(trial), str(trial.run_metadata.num_failures) + ('' if trial.status == Trial.ERROR else '*'), trial.error_file] for trial in failed[:max_rows]]
        messages.append(tabulate(fail_table_data, headers=fail_header, tablefmt=fmt, showindex=False, colalign=('left', 'right', 'left')))
        if any((trial.status == Trial.TERMINATED for trial in failed[:max_rows])):
            messages.append('* The trial terminated successfully after retrying.')
    delim = '<br>' if fmt == 'html' else '\n'
    return delim.join(messages)

def _best_trial_str(trial: Trial, metric: str, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None):
    if False:
        i = 10
        return i + 15
    'Returns a readable message stating the current best trial.'
    val = unflattened_lookup(metric, trial.last_result, default=None)
    config = trial.last_result.get('config', {})
    parameter_columns = parameter_columns or list(config.keys())
    if isinstance(parameter_columns, Mapping):
        parameter_columns = parameter_columns.keys()
    params = {p: unflattened_lookup(p, config) for p in parameter_columns}
    return f'Current best trial: {trial.trial_id} with {metric}={val} and parameters={params}'

def _fair_filter_trials(trials_by_state: Dict[str, List[Trial]], max_trials: int, sort_by_metric: bool=False):
    if False:
        while True:
            i = 10
    'Filters trials such that each state is represented fairly.\n\n    The oldest trials are truncated if necessary.\n\n    Args:\n        trials_by_state: Maximum number of trials to return.\n    Returns:\n        Dict mapping state to List of fairly represented trials.\n    '
    num_trials_by_state = collections.defaultdict(int)
    no_change = False
    while max_trials > 0 and (not no_change):
        no_change = True
        for state in sorted(trials_by_state):
            if num_trials_by_state[state] < len(trials_by_state[state]):
                no_change = False
                max_trials -= 1
                num_trials_by_state[state] += 1
    sorted_trials_by_state = dict()
    for state in sorted(trials_by_state):
        if state == Trial.TERMINATED and sort_by_metric:
            sorted_trials_by_state[state] = trials_by_state[state]
        else:
            sorted_trials_by_state[state] = sorted(trials_by_state[state], reverse=False, key=lambda t: t.trial_id)
    filtered_trials = {state: sorted_trials_by_state[state][:num_trials_by_state[state]] for state in sorted(trials_by_state)}
    return filtered_trials

def _get_trial_location(trial: Trial, result: dict) -> _Location:
    if False:
        while True:
            i = 10
    (node_ip, pid) = (result.get(NODE_IP, None), result.get(PID, None))
    if node_ip and pid:
        location = _Location(node_ip, pid)
    else:
        location = trial.temporary_state.location
    return location

def _get_trial_info(trial: Trial, parameters: List[str], metrics: List[str], max_column_length: int=20):
    if False:
        i = 10
        return i + 15
    'Returns the following information about a trial:\n\n    name | status | loc | params... | metrics...\n\n    Args:\n        trial: Trial to get information for.\n        parameters: Names of trial parameters to include.\n        metrics: Names of metrics to include.\n        max_column_length: Maximum column length (in characters).\n    '
    result = trial.last_result
    config = trial.config
    location = _get_trial_location(trial, result)
    trial_info = [str(trial), trial.status, str(location)]
    trial_info += [_max_len(unflattened_lookup(param, config, default=None), max_len=max_column_length, add_addr=True) for param in parameters]
    trial_info += [_max_len(unflattened_lookup(metric, result, default=None), max_len=max_column_length, add_addr=True) for metric in metrics]
    return trial_info

@DeveloperAPI
class TrialProgressCallback(Callback):
    """Reports (prints) intermediate trial progress.

    This callback is automatically added to the callback stack. When a
    result is obtained, this callback will print the results according to
    the specified verbosity level.

    For ``Verbosity.V3_TRIAL_DETAILS``, a full result list is printed.

    For ``Verbosity.V2_TRIAL_NORM``, only one line is printed per received
    result.

    All other verbosity levels do not print intermediate trial progress.

    Result printing is throttled on a per-trial basis. Per default, results are
    printed only once every 30 seconds. Results are always printed when a trial
    finished or errored.

    """

    def __init__(self, metric: Optional[str]=None, progress_metrics: Optional[List[str]]=None):
        if False:
            print('Hello World!')
        self._last_print = collections.defaultdict(float)
        self._last_print_iteration = collections.defaultdict(int)
        self._completed_trials = set()
        self._last_result_str = {}
        self._metric = metric
        self._progress_metrics = set(progress_metrics or [])
        if self._metric and self._progress_metrics:
            self._progress_metrics.add(self._metric)
        self._last_result = {}
        self._display_handle = None

    def _print(self, msg: str):
        if False:
            return 10
        safe_print(msg)

    def on_trial_result(self, iteration: int, trials: List['Trial'], trial: 'Trial', result: Dict, **info):
        if False:
            for i in range(10):
                print('nop')
        self.log_result(trial, result, error=False)

    def on_trial_error(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        if False:
            for i in range(10):
                print('nop')
        self.log_result(trial, trial.last_result, error=True)

    def on_trial_complete(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        if False:
            while True:
                i = 10
        if trial not in self._completed_trials:
            self._completed_trials.add(trial)
            print_result_str = self._print_result(trial.last_result)
            last_result_str = self._last_result_str.get(trial, '')
            if print_result_str != last_result_str:
                self.log_result(trial, trial.last_result, error=False)
            else:
                self._print(f'Trial {trial} completed. Last result: {print_result_str}')

    def log_result(self, trial: 'Trial', result: Dict, error: bool=False):
        if False:
            for i in range(10):
                print('nop')
        done = result.get('done', False) is True
        last_print = self._last_print[trial]
        should_print = done or error or time.time() - last_print > DEBUG_PRINT_INTERVAL
        if done and trial not in self._completed_trials:
            self._completed_trials.add(trial)
        if should_print:
            if IS_NOTEBOOK:
                self.display_result(trial, result, error, done)
            else:
                self.print_result(trial, result, error, done)
            self._last_print[trial] = time.time()
            if TRAINING_ITERATION in result:
                self._last_print_iteration[trial] = result[TRAINING_ITERATION]

    def print_result(self, trial: Trial, result: Dict, error: bool, done: bool):
        if False:
            while True:
                i = 10
        'Print the most recent results for the given trial to stdout.\n\n        Args:\n            trial: Trial for which results are to be printed\n            result: Result to be printed\n            error: True if an error has occurred, False otherwise\n            done: True if the trial is finished, False otherwise\n        '
        last_print_iteration = self._last_print_iteration[trial]
        if has_verbosity(Verbosity.V3_TRIAL_DETAILS):
            if result.get(TRAINING_ITERATION) != last_print_iteration:
                self._print(f'Result for {trial}:')
                self._print('  {}'.format(pretty_print(result).replace('\n', '\n  ')))
            if done:
                self._print(f'Trial {trial} completed.')
        elif has_verbosity(Verbosity.V2_TRIAL_NORM):
            metric_name = self._metric or '_metric'
            metric_value = result.get(metric_name, -99.0)
            error_file = os.path.join(trial.local_path, EXPR_ERROR_FILE)
            info = ''
            if done:
                info = ' This trial completed.'
            print_result_str = self._print_result(result)
            self._last_result_str[trial] = print_result_str
            if error:
                message = f'The trial {trial} errored with parameters={trial.config}. Error file: {error_file}'
            elif self._metric:
                message = f'Trial {trial} reported {metric_name}={metric_value:.2f} with parameters={trial.config}.{info}'
            else:
                message = f'Trial {trial} reported {print_result_str} with parameters={trial.config}.{info}'
            self._print(message)

    def generate_trial_table(self, trials: Dict[Trial, Dict], columns: List[str]) -> str:
        if False:
            while True:
                i = 10
        'Generate an HTML table of trial progress info.\n\n        Trials (rows) are sorted by name; progress stats (columns) are sorted\n        as well.\n\n        Args:\n            trials: Trials and their associated latest results\n            columns: Columns to show in the table; must be a list of valid\n                keys for each Trial result\n\n        Returns:\n            HTML template containing a rendered table of progress info\n        '
        data = []
        columns = sorted(columns)
        sorted_trials = collections.OrderedDict(sorted(self._last_result.items(), key=lambda item: str(item[0])))
        for (trial, result) in sorted_trials.items():
            data.append([str(trial)] + [result.get(col, '') for col in columns])
        return Template('trial_progress.html.j2').render(table=tabulate(data, tablefmt='html', headers=['Trial name'] + columns, showindex=False))

    def display_result(self, trial: Trial, result: Dict, error: bool, done: bool):
        if False:
            print('Hello World!')
        'Display a formatted HTML table of trial progress results.\n\n        Trial progress is only shown if verbosity is set to level 2 or 3.\n\n        Args:\n            trial: Trial for which results are to be printed\n            result: Result to be printed\n            error: True if an error has occurred, False otherwise\n            done: True if the trial is finished, False otherwise\n        '
        from IPython.display import display, HTML
        self._last_result[trial] = result
        if has_verbosity(Verbosity.V3_TRIAL_DETAILS):
            ignored_keys = {'config', 'hist_stats'}
        elif has_verbosity(Verbosity.V2_TRIAL_NORM):
            ignored_keys = {'config', 'hist_stats', 'trial_id', 'experiment_tag', 'done'} | set(AUTO_RESULT_KEYS)
        else:
            return
        table = self.generate_trial_table(self._last_result, set(result.keys()) - ignored_keys)
        if not self._display_handle:
            self._display_handle = display(HTML(table), display_id=True)
        else:
            self._display_handle.update(HTML(table))

    def _print_result(self, result: Dict):
        if False:
            for i in range(10):
                print('nop')
        if self._progress_metrics:
            flat_result = flatten_dict(result)
            print_result = {}
            for metric in self._progress_metrics:
                print_result[metric] = flat_result.get(metric)
        else:
            print_result = result.copy()
            for skip_result in SKIP_RESULTS_IN_REPORT:
                print_result.pop(skip_result, None)
            for auto_result in AUTO_RESULT_KEYS:
                print_result.pop(auto_result, None)
        print_result_str = ','.join([f'{k}={v}' for (k, v) in print_result.items() if v is not None])
        return print_result_str

def _detect_reporter(**kwargs) -> TuneReporterBase:
    if False:
        i = 10
        return i + 15
    'Detect progress reporter class.\n\n    Will return a :class:`JupyterNotebookReporter` if a IPython/Jupyter-like\n    session was detected, and a :class:`CLIReporter` otherwise.\n\n    Keyword arguments are passed on to the reporter class.\n    '
    if IS_NOTEBOOK:
        kwargs.setdefault('overwrite', not has_verbosity(Verbosity.V2_TRIAL_NORM))
        progress_reporter = JupyterNotebookReporter(**kwargs)
    else:
        progress_reporter = CLIReporter(**kwargs)
    return progress_reporter

def _detect_progress_metrics(trainable: Optional[Union['Trainable', Callable]]) -> Optional[Collection[str]]:
    if False:
        while True:
            i = 10
    'Detect progress metrics to report.'
    if not trainable:
        return None
    return getattr(trainable, '_progress_metrics', None)

def _prepare_progress_reporter_for_ray_client(progress_reporter: ProgressReporter, verbosity: Union[int, Verbosity], string_queue: Optional[Queue]=None) -> Tuple[ProgressReporter, Queue]:
    if False:
        print('Hello World!')
    "Prepares progress reported for Ray Client by setting the string queue.\n\n    The string queue will be created if it's None."
    set_verbosity(verbosity)
    progress_reporter = progress_reporter or _detect_reporter()
    if isinstance(progress_reporter, RemoteReporterMixin):
        if string_queue is None:
            string_queue = Queue(actor_options={'num_cpus': 0, **_force_on_current_node(None)})
        progress_reporter.output_queue = string_queue
    return (progress_reporter, string_queue)

def _stream_client_output(remote_future: ray.ObjectRef, progress_reporter: ProgressReporter, string_queue: Queue) -> Any:
    if False:
        print('Hello World!')
    '\n    Stream items from string queue to progress_reporter until remote_future resolves\n    '
    if string_queue is None:
        return

    def get_next_queue_item():
        if False:
            print('Hello World!')
        try:
            return string_queue.get(block=False)
        except Empty:
            return None

    def _handle_string_queue():
        if False:
            print('Hello World!')
        string_item = get_next_queue_item()
        while string_item is not None:
            progress_reporter.display(string_item)
            string_item = get_next_queue_item()
    while ray.wait([remote_future], timeout=0.2)[1]:
        _handle_string_queue()
    _handle_string_queue()