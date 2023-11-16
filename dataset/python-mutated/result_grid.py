import os
import pandas as pd
import pyarrow
from typing import Optional, Union
from ray.air.result import Result
from ray.cloudpickle import cloudpickle
from ray.exceptions import RayTaskError
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.error import TuneError
from ray.tune.experiment import Trial
from ray.util import PublicAPI

@PublicAPI(stability='beta')
class ResultGrid:
    """A set of ``Result`` objects for interacting with Ray Tune results.

    You can use it to inspect the trials and obtain the best result.

    The constructor is a private API. This object can only be created as a result of
    ``Tuner.fit()``.

    Example:
    .. testcode::

        import random
        from ray import train, tune
        def random_error_trainable(config):
            if random.random() < 0.5:
                return {"loss": 0.0}
            else:
                raise ValueError("This is an error")
        tuner = tune.Tuner(
            random_error_trainable,
            run_config=train.RunConfig(name="example-experiment"),
            tune_config=tune.TuneConfig(num_samples=10),
        )
        try:
            result_grid = tuner.fit()
        except ValueError:
            pass
        for i in range(len(result_grid)):
            result = result_grid[i]
            if not result.error:
                    print(f"Trial finishes successfully with metrics"
                       f"{result.metrics}.")
            else:
                    print(f"Trial failed with error {result.error}.")

    .. testoutput::
        :hide:

        ...

    You can also use ``result_grid`` for more advanced analysis.

    >>> # Get the best result based on a particular metric.
    >>> best_result = result_grid.get_best_result( # doctest: +SKIP
    ...     metric="loss", mode="min")
    >>> # Get the best checkpoint corresponding to the best result.
    >>> best_checkpoint = best_result.checkpoint # doctest: +SKIP
    >>> # Get a dataframe for the last reported results of all of the trials
    >>> df = result_grid.get_dataframe() # doctest: +SKIP
    >>> # Get a dataframe for the minimum loss seen for each trial
    >>> df = result_grid.get_dataframe(metric="loss", mode="min") # doctest: +SKIP

    Note that trials of all statuses are included in the final result grid.
    If a trial is not in terminated state, its latest result and checkpoint as
    seen by Tune will be provided.

    See :doc:`/tune/examples/tune_analyze_results` for more usage examples.
    """

    def __init__(self, experiment_analysis: ExperimentAnalysis):
        if False:
            for i in range(10):
                print('nop')
        self._experiment_analysis = experiment_analysis
        self._results = [self._trial_to_result(trial) for trial in self._experiment_analysis.trials]

    @property
    def experiment_path(self) -> str:
        if False:
            print('Hello World!')
        'Path pointing to the experiment directory on persistent storage.\n\n        This can point to a remote storage location (e.g. S3) or to a local\n        location (path on the head node).'
        return self._experiment_analysis.experiment_path

    @property
    def filesystem(self) -> pyarrow.fs.FileSystem:
        if False:
            return 10
        'Return the filesystem that can be used to access the experiment path.\n\n        Returns:\n            pyarrow.fs.FileSystem implementation.\n        '
        return self._experiment_analysis._fs

    def get_best_result(self, metric: Optional[str]=None, mode: Optional[str]=None, scope: str='last', filter_nan_and_inf: bool=True) -> Result:
        if False:
            i = 10
            return i + 15
        "Get the best result from all the trials run.\n\n        Args:\n            metric: Key for trial info to order on. Defaults to\n                the metric specified in your Tuner's ``TuneConfig``.\n            mode: One of [min, max]. Defaults to the mode specified\n                in your Tuner's ``TuneConfig``.\n            scope: One of [all, last, avg, last-5-avg, last-10-avg].\n                If `scope=last`, only look at each trial's final step for\n                `metric`, and compare across trials based on `mode=[min,max]`.\n                If `scope=avg`, consider the simple average over all steps\n                for `metric` and compare across trials based on\n                `mode=[min,max]`. If `scope=last-5-avg` or `scope=last-10-avg`,\n                consider the simple average over the last 5 or 10 steps for\n                `metric` and compare across trials based on `mode=[min,max]`.\n                If `scope=all`, find each trial's min/max score for `metric`\n                based on `mode`, and compare trials based on `mode=[min,max]`.\n            filter_nan_and_inf: If True (default), NaN or infinite\n                values are disregarded and these trials are never selected as\n                the best trial.\n        "
        if len(self._experiment_analysis.trials) == 1:
            return self._trial_to_result(self._experiment_analysis.trials[0])
        if not metric and (not self._experiment_analysis.default_metric):
            raise ValueError('No metric is provided. Either pass in a `metric` arg to `get_best_result` or specify a metric in the `TuneConfig` of your `Tuner`.')
        if not mode and (not self._experiment_analysis.default_mode):
            raise ValueError('No mode is provided. Either pass in a `mode` arg to `get_best_result` or specify a mode in the `TuneConfig` of your `Tuner`.')
        best_trial = self._experiment_analysis.get_best_trial(metric=metric, mode=mode, scope=scope, filter_nan_and_inf=filter_nan_and_inf)
        if not best_trial:
            error_msg = f'No best trial found for the given metric: {metric or self._experiment_analysis.default_metric}. This means that no trial has reported this metric'
            error_msg += ', or all values reported for this metric are NaN. To not ignore NaN values, you can set the `filter_nan_and_inf` arg to False.' if filter_nan_and_inf else '.'
            raise RuntimeError(error_msg)
        return self._trial_to_result(best_trial)

    def get_dataframe(self, filter_metric: Optional[str]=None, filter_mode: Optional[str]=None) -> pd.DataFrame:
        if False:
            return 10
        'Return dataframe of all trials with their configs and reported results.\n\n        Per default, this returns the last reported results for each trial.\n\n        If ``filter_metric`` and ``filter_mode`` are set, the results from each\n        trial are filtered for this metric and mode. For example, if\n        ``filter_metric="some_metric"`` and ``filter_mode="max"``, for each trial,\n        every received result is checked, and the one where ``some_metric`` is\n        maximal is returned.\n\n\n        Example:\n\n            .. testcode::\n\n                from ray import train\n                from ray.train import RunConfig\n                from ray.tune import Tuner\n\n                def training_loop_per_worker(config):\n                    train.report({"accuracy": 0.8})\n\n                result_grid = Tuner(\n                    trainable=training_loop_per_worker,\n                    run_config=RunConfig(name="my_tune_run")\n                ).fit()\n\n                # Get last reported results per trial\n                df = result_grid.get_dataframe()\n\n                # Get best ever reported accuracy per trial\n                df = result_grid.get_dataframe(\n                    filter_metric="accuracy", filter_mode="max"\n                )\n\n            .. testoutput::\n                :hide:\n\n                ...\n\n        Args:\n            filter_metric: Metric to filter best result for.\n            filter_mode: If ``filter_metric`` is given, one of ``["min", "max"]``\n                to specify if we should find the minimum or maximum result.\n\n        Returns:\n            Pandas DataFrame with each trial as a row and their results as columns.\n        '
        return self._experiment_analysis.dataframe(metric=filter_metric, mode=filter_mode)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._results)

    def __getitem__(self, i: int) -> Result:
        if False:
            for i in range(10):
                print('nop')
        "Returns the i'th result in the grid."
        return self._results[i]

    @property
    def errors(self):
        if False:
            while True:
                i = 10
        'Returns the exceptions of errored trials.'
        return [result.error for result in self if result.error]

    @property
    def num_errors(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of errored trials.'
        return len([t for t in self._experiment_analysis.trials if t.status == Trial.ERROR])

    @property
    def num_terminated(self):
        if False:
            print('Hello World!')
        'Returns the number of terminated (but not errored) trials.'
        return len([t for t in self._experiment_analysis.trials if t.status == Trial.TERMINATED])

    @staticmethod
    def _populate_exception(trial: Trial) -> Optional[Union[TuneError, RayTaskError]]:
        if False:
            for i in range(10):
                print('nop')
        if trial.status == Trial.TERMINATED:
            return None
        if trial.pickled_error_file and os.path.exists(trial.pickled_error_file):
            with open(trial.pickled_error_file, 'rb') as f:
                e = cloudpickle.load(f)
                return e
        elif trial.error_file and os.path.exists(trial.error_file):
            with open(trial.error_file, 'r') as f:
                return TuneError(f.read())
        return None

    def _trial_to_result(self, trial: Trial) -> Result:
        if False:
            while True:
                i = 10
        cpm = trial.run_metadata.checkpoint_manager
        checkpoint = None
        if cpm.latest_checkpoint_result:
            checkpoint = cpm.latest_checkpoint_result.checkpoint
        best_checkpoint_results = cpm.best_checkpoint_results
        best_checkpoints = [(checkpoint_result.checkpoint, checkpoint_result.metrics) for checkpoint_result in best_checkpoint_results]
        metrics_df = self._experiment_analysis.trial_dataframes.get(trial.trial_id)
        result = Result(checkpoint=checkpoint, metrics=trial.last_result.copy(), error=self._populate_exception(trial), _local_path=trial.local_path, _remote_path=trial.path, _storage_filesystem=self._experiment_analysis._fs if isinstance(self._experiment_analysis, ExperimentAnalysis) else None, metrics_dataframe=metrics_df, best_checkpoints=best_checkpoints)
        return result

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        all_results_repr = [result._repr(indent=2) for result in self]
        all_results_repr = ',\n'.join(all_results_repr)
        return f'ResultGrid<[\n{all_results_repr}\n]>'