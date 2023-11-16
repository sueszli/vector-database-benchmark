import copy
import glob
import logging
import os
import warnings
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from ray.air._internal.usage import tag_searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
if TYPE_CHECKING:
    from ray.tune.experiment import Trial
    from ray.tune.analysis import ExperimentAnalysis
logger = logging.getLogger(__name__)

@DeveloperAPI
class Searcher:
    """Abstract class for wrapping suggesting algorithms.

    Custom algorithms can extend this class easily by overriding the
    `suggest` method provide generated parameters for the trials.

    Any subclass that implements ``__init__`` must also call the
    constructor of this class: ``super(Subclass, self).__init__(...)``.

    To track suggestions and their corresponding evaluations, the method
    `suggest` will be passed a trial_id, which will be used in
    subsequent notifications.

    Not all implementations support multi objectives.

    Note to Tune developers: If a new searcher is added, please update
    `air/_internal/usage.py`.

    Args:
        metric: The training result objective value attribute. If
            list then list of training result objective value attributes
        mode: If string One of {min, max}. If list then
            list of max and min, determines whether objective is minimizing
            or maximizing the metric attribute. Must match type of metric.

    .. code-block:: python

        class ExampleSearch(Searcher):
            def __init__(self, metric="mean_loss", mode="min", **kwargs):
                super(ExampleSearch, self).__init__(
                    metric=metric, mode=mode, **kwargs)
                self.optimizer = Optimizer()
                self.configurations = {}

            def suggest(self, trial_id):
                configuration = self.optimizer.query()
                self.configurations[trial_id] = configuration

            def on_trial_complete(self, trial_id, result, **kwargs):
                configuration = self.configurations[trial_id]
                if result and self.metric in result:
                    self.optimizer.update(configuration, result[self.metric])

        tuner = tune.Tuner(
            trainable_function,
            tune_config=tune.TuneConfig(
                search_alg=ExampleSearch()
            )
        )
        tuner.fit()


    """
    FINISHED = 'FINISHED'
    CKPT_FILE_TMPL = 'searcher-state-{}.pkl'

    def __init__(self, metric: Optional[str]=None, mode: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        tag_searcher(self)
        self._metric = metric
        self._mode = mode
        if not mode or not metric:
            return
        assert isinstance(metric, type(mode)), 'metric and mode must be of the same type'
        if isinstance(mode, str):
            assert mode in ['min', 'max'], "if `mode` is a str must be 'min' or 'max'!"
        elif isinstance(mode, list):
            assert len(mode) == len(metric), 'Metric and mode must be the same length'
            assert all((mod in ['min', 'max', 'obs'] for mod in mode)), "All of mode must be 'min' or 'max' or 'obs'!"
        else:
            raise ValueError('Mode most either be a list or string')

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Pass search properties to searcher.\n\n        This method acts as an alternative to instantiating search algorithms\n        with their own specific search spaces. Instead they can accept a\n        Tune config through this method. A searcher should return ``True``\n        if setting the config was successful, or ``False`` if it was\n        unsuccessful, e.g. when the search space has already been set.\n\n        Args:\n            metric: Metric to optimize\n            mode: One of ["min", "max"]. Direction to optimize.\n            config: Tune config dict.\n            **spec: Any kwargs for forward compatiblity.\n                Info like Experiment.PUBLIC_KEYS is provided through here.\n        '
        return False

    def on_trial_result(self, trial_id: str, result: Dict) -> None:
        if False:
            return 10
        'Optional notification for result during training.\n\n        Note that by default, the result dict may include NaNs or\n        may not include the optimization metric. It is up to the\n        subclass implementation to preprocess the result to\n        avoid breaking the optimization process.\n\n        Args:\n            trial_id: A unique string ID for the trial.\n            result: Dictionary of metrics for current training progress.\n                Note that the result dict may include NaNs or\n                may not include the optimization metric. It is up to the\n                subclass implementation to preprocess the result to\n                avoid breaking the optimization process.\n        '
        pass

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Notification for the completion of trial.\n\n        Typically, this method is used for notifying the underlying\n        optimizer of the result.\n\n        Args:\n            trial_id: A unique string ID for the trial.\n            result: Dictionary of metrics for current training progress.\n                Note that the result dict may include NaNs or\n                may not include the optimization metric. It is up to the\n                subclass implementation to preprocess the result to\n                avoid breaking the optimization process. Upon errors, this\n                may also be None.\n            error: True if the training process raised an error.\n\n        '
        raise NotImplementedError

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            return 10
        'Queries the algorithm to retrieve the next set of parameters.\n\n        Arguments:\n            trial_id: Trial ID used for subsequent notifications.\n\n        Returns:\n            dict | FINISHED | None: Configuration for a trial, if possible.\n                If FINISHED is returned, Tune will be notified that\n                no more suggestions/configurations will be provided.\n                If None is returned, Tune will skip the querying of the\n                searcher for this step.\n\n        '
        raise NotImplementedError

    def add_evaluated_point(self, parameters: Dict, value: float, error: bool=False, pruned: bool=False, intermediate_values: Optional[List[float]]=None):
        if False:
            for i in range(10):
                print('nop')
        'Pass results from a point that has been evaluated separately.\n\n        This method allows for information from outside the\n        suggest - on_trial_complete loop to be passed to the search\n        algorithm.\n        This functionality depends on the underlying search algorithm\n        and may not be always available.\n\n        Args:\n            parameters: Parameters used for the trial.\n            value: Metric value obtained in the trial.\n            error: True if the training process raised an error.\n            pruned: True if trial was pruned.\n            intermediate_values: List of metric values for\n                intermediate iterations of the result. None if not\n                applicable.\n\n        '
        raise NotImplementedError

    def add_evaluated_trials(self, trials_or_analysis: Union['Trial', List['Trial'], 'ExperimentAnalysis'], metric: str):
        if False:
            i = 10
            return i + 15
        'Pass results from trials that have been evaluated separately.\n\n        This method allows for information from outside the\n        suggest - on_trial_complete loop to be passed to the search\n        algorithm.\n        This functionality depends on the underlying search algorithm\n        and may not be always available (same as ``add_evaluated_point``.)\n\n        Args:\n            trials_or_analysis: Trials to pass results form to the searcher.\n            metric: Metric name reported by trials used for\n                determining the objective value.\n\n        '
        if self.add_evaluated_point == Searcher.add_evaluated_point:
            raise NotImplementedError
        from ray.tune.experiment import Trial
        from ray.tune.analysis import ExperimentAnalysis
        from ray.tune.result import DONE
        if isinstance(trials_or_analysis, (list, tuple)):
            trials = trials_or_analysis
        elif isinstance(trials_or_analysis, Trial):
            trials = [trials_or_analysis]
        elif isinstance(trials_or_analysis, ExperimentAnalysis):
            trials = trials_or_analysis.trials
        else:
            raise NotImplementedError(f'Expected input to be a `Trial`, a list of `Trial`s, or `ExperimentAnalysis`, got: {trials_or_analysis}')
        any_trial_had_metric = False

        def trial_to_points(trial: Trial) -> Dict[str, Any]:
            if False:
                print('Hello World!')
            nonlocal any_trial_had_metric
            has_trial_been_pruned = trial.status == Trial.TERMINATED and (not trial.last_result.get(DONE, False))
            has_trial_finished = trial.status == Trial.TERMINATED and trial.last_result.get(DONE, False)
            if not any_trial_had_metric:
                any_trial_had_metric = metric in trial.last_result and has_trial_finished
            if Trial.TERMINATED and metric not in trial.last_result:
                return None
            return dict(parameters=trial.config, value=trial.last_result.get(metric, None), error=trial.status == Trial.ERROR, pruned=has_trial_been_pruned, intermediate_values=None)
        for trial in trials:
            kwargs = trial_to_points(trial)
            if kwargs:
                self.add_evaluated_point(**kwargs)
        if not any_trial_had_metric:
            warnings.warn('No completed trial returned the specified metric. Make sure the name you have passed is correct. ')

    def save(self, checkpoint_path: str):
        if False:
            for i in range(10):
                print('nop')
        'Save state to path for this search algorithm.\n\n        Args:\n            checkpoint_path: File where the search algorithm\n                state is saved. This path should be used later when\n                restoring from file.\n\n        Example:\n\n        .. code-block:: python\n\n            search_alg = Searcher(...)\n\n            tuner = tune.Tuner(\n                cost,\n                tune_config=tune.TuneConfig(\n                    search_alg=search_alg,\n                    num_samples=5\n                ),\n                param_space=config\n            )\n            results = tuner.fit()\n\n            search_alg.save("./my_favorite_path.pkl")\n\n        .. versionchanged:: 0.8.7\n            Save is automatically called by `Tuner().fit()`. You can use\n            `Tuner().restore()` to restore from an experiment directory\n            such as `~/ray_results/trainable`.\n\n        '
        raise NotImplementedError

    def restore(self, checkpoint_path: str):
        if False:
            for i in range(10):
                print('nop')
        'Restore state for this search algorithm\n\n\n        Args:\n            checkpoint_path: File where the search algorithm\n                state is saved. This path should be the same\n                as the one provided to "save".\n\n        Example:\n\n        .. code-block:: python\n\n            search_alg.save("./my_favorite_path.pkl")\n\n            search_alg2 = Searcher(...)\n            search_alg2 = ConcurrencyLimiter(search_alg2, 1)\n            search_alg2.restore(checkpoint_path)\n            tuner = tune.Tuner(\n                cost,\n                tune_config=tune.TuneConfig(\n                    search_alg=search_alg2,\n                    num_samples=5\n                ),\n            )\n            tuner.fit()\n\n        '
        raise NotImplementedError

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        if False:
            while True:
                i = 10
        'Set max concurrent trials this searcher can run.\n\n        This method will be called on the wrapped searcher by the\n        ``ConcurrencyLimiter``. It is intended to allow for searchers\n        which have custom, internal logic handling max concurrent trials\n        to inherit the value passed to ``ConcurrencyLimiter``.\n\n        If this method returns False, it signifies that no special\n        logic for handling this case is present in the searcher.\n\n        Args:\n            max_concurrent: Number of maximum concurrent trials.\n        '
        return False

    def get_state(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def set_state(self, state: Dict):
        if False:
            return 10
        raise NotImplementedError

    def save_to_dir(self, checkpoint_dir: str, session_str: str='default'):
        if False:
            print('Hello World!')
        'Automatically saves the given searcher to the checkpoint_dir.\n\n        This is automatically used by Tuner().fit() during a Tune job.\n\n        Args:\n            checkpoint_dir: Filepath to experiment dir.\n            session_str: Unique identifier of the current run\n                session.\n        '
        tmp_search_ckpt_path = os.path.join(checkpoint_dir, '.tmp_searcher_ckpt')
        success = True
        try:
            self.save(tmp_search_ckpt_path)
        except NotImplementedError:
            if log_once('suggest:save_to_dir'):
                logger.warning('save not implemented for Searcher. Skipping save.')
            success = False
        if success and os.path.exists(tmp_search_ckpt_path):
            os.replace(tmp_search_ckpt_path, os.path.join(checkpoint_dir, self.CKPT_FILE_TMPL.format(session_str)))

    def restore_from_dir(self, checkpoint_dir: str):
        if False:
            i = 10
            return i + 15
        'Restores the state of a searcher from a given checkpoint_dir.\n\n        Typically, you should use this function to restore from an\n        experiment directory such as `~/ray_results/trainable`.\n\n        .. code-block:: python\n\n            tuner = tune.Tuner(\n                cost,\n                run_config=train.RunConfig(\n                    name=self.experiment_name,\n                    local_dir="~/my_results",\n                ),\n                tune_config=tune.TuneConfig(\n                    search_alg=search_alg,\n                    num_samples=5\n                ),\n                param_space=config\n            )\n            tuner.fit()\n\n            search_alg2 = Searcher()\n            search_alg2.restore_from_dir(\n                os.path.join("~/my_results", self.experiment_name)\n        '
        pattern = self.CKPT_FILE_TMPL.format('*')
        full_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
        if not full_paths:
            raise RuntimeError('Searcher unable to find checkpoint in {}'.format(checkpoint_dir))
        most_recent_checkpoint = max(full_paths)
        self.restore(most_recent_checkpoint)

    @property
    def metric(self) -> str:
        if False:
            return 10
        'The training result objective value attribute.'
        return self._metric

    @property
    def mode(self) -> str:
        if False:
            i = 10
            return i + 15
        'Specifies if minimizing or maximizing the metric.'
        return self._mode

@PublicAPI
class ConcurrencyLimiter(Searcher):
    """A wrapper algorithm for limiting the number of concurrent trials.

    Certain Searchers have their own internal logic for limiting
    the number of concurrent trials. If such a Searcher is passed to a
    ``ConcurrencyLimiter``, the ``max_concurrent`` of the
    ``ConcurrencyLimiter`` will override the ``max_concurrent`` value
    of the Searcher. The ``ConcurrencyLimiter`` will then let the
    Searcher's internal logic take over.

    Args:
        searcher: Searcher object that the
            ConcurrencyLimiter will manage.
        max_concurrent: Maximum concurrent samples from the underlying
            searcher.
        batch: Whether to wait for all concurrent samples
            to finish before updating the underlying searcher.

    Example:

    .. code-block:: python

        from ray.tune.search import ConcurrencyLimiter
        search_alg = HyperOptSearch(metric="accuracy")
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
        tuner = tune.Tuner(
            trainable_function,
            tune_config=tune.TuneConfig(
                search_alg=search_alg
            ),
        )
        tuner.fit()

    """

    def __init__(self, searcher: Searcher, max_concurrent: int, batch: bool=False):
        if False:
            print('Hello World!')
        assert type(max_concurrent) is int and max_concurrent > 0
        self.searcher = searcher
        self.max_concurrent = max_concurrent
        self.batch = batch
        self.live_trials = set()
        self.num_unfinished_live_trials = 0
        self.cached_results = {}
        self._limit_concurrency = True
        if not isinstance(searcher, Searcher):
            raise RuntimeError(f'The `ConcurrencyLimiter` only works with `Searcher` objects (got {type(searcher)}). Please try to pass `max_concurrent` to the search generator directly.')
        self._set_searcher_max_concurrency()
        super(ConcurrencyLimiter, self).__init__(metric=self.searcher.metric, mode=self.searcher.mode)

    def _set_searcher_max_concurrency(self):
        if False:
            return 10
        self._limit_concurrency = not self.searcher.set_max_concurrency(self.max_concurrent)

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self.max_concurrent = max_concurrent
        return True

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        if False:
            while True:
                i = 10
        self._set_searcher_max_concurrency()
        return _set_search_properties_backwards_compatible(self.searcher.set_search_properties, metric, mode, config, **spec)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            for i in range(10):
                print('nop')
        if not self._limit_concurrency:
            return self.searcher.suggest(trial_id)
        assert trial_id not in self.live_trials, f'Trial ID {trial_id} must be unique: already found in set.'
        if len(self.live_trials) >= self.max_concurrent:
            logger.debug(f'Not providing a suggestion for {trial_id} due to concurrency limit: %s/%s.', len(self.live_trials), self.max_concurrent)
            return
        suggestion = self.searcher.suggest(trial_id)
        if suggestion not in (None, Searcher.FINISHED):
            self.live_trials.add(trial_id)
            self.num_unfinished_live_trials += 1
        return suggestion

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if False:
            return 10
        if not self._limit_concurrency:
            return self.searcher.on_trial_complete(trial_id, result=result, error=error)
        if trial_id not in self.live_trials:
            return
        elif self.batch:
            self.cached_results[trial_id] = (result, error)
            self.num_unfinished_live_trials -= 1
            if self.num_unfinished_live_trials <= 0:
                for (trial_id, (result, error)) in self.cached_results.items():
                    self.searcher.on_trial_complete(trial_id, result=result, error=error)
                    self.live_trials.remove(trial_id)
                self.cached_results = {}
                self.num_unfinished_live_trials = 0
            else:
                return
        else:
            self.searcher.on_trial_complete(trial_id, result=result, error=error)
            self.live_trials.remove(trial_id)
            self.num_unfinished_live_trials -= 1

    def on_trial_result(self, trial_id: str, result: Dict) -> None:
        if False:
            while True:
                i = 10
        self.searcher.on_trial_result(trial_id, result)

    def add_evaluated_point(self, parameters: Dict, value: float, error: bool=False, pruned: bool=False, intermediate_values: Optional[List[float]]=None):
        if False:
            return 10
        return self.searcher.add_evaluated_point(parameters, value, error, pruned, intermediate_values)

    def get_state(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        state = self.__dict__.copy()
        del state['searcher']
        return copy.deepcopy(state)

    def set_state(self, state: Dict):
        if False:
            return 10
        self.__dict__.update(state)

    def save(self, checkpoint_path: str):
        if False:
            i = 10
            return i + 15
        self.searcher.save(checkpoint_path)

    def restore(self, checkpoint_path: str):
        if False:
            for i in range(10):
                print('nop')
        self.searcher.restore(checkpoint_path)