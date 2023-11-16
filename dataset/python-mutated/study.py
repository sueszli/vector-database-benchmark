import types
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from optuna import logging
from optuna import multi_objective
from optuna._deprecated import deprecated_class
from optuna._deprecated import deprecated_func
from optuna.pruners import NopPruner
from optuna.storages import BaseStorage
from optuna.study import create_study as _create_study
from optuna.study import load_study as _load_study
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
ObjectiveFuncType = Callable[['multi_objective.trial.MultiObjectiveTrial'], Sequence[float]]
CallbackFuncType = Callable[['multi_objective.study.MultiObjectiveStudy', 'multi_objective.trial.FrozenMultiObjectiveTrial'], None]
_logger = logging.get_logger(__name__)

@deprecated_func('2.4.0', '4.0.0')
def create_study(directions: List[str], study_name: Optional[str]=None, storage: Optional[Union[str, BaseStorage]]=None, sampler: Optional['multi_objective.samplers.BaseMultiObjectiveSampler']=None, load_if_exists: bool=False) -> 'multi_objective.study.MultiObjectiveStudy':
    if False:
        while True:
            i = 10
    'Create a new :class:`~optuna.multi_objective.study.MultiObjectiveStudy`.\n\n    Example:\n\n        .. testcode::\n\n            import optuna\n\n\n            def objective(trial):\n                # Binh and Korn function.\n                x = trial.suggest_float("x", 0, 5)\n                y = trial.suggest_float("y", 0, 3)\n\n                v0 = 4 * x**2 + 4 * y**2\n                v1 = (x - 5) ** 2 + (y - 5) ** 2\n                return v0, v1\n\n\n            study = optuna.multi_objective.create_study(["minimize", "minimize"])\n            study.optimize(objective, n_trials=3)\n\n    Args:\n        directions:\n            Optimization direction for each objective value.\n            Set ``minimize`` for minimization and ``maximize`` for maximization.\n        study_name:\n            Study\'s name. If this argument is set to None, a unique name is generated\n            automatically.\n        storage:\n            Database URL. If this argument is set to None, in-memory storage is used, and the\n            :class:`~optuna.study.Study` will not be persistent.\n\n            .. note::\n                When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle\n                the database. Please refer to `SQLAlchemy\'s document`_ for further details.\n                If you want to specify non-default options to `SQLAlchemy Engine`_, you can\n                instantiate :class:`~optuna.storages.RDBStorage` with your desired options and\n                pass it to the ``storage`` argument instead of a URL.\n\n             .. _SQLAlchemy: https://www.sqlalchemy.org/\n             .. _SQLAlchemy\'s document:\n                 https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls\n             .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html\n\n        sampler:\n            A sampler object that implements background algorithm for value suggestion.\n            If :obj:`None` is specified,\n            :class:`~optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler` is used\n            as the default. See also :class:`~optuna.multi_objective.samplers`.\n        load_if_exists:\n            Flag to control the behavior to handle a conflict of study names.\n            In the case where a study named ``study_name`` already exists in the ``storage``,\n            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is\n            set to :obj:`False`.\n            Otherwise, the creation of the study is skipped, and the existing one is returned.\n\n    Returns:\n        A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object.\n    '
    mo_sampler = sampler or multi_objective.samplers.NSGAIIMultiObjectiveSampler()
    sampler_adapter = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)
    if not isinstance(directions, Iterable):
        raise TypeError('`directions` must be a list or other iterable types.')
    if not all((d in ['minimize', 'maximize'] for d in directions)):
        raise ValueError('`directions` includes unknown direction names.')
    study = _create_study(study_name=study_name, storage=storage, sampler=sampler_adapter, pruner=NopPruner(), load_if_exists=load_if_exists)
    study._storage.set_study_system_attr(study._study_id, 'multi_objective:study:directions', list(directions))
    return MultiObjectiveStudy(study)

@deprecated_func('2.4.0', '4.0.0')
def load_study(study_name: str, storage: Union[str, BaseStorage], sampler: Optional['multi_objective.samplers.BaseMultiObjectiveSampler']=None) -> 'multi_objective.study.MultiObjectiveStudy':
    if False:
        i = 10
        return i + 15
    'Load the existing :class:`MultiObjectiveStudy` that has the specified name.\n\n    Example:\n\n        .. testsetup::\n\n            import os\n\n            if os.path.exists("example.db"):\n                raise RuntimeError("\'example.db\' already exists. Please remove it.")\n\n        .. testcode::\n\n            import optuna\n\n\n            def objective(trial):\n                # Binh and Korn function.\n                x = trial.suggest_float("x", 0, 5)\n                y = trial.suggest_float("y", 0, 3)\n\n                v0 = 4 * x**2 + 4 * y**2\n                v1 = (x - 5) ** 2 + (y - 5) ** 2\n                return v0, v1\n\n\n            study = optuna.multi_objective.create_study(\n                directions=["minimize", "minimize"],\n                study_name="my_study",\n                storage="sqlite:///example.db",\n            )\n            study.optimize(objective, n_trials=3)\n\n            loaded_study = optuna.multi_objective.study.load_study(\n                study_name="my_study", storage="sqlite:///example.db"\n            )\n            assert len(loaded_study.trials) == len(study.trials)\n\n        .. testcleanup::\n\n            os.remove("example.db")\n\n    Args:\n        study_name:\n            Study\'s name. Each study has a unique name as an identifier.\n        storage:\n            Database URL such as ``sqlite:///example.db``. Please see also the documentation of\n            :func:`~optuna.multi_objective.study.create_study` for further details.\n        sampler:\n            A sampler object that implements background algorithm for value suggestion.\n            If :obj:`None` is specified,\n            :class:`~optuna.multi_objective.samplers.RandomMultiObjectiveSampler` is used\n            as the default. See also :class:`~optuna.multi_objective.samplers`.\n\n    Returns:\n        A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object.\n    '
    mo_sampler = sampler or multi_objective.samplers.RandomMultiObjectiveSampler()
    sampler_adapter = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)
    study = _load_study(study_name=study_name, storage=storage, sampler=sampler_adapter)
    return MultiObjectiveStudy(study)

@deprecated_class('2.4.0', '4.0.0')
class MultiObjectiveStudy:
    """A study corresponds to a multi-objective optimization task, i.e., a set of trials.

    This object provides interfaces to run a new
    :class:`~optuna.multi_objective.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    Note that the direct use of this constructor is not recommended.
    To create and load a study, please refer to the documentation of
    :func:`~optuna.multi_objective.study.create_study` and
    :func:`~optuna.multi_objective.study.load_study` respectively.
    """

    def __init__(self, study: Study):
        if False:
            return 10
        self._study = study
        self._directions = []
        for d in study._storage.get_study_system_attrs(study._study_id)['multi_objective:study:directions']:
            if d == 'minimize':
                self._directions.append(StudyDirection.MINIMIZE)
            elif d == 'maximize':
                self._directions.append(StudyDirection.MAXIMIZE)
            else:
                raise ValueError('Unknown direction ({}) is specified.'.format(d))
        n_objectives = len(self._directions)
        if n_objectives < 1:
            raise ValueError('The number of objectives must be greater than 0.')
        self._study._log_completed_trial = types.MethodType(_log_completed_trial, self._study)

    @property
    def n_objectives(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of objectives.\n\n        Returns:\n            Number of objectives.\n        '
        return len(self._directions)

    @property
    def directions(self) -> List[StudyDirection]:
        if False:
            return 10
        'Return the optimization direction list.\n\n        Returns:\n            A list that contains the optimization direction for each objective value.\n        '
        return self._directions

    @property
    def sampler(self) -> 'multi_objective.samplers.BaseMultiObjectiveSampler':
        if False:
            print('Hello World!')
        'Return the sampler.\n\n        Returns:\n            A :class:`~multi_objective.samplers.BaseMultiObjectiveSampler` object.\n        '
        adapter = self._study.sampler
        assert isinstance(adapter, multi_objective.samplers._MultiObjectiveSamplerAdapter)
        return adapter._mo_sampler

    def optimize(self, objective: ObjectiveFuncType, timeout: Optional[int]=None, n_trials: Optional[int]=None, n_jobs: int=1, catch: Tuple[Type[Exception], ...]=(), callbacks: Optional[List[CallbackFuncType]]=None, gc_after_trial: bool=True, show_progress_bar: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Optimize an objective function.\n\n        This method is the same as :func:`optuna.study.Study.optimize` except for\n        taking an objective function that returns multi-objective values as the argument.\n\n        Please refer to the documentation of :func:`optuna.study.Study.optimize`\n        for further details.\n\n        Example:\n\n            .. testcode::\n\n                import optuna\n\n\n                def objective(trial):\n                    # Binh and Korn function.\n                    x = trial.suggest_float("x", 0, 5)\n                    y = trial.suggest_float("y", 0, 3)\n\n                    v0 = 4 * x**2 + 4 * y**2\n                    v1 = (x - 5) ** 2 + (y - 5) ** 2\n                    return v0, v1\n\n\n                study = optuna.multi_objective.create_study(["minimize", "minimize"])\n                study.optimize(objective, n_trials=3)\n        '

        def mo_objective(trial: Trial) -> float:
            if False:
                while True:
                    i = 10
            mo_trial = multi_objective.trial.MultiObjectiveTrial(trial)
            values = objective(mo_trial)
            mo_trial._report_complete_values(values)
            return 0.0

        def wrap_mo_callback(callback: CallbackFuncType) -> Callable[[Study, FrozenTrial], None]:
            if False:
                print('Hello World!')
            return lambda study, trial: callback(MultiObjectiveStudy(study), multi_objective.trial.FrozenMultiObjectiveTrial(self.n_objectives, trial))
        if callbacks is None:
            wrapped_callbacks = None
        else:
            wrapped_callbacks = [wrap_mo_callback(callback) for callback in callbacks]
        self._study.optimize(mo_objective, timeout=timeout, n_trials=n_trials, n_jobs=n_jobs, catch=catch, callbacks=wrapped_callbacks, gc_after_trial=gc_after_trial, show_progress_bar=show_progress_bar)

    @property
    def user_attrs(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Return user attributes.\n\n        Returns:\n            A dictionary containing all user attributes.\n        '
        return self._study.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return system attributes.\n\n        Returns:\n            A dictionary containing all system attributes.\n        '
        return self._study._storage.get_study_system_attrs(self._study._study_id)

    def set_user_attr(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        'Set a user attribute to the study.\n\n        Args:\n            key: A key string of the attribute.\n            value: A value of the attribute. The value should be JSON serializable.\n        '
        self._study.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set a system attribute to the study.\n\n        Note that Optuna internally uses this method to save system messages. Please use\n        :func:`~optuna.multi_objective.study.MultiObjectiveStudy.set_user_attr`\n        to set users' attributes.\n\n        Args:\n            key: A key string of the attribute.\n            value: A value of the attribute. The value should be JSON serializable.\n\n        "
        self._study._storage.set_study_system_attr(self._study._study_id, key, value)

    def enqueue_trial(self, params: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Enqueue a trial with given parameter values.\n\n        You can fix the next sampling parameters which will be evaluated in your\n        objective function.\n\n        Please refer to the documentation of :func:`optuna.study.Study.enqueue_trial`\n        for further details.\n\n        Args:\n            params:\n                Parameter values to pass your objective function.\n        '
        self._study.enqueue_trial(params, skip_if_exists=False)

    @property
    def trials(self) -> List['multi_objective.trial.FrozenMultiObjectiveTrial']:
        if False:
            return 10
        'Return all trials in the study.\n\n        The returned trials are ordered by trial number.\n\n        This is a short form of ``self.get_trials(deepcopy=True, states=None)``.\n\n        Returns:\n            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.\n        '
        return self.get_trials(deepcopy=True, states=None)

    def get_trials(self, deepcopy: bool=True, states: Optional[Container[TrialState]]=None) -> List['multi_objective.trial.FrozenMultiObjectiveTrial']:
        if False:
            while True:
                i = 10
        "Return all trials in the study.\n\n        The returned trials are ordered by trial number.\n\n        Args:\n            deepcopy:\n                Flag to control whether to apply ``copy.deepcopy()`` to the trials.\n                Note that if you set the flag to :obj:`False`, you shouldn't mutate\n                any fields of the returned trial. Otherwise the internal state of\n                the study may corrupt and unexpected behavior may happen.\n            states:\n                Trial states to filter on. If :obj:`None`, include all states.\n\n        Returns:\n            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.\n        "
        return [multi_objective.trial.FrozenMultiObjectiveTrial(self.n_objectives, t) for t in self._study.get_trials(deepcopy=deepcopy, states=states)]

    def get_pareto_front_trials(self) -> List['multi_objective.trial.FrozenMultiObjectiveTrial']:
        if False:
            i = 10
            return i + 15
        "Return trials located at the pareto front in the study.\n\n        A trial is located at the pareto front if there are no trials that dominate the trial.\n        It's called that a trial ``t0`` dominates another trial ``t1`` if\n        ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and\n        ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.\n\n        Returns:\n            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.\n        "
        pareto_front = []
        trials = [t for t in self.trials if t.state == TrialState.COMPLETE]
        for trial in trials:
            dominated = False
            for other in trials:
                if other._dominates(trial, self.directions):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(trial)
        return pareto_front

    @property
    def _storage(self) -> BaseStorage:
        if False:
            i = 10
            return i + 15
        return self._study._storage

    @property
    def _study_id(self) -> int:
        if False:
            return 10
        return self._study._study_id

def _log_completed_trial(self: Study, trial: FrozenTrial) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not _logger.isEnabledFor(logging.INFO):
        return
    n_objectives = len(self.directions)
    frozen_multi_objective_trial = multi_objective.trial.FrozenMultiObjectiveTrial(n_objectives, trial)
    actual_values = frozen_multi_objective_trial.values
    _logger.info('Trial {} finished with values: {} with parameters: {}.'.format(trial.number, actual_values, trial.params))