from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Union
from optuna import multi_objective
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_class
from optuna.distributions import BaseDistribution
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
from optuna.trial._base import _SUGGEST_INT_POSITIONAL_ARGS
CategoricalChoiceType = Union[None, bool, int, float, str]

@deprecated_class('2.4.0', '4.0.0')
class MultiObjectiveTrial:
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.multi_objective.study.MultiObjectiveStudy.optimize()` method;
    hence library users do not care about instantiation of this object.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object.
    """

    def __init__(self, trial: Trial):
        if False:
            for i in range(10):
                print('nop')
        self._trial = trial
        self._n_objectives = multi_objective.study.MultiObjectiveStudy(trial.study).n_objectives

    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float]=None, log: bool=False) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Suggest a value for the floating point parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_float`\n        for further details.\n        '
        return self._trial.suggest_float(name, low, high, step=step, log=log)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        if False:
            while True:
                i = 10
        'Suggest a value for the continuous parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_uniform`\n        for further details.\n        '
        return self._trial.suggest_uniform(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        if False:
            i = 10
            return i + 15
        'Suggest a value for the continuous parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_loguniform`\n        for further details.\n        '
        return self._trial.suggest_loguniform(name, low, high)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Suggest a value for the discrete parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_discrete_uniform`\n        for further details.\n        '
        return self._trial.suggest_discrete_uniform(name, low, high, q)

    @convert_positional_args(previous_positional_arg_names=_SUGGEST_INT_POSITIONAL_ARGS)
    def suggest_int(self, name: str, low: int, high: int, *, step: int=1, log: bool=False) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Suggest a value for the integer parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_int`\n        for further details.\n        '
        return self._trial.suggest_int(name, low, high, step=step, log=log)

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool:
        if False:
            print('Hello World!')
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int:
        if False:
            print('Hello World!')
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float:
        if False:
            print('Hello World!')
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        if False:
            while True:
                i = 10
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        if False:
            i = 10
            return i + 15
        ...

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        if False:
            while True:
                i = 10
        'Suggest a value for the categorical parameter.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_categorical`\n        for further details.\n        '
        return self._trial.suggest_categorical(name, choices)

    def report(self, values: Sequence[float], step: int) -> None:
        if False:
            while True:
                i = 10
        'Report intermediate objective function values for a given step.\n\n        The reported values are used by the pruners to determine whether this trial should be\n        pruned.\n\n        .. seealso::\n            Please refer to :class:`~optuna.pruners.BasePruner`.\n\n        .. note::\n            The reported values are converted to ``float`` type by applying ``float()``\n            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).\n            If the conversion fails, a ``TypeError`` is raised.\n\n        Args:\n            values:\n                Intermediate objective function values for a given step.\n            step:\n                Step of the trial (e.g., Epoch of neural network training).\n        '
        if len(values) != self._n_objectives:
            raise ValueError('The number of the intermediate values {} at step {} is mismatched withthe number of the objectives {}.', len(values), step, self._n_objectives)
        for (i, value) in enumerate(values):
            self._trial.report(value, self._n_objectives * (step + 1) + i)

    def _report_complete_values(self, values: Sequence[float]) -> None:
        if False:
            return 10
        if len(values) != self._n_objectives:
            raise ValueError('The number of the values {} is mismatched with the number of the objectives {}.', len(values), self._n_objectives)
        for (i, value) in enumerate(values):
            self._trial.report(value, i)

    def set_user_attr(self, key: str, value: Any) -> None:
        if False:
            print('Hello World!')
        'Set user attributes to the trial.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.set_user_attr`\n        for further details.\n        '
        self._trial.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        'Set system attributes to the trial.\n\n        Please refer to the documentation of :func:`optuna.trial.Trial.set_system_attr`\n        for further details.\n        '
        self._trial.storage.set_trial_system_attr(self._trial._trial_id, key, value)

    @property
    def number(self) -> int:
        if False:
            print('Hello World!')
        "Return trial's number which is consecutive and unique in a study.\n\n        Returns:\n            A trial number.\n        "
        return self._trial.number

    @property
    def params(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Return parameters to be optimized.\n\n        Returns:\n            A dictionary containing all parameters.\n        '
        return self._trial.params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        if False:
            i = 10
            return i + 15
        'Return distributions of parameters to be optimized.\n\n        Returns:\n            A dictionary containing all distributions.\n        '
        return self._trial.distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return user attributes.\n\n        Returns:\n            A dictionary containing all user attributes.\n        '
        return self._trial.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Return system attributes.\n\n        Returns:\n            A dictionary containing all system attributes.\n        '
        return self._trial.system_attrs

    @property
    def datetime_start(self) -> Optional[datetime]:
        if False:
            for i in range(10):
                print('nop')
        'Return start datetime.\n\n        Returns:\n            Datetime where the :class:`~optuna.trial.Trial` started.\n        '
        return self._trial.datetime_start

@deprecated_class('2.4.0', '4.0.0')
class FrozenMultiObjectiveTrial:
    """Status and results of a :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.

    Attributes:
        number:
            Unique and consecutive number of
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` for each
            :class:`~optuna.multi_objective.study.MultiObjectiveStudy`.
            Note that this field uses zero-based numbering.
        state:
            :class:`~optuna.trial.TrialState` of the
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.
        values:
            Objective values of the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.
        datetime_start:
            Datetime where the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` started.
        datetime_complete:
            Datetime where the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` finished.
        params:
            Dictionary that contains suggested parameters.
        distributions:
            Dictionary that contains the distributions of :attr:`params`.
        user_attrs:
            Dictionary that contains the attributes of the
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` set with
            :func:`optuna.multi_objective.trial.MultiObjectiveTrial.set_user_attr`.
        intermediate_values:
            Intermediate objective values set with
            :func:`optuna.multi_objective.trial.MultiObjectiveTrial.report`.
    """

    def __init__(self, n_objectives: int, trial: FrozenTrial):
        if False:
            while True:
                i = 10
        self.n_objectives = n_objectives
        self._trial = trial
        self.values = tuple((trial.intermediate_values.get(i) for i in range(n_objectives)))
        intermediate_values: Dict[int, List[Optional[float]]] = {}
        for (key, value) in trial.intermediate_values.items():
            if key < n_objectives:
                continue
            step = key // n_objectives - 1
            if step not in intermediate_values:
                intermediate_values[step] = [None for _ in range(n_objectives)]
            intermediate_values[step][key % n_objectives] = value
        self.intermediate_values = {k: tuple(v) for (k, v) in intermediate_values.items()}

    @property
    def number(self) -> int:
        if False:
            return 10
        return self._trial.number

    @property
    def _trial_id(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._trial._trial_id

    @property
    def state(self) -> TrialState:
        if False:
            for i in range(10):
                print('nop')
        return self._trial.state

    @property
    def datetime_start(self) -> Optional[datetime]:
        if False:
            print('Hello World!')
        return self._trial.datetime_start

    @property
    def datetime_complete(self) -> Optional[datetime]:
        if False:
            while True:
                i = 10
        return self._trial.datetime_complete

    @property
    def params(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._trial.params

    @property
    def user_attrs(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._trial.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._trial.system_attrs

    @property
    def last_step(self) -> Optional[int]:
        if False:
            print('Hello World!')
        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        if False:
            while True:
                i = 10
        return self._trial.distributions

    def _dominates(self, other: 'multi_objective.trial.FrozenMultiObjectiveTrial', directions: List[StudyDirection]) -> bool:
        if False:
            while True:
                i = 10
        if len(self.values) != len(other.values):
            raise ValueError('Trials with different numbers of objectives cannot be compared.')
        if len(self.values) != len(directions):
            raise ValueError('The number of the values and the number of the objectives are mismatched.')
        if self.state != TrialState.COMPLETE:
            return False
        if other.state != TrialState.COMPLETE:
            return True
        values0 = [_normalize_value(v, d) for (v, d) in zip(self.values, directions)]
        values1 = [_normalize_value(v, d) for (v, d) in zip(other.values, directions)]
        if values0 == values1:
            return False
        return all((v0 <= v1 for (v0, v1) in zip(values0, values1)))

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented
        return self._trial == other._trial

    def __lt__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented
        return self._trial < other._trial

    def __le__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented
        return self._trial <= other._trial

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash(self._trial)

def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if False:
        while True:
            i = 10
    if value is None:
        value = float('inf')
    if direction is StudyDirection.MAXIMIZE:
        value = -value
    return value