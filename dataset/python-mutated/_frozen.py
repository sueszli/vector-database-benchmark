import datetime
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
import warnings
from optuna import distributions
from optuna import logging
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_func
from optuna._typing import JSONSerializable
from optuna.distributions import _convert_old_distribution_to_new_distribution
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial._base import _SUGGEST_INT_POSITIONAL_ARGS
from optuna.trial._base import BaseTrial
from optuna.trial._state import TrialState
_logger = logging.get_logger(__name__)
_suggest_deprecated_msg = 'Use suggest_float{args} instead.'

class FrozenTrial(BaseTrial):
    """Status and results of a :class:`~optuna.trial.Trial`.

    An object of this class has the same methods as :class:`~optuna.trial.Trial`, but is not
    associated with, nor has any references to a :class:`~optuna.study.Study`.

    It is therefore not possible to make persistent changes to a storage from this object by
    itself, for instance by using :func:`~optuna.trial.FrozenTrial.set_user_attr`.

    It will suggest the parameter values stored in :attr:`params` and will not sample values from
    any distributions.

    It can be passed to objective functions (see :func:`~optuna.study.Study.optimize`) and is
    useful for deploying optimization results.

    Example:

        Re-evaluate an objective function with parameter values optimized study.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x**2


            study = optuna.create_study()
            study.optimize(objective, n_trials=3)

            assert objective(study.best_trial) == study.best_value

    .. note::
        Instances are mutable, despite the name.
        For instance, :func:`~optuna.trial.FrozenTrial.set_user_attr` will update user attributes
        of objects in-place.


        Example:

            Overwritten attributes.

            .. testcode::

                import copy
                import datetime

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", -1, 1)

                    # this user attribute always differs
                    trial.set_user_attr("evaluation time", datetime.datetime.now())

                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                best_trial = study.best_trial
                best_trial_copy = copy.deepcopy(best_trial)

                # re-evaluate
                objective(best_trial)

                # the user attribute is overwritten by re-evaluation
                assert best_trial.user_attrs != best_trial_copy.user_attrs

    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.


    Attributes:
        number:
            Unique and consecutive number of :class:`~optuna.trial.Trial` for each
            :class:`~optuna.study.Study`. Note that this field uses zero-based numbering.
        state:
            :class:`TrialState` of the :class:`~optuna.trial.Trial`.
        value:
            Objective value of the :class:`~optuna.trial.Trial`.
            ``value`` and ``values`` must not be specified at the same time.
        values:
            Sequence of objective values of the :class:`~optuna.trial.Trial`.
            The length is greater than 1 if the problem is multi-objective optimization.
            ``value`` and ``values`` must not be specified at the same time.
        datetime_start:
            Datetime where the :class:`~optuna.trial.Trial` started.
        datetime_complete:
            Datetime where the :class:`~optuna.trial.Trial` finished.
        params:
            Dictionary that contains suggested parameters.
        distributions:
            Dictionary that contains the distributions of :attr:`params`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` set with
            :func:`optuna.trial.Trial.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` set with
            :func:`optuna.trial.Trial.set_system_attr`.
        intermediate_values:
            Intermediate objective values set with :func:`optuna.trial.Trial.report`.
    """

    def __init__(self, number: int, state: TrialState, value: Optional[float], datetime_start: Optional[datetime.datetime], datetime_complete: Optional[datetime.datetime], params: Dict[str, Any], distributions: Dict[str, BaseDistribution], user_attrs: Dict[str, Any], system_attrs: Dict[str, Any], intermediate_values: Dict[int, float], trial_id: int, *, values: Optional[Sequence[float]]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._number = number
        self.state = state
        self._values: Optional[List[float]] = None
        if value is not None and values is not None:
            raise ValueError('Specify only one of `value` and `values`.')
        elif value is not None:
            self._values = [value]
        elif values is not None:
            self._values = list(values)
        self._datetime_start = datetime_start
        self.datetime_complete = datetime_complete
        self._params = params
        self._user_attrs = user_attrs
        self._system_attrs = system_attrs
        self.intermediate_values = intermediate_values
        self._distributions = distributions
        self._trial_id = trial_id

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:
        if False:
            return 10
        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return self.number < other.number

    def __le__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return self.number <= other.number

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(tuple((getattr(self, field) for field in self.__dict__)))

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '{cls}({kwargs})'.format(cls=self.__class__.__name__, kwargs=', '.join(('{field}={value}'.format(field=field if not field.startswith('_') else field[1:], value=repr(getattr(self, field))) for field in self.__dict__)) + ', value=None')

    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float]=None, log: bool=False) -> float:
        if False:
            print('Hello World!')
        return self._suggest(name, FloatDistribution(low, high, log=log, step=step))

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args=''))
    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        if False:
            while True:
                i = 10
        return self.suggest_float(name, low, high)

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args='(..., log=True)'))
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        if False:
            print('Hello World!')
        return self.suggest_float(name, low, high, log=True)

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args='(..., step=...)'))
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        if False:
            i = 10
            return i + 15
        return self.suggest_float(name, low, high, step=q)

    @convert_positional_args(previous_positional_arg_names=_SUGGEST_INT_POSITIONAL_ARGS)
    def suggest_int(self, name: str, low: int, high: int, *, step: int=1, log: bool=False) -> int:
        if False:
            while True:
                i = 10
        return int(self._suggest(name, IntDistribution(low, high, log=log, step=step)))

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool:
        if False:
            return 10
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float:
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        ...

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        if False:
            print('Hello World!')
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        if False:
            i = 10
            return i + 15
        'Interface of report function.\n\n        Since :class:`~optuna.trial.FrozenTrial` is not pruned,\n        this report function does nothing.\n\n        .. seealso::\n            Please refer to :func:`~optuna.trial.FrozenTrial.should_prune`.\n\n        Args:\n            value:\n                A value returned from the objective function.\n            step:\n                Step of the trial (e.g., Epoch of neural network training). Note that pruners\n                assume that ``step`` starts at zero. For example,\n                :class:`~optuna.pruners.MedianPruner` simply checks if ``step`` is less than\n                ``n_warmup_steps`` as the warmup mechanism.\n        '
        pass

    def should_prune(self) -> bool:
        if False:
            return 10
        'Suggest whether the trial should be pruned or not.\n\n        The suggestion is always :obj:`False` regardless of a pruning algorithm.\n\n        .. note::\n            :class:`~optuna.trial.FrozenTrial` only samples one combination of parameters.\n\n        Returns:\n            :obj:`False`.\n        '
        return False

    def set_user_attr(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        self._user_attrs[key] = value

    @deprecated_func('3.1.0', '5.0.0')
    def set_system_attr(self, key: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._system_attrs[key] = value

    def _validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.state != TrialState.WAITING and self.datetime_start is None:
            raise ValueError('`datetime_start` is supposed to be set when the trial state is not waiting.')
        if self.state.is_finished():
            if self.datetime_complete is None:
                raise ValueError('`datetime_complete` is supposed to be set for a finished trial.')
        elif self.datetime_complete is not None:
            raise ValueError('`datetime_complete` is supposed to be None for an unfinished trial.')
        if self.state == TrialState.COMPLETE and self._values is None:
            raise ValueError('`value` is supposed to be set for a complete trial.')
        if set(self.params.keys()) != set(self.distributions.keys()):
            raise ValueError('Inconsistent parameters {} and distributions {}.'.format(set(self.params.keys()), set(self.distributions.keys())))
        for (param_name, param_value) in self.params.items():
            distribution = self.distributions[param_name]
            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_value_in_internal_repr):
                raise ValueError("The value {} of parameter '{}' isn't contained in the distribution {}.".format(param_value, param_name, distribution))

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        if False:
            return 10
        if name not in self._params:
            raise ValueError("The value of the parameter '{}' is not found. Please set it at the construction of the FrozenTrial object.".format(name))
        value = self._params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            warnings.warn("The value {} of the parameter '{}' is out of the range of the distribution {}.".format(value, name, distribution))
        if name in self._distributions:
            distributions.check_distribution_compatibility(self._distributions[name], distribution)
        self._distributions[name] = distribution
        return value

    @property
    def number(self) -> int:
        if False:
            print('Hello World!')
        return self._number

    @number.setter
    def number(self, value: int) -> None:
        if False:
            return 10
        self._number = value

    @property
    def value(self) -> Optional[float]:
        if False:
            return 10
        if self._values is not None:
            if len(self._values) > 1:
                raise RuntimeError('This attribute is not available during multi-objective optimization.')
            return self._values[0]
        return None

    @value.setter
    def value(self, v: Optional[float]) -> None:
        if False:
            while True:
                i = 10
        if self._values is not None:
            if len(self._values) > 1:
                raise RuntimeError('This attribute is not available during multi-objective optimization.')
        if v is not None:
            self._values = [v]
        else:
            self._values = None

    def _get_values(self) -> Optional[List[float]]:
        if False:
            return 10
        return self._values

    def _set_values(self, v: Optional[Sequence[float]]) -> None:
        if False:
            while True:
                i = 10
        if v is not None:
            self._values = list(v)
        else:
            self._values = None
    values = property(_get_values, _set_values)

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        if False:
            return 10
        return self._datetime_start

    @datetime_start.setter
    def datetime_start(self, value: Optional[datetime.datetime]) -> None:
        if False:
            print('Hello World!')
        self._datetime_start = value

    @property
    def params(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._params

    @params.setter
    def params(self, params: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._params = params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        if False:
            return 10
        return self._distributions

    @distributions.setter
    def distributions(self, value: Dict[str, BaseDistribution]) -> None:
        if False:
            i = 10
            return i + 15
        self._distributions = value

    @property
    def user_attrs(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return self._user_attrs

    @user_attrs.setter
    def user_attrs(self, value: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        self._user_attrs = value

    @property
    def system_attrs(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._system_attrs

    @system_attrs.setter
    def system_attrs(self, value: Mapping[str, JSONSerializable]) -> None:
        if False:
            i = 10
            return i + 15
        self._system_attrs = cast(Dict[str, Any], value)

    @property
    def last_step(self) -> Optional[int]:
        if False:
            return 10
        'Return the maximum step of :attr:`intermediate_values` in the trial.\n\n        Returns:\n            The maximum step of intermediates.\n        '
        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        if False:
            while True:
                i = 10
        'Return the elapsed time taken to complete the trial.\n\n        Returns:\n            The duration.\n        '
        if self.datetime_start and self.datetime_complete:
            return self.datetime_complete - self.datetime_start
        else:
            return None

def create_trial(*, state: TrialState=TrialState.COMPLETE, value: Optional[float]=None, values: Optional[Sequence[float]]=None, params: Optional[Dict[str, Any]]=None, distributions: Optional[Dict[str, BaseDistribution]]=None, user_attrs: Optional[Dict[str, Any]]=None, system_attrs: Optional[Dict[str, Any]]=None, intermediate_values: Optional[Dict[int, float]]=None) -> FrozenTrial:
    if False:
        return 10
    'Create a new :class:`~optuna.trial.FrozenTrial`.\n\n    Example:\n\n        .. testcode::\n\n            import optuna\n            from optuna.distributions import CategoricalDistribution\n            from optuna.distributions import FloatDistribution\n\n            trial = optuna.trial.create_trial(\n                params={"x": 1.0, "y": 0},\n                distributions={\n                    "x": FloatDistribution(0, 10),\n                    "y": CategoricalDistribution([-1, 0, 1]),\n                },\n                value=5.0,\n            )\n\n            assert isinstance(trial, optuna.trial.FrozenTrial)\n            assert trial.value == 5.0\n            assert trial.params == {"x": 1.0, "y": 0}\n\n    .. seealso::\n\n        See :func:`~optuna.study.Study.add_trial` for how this function can be used to create a\n        study from existing trials.\n\n    .. note::\n\n        Please note that this is a low-level API. In general, trials that are passed to objective\n        functions are created inside :func:`~optuna.study.Study.optimize`.\n\n    .. note::\n        When ``state`` is :class:`TrialState.COMPLETE`, the following parameters are\n        required:\n\n        * ``params``\n        * ``distributions``\n        * ``value`` or ``values``\n\n    Args:\n        state:\n            Trial state.\n        value:\n            Trial objective value. Must be specified if ``state`` is :class:`TrialState.COMPLETE`.\n            ``value`` and ``values`` must not be specified at the same time.\n        values:\n            Sequence of the trial objective values. The length is greater than 1 if the problem is\n            multi-objective optimization.\n            Must be specified if ``state`` is :class:`TrialState.COMPLETE`.\n            ``value`` and ``values`` must not be specified at the same time.\n        params:\n            Dictionary with suggested parameters of the trial.\n        distributions:\n            Dictionary with parameter distributions of the trial.\n        user_attrs:\n            Dictionary with user attributes.\n        system_attrs:\n            Dictionary with system attributes. Should not have to be used for most users.\n        intermediate_values:\n            Dictionary with intermediate objective values of the trial.\n\n    Returns:\n        Created trial.\n    '
    params = params or {}
    distributions = distributions or {}
    distributions = {key: _convert_old_distribution_to_new_distribution(dist) for (key, dist) in distributions.items()}
    user_attrs = user_attrs or {}
    system_attrs = system_attrs or {}
    intermediate_values = intermediate_values or {}
    if state == TrialState.WAITING:
        datetime_start = None
    else:
        datetime_start = datetime.datetime.now()
    if state.is_finished():
        datetime_complete: Optional[datetime.datetime] = datetime_start
    else:
        datetime_complete = None
    trial = FrozenTrial(number=-1, trial_id=-1, state=state, value=value, values=values, datetime_start=datetime_start, datetime_complete=datetime_complete, params=params, distributions=distributions, user_attrs=user_attrs, system_attrs=system_attrs, intermediate_values=intermediate_values)
    trial._validate()
    return trial