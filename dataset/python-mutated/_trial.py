from __future__ import annotations
from collections import UserDict
import copy
import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import overload
from typing import Sequence
import warnings
import optuna
from optuna import distributions
from optuna import logging
from optuna import pruners
from optuna._convert_positional_args import convert_positional_args
from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial
from optuna.trial._base import _SUGGEST_INT_POSITIONAL_ARGS
from optuna.trial._base import BaseTrial
_logger = logging.get_logger(__name__)
_suggest_deprecated_msg = 'Use suggest_float{args} instead.'

class Trial(BaseTrial):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.study.Study.optimize()` method; hence library users do not care about
    instantiation of this object.

    Args:
        study:
            A :class:`~optuna.study.Study` object.
        trial_id:
            A trial ID that is automatically generated.

    """

    def __init__(self, study: 'optuna.study.Study', trial_id: int) -> None:
        if False:
            i = 10
            return i + 15
        self.study = study
        self._trial_id = trial_id
        self.storage = self.study._storage
        self._cached_frozen_trial = self.storage.get_trial(self._trial_id)
        study = pruners._filter_study(self.study, self._cached_frozen_trial)
        self.study.sampler.before_trial(study, self._cached_frozen_trial)
        self.relative_search_space = self.study.sampler.infer_relative_search_space(study, self._cached_frozen_trial)
        self._relative_params: Optional[Dict[str, Any]] = None
        self._fixed_params = self._cached_frozen_trial.system_attrs.get('fixed_params', {})

    @property
    def relative_params(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        if self._relative_params is None:
            study = pruners._filter_study(self.study, self._cached_frozen_trial)
            self._relative_params = self.study.sampler.sample_relative(study, self._cached_frozen_trial, self.relative_search_space)
        return self._relative_params

    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float]=None, log: bool=False) -> float:
        if False:
            print('Hello World!')
        'Suggest a value for the floating point parameter.\n\n        Example:\n\n            Suggest a momentum, learning rate and scaling factor of learning rate\n            for neural network training.\n\n            .. testcode::\n\n                import numpy as np\n                from sklearn.datasets import load_iris\n                from sklearn.model_selection import train_test_split\n                from sklearn.neural_network import MLPClassifier\n\n                import optuna\n\n                X, y = load_iris(return_X_y=True)\n                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)\n\n\n                def objective(trial):\n                    momentum = trial.suggest_float("momentum", 0.0, 1.0)\n                    learning_rate_init = trial.suggest_float(\n                        "learning_rate_init", 1e-5, 1e-3, log=True\n                    )\n                    power_t = trial.suggest_float("power_t", 0.2, 0.8, step=0.1)\n                    clf = MLPClassifier(\n                        hidden_layer_sizes=(100, 50),\n                        momentum=momentum,\n                        learning_rate_init=learning_rate_init,\n                        solver="sgd",\n                        random_state=0,\n                        power_t=power_t,\n                    )\n                    clf.fit(X_train, y_train)\n\n                    return clf.score(X_valid, y_valid)\n\n\n                study = optuna.create_study(direction="maximize")\n                study.optimize(objective, n_trials=3)\n\n        Args:\n            name:\n                A parameter name.\n            low:\n                Lower endpoint of the range of suggested values. ``low`` is included in the range.\n                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,\n                ``low`` must be larger than 0.\n            high:\n                Upper endpoint of the range of suggested values. ``high`` is included in the range.\n                ``high`` must be greater than or equal to ``low``.\n            step:\n                A step of discretization.\n\n                .. note::\n                    The ``step`` and ``log`` arguments cannot be used at the same time. To set\n                    the ``step`` argument to a float number, set the ``log`` argument to\n                    :obj:`False`.\n            log:\n                A flag to sample the value from the log domain or not.\n                If ``log`` is true, the value is sampled from the range in the log domain.\n                Otherwise, the value is sampled from the range in the linear domain.\n\n                .. note::\n                    The ``step`` and ``log`` arguments cannot be used at the same time. To set\n                    the ``log`` argument to :obj:`True`, set the ``step`` argument to :obj:`None`.\n\n        Returns:\n            A suggested float value.\n\n        .. seealso::\n            :ref:`configurations` tutorial describes more details and flexible usages.\n        '
        distribution = FloatDistribution(low, high, log=log, step=step)
        suggested_value = self._suggest(name, distribution)
        self._check_distribution(name, distribution)
        return suggested_value

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args=''))
    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        if False:
            return 10
        'Suggest a value for the continuous parameter.\n\n        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`\n        in the linear domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of\n        :math:`\\mathsf{low}` will be returned.\n\n        Args:\n            name:\n                A parameter name.\n            low:\n                Lower endpoint of the range of suggested values. ``low`` is included in the range.\n            high:\n                Upper endpoint of the range of suggested values. ``high`` is included in the range.\n\n        Returns:\n            A suggested float value.\n        '
        return self.suggest_float(name, low, high)

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args='(..., log=True)'))
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        if False:
            i = 10
            return i + 15
        'Suggest a value for the continuous parameter.\n\n        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high})`\n        in the log domain. When :math:`\\mathsf{low} = \\mathsf{high}`, the value of\n        :math:`\\mathsf{low}` will be returned.\n\n        Args:\n            name:\n                A parameter name.\n            low:\n                Lower endpoint of the range of suggested values. ``low`` is included in the range.\n            high:\n                Upper endpoint of the range of suggested values. ``high`` is included in the range.\n\n        Returns:\n            A suggested float value.\n        '
        return self.suggest_float(name, low, high, log=True)

    @deprecated_func('3.0.0', '6.0.0', text=_suggest_deprecated_msg.format(args='(..., step=...)'))
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        if False:
            return 10
        'Suggest a value for the discrete parameter.\n\n        The value is sampled from the range :math:`[\\mathsf{low}, \\mathsf{high}]`,\n        and the step of discretization is :math:`q`. More specifically,\n        this method returns one of the values in the sequence\n        :math:`\\mathsf{low}, \\mathsf{low} + q, \\mathsf{low} + 2 q, \\dots,\n        \\mathsf{low} + k q \\le \\mathsf{high}`,\n        where :math:`k` denotes an integer. Note that :math:`high` may be changed due to round-off\n        errors if :math:`q` is not an integer. Please check warning messages to find the changed\n        values.\n\n        Args:\n            name:\n                A parameter name.\n            low:\n                Lower endpoint of the range of suggested values. ``low`` is included in the range.\n            high:\n                Upper endpoint of the range of suggested values. ``high`` is included in the range.\n            q:\n                A step of discretization.\n\n        Returns:\n            A suggested float value.\n        '
        return self.suggest_float(name, low, high, step=q)

    @convert_positional_args(previous_positional_arg_names=_SUGGEST_INT_POSITIONAL_ARGS)
    def suggest_int(self, name: str, low: int, high: int, *, step: int=1, log: bool=False) -> int:
        if False:
            print('Hello World!')
        'Suggest a value for the integer parameter.\n\n        The value is sampled from the integers in :math:`[\\mathsf{low}, \\mathsf{high}]`.\n\n        Example:\n\n            Suggest the number of trees in `RandomForestClassifier <https://scikit-learn.org/\n            stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.\n\n            .. testcode::\n\n                import numpy as np\n                from sklearn.datasets import load_iris\n                from sklearn.ensemble import RandomForestClassifier\n                from sklearn.model_selection import train_test_split\n\n                import optuna\n\n                X, y = load_iris(return_X_y=True)\n                X_train, X_valid, y_train, y_valid = train_test_split(X, y)\n\n\n                def objective(trial):\n                    n_estimators = trial.suggest_int("n_estimators", 50, 400)\n                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)\n                    clf.fit(X_train, y_train)\n                    return clf.score(X_valid, y_valid)\n\n\n                study = optuna.create_study(direction="maximize")\n                study.optimize(objective, n_trials=3)\n\n        Args:\n            name:\n                A parameter name.\n            low:\n                Lower endpoint of the range of suggested values. ``low`` is included in the range.\n                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,\n                ``low`` must be larger than 0.\n            high:\n                Upper endpoint of the range of suggested values. ``high`` is included in the range.\n                ``high`` must be greater than or equal to ``low``.\n            step:\n                A step of discretization.\n\n                .. note::\n                    Note that :math:`\\mathsf{high}` is modified if the range is not divisible by\n                    :math:`\\mathsf{step}`. Please check the warning messages to find the changed\n                    values.\n\n                .. note::\n                    The method returns one of the values in the sequence\n                    :math:`\\mathsf{low}, \\mathsf{low} + \\mathsf{step}, \\mathsf{low} + 2 *\n                    \\mathsf{step}, \\dots, \\mathsf{low} + k * \\mathsf{step} \\le\n                    \\mathsf{high}`, where :math:`k` denotes an integer.\n\n                .. note::\n                    The ``step != 1`` and ``log`` arguments cannot be used at the same time.\n                    To set the ``step`` argument :math:`\\mathsf{step} \\ge 2`, set the\n                    ``log`` argument to :obj:`False`.\n            log:\n                A flag to sample the value from the log domain or not.\n\n                .. note::\n                    If ``log`` is true, at first, the range of suggested values is divided into\n                    grid points of width 1. The range of suggested values is then converted to\n                    a log domain, from which a value is sampled. The uniformly sampled\n                    value is re-converted to the original domain and rounded to the nearest grid\n                    point that we just split, and the suggested value is determined.\n                    For example, if `low = 2` and `high = 8`, then the range of suggested values is\n                    `[2, 3, 4, 5, 6, 7, 8]` and lower values tend to be more sampled than higher\n                    values.\n\n                .. note::\n                    The ``step != 1`` and ``log`` arguments cannot be used at the same time.\n                    To set the ``log`` argument to :obj:`True`, set the ``step`` argument to 1.\n\n        .. seealso::\n            :ref:`configurations` tutorial describes more details and flexible usages.\n        '
        distribution = IntDistribution(low=low, high=high, log=log, step=step)
        suggested_value = int(self._suggest(name, distribution))
        self._check_distribution(name, distribution)
        return suggested_value

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool:
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        if False:
            print('Hello World!')
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        if False:
            for i in range(10):
                print('nop')
        ...

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        if False:
            for i in range(10):
                print('nop')
        'Suggest a value for the categorical parameter.\n\n        The value is sampled from ``choices``.\n\n        Example:\n\n            Suggest a kernel function of `SVC <https://scikit-learn.org/stable/modules/generated/\n            sklearn.svm.SVC.html>`_.\n\n            .. testcode::\n\n                import numpy as np\n                from sklearn.datasets import load_iris\n                from sklearn.model_selection import train_test_split\n                from sklearn.svm import SVC\n\n                import optuna\n\n                X, y = load_iris(return_X_y=True)\n                X_train, X_valid, y_train, y_valid = train_test_split(X, y)\n\n\n                def objective(trial):\n                    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])\n                    clf = SVC(kernel=kernel, gamma="scale", random_state=0)\n                    clf.fit(X_train, y_train)\n                    return clf.score(X_valid, y_valid)\n\n\n                study = optuna.create_study(direction="maximize")\n                study.optimize(objective, n_trials=3)\n\n\n        Args:\n            name:\n                A parameter name.\n            choices:\n                Parameter value candidates.\n\n        .. seealso::\n            :class:`~optuna.distributions.CategoricalDistribution`.\n\n        Returns:\n            A suggested value.\n\n        .. seealso::\n            :ref:`configurations` tutorial describes more details and flexible usages.\n        '
        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        if False:
            print('Hello World!')
        'Report an objective function value for a given step.\n\n        The reported values are used by the pruners to determine whether this trial should be\n        pruned.\n\n        .. seealso::\n            Please refer to :class:`~optuna.pruners.BasePruner`.\n\n        .. note::\n            The reported value is converted to ``float`` type by applying ``float()``\n            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).\n            If the conversion fails, a ``TypeError`` is raised.\n\n        .. note::\n            If this method is called multiple times at the same ``step`` in a trial,\n            the reported ``value`` only the first time is stored and the reported values\n            from the second time are ignored.\n\n        .. note::\n            :func:`~optuna.trial.Trial.report` does not support multi-objective\n            optimization.\n\n        Example:\n\n            Report intermediate scores of `SGDClassifier <https://scikit-learn.org/stable/modules/\n            generated/sklearn.linear_model.SGDClassifier.html>`_ training.\n\n            .. testcode::\n\n                import numpy as np\n                from sklearn.datasets import load_iris\n                from sklearn.linear_model import SGDClassifier\n                from sklearn.model_selection import train_test_split\n\n                import optuna\n\n                X, y = load_iris(return_X_y=True)\n                X_train, X_valid, y_train, y_valid = train_test_split(X, y)\n\n\n                def objective(trial):\n                    clf = SGDClassifier(random_state=0)\n                    for step in range(100):\n                        clf.partial_fit(X_train, y_train, np.unique(y))\n                        intermediate_value = clf.score(X_valid, y_valid)\n                        trial.report(intermediate_value, step=step)\n                        if trial.should_prune():\n                            raise optuna.TrialPruned()\n\n                    return clf.score(X_valid, y_valid)\n\n\n                study = optuna.create_study(direction="maximize")\n                study.optimize(objective, n_trials=3)\n\n\n        Args:\n            value:\n                A value returned from the objective function.\n            step:\n                Step of the trial (e.g., Epoch of neural network training). Note that pruners\n                assume that ``step`` starts at zero. For example,\n                :class:`~optuna.pruners.MedianPruner` simply checks if ``step`` is less than\n                ``n_warmup_steps`` as the warmup mechanism.\n                ``step`` must be a positive integer.\n        '
        if len(self.study.directions) > 1:
            raise NotImplementedError('Trial.report is not supported for multi-objective optimization.')
        try:
            value = float(value)
        except (TypeError, ValueError):
            message = "The `value` argument is of type '{}' but supposed to be a float.".format(type(value).__name__)
            raise TypeError(message) from None
        if step < 0:
            raise ValueError('The `step` argument is {} but cannot be negative.'.format(step))
        if step in self._cached_frozen_trial.intermediate_values:
            warnings.warn('The reported value is ignored because this `step` {} is already reported.'.format(step))
            return
        self.storage.set_trial_intermediate_value(self._trial_id, step, value)
        self._cached_frozen_trial.intermediate_values[step] = value

    def should_prune(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Suggest whether the trial should be pruned or not.\n\n        The suggestion is made by a pruning algorithm associated with the trial and is based on\n        previously reported values. The algorithm can be specified when constructing a\n        :class:`~optuna.study.Study`.\n\n        .. note::\n            If no values have been reported, the algorithm cannot make meaningful suggestions.\n            Similarly, if this method is called multiple times with the exact same set of reported\n            values, the suggestions will be the same.\n\n        .. seealso::\n            Please refer to the example code in :func:`optuna.trial.Trial.report`.\n\n        .. note::\n            :func:`~optuna.trial.Trial.should_prune` does not support multi-objective\n            optimization.\n\n        Returns:\n            A boolean value. If :obj:`True`, the trial should be pruned according to the\n            configured pruning algorithm. Otherwise, the trial should continue.\n        '
        if len(self.study.directions) > 1:
            raise NotImplementedError('Trial.should_prune is not supported for multi-objective optimization.')
        trial = self._get_latest_trial()
        return self.study.pruner.prune(self.study, trial)

    def set_user_attr(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        'Set user attributes to the trial.\n\n        The user attributes in the trial can be access via :func:`optuna.trial.Trial.user_attrs`.\n\n        .. seealso::\n\n            See the recipe on :ref:`attributes`.\n\n        Example:\n\n            Save fixed hyperparameters of neural network training.\n\n            .. testcode::\n\n                import numpy as np\n                from sklearn.datasets import load_iris\n                from sklearn.model_selection import train_test_split\n                from sklearn.neural_network import MLPClassifier\n\n                import optuna\n\n                X, y = load_iris(return_X_y=True)\n                X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)\n\n\n                def objective(trial):\n                    trial.set_user_attr("BATCHSIZE", 128)\n                    momentum = trial.suggest_float("momentum", 0, 1.0)\n                    clf = MLPClassifier(\n                        hidden_layer_sizes=(100, 50),\n                        batch_size=trial.user_attrs["BATCHSIZE"],\n                        momentum=momentum,\n                        solver="sgd",\n                        random_state=0,\n                    )\n                    clf.fit(X_train, y_train)\n\n                    return clf.score(X_valid, y_valid)\n\n\n                study = optuna.create_study(direction="maximize")\n                study.optimize(objective, n_trials=3)\n                assert "BATCHSIZE" in study.best_trial.user_attrs.keys()\n                assert study.best_trial.user_attrs["BATCHSIZE"] == 128\n\n\n        Args:\n            key:\n                A key string of the attribute.\n            value:\n                A value of the attribute. The value should be JSON serializable.\n        '
        self.storage.set_trial_user_attr(self._trial_id, key, value)
        self._cached_frozen_trial.user_attrs[key] = value

    @deprecated_func('3.1.0', '5.0.0')
    def set_system_attr(self, key: str, value: Any) -> None:
        if False:
            return 10
        "Set system attributes to the trial.\n\n        Note that Optuna internally uses this method to save system messages such as failure\n        reason of trials. Please use :func:`~optuna.trial.Trial.set_user_attr` to set users'\n        attributes.\n\n        Args:\n            key:\n                A key string of the attribute.\n            value:\n                A value of the attribute. The value should be JSON serializable.\n        "
        self.storage.set_trial_system_attr(self._trial_id, key, value)
        self._cached_frozen_trial.system_attrs[key] = value

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        if False:
            i = 10
            return i + 15
        storage = self.storage
        trial_id = self._trial_id
        trial = self._get_latest_trial()
        if name in trial.distributions:
            distributions.check_distribution_compatibility(trial.distributions[name], distribution)
            param_value = trial.params[name]
        else:
            if self._is_fixed_param(name, distribution):
                param_value = self._fixed_params[name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            elif self._is_relative_param(name, distribution):
                param_value = self.relative_params[name]
            else:
                study = pruners._filter_study(self.study, trial)
                param_value = self.study.sampler.sample_independent(study, trial, name, distribution)
            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)
            self._cached_frozen_trial.distributions[name] = distribution
            self._cached_frozen_trial.params[name] = param_value
        return param_value

    def _is_fixed_param(self, name: str, distribution: BaseDistribution) -> bool:
        if False:
            i = 10
            return i + 15
        if name not in self._fixed_params:
            return False
        param_value = self._fixed_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn("Fixed parameter '{}' with value {} is out of range for distribution {}.".format(name, param_value, distribution))
        return True

    def _is_relative_param(self, name: str, distribution: BaseDistribution) -> bool:
        if False:
            return 10
        if name not in self.relative_params:
            return False
        if name not in self.relative_search_space:
            raise ValueError("The parameter '{}' was sampled by `sample_relative` method but it is not contained in the relative search space.".format(name))
        relative_distribution = self.relative_search_space[name]
        distributions.check_distribution_compatibility(relative_distribution, distribution)
        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)

    def _check_distribution(self, name: str, distribution: BaseDistribution) -> None:
        if False:
            for i in range(10):
                print('nop')
        old_distribution = self._cached_frozen_trial.distributions.get(name, distribution)
        if old_distribution != distribution:
            warnings.warn('Inconsistent parameter values for distribution with name "{}"! This might be a configuration mistake. Optuna allows to call the same distribution with the same name more than once in a trial. When the parameter values are inconsistent optuna only uses the values of the first call and ignores all following. Using these values: {}'.format(name, old_distribution._asdict()), RuntimeWarning)

    def _get_latest_trial(self) -> FrozenTrial:
        if False:
            return 10
        latest_trial = copy.copy(self._cached_frozen_trial)
        latest_trial.system_attrs = _LazyTrialSystemAttrs(self._trial_id, self.storage)
        return latest_trial

    @property
    def params(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Return parameters to be optimized.\n\n        Returns:\n            A dictionary containing all parameters.\n        '
        return copy.deepcopy(self._cached_frozen_trial.params)

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        if False:
            for i in range(10):
                print('nop')
        'Return distributions of parameters to be optimized.\n\n        Returns:\n            A dictionary containing all distributions.\n        '
        return copy.deepcopy(self._cached_frozen_trial.distributions)

    @property
    def user_attrs(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Return user attributes.\n\n        Returns:\n            A dictionary containing all user attributes.\n        '
        return copy.deepcopy(self._cached_frozen_trial.user_attrs)

    @property
    @deprecated_func('3.1.0', '5.0.0')
    def system_attrs(self) -> Dict[str, Any]:
        if False:
            return 10
        'Return system attributes.\n\n        Returns:\n            A dictionary containing all system attributes.\n        '
        return copy.deepcopy(self.storage.get_trial_system_attrs(self._trial_id))

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        if False:
            return 10
        'Return start datetime.\n\n        Returns:\n            Datetime where the :class:`~optuna.trial.Trial` started.\n        '
        return self._cached_frozen_trial.datetime_start

    @property
    def number(self) -> int:
        if False:
            while True:
                i = 10
        "Return trial's number which is consecutive and unique in a study.\n\n        Returns:\n            A trial number.\n        "
        return self._cached_frozen_trial.number

class _LazyTrialSystemAttrs(UserDict):

    def __init__(self, trial_id: int, storage: optuna.storages.BaseStorage) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._trial_id = trial_id
        self._storage = storage
        self._initialized = False

    def __getattribute__(self, key: str) -> Any:
        if False:
            i = 10
            return i + 15
        if key == 'data':
            if not self._initialized:
                self._initialized = True
                super().update(self._storage.get_trial_system_attrs(self._trial_id))
        return super().__getattribute__(key)