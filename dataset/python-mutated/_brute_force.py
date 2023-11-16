from dataclasses import dataclass
import decimal
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import numpy as np
from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

@dataclass
class _TreeNode:
    param_name: Optional[str] = None
    children: Optional[Dict[float, '_TreeNode']] = None

    def expand(self, param_name: Optional[str], search_space: Iterable[float]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.children is None:
            self.param_name = param_name
            self.children = {value: _TreeNode() for value in search_space}
        else:
            if self.param_name != param_name:
                raise ValueError(f'param_name mismatch: {self.param_name} != {param_name}')
            if self.children.keys() != set(search_space):
                raise ValueError(f'search_space mismatch: {set(self.children.keys())} != {set(search_space)}')

    def set_leaf(self) -> None:
        if False:
            return 10
        self.expand(None, [])

    def add_path(self, params_and_search_spaces: Iterable[Tuple[str, Iterable[float], float]]) -> Optional['_TreeNode']:
        if False:
            return 10
        current_node = self
        for (param_name, search_space, value) in params_and_search_spaces:
            current_node.expand(param_name, search_space)
            assert current_node.children is not None
            if value not in current_node.children:
                return None
            current_node = current_node.children[value]
        return current_node

    def count_unexpanded(self) -> int:
        if False:
            return 10
        return 1 if self.children is None else sum((child.count_unexpanded() for child in self.children.values()))

    def sample_child(self, rng: np.random.RandomState) -> float:
        if False:
            return 10
        assert self.children is not None
        weights = np.array([child.count_unexpanded() for child in self.children.values()], dtype=np.float64)
        weights /= weights.sum()
        return rng.choice(list(self.children.keys()), p=weights)

@experimental_class('3.1.0')
class BruteForceSampler(BaseSampler):
    """Sampler using brute force.

    This sampler performs exhaustive search on the defined search space.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                c = trial.suggest_categorical("c", ["float", "int"])
                if c == "float":
                    return trial.suggest_float("x", 1, 3, step=0.5)
                elif c == "int":
                    a = trial.suggest_int("a", 1, 3)
                    b = trial.suggest_int("b", a, 3)
                    return a + b


            study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
            study.optimize(objective)

    Note:
        The defined search space must be finite. Therefore, when using
        :class:`~optuna.distributions.FloatDistribution` or
        :func:`~optuna.trial.Trial.suggest_float`, ``step=None`` is not allowed.

    Note:
        The sampler may fail to try the entire search space in when the suggestion ranges or
        parameters are changed in the same :class:`~optuna.study.Study`.

    Args:
        seed:
            A seed to fix the order of trials as the search order randomly shuffled. Please note
            that it is not recommended using this option in distributed optimization settings since
            this option cannot ensure the order of trials and may increase the number of duplicate
            suggestions during distributed optimization.
    """

    def __init__(self, seed: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._rng = LazyRandomState(seed)

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> Dict[str, BaseDistribution]:
        if False:
            print('Hello World!')
        return {}

    def sample_relative(self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {}

    @staticmethod
    def _populate_tree(tree: _TreeNode, trials: Iterable[FrozenTrial], params: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        incomplete_leaves: List[_TreeNode] = []
        for trial in trials:
            if not all((p in trial.params and trial.params[p] == v for (p, v) in params.items())):
                continue
            leaf = tree.add_path(((param_name, _enumerate_candidates(param_distribution), param_distribution.to_internal_repr(trial.params[param_name])) for (param_name, param_distribution) in trial.distributions.items() if param_name not in params))
            if leaf is not None:
                if trial.state.is_finished():
                    leaf.set_leaf()
                else:
                    incomplete_leaves.append(leaf)
        for leaf in incomplete_leaves:
            if leaf.children is None:
                leaf.set_leaf()

    def sample_independent(self, study: Study, trial: FrozenTrial, param_name: str, param_distribution: BaseDistribution) -> Any:
        if False:
            print('Hello World!')
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL))
        tree = _TreeNode()
        candidates = _enumerate_candidates(param_distribution)
        tree.expand(param_name, candidates)
        self._populate_tree(tree, (t for t in trials if t.number != trial.number), trial.params)
        if tree.count_unexpanded() == 0:
            return param_distribution.to_external_repr(self._rng.rng.choice(candidates))
        else:
            return param_distribution.to_external_repr(tree.sample_child(self._rng.rng))

    def after_trial(self, study: Study, trial: FrozenTrial, state: TrialState, values: Optional[Sequence[float]]) -> None:
        if False:
            i = 10
            return i + 15
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL))
        tree = _TreeNode()
        self._populate_tree(tree, (t if t.number != trial.number else create_trial(state=state, values=values, params=trial.params, distributions=trial.distributions) for t in trials), {})
        if tree.count_unexpanded() == 0:
            study.stop()

def _enumerate_candidates(param_distribution: BaseDistribution) -> Sequence[float]:
    if False:
        return 10
    if isinstance(param_distribution, FloatDistribution):
        if param_distribution.step is None:
            raise ValueError('FloatDistribution.step must be given for BruteForceSampler (otherwise, the search space will be infinite).')
        low = decimal.Decimal(str(param_distribution.low))
        high = decimal.Decimal(str(param_distribution.high))
        step = decimal.Decimal(str(param_distribution.step))
        ret = []
        value = low
        while value <= high:
            ret.append(float(value))
            value += step
        return ret
    elif isinstance(param_distribution, IntDistribution):
        return list(range(param_distribution.low, param_distribution.high + 1, param_distribution.step))
    elif isinstance(param_distribution, CategoricalDistribution):
        return list(range(len(param_distribution.choices)))
    else:
        raise ValueError(f'Unknown distribution {param_distribution}.')