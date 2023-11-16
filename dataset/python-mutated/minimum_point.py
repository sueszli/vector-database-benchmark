"""Check if the DAG has reached a relative semi-stable point over previous runs."""
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Tuple
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

class MinimumPoint(TransformationPass):
    """Check if the DAG has reached a relative semi-stable point over previous runs

    This pass is similar to the :class:`~.FixedPoint` transpiler pass and is intended
    primarily to be used to set a loop break condition in the property set.
    However, unlike the :class:`~.FixedPoint` class which only sets the
    condition if 2 consecutive runs have the same value property set value
    this pass is designed to find a local minimum and use that instead. This
    pass is designed for an optimization loop where a fixed point may never
    get reached (for example if synthesis is used and there are multiple
    equivalent outputs for some cases).

    This pass will track the state of fields in the property set over its past
    executions and set a boolean field when either a fixed point is reached
    over the backtracking depth or selecting the minimum value found if the
    backtracking depth is reached. To do this it stores a deep copy of the
    current minimum DAG in the property set and when ``backtrack_depth`` number
    of executions is reached since the last minimum the output dag is set to
    that copy of the earlier minimum.

    Fields used by this pass in the property set are (all relative to the ``prefix``
    argument):

    * ``{prefix}_minimum_point_state`` - Used to track the state of the minimum point search
    * ``{prefix}_minimum_point`` - This value gets set to ``True`` when either a fixed point
        is reached over the ``backtrack_depth`` executions, or ``backtrack_depth`` was exceeded
        and an earlier minimum is restored.
    """

    def __init__(self, property_set_list, prefix, backtrack_depth=5):
        if False:
            while True:
                i = 10
        "Initialize an instance of this pass\n\n        Args:\n            property_set_list (list): A list of property set keys that will\n                be used to evaluate the local minimum. The values of these\n                property set keys will be used as a tuple for comparison\n            prefix (str): The prefix to use for the property set key that is used\n                for tracking previous evaluations\n            backtrack_depth (int): The maximum number of entries to store. If\n                this number is reached and the next iteration doesn't have\n                a decrease in the number of values the minimum of the previous\n                n will be set as the output dag and ``minimum_point`` will be set to\n                ``True`` in the property set\n        "
        super().__init__()
        self.property_set_list = property_set_list
        self.backtrack_name = f'{prefix}_minimum_point_state'
        self.minimum_reached = f'{prefix}_minimum_point'
        self.backtrack_depth = backtrack_depth

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the MinimumPoint pass on `dag`.'
        score = tuple((self.property_set[x] for x in self.property_set_list))
        state = self.property_set[self.backtrack_name]
        if state is None:
            self.property_set[self.backtrack_name] = _MinimumPointState(dag=None, score=(math.inf,) * len(self.property_set_list), since=0)
        elif score > state.score:
            state.since += 1
            if state.since == self.backtrack_depth:
                self.property_set[self.minimum_reached] = True
                return self.property_set[self.backtrack_name].dag
        elif score < state.score:
            state.since = 1
            state.score = score
            state.dag = deepcopy(dag)
        elif score == state.score:
            self.property_set[self.minimum_reached] = True
            return dag
        return dag

@dataclass
class _MinimumPointState:
    __slots__ = ('dag', 'score', 'since')
    dag: DAGCircuit
    score: Tuple[float, ...]
    since: int