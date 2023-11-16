"""Reduce 1Q gate complexity by commuting through 2Q gates and resynthesizing."""
from copy import copy
import logging
from collections import deque
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import Optimize1qGatesDecomposition
logger = logging.getLogger(__name__)
commutation_table = {RZXGate: (['rz', 'p'], ['x', 'sx', 'rx']), CXGate: (['rz', 'p'], ['x', 'sx', 'rx'])}
'\nSimple commutation rules: G belongs to commutation_table[barrier_type][qubit_preindex] when G\ncommutes with the indicated barrier on that qubit wire.\n\nNOTE: Does not cover identities like\n          (X (x) I) .   CX = CX .   (X (x) X) ,  (duplication)\n          (U (x) I) . SWAP = SWAP . (I (x) U) .  (permutation)\n\nNOTE: These rules are _symmetric_, so that they may be applied in reverse.\n'

class Optimize1qGatesSimpleCommutation(TransformationPass):
    """
    Optimizes 1Q gate strings interrupted by 2Q gates by commuting the components and
    resynthesizing the results.  The commutation rules are stored in ``commutation_table``.

    NOTE: In addition to those mentioned in ``commutation_table``, this pass has some limitations:
          + Does not handle multiple commutations in a row without intermediate progress.
          + Can only commute into positions where there are pre-existing runs.
          + Does not exhaustively test all the different ways commuting gates can be assigned to
            either side of a barrier to try to find low-depth configurations.  (This is particularly
            evident if all the gates in a run commute with both the predecessor and the successor
            barriers.)
    """

    def __init__(self, basis=None, run_to_completion=False, target=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            basis (List[str]): See also `Optimize1qGatesDecomposition`.\n            run_to_completion (bool): If `True`, this pass retries until it is unable to do any more\n                work.  If `False`, it finds and performs one optimization, and for full optimization\n                the user is obligated to re-call the pass until the output stabilizes.\n            target (Target): The :class:`~.Target` representing the target backend, if both\n                ``basis`` and this are specified then this argument will take\n                precedence and ``basis`` will be ignored.\n        '
        super().__init__()
        self._optimize1q = Optimize1qGatesDecomposition(basis=basis, target=target)
        self._run_to_completion = run_to_completion

    @staticmethod
    def _find_adjoining_run(dag, runs, run, front=True):
        if False:
            while True:
                i = 10
        '\n        Finds the run which abuts `run` from the front (or the rear if `front == False`), separated\n        by a blocking node.\n\n        Returns a pair of the abutting multiqubit gate and the run which it separates from this\n        one. The next run can be the empty list `[]` if it is absent.\n        '
        edge_node = run[0] if front else run[-1]
        blocker = next(dag.predecessors(edge_node) if front else dag.successors(edge_node))
        possibilities = dag.predecessors(blocker) if front else dag.successors(blocker)
        adjoining_run = []
        for possibility in possibilities:
            if isinstance(possibility, DAGOpNode) and possibility.qargs == edge_node.qargs:
                adjoining_run = []
                for single_run in runs:
                    if len(single_run) != 0 and single_run[0].qargs == possibility.qargs:
                        if possibility in single_run:
                            adjoining_run = single_run
                            break
                break
        return (blocker, adjoining_run)

    @staticmethod
    def _commute_through(blocker, run, front=True):
        if False:
            while True:
                i = 10
        '\n        Pulls `DAGOpNode`s from the front of `run` (or the back, if `front == False`) until it\n        encounters a gate which does not commute with `blocker`.\n\n        Returns a pair of lists whose concatenation is `run`.\n        '
        if run == []:
            return ([], [])
        run_clone = deque(run)
        commuted = deque([])
        (preindex, commutation_rule) = (None, None)
        if isinstance(blocker, DAGOpNode):
            preindex = None
            for (i, q) in enumerate(blocker.qargs):
                if q == run[0].qargs[0]:
                    preindex = i
            commutation_rule = None
            if preindex is not None and isinstance(blocker, DAGOpNode) and (blocker.op.base_class in commutation_table):
                commutation_rule = commutation_table[blocker.op.base_class][preindex]
        if commutation_rule is not None:
            while run_clone:
                next_gate = run_clone[0] if front else run_clone[-1]
                if next_gate.name not in commutation_rule:
                    break
                if front:
                    run_clone.popleft()
                    commuted.append(next_gate)
                else:
                    run_clone.pop()
                    commuted.appendleft(next_gate)
        if front:
            return (list(commuted), list(run_clone))
        else:
            return (list(run_clone), list(commuted))

    def _resynthesize(self, run, qubit):
        if False:
            i = 10
            return i + 15
        '\n        Synthesizes an efficient circuit from a sequence `run` of `DAGOpNode`s.\n\n        NOTE: Returns None when resynthesis is not possible.\n        '
        if len(run) == 0:
            dag = DAGCircuit()
            dag.add_qreg(QuantumRegister(1))
            return dag
        operator = run[0].op.to_matrix()
        for gate in run[1:]:
            operator = gate.op.to_matrix().dot(operator)
        return self._optimize1q._gate_sequence_to_dag(self._optimize1q._resynthesize_run(operator, qubit))

    @staticmethod
    def _replace_subdag(dag, old_run, new_dag):
        if False:
            return 10
        '\n        Replaces a nonempty sequence `old_run` of `DAGNode`s, assumed to be a complete chain in\n        `dag`, with the circuit `new_circ`.\n        '
        node_map = dag.substitute_node_with_dag(old_run[0], new_dag)
        for node in old_run[1:]:
            dag.remove_op_node(node)
        spliced_run = [node_map[node._node_id] for node in new_dag.topological_op_nodes()]
        mov_list(old_run, spliced_run)

    def _step(self, dag):
        if False:
            return 10
        '\n        Performs one full pass of optimization work.\n\n        Returns True if `dag` changed, False if no work on `dag` was possible.\n        '
        runs = dag.collect_1q_runs()
        did_work = False
        for run in runs:
            run_clone = copy(run)
            if run == []:
                continue
            (preceding_blocker, preceding_run) = self._find_adjoining_run(dag, runs, run)
            commuted_preceding = []
            if preceding_run != []:
                (commuted_preceding, run_clone) = self._commute_through(preceding_blocker, run_clone)
            (succeeding_blocker, succeeding_run) = self._find_adjoining_run(dag, runs, run, front=False)
            commuted_succeeding = []
            if succeeding_run != []:
                (run_clone, commuted_succeeding) = self._commute_through(succeeding_blocker, run_clone, front=False)
            qubit = dag.find_bit(run[0].qargs[0]).index
            new_preceding_run = self._resynthesize(preceding_run + commuted_preceding, qubit)
            new_succeeding_run = self._resynthesize(commuted_succeeding + succeeding_run, qubit)
            new_run = self._resynthesize(run_clone, qubit)
            if self._optimize1q._substitution_checks(dag, (preceding_run or []) + run + (succeeding_run or []), new_preceding_run.op_nodes() + new_run.op_nodes() + new_succeeding_run.op_nodes(), self._optimize1q._basis_gates, dag.find_bit(run[0].qargs[0]).index):
                if preceding_run and new_preceding_run is not None:
                    self._replace_subdag(dag, preceding_run, new_preceding_run)
                if succeeding_run and new_succeeding_run is not None:
                    self._replace_subdag(dag, succeeding_run, new_succeeding_run)
                if new_run is not None:
                    self._replace_subdag(dag, run, new_run)
                did_work = True
        return did_work

    def run(self, dag):
        if False:
            print('Hello World!')
        '\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        while True:
            did_work = self._step(dag)
            if not self._run_to_completion or not did_work:
                break
        return dag

def mov_list(destination, source):
    if False:
        while True:
            i = 10
    '\n    Replace `destination` in-place with `source`.\n    '
    while destination:
        del destination[0]
    destination += source