"""Map (with minimum effort) a DAGCircuit onto a ``coupling_map`` adding swap gates."""
from __future__ import annotations
import numpy as np
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.transpiler.target import Target

class LayoutTransformation(TransformationPass):
    """Adds a Swap circuit for a given (partial) permutation to the circuit.

    This circuit is found by a 4-approximation algorithm for Token Swapping.
    More details are available in the routing code.
    """

    def __init__(self, coupling_map: CouplingMap | Target | None, from_layout: Layout | str, to_layout: Layout | str, seed: int | np.random.Generator | None=None, trials=4):
        if False:
            i = 10
            return i + 15
        'LayoutTransformation initializer.\n\n        Args:\n            coupling_map:\n                Directed graph representing a coupling map.\n\n            from_layout (Union[Layout, str]):\n                The starting layout of qubits onto physical qubits.\n                If the type is str, look up `property_set` when this pass runs.\n\n            to_layout (Union[Layout, str]):\n                The final layout of qubits on physical qubits.\n                If the type is str, look up ``property_set`` when this pass runs.\n\n            seed (Union[int, np.random.default_rng]):\n                Seed to use for random trials.\n\n            trials (int):\n                How many randomized trials to perform, taking the best circuit as output.\n        '
        super().__init__()
        self.from_layout = from_layout
        self.to_layout = to_layout
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        if self.coupling_map is None:
            self.coupling_map = CouplingMap.from_full(len(to_layout))
        graph = self.coupling_map.graph.to_undirected()
        self.token_swapper = ApproximateTokenSwapper(graph, seed)
        self.trials = trials

    def run(self, dag):
        if False:
            print('Hello World!')
        'Apply the specified partial permutation to the circuit.\n\n        Args:\n            dag (DAGCircuit): DAG to transform the layout of.\n\n        Returns:\n            DAGCircuit: The DAG with transformed layout.\n\n        Raises:\n            TranspilerError: if the coupling map or the layout are not compatible with the DAG.\n                Or if either of string from/to_layout is not found in `property_set`.\n        '
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('LayoutTransform runs on physical circuits only')
        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')
        from_layout = self.from_layout
        if isinstance(from_layout, str):
            try:
                from_layout = self.property_set[from_layout]
            except Exception as ex:
                raise TranspilerError(f'No {from_layout} (from_layout) in property_set.') from ex
        to_layout = self.to_layout
        if isinstance(to_layout, str):
            try:
                to_layout = self.property_set[to_layout]
            except Exception as ex:
                raise TranspilerError(f'No {to_layout} (to_layout) in property_set.') from ex
        permutation = {pqubit: to_layout.get_virtual_bits()[vqubit] for (vqubit, pqubit) in from_layout.get_virtual_bits().items()}
        perm_circ = self.token_swapper.permutation_circuit(permutation, self.trials)
        qubits = [dag.qubits[i[0]] for i in sorted(perm_circ.inputmap.items(), key=lambda x: x[0])]
        dag.compose(perm_circ.circuit, qubits=qubits)
        return dag