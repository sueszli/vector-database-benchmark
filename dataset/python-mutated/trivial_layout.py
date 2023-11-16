"""Choose a Layout by assigning ``n`` circuit qubits to device qubits ``0, .., n-1``."""
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

class TrivialLayout(AnalysisPass):
    """Choose a Layout by assigning ``n`` circuit qubits to device qubits ``0, .., n-1``.

    A pass for choosing a Layout of a circuit onto a Coupling graph, using a simple
    round-robin order.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.

    Does not assume any ancilla.
    """

    def __init__(self, coupling_map):
        if False:
            while True:
                i = 10
        'TrivialLayout initializer.\n\n        Args:\n            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.\n\n        Raises:\n            TranspilerError: if invalid options\n        '
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the TrivialLayout pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): DAG to find layout for.\n\n        Raises:\n            TranspilerError: if dag wider than the target backend\n        '
        if self.target is not None:
            if dag.num_qubits() > self.target.num_qubits:
                raise TranspilerError('Number of qubits greater than device.')
        elif dag.num_qubits() > self.coupling_map.size():
            raise TranspilerError('Number of qubits greater than device.')
        self.property_set['layout'] = Layout.generate_trivial_layout(*dag.qubits + list(dag.qregs.values()))