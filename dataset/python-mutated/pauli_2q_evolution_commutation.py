"""An analysis pass to find evolution gates in which the Paulis commute."""
from typing import Tuple
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import Commuting2qBlock

class FindCommutingPauliEvolutions(TransformationPass):
    """Finds :class:`.PauliEvolutionGate`s where the operators, that are evolved, all commute."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            print('Hello World!')
        'Check for :class:`.PauliEvolutionGate`s where the summands all commute.\n\n        Args:\n            The DAG circuit in which to look for the commuting evolutions.\n\n        Returns:\n            The dag in which :class:`.PauliEvolutionGate`s made of commuting two-qubit Paulis\n            have been replaced with :class:`.Commuting2qBlocks`` gate instructions. These gates\n            contain nodes of two-qubit :class:`.PauliEvolutionGate`s.\n        '
        for node in dag.op_nodes():
            if isinstance(node.op, PauliEvolutionGate):
                operator = node.op.operator
                if self.single_qubit_terms_only(operator):
                    continue
                if self.summands_commute(node.op.operator):
                    sub_dag = self._decompose_to_2q(dag, node.op)
                    block_op = Commuting2qBlock(set(sub_dag.op_nodes()))
                    wire_order = {wire: idx for (idx, wire) in enumerate(dag.qubits)}
                    dag.replace_block_with_op([node], block_op, wire_order)
        return dag

    @staticmethod
    def single_qubit_terms_only(operator: SparsePauliOp) -> bool:
        if False:
            return 10
        'Determine if the Paulis are made of single qubit terms only.\n\n        Args:\n            operator: The operator to check if it consists only of single qubit terms.\n\n        Returns:\n            True if the operator consists of only single qubit terms (like ``IIX + IZI``),\n            and False otherwise.\n        '
        for pauli in operator.paulis:
            if sum(np.logical_or(pauli.x, pauli.z)) > 1:
                return False
        return True

    @staticmethod
    def summands_commute(operator: SparsePauliOp) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if all summands in the evolved operator commute.\n\n        Args:\n            operator: The operator to check if all its summands commute.\n\n        Returns:\n            True if all summands commute, False otherwise.\n        '
        commuting_subparts = operator.paulis.group_qubit_wise_commuting()
        return len(commuting_subparts) == 1

    @staticmethod
    def _pauli_to_edge(pauli: Pauli) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        'Convert a pauli to an edge.\n\n        Args:\n            pauli: A pauli that is converted to a string to find out where non-identity\n                Paulis are.\n\n        Returns:\n            A tuple representing where the Paulis are. For example, the Pauli "IZIZ" will\n            return (0, 2) since virtual qubits 0 and 2 interact.\n\n        Raises:\n            QiskitError: If the pauli does not exactly have two non-identity terms.\n        '
        edge = tuple(np.logical_or(pauli.x, pauli.z).nonzero()[0])
        if len(edge) != 2:
            raise QiskitError(f'{pauli} does not have length two.')
        return edge

    def _decompose_to_2q(self, dag: DAGCircuit, op: PauliEvolutionGate) -> DAGCircuit:
        if False:
            i = 10
            return i + 15
        'Decompose the PauliSumOp into two-qubit.\n\n        Args:\n            dag: The dag needed to get access to qubits.\n            op: The operator with all the Pauli terms we need to apply.\n\n        Returns:\n            A dag made of two-qubit :class:`.PauliEvolutionGate`.\n        '
        sub_dag = dag.copy_empty_like()
        required_paulis = {self._pauli_to_edge(pauli): (pauli, coeff) for (pauli, coeff) in zip(op.operator.paulis, op.operator.coeffs)}
        for (edge, (pauli, coeff)) in required_paulis.items():
            qubits = [dag.qubits[edge[0]], dag.qubits[edge[1]]]
            simple_pauli = Pauli(pauli.to_label().replace('I', ''))
            pauli_2q = PauliEvolutionGate(simple_pauli, op.time * np.real(coeff))
            sub_dag.apply_operation_back(pauli_2q, qubits)
        return sub_dag