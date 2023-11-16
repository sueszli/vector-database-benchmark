"""Translate parameterized gates only, and leave others as they are."""
from __future__ import annotations
from qiskit.circuit import Instruction, ParameterExpression, Qubit, Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.equivalence_library import EquivalenceLibrary
from qiskit.exceptions import QiskitError
from qiskit.transpiler import Target
from qiskit.transpiler.basepasses import TransformationPass
from .basis_translator import BasisTranslator

class TranslateParameterizedGates(TransformationPass):
    """Translate parameterized gates to a supported basis set.

    Once a parameterized instruction is found that is not in the ``supported_gates`` list,
    the instruction is decomposed one level and the parameterized sub-blocks are recursively
    decomposed. The recursion is stopped once all parameterized gates are in ``supported_gates``,
    or if a gate has no definition and a translation to the basis is attempted (this might happen
    e.g. for the ``UGate`` if it's not in the specified gate list).

    Example:

        The following, multiply nested circuit::

            from qiskit.circuit import QuantumCircuit, ParameterVector
            from qiskit.transpiler.passes import TranslateParameterizedGates

            x = ParameterVector("x", 4)
            block1 = QuantumCircuit(1)
            block1.rx(x[0], 0)

            sub_block = QuantumCircuit(2)
            sub_block.cx(0, 1)
            sub_block.rz(x[2], 0)

            block2 = QuantumCircuit(2)
            block2.ry(x[1], 0)
            block2.append(sub_block.to_gate(), [0, 1])

            block3 = QuantumCircuit(3)
            block3.ccx(0, 1, 2)

            circuit = QuantumCircuit(3)
            circuit.append(block1.to_gate(), [1])
            circuit.append(block2.to_gate(), [0, 1])
            circuit.append(block3.to_gate(), [0, 1, 2])
            circuit.cry(x[3], 0, 2)

            supported_gates = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]
            unrolled = TranslateParameterizedGates(supported_gates)(circuit)

        is decomposed to::

                 ┌──────────┐     ┌──────────┐┌─────────────┐
            q_0: ┤ Ry(x[1]) ├──■──┤ Rz(x[2]) ├┤0            ├─────■──────
                 ├──────────┤┌─┴─┐└──────────┘│             │     │
            q_1: ┤ Rx(x[0]) ├┤ X ├────────────┤1 circuit-92 ├─────┼──────
                 └──────────┘└───┘            │             │┌────┴─────┐
            q_2: ─────────────────────────────┤2            ├┤ Ry(x[3]) ├
                                              └─────────────┘└──────────┘

    """

    def __init__(self, supported_gates: list[str] | None=None, equivalence_library: EquivalenceLibrary | None=None, target: Target | None=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            supported_gates: A list of suppported basis gates specified as string. If ``None``,\n                a ``target`` must be provided.\n            equivalence_library: The equivalence library to translate the gates. Defaults\n                to the equivalence library of all Qiskit standard gates.\n            target: A :class:`.Target` containing the supported operations. If ``None``,\n                ``supported_gates`` must be set. Note that this argument takes precedence over\n                ``supported_gates``, if both are set.\n\n        Raises:\n            ValueError: If neither of ``supported_gates`` and ``target`` are passed.\n        '
        super().__init__()
        if equivalence_library is None:
            from qiskit.circuit.library.standard_gates.equivalence_library import _sel
            equivalence_library = _sel
        if target is not None:
            supported_gates = target.operation_names
        elif supported_gates is None:
            raise ValueError('One of ``supported_gates`` or ``target`` must be specified.')
        self._supported_gates = supported_gates
        self._target = target
        self._translator = BasisTranslator(equivalence_library, supported_gates, target=target)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            print('Hello World!')
        'Run the transpiler pass.\n\n        Args:\n            dag: The DAG circuit in which the parameterized gates should be unrolled.\n\n        Returns:\n            A DAG where the parameterized gates have been unrolled.\n\n        Raises:\n            QiskitError: If the circuit cannot be unrolled.\n        '
        for node in dag.op_nodes():
            if _is_parameterized(node.op) and (not _is_supported(node, self._supported_gates, self._target)):
                definition = node.op.definition
                if definition is not None:
                    unrolled = self.run(circuit_to_dag(definition))
                else:
                    try:
                        unrolled = self._translator.run(_instruction_to_dag(node.op))
                    except Exception as exc:
                        raise QiskitError('Failed to translate final block.') from exc
                dag.substitute_node_with_dag(node, unrolled)
        return dag

def _is_parameterized(op: Instruction) -> bool:
    if False:
        print('Hello World!')
    return any((isinstance(param, ParameterExpression) and len(param.parameters) > 0 for param in op.params))

def _is_supported(node: DAGOpNode, supported_gates: list[str], target: Target | None) -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether the node is supported.\n\n    If the target is provided, check using the target, otherwise the supported_gates are used.\n    '
    if target is not None:
        return target.instruction_supported(node.op.name)
    return node.op.name in supported_gates

def _instruction_to_dag(op: Instruction) -> DAGCircuit:
    if False:
        while True:
            i = 10
    dag = DAGCircuit()
    dag.add_qubits([Qubit() for _ in range(op.num_qubits)])
    dag.add_qubits([Clbit() for _ in range(op.num_clbits)])
    dag.apply_operation_back(op, dag.qubits, dag.clbits, check=False)
    return dag