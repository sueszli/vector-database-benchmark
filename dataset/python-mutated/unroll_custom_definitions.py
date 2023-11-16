"""Unrolls instructions with custom definitions."""
from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag

class UnrollCustomDefinitions(TransformationPass):
    """Unrolls instructions with custom definitions."""

    def __init__(self, equivalence_library, basis_gates=None, target=None, min_qubits=0):
        if False:
            for i in range(10):
                print('nop')
        "Unrolls instructions with custom definitions.\n\n        Args:\n            equivalence_library (EquivalenceLibrary): The equivalence library\n                which will be used by the BasisTranslator pass. (Instructions in\n                this library will not be unrolled by this pass.)\n            basis_gates (Optional[list[str]]): Target basis names to unroll to, e.g. ``['u3', 'cx']``.\n                Ignored if ``target`` is also specified.\n            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation\n                target. When specified, any argument specified for ``basis_gates`` is ignored.\n             min_qubits (int): The minimum number of qubits for operations in the input\n                 dag to translate.\n        "
        super().__init__()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._target = target
        self._min_qubits = min_qubits

    def run(self, dag):
        if False:
            i = 10
            return i + 15
        'Run the UnrollCustomDefinitions pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): input dag\n\n        Raises:\n            QiskitError: if unable to unroll given the basis due to undefined\n            decomposition rules (such as a bad basis) or excessive recursion.\n\n        Returns:\n            DAGCircuit: output unrolled dag\n        '
        if self._basis_gates is None and self._target is None:
            return dag
        if self._target is None:
            basic_insts = {'measure', 'reset', 'barrier', 'snapshot', 'delay'}
            device_insts = basic_insts | set(self._basis_gates)
        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue
            if getattr(node.op, '_directive', False):
                continue
            if dag.has_calibration_for(node) or len(node.qargs) < self._min_qubits:
                continue
            controlled_gate_open_ctrl = isinstance(node.op, ControlledGate) and node.op._open_ctrl
            if not controlled_gate_open_ctrl:
                inst_supported = self._target.instruction_supported(operation_name=node.op.name, qargs=tuple((dag.find_bit(x).index for x in node.qargs))) if self._target is not None else node.name in device_insts
                if inst_supported or self._equiv_lib.has_entry(node.op):
                    continue
            try:
                unrolled = getattr(node.op, 'definition', None)
            except TypeError as err:
                raise QiskitError(f'Error decomposing node {node.name}: {err}') from err
            if unrolled is None:
                raise QiskitError('Cannot unroll the circuit to the given basis, %s. Instruction %s not found in equivalence library and no rule found to expand.' % (str(self._basis_gates), node.op.name))
            decomposition = circuit_to_dag(unrolled, copy_operations=False)
            unrolled_dag = self.run(decomposition)
            dag.substitute_node_with_dag(node, unrolled_dag)
        return dag