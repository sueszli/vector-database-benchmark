"""Replace each block of consecutive gates by a single Unitary node."""
from __future__ import annotations
import numpy as np
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.synthesis import unitary_synthesis
from qiskit.transpiler.passes.utils import _block_to_matrix
from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks

class ConsolidateBlocks(TransformationPass):
    """Replace each block of consecutive gates by a single Unitary node.

    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a Unitary node, to be resynthesized later,
    to a potentially more optimal subcircuit.

    Notes:
        This pass assumes that the 'blocks_list' property that it reads is
        given such that blocks are in topological order. The blocks are
        collected by a previous pass, such as `Collect2qBlocks`.
    """

    def __init__(self, kak_basis_gate=None, force_consolidate=False, basis_gates=None, approximation_degree=1.0, target=None):
        if False:
            i = 10
            return i + 15
        'ConsolidateBlocks initializer.\n\n        If ``kak_basis_gate`` is not ``None`` it will be used as the basis gate for KAK decomposition.\n        Otherwise, if ``basis_gates`` is not ``None`` a basis gate will be chosen from this list.\n        Otherwise, the basis gate will be :class:`.CXGate`.\n\n        Args:\n            kak_basis_gate (Gate): Basis gate for KAK decomposition.\n            force_consolidate (bool): Force block consolidation.\n            basis_gates (List(str)): Basis gates from which to choose a KAK gate.\n            approximation_degree (float): a float between $[0.0, 1.0]$. Lower approximates more.\n            target (Target): The target object for the compilation target backend.\n        '
        super().__init__()
        self.basis_gates = None
        self.target = target
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)
        self.force_consolidate = force_consolidate
        if kak_basis_gate is not None:
            self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
        elif basis_gates is not None:
            self.decomposer = unitary_synthesis._decomposer_2q_from_basis_gates(basis_gates, approximation_degree=approximation_degree)
        else:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the ConsolidateBlocks pass on `dag`.\n\n        Iterate over each block and replace it with an equivalent Unitary\n        on the same wires.\n        '
        if self.decomposer is None:
            return dag
        blocks = self.property_set['block_list'] or []
        basis_gate_name = self.decomposer.gate.name
        all_block_gates = set()
        for block in blocks:
            if len(block) == 1 and self._check_not_in_basis(dag, block[0].name, block[0].qargs):
                all_block_gates.add(block[0])
                dag.substitute_node(block[0], UnitaryGate(block[0].op.to_matrix()))
            else:
                basis_count = 0
                outside_basis = False
                block_qargs = set()
                block_cargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                    if isinstance(nd, DAGOpNode) and getattr(nd.op, 'condition', None):
                        block_cargs |= set(getattr(nd.op, 'condition', None)[0])
                    all_block_gates.add(nd)
                block_index_map = self._block_qargs_to_indices(dag, block_qargs)
                for nd in block:
                    if nd.op.name == basis_gate_name:
                        basis_count += 1
                    if self._check_not_in_basis(dag, nd.op.name, nd.qargs):
                        outside_basis = True
                if len(block_qargs) > 2:
                    q = QuantumRegister(len(block_qargs))
                    qc = QuantumCircuit(q)
                    if block_cargs:
                        c = ClassicalRegister(len(block_cargs))
                        qc.add_register(c)
                    for nd in block:
                        qc.append(nd.op, [q[block_index_map[i]] for i in nd.qargs])
                    unitary = UnitaryGate(Operator(qc), check_input=False)
                else:
                    matrix = _block_to_matrix(block, block_index_map)
                    unitary = UnitaryGate(matrix, check_input=False)
                max_2q_depth = 20
                if self.force_consolidate or unitary.num_qubits > 2 or self.decomposer.num_basis_gates(unitary) < basis_count or (len(block) > max_2q_depth) or (self.basis_gates is not None and outside_basis) or (self.target is not None and outside_basis):
                    identity = np.eye(2 ** unitary.num_qubits)
                    if np.allclose(identity, unitary.to_matrix()):
                        for node in block:
                            dag.remove_op_node(node)
                    else:
                        dag.replace_block_with_op(block, unitary, block_index_map, cycle_check=False)
        runs = self.property_set['run_list'] or []
        identity_1q = np.eye(2)
        for run in runs:
            if any((gate in all_block_gates for gate in run)):
                continue
            if len(run) == 1 and (not self._check_not_in_basis(dag, run[0].name, run[0].qargs)):
                dag.substitute_node(run[0], UnitaryGate(run[0].op.to_matrix(), check_input=False))
            else:
                qubit = run[0].qargs[0]
                operator = run[0].op.to_matrix()
                already_in_block = False
                for gate in run[1:]:
                    if gate in all_block_gates:
                        already_in_block = True
                    operator = gate.op.to_matrix().dot(operator)
                if already_in_block:
                    continue
                unitary = UnitaryGate(operator, check_input=False)
                if np.allclose(identity_1q, unitary.to_matrix()):
                    for node in run:
                        dag.remove_op_node(node)
                else:
                    dag.replace_block_with_op(run, unitary, {qubit: 0}, cycle_check=False)
        dag = self._handle_control_flow_ops(dag)
        if 'run_list' in self.property_set:
            del self.property_set['run_list']
        if 'block_list' in self.property_set:
            del self.property_set['block_list']
        return dag

    def _handle_control_flow_ops(self, dag):
        if False:
            i = 10
            return i + 15
        '\n        This is similar to transpiler/passes/utils/control_flow.py except that the\n        collect blocks is redone for the control flow blocks.\n        '
        pass_manager = PassManager()
        if 'run_list' in self.property_set:
            pass_manager.append(Collect1qRuns())
        if 'block_list' in self.property_set:
            pass_manager.append(Collect2qBlocks())
        pass_manager.append(self)
        for node in dag.op_nodes(ControlFlowOp):
            node.op = node.op.replace_blocks((pass_manager.run(block) for block in node.op.blocks))
        return dag

    def _check_not_in_basis(self, dag, gate_name, qargs):
        if False:
            for i in range(10):
                print('nop')
        if self.target is not None:
            return not self.target.instruction_supported(gate_name, tuple((dag.find_bit(qubit).index for qubit in qargs)))
        else:
            return self.basis_gates and gate_name not in self.basis_gates

    def _block_qargs_to_indices(self, dag, block_qargs):
        if False:
            return 10
        "Map each qubit in block_qargs to its wire position among the block's wires.\n        Args:\n            block_qargs (list): list of qubits that a block acts on\n            global_index_map (dict): mapping from each qubit in the\n                circuit to its wire position within that circuit\n        Returns:\n            dict: mapping from qarg to position in block\n        "
        block_indices = [dag.find_bit(q).index for q in block_qargs]
        ordered_block_indices = {bit: index for (index, bit) in enumerate(sorted(block_indices))}
        block_positions = {q: ordered_block_indices[dag.find_bit(q).index] for q in block_qargs}
        return block_positions