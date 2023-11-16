"""Weyl decomposition of two-qubit gates in terms of echoed cross-resonance gates."""
from typing import Tuple
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import RZXGate, HGate, XGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.calibration.rzx_builder import _check_calibration_type, CRCalType
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag

class EchoRZXWeylDecomposition(TransformationPass):
    """Rewrite two-qubit gates using the Weyl decomposition.

    This transpiler pass rewrites two-qubit gates in terms of echoed cross-resonance gates according
    to the Weyl decomposition. A two-qubit gate will be replaced with at most six non-echoed RZXGates.
    Each pair of RZXGates forms an echoed RZXGate.
    """

    def __init__(self, instruction_schedule_map=None, target=None):
        if False:
            print('Hello World!')
        'EchoRZXWeylDecomposition pass.\n\n        Args:\n            instruction_schedule_map (InstructionScheduleMap): the mapping from circuit\n                :class:`~.circuit.Instruction` names and arguments to :class:`.Schedule`\\ s.\n            target (Target): The :class:`~.Target` representing the target backend, if both\n                ``instruction_schedule_map`` and ``target`` are specified then this argument will take\n                precedence and ``instruction_schedule_map`` will be ignored.\n        '
        super().__init__()
        self._inst_map = instruction_schedule_map
        if target is not None:
            self._inst_map = target.instruction_schedule_map()

    def _is_native(self, qubit_pair: Tuple) -> bool:
        if False:
            while True:
                i = 10
        'Return the direction of the qubit pair that is native.'
        (cal_type, _, _) = _check_calibration_type(self._inst_map, qubit_pair)
        return cal_type in [CRCalType.ECR_CX_FORWARD, CRCalType.ECR_FORWARD, CRCalType.DIRECT_CX_FORWARD]

    @staticmethod
    def _echo_rzx_dag(theta):
        if False:
            for i in range(10):
                print('nop')
        'Return the following circuit\n\n        .. parsed-literal::\n\n                 ┌───────────────┐┌───┐┌────────────────┐┌───┐\n            q_0: ┤0              ├┤ X ├┤0               ├┤ X ├\n                 │  Rzx(theta/2) │└───┘│  Rzx(-theta/2) │└───┘\n            q_1: ┤1              ├─────┤1               ├─────\n                 └───────────────┘     └────────────────┘\n        '
        rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        rzx_dag.add_qreg(qr)
        rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[0], qr[1]], [])
        rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
        rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[0], qr[1]], [])
        rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
        return rzx_dag

    @staticmethod
    def _reverse_echo_rzx_dag(theta):
        if False:
            i = 10
            return i + 15
        'Return the following circuit\n\n        .. parsed-literal::\n\n                 ┌───┐┌───────────────┐     ┌────────────────┐┌───┐\n            q_0: ┤ H ├┤1              ├─────┤1               ├┤ H ├─────\n                 ├───┤│  Rzx(theta/2) │┌───┐│  Rzx(-theta/2) │├───┤┌───┐\n            q_1: ┤ H ├┤0              ├┤ X ├┤0               ├┤ X ├┤ H ├\n                 └───┘└───────────────┘└───┘└────────────────┘└───┘└───┘\n        '
        reverse_rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        reverse_rzx_dag.add_qreg(qr)
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[1], qr[0]], [])
        reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[1], qr[0]], [])
        reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        return reverse_rzx_dag

    def run(self, dag: DAGCircuit):
        if False:
            return 10
        'Run the EchoRZXWeylDecomposition pass on `dag`.\n\n        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance\n        gates by computing the Weyl decomposition of the corresponding unitary. Modifies the\n        input dag.\n\n        Args:\n            dag (DAGCircuit): DAG to rewrite.\n\n        Returns:\n            DAGCircuit: The modified dag.\n\n        Raises:\n            TranspilerError: If the circuit cannot be rewritten.\n        '
        from qiskit.quantum_info import Operator
        from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitControlledUDecomposer
        if len(dag.qregs) > 1:
            raise TranspilerError(f'EchoRZXWeylDecomposition expects a single qreg input DAG,but input DAG had qregs: {dag.qregs}.')
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        decomposer = TwoQubitControlledUDecomposer(RZXGate)
        for node in dag.two_qubit_ops():
            unitary = Operator(node.op).data
            dag_weyl = circuit_to_dag(decomposer(unitary))
            dag.substitute_node_with_dag(node, dag_weyl)
        for node in dag.two_qubit_ops():
            if node.name == 'rzx':
                control = node.qargs[0]
                target = node.qargs[1]
                physical_q0 = trivial_layout[control]
                physical_q1 = trivial_layout[target]
                is_native = self._is_native((physical_q0, physical_q1))
                theta = node.op.params[0]
                if is_native:
                    dag.substitute_node_with_dag(node, self._echo_rzx_dag(theta))
                else:
                    dag.substitute_node_with_dag(node, self._reverse_echo_rzx_dag(theta))
        return dag