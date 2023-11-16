"""
Decompose a single-qubit unitary via Euler angles.
"""
from __future__ import annotations
import numpy as np
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.library.standard_gates import UGate, PhaseGate, U3Gate, U2Gate, U1Gate, RXGate, RYGate, RZGate, RGate, SXGate, XGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.operator import Operator
DEFAULT_ATOL = 1e-12
ONE_QUBIT_EULER_BASIS_GATES = {'U3': ['u3'], 'U321': ['u3', 'u2', 'u1'], 'U': ['u'], 'PSX': ['p', 'sx'], 'U1X': ['u1', 'rx'], 'RR': ['r'], 'ZYZ': ['rz', 'ry'], 'ZXZ': ['rz', 'rx'], 'XZX': ['rz', 'rx'], 'XYX': ['rx', 'ry'], 'ZSXX': ['rz', 'sx', 'x'], 'ZSX': ['rz', 'sx']}
NAME_MAP = {'u': UGate, 'u1': U1Gate, 'u2': U2Gate, 'u3': U3Gate, 'p': PhaseGate, 'rx': RXGate, 'ry': RYGate, 'rz': RZGate, 'r': RGate, 'sx': SXGate, 'x': XGate}

class OneQubitEulerDecomposer:
    """A class for decomposing 1-qubit unitaries into Euler angle rotations.

    The resulting decomposition is parameterized by 3 Euler rotation angle
    parameters :math:`(\\theta, \\phi, \\lambda)`, and a phase parameter
    :math:`\\gamma`. The value of the parameters for an input unitary depends
    on the decomposition basis. Allowed bases and the resulting circuits are
    shown in the following table. Note that for the non-Euler bases (U3, U1X,
    RR), the ZYZ Euler parameters are used.

    .. list-table:: Supported circuit bases
        :widths: auto
        :header-rows: 1

        * - Basis
          - Euler Angle Basis
          - Decomposition Circuit
        * - 'ZYZ'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} R_Z(\\phi).R_Y(\\theta).R_Z(\\lambda)`
        * - 'ZXZ'
          - :math:`Z(\\phi) X(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} R_Z(\\phi).R_X(\\theta).R_Z(\\lambda)`
        * - 'XYX'
          - :math:`X(\\phi) Y(\\theta) X(\\lambda)`
          - :math:`e^{i\\gamma} R_X(\\phi).R_Y(\\theta).R_X(\\lambda)`
        * - 'XZX'
          - :math:`X(\\phi) Z(\\theta) X(\\lambda)`
          - :math:`e^{i\\gamma} R_X(\\phi).R_Z(\\theta).R_X(\\lambda)`
        * - 'U3'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} U_3(\\theta,\\phi,\\lambda)`
        * - 'U321'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} U_3(\\theta,\\phi,\\lambda)`
        * - 'U'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} U_3(\\theta,\\phi,\\lambda)`
        * - 'PSX'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} U_1(\\phi+\\pi).R_X\\left(\\frac{\\pi}{2}\\right).`
            :math:`U_1(\\theta+\\pi).R_X\\left(\\frac{\\pi}{2}\\right).U_1(\\lambda)`
        * - 'ZSX'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} R_Z(\\phi+\\pi).\\sqrt{X}.`
            :math:`R_Z(\\theta+\\pi).\\sqrt{X}.R_Z(\\lambda)`
        * - 'ZSXX'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} R_Z(\\phi+\\pi).\\sqrt{X}.R_Z(\\theta+\\pi).\\sqrt{X}.R_Z(\\lambda)`
            or
            :math:`e^{i\\gamma} R_Z(\\phi+\\pi).X.R_Z(\\lambda)`
        * - 'U1X'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} U_1(\\phi+\\pi).R_X\\left(\\frac{\\pi}{2}\\right).`
            :math:`U_1(\\theta+\\pi).R_X\\left(\\frac{\\pi}{2}\\right).U_1(\\lambda)`
        * - 'RR'
          - :math:`Z(\\phi) Y(\\theta) Z(\\lambda)`
          - :math:`e^{i\\gamma} R\\left(-\\pi,\\frac{\\phi-\\lambda+\\pi}{2}\\right).`
            :math:`R\\left(\\theta+\\pi,\\frac{\\pi}{2}-\\lambda\\right)`
    """

    def __init__(self, basis: str='U3', use_dag: bool=False):
        if False:
            print('Hello World!')
        "Initialize decomposer\n\n        Supported bases are: 'U', 'PSX', 'ZSXX', 'ZSX', 'U321', 'U3', 'U1X', 'RR', 'ZYZ', 'ZXZ',\n        'XYX', 'XZX'.\n\n        Args:\n            basis (str): the decomposition basis [Default: 'U3']\n            use_dag (bool): If true the output from calls to the decomposer\n                will be a :class:`~qiskit.dagcircuit.DAGCircuit` object instead of\n                :class:`~qiskit.circuit.QuantumCircuit`.\n\n        Raises:\n            QiskitError: If input basis is not recognized.\n        "
        self.basis = basis
        self.use_dag = use_dag

    def build_circuit(self, gates, global_phase):
        if False:
            while True:
                i = 10
        'Return the circuit or dag object from a list of gates.'
        qr = [Qubit()]
        lookup_gate = False
        if len(gates) > 0 and isinstance(gates[0], tuple):
            lookup_gate = True
        if self.use_dag:
            from qiskit.dagcircuit import dagcircuit
            dag = dagcircuit.DAGCircuit()
            dag.global_phase = global_phase
            dag.add_qubits(qr)
            for gate_entry in gates:
                if lookup_gate:
                    gate = NAME_MAP[gate_entry[0]](*gate_entry[1])
                else:
                    gate = gate_entry
                dag.apply_operation_back(gate, (qr[0],), check=False)
            return dag
        else:
            circuit = QuantumCircuit(qr, global_phase=global_phase)
            for gate_entry in gates:
                if lookup_gate:
                    gate = NAME_MAP[gate_entry[0]](*gate_entry[1])
                else:
                    gate = gate_entry
                circuit._append(gate, [qr[0]], [])
            return circuit

    def __call__(self, unitary: Operator | Gate | np.ndarray, simplify: bool=True, atol: float=DEFAULT_ATOL) -> QuantumCircuit:
        if False:
            for i in range(10):
                print('nop')
        'Decompose single qubit gate into a circuit.\n\n        Args:\n            unitary (Operator or Gate or array): 1-qubit unitary matrix\n            simplify (bool): reduce gate count in decomposition [Default: True].\n            atol (float): absolute tolerance for checking angles when simplifying\n                         returned circuit [Default: 1e-12].\n        Returns:\n            QuantumCircuit: the decomposed single-qubit gate circuit\n\n        Raises:\n            QiskitError: if input is invalid or synthesis fails.\n        '
        if hasattr(unitary, 'to_operator'):
            unitary = unitary.to_operator().data
        elif hasattr(unitary, 'to_matrix'):
            unitary = unitary.to_matrix()
        unitary = np.asarray(unitary, dtype=complex)
        if unitary.shape != (2, 2):
            raise QiskitError('OneQubitEulerDecomposer: expected 2x2 input matrix')
        if not is_unitary_matrix(unitary):
            raise QiskitError('OneQubitEulerDecomposer: input matrix is not unitary.')
        return self._decompose(unitary, simplify=simplify, atol=atol)

    def _decompose(self, unitary, simplify=True, atol=DEFAULT_ATOL):
        if False:
            print('Hello World!')
        circuit_sequence = euler_one_qubit_decomposer.unitary_to_gate_sequence(unitary, [self.basis], 0, None, simplify, atol)
        circuit = self.build_circuit(circuit_sequence, circuit_sequence.global_phase)
        return circuit

    @property
    def basis(self):
        if False:
            for i in range(10):
                print('nop')
        'The decomposition basis.'
        return self._basis

    @basis.setter
    def basis(self, basis):
        if False:
            print('Hello World!')
        'Set the decomposition basis.'
        basis_methods = {'U321': self._params_u3, 'U3': self._params_u3, 'U': self._params_u3, 'PSX': self._params_u1x, 'ZSX': self._params_u1x, 'ZSXX': self._params_u1x, 'U1X': self._params_u1x, 'RR': self._params_zyz, 'ZYZ': self._params_zyz, 'ZXZ': self._params_zxz, 'XYX': self._params_xyx, 'XZX': self._params_xzx}
        if basis not in basis_methods:
            raise QiskitError(f'OneQubitEulerDecomposer: unsupported basis {basis}')
        self._basis = basis
        self._params = basis_methods[basis]

    def angles(self, unitary: np.ndarray) -> tuple:
        if False:
            return 10
        'Return the Euler angles for input array.\n\n        Args:\n            unitary (np.ndarray): 2x2 unitary matrix.\n\n        Returns:\n            tuple: (theta, phi, lambda).\n        '
        unitary = np.asarray(unitary, dtype=complex)
        (theta, phi, lam, _) = self._params(unitary)
        return (theta, phi, lam)

    def angles_and_phase(self, unitary: np.ndarray) -> tuple:
        if False:
            print('Hello World!')
        'Return the Euler angles and phase for input array.\n\n        Args:\n            unitary (np.ndarray): 2x2 unitary matrix.\n\n        Returns:\n            tuple: (theta, phi, lambda, phase).\n        '
        unitary = np.asarray(unitary, dtype=complex)
        return self._params(unitary)
    _params_zyz = staticmethod(euler_one_qubit_decomposer.params_zyz)
    _params_zxz = staticmethod(euler_one_qubit_decomposer.params_zxz)
    _params_xyx = staticmethod(euler_one_qubit_decomposer.params_xyx)
    _params_xzx = staticmethod(euler_one_qubit_decomposer.params_xzx)
    _params_u3 = staticmethod(euler_one_qubit_decomposer.params_u3)
    _params_u1x = staticmethod(euler_one_qubit_decomposer.params_u1x)