"""
Gate described by the time evolution of a Hermitian Hamiltonian operator.
"""
from __future__ import annotations
import typing
from numbers import Number
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.utils.deprecation import deprecate_func
from .generalized_gates.unitary import UnitaryGate
if typing.TYPE_CHECKING:
    from qiskit.quantum_info.operators.base_operator import BaseOperator

class HamiltonianGate(Gate):
    """Class for representing evolution by a Hamiltonian operator as a gate.

    This gate resolves to a :class:`~.library.UnitaryGate` as :math:`U(t) = \\exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits.
    """

    def __init__(self, data: np.ndarray | Gate | BaseOperator, time: float | ParameterExpression, label: str | None=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            data: A hermitian operator.\n            time: Time evolution parameter.\n            label: Unitary name for backend [Default: None].\n\n        Raises:\n            ValueError: if input data is not an N-qubit unitary operator.\n        '
        if hasattr(data, 'to_matrix'):
            data = data.to_matrix()
        elif hasattr(data, 'to_operator'):
            data = data.to_operator().data
        data = np.array(data, dtype=complex)
        if not is_hermitian_matrix(data):
            raise ValueError('Input matrix is not Hermitian.')
        if isinstance(time, Number) and time != np.real(time):
            raise ValueError('Evolution time is not real.')
        (input_dim, output_dim) = data.shape
        num_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2 ** num_qubits != input_dim:
            raise ValueError('Input matrix is not an N-qubit operator.')
        super().__init__('hamiltonian', num_qubits, [data, time], label=label)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, HamiltonianGate):
            return False
        if self.label != other.label:
            return False
        operators_eq = matrix_equal(self.params[0], other.params[0], ignore_phase=False)
        times_eq = self.params[1] == other.params[1]
        return operators_eq and times_eq

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        'Return matrix for the unitary.'
        import scipy.linalg
        try:
            return scipy.linalg.expm(-1j * self.params[0] * float(self.params[1]))
        except TypeError as ex:
            raise TypeError('Unable to generate Unitary matrix for unbound t parameter {}'.format(self.params[1])) from ex

    def inverse(self):
        if False:
            print('Hello World!')
        'Return the adjoint of the unitary.'
        return self.adjoint()

    def conjugate(self):
        if False:
            return 10
        'Return the conjugate of the Hamiltonian.'
        return HamiltonianGate(np.conj(self.params[0]), -self.params[1])

    def adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the adjoint of the unitary.'
        return HamiltonianGate(self.params[0], -self.params[1])

    def transpose(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the transpose of the Hamiltonian.'
        return HamiltonianGate(np.transpose(self.params[0]), self.params[1])

    def _define(self):
        if False:
            print('Hello World!')
        'Calculate a subcircuit that implements this unitary.'
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc._append(UnitaryGate(self.to_matrix()), q[:], [])
        self.definition = qc

    @deprecate_func(since='0.25.0', package_name='qiskit-terra')
    def qasm(self):
        if False:
            return 10
        'Raise an error, as QASM is not defined for the HamiltonianGate.'
        raise CircuitError('HamiltonianGate has no OpenQASM 2 definition.')

    def validate_parameter(self, parameter):
        if False:
            print('Hello World!')
        'Hamiltonian parameter has to be an ndarray, operator or float.'
        if isinstance(parameter, (float, int, np.ndarray)):
            return parameter
        elif isinstance(parameter, ParameterExpression) and len(parameter.parameters) == 0:
            return float(parameter)
        else:
            raise CircuitError(f'invalid param type {type(parameter)} for gate {self.name}')