"""CircuitStateFn Class"""
from typing import Dict, List, Optional, Set, Union, cast
import numpy as np
from qiskit import BasicAer, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import IGate, StatePreparation
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class CircuitStateFn(StateFn):
    """
    Deprecated: A class for state functions and measurements which are defined by the action of a
    QuantumCircuit starting from \\|0⟩, and stored using Terra's ``QuantumCircuit`` class.
    """
    primitive: QuantumCircuit

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, primitive: Union[QuantumCircuit, Instruction]=None, coeff: Union[complex, ParameterExpression]=1.0, is_measurement: bool=False, from_operator: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            primitive: The ``QuantumCircuit`` (or ``Instruction``, which will be converted) which\n                defines the behavior of the underlying function.\n            coeff: A coefficient multiplying the state function.\n            is_measurement: Whether the StateFn is a measurement operator.\n            from_operator: if True the StateFn is derived from OperatorStateFn. (Default: False)\n\n        Raises:\n            TypeError: Unsupported primitive, or primitive has ClassicalRegisters.\n        '
        if isinstance(primitive, Instruction):
            qc = QuantumCircuit(primitive.num_qubits)
            qc.append(primitive, qargs=range(primitive.num_qubits))
            primitive = qc
        if not isinstance(primitive, QuantumCircuit):
            raise TypeError('CircuitStateFn can only be instantiated with QuantumCircuit, not {}'.format(type(primitive)))
        if len(primitive.clbits) != 0:
            raise TypeError('CircuitOp does not support QuantumCircuits with ClassicalRegisters.')
        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)
        self.from_operator = from_operator

    @staticmethod
    def from_dict(density_dict: dict) -> 'CircuitStateFn':
        if False:
            for i in range(10):
                print('nop')
        'Construct the CircuitStateFn from a dict mapping strings to probability densities.\n\n        Args:\n            density_dict: The dict representing the desired state.\n\n        Returns:\n            The CircuitStateFn created from the dict.\n        '
        if len(density_dict) <= len(list(density_dict.keys())[0]):
            statefn_circuits = []
            for (bstr, prob) in density_dict.items():
                qc = QuantumCircuit(len(bstr))
                for (index, bit) in enumerate(reversed(bstr)):
                    if bit == '1':
                        qc.x(index)
                sf_circuit = CircuitStateFn(qc, coeff=prob)
                statefn_circuits += [sf_circuit]
            if len(statefn_circuits) == 1:
                return statefn_circuits[0]
            else:
                return cast(CircuitStateFn, SummedOp(cast(List[OperatorBase], statefn_circuits)))
        else:
            sf_dict = StateFn(density_dict)
            return CircuitStateFn.from_vector(sf_dict.to_matrix())

    @staticmethod
    def from_vector(statevector: np.ndarray) -> 'CircuitStateFn':
        if False:
            i = 10
            return i + 15
        'Construct the CircuitStateFn from a vector representing the statevector.\n\n        Args:\n            statevector: The statevector representing the desired state.\n\n        Returns:\n            The CircuitStateFn created from the vector.\n        '
        normalization_coeff = np.linalg.norm(statevector)
        normalized_sv = statevector / normalization_coeff
        return CircuitStateFn(StatePreparation(normalized_sv), coeff=normalization_coeff)

    def primitive_strings(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return {'QuantumCircuit'}

    @property
    def settings(self) -> Dict:
        if False:
            i = 10
            return i + 15
        'Return settings.'
        data = super().settings
        data['from_operator'] = self.from_operator
        return data

    @property
    def num_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over operators with different numbers of qubits, {} and {}, is not well defined'.format(self.num_qubits, other.num_qubits))
        if isinstance(other, CircuitStateFn) and self.primitive == other.primitive:
            return CircuitStateFn(self.primitive, coeff=self.coeff + other.coeff)
        return SummedOp([self, other])

    def adjoint(self) -> 'CircuitStateFn':
        if False:
            print('Hello World!')
        try:
            inverse = self.primitive.inverse()
        except CircuitError as missing_inverse:
            raise OpflowError('Failed to take the inverse of the underlying circuit, the circuit is likely not unitary and can therefore not be inverted.') from missing_inverse
        return CircuitStateFn(inverse, coeff=self.coeff.conjugate(), is_measurement=not self.is_measurement)

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        if not self.is_measurement and (not front):
            raise ValueError('Composition with a Statefunctions in the first operand is not defined.')
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        new_self.from_operator = self.from_operator
        if front:
            return other.compose(new_self)
        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            op_circuit_self = CircuitOp(self.primitive)
            composed_op_circs = cast(CircuitOp, op_circuit_self.compose(other.to_circuit_op()))
            return CircuitStateFn(composed_op_circs.primitive, is_measurement=self.is_measurement, coeff=self.coeff * other.coeff, from_operator=self.from_operator)
        if isinstance(other, CircuitStateFn) and self.is_measurement:
            from ..operator_globals import Zero
            return self.compose(CircuitOp(other.primitive)).compose((Zero ^ self.num_qubits) * other.coeff)
        return ComposedOp([new_self, other])

    def tensor(self, other: OperatorBase) -> Union['CircuitStateFn', TensoredOp]:
        if False:
            return 10
        "\n        Return tensor product between self and other, overloaded by ``^``.\n        Note: You must be conscious of Qiskit's big-endian bit printing convention.\n        Meaning, Plus.tensor(Zero)\n        produces a \\|+⟩ on qubit 0 and a \\|0⟩ on qubit 1, or \\|+⟩⨂\\|0⟩, but would produce\n        a QuantumCircuit like:\n\n            \\|0⟩--\n            \\|+⟩--\n\n        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.\n\n        Args:\n            other: The ``OperatorBase`` to tensor product with self.\n\n        Returns:\n            An ``OperatorBase`` equivalent to the tensor product of self and other.\n        "
        if isinstance(other, CircuitStateFn) and other.is_measurement == self.is_measurement:
            c_op_self = CircuitOp(self.primitive, self.coeff)
            c_op_other = CircuitOp(other.primitive, other.coeff)
            c_op = c_op_self.tensor(c_op_other)
            if isinstance(c_op, CircuitOp):
                return CircuitStateFn(primitive=c_op.primitive, coeff=c_op.coeff, is_measurement=self.is_measurement)
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Return numpy matrix of density operator, warn if more than 16 qubits to\n        force the user to set\n        massive=True if they want such a large matrix. Generally big methods like this\n        should require the use of a\n        converter, but in this case a convenience method for quick hacking and access\n        to classical tools is\n        appropriate.\n        '
        OperatorBase._check_massive('to_density_matrix', True, self.num_qubits, massive)
        return VectorStateFn(self.to_matrix(massive=massive) * self.coeff).to_density_matrix()

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        OperatorBase._check_massive('to_matrix', False, self.num_qubits, massive)
        if self.is_measurement:
            return np.conj(self.adjoint().to_matrix(massive=massive))
        qc = self.to_circuit(meas=False)
        statevector_backend = BasicAer.get_backend('statevector_simulator')
        transpiled = transpile(qc, statevector_backend, optimization_level=0)
        statevector = statevector_backend.run(transpiled).result().get_statevector()
        from ..operator_globals import EVAL_SIG_DIGITS
        return np.round(statevector * self.coeff, decimals=EVAL_SIG_DIGITS)

    def __str__(self) -> str:
        if False:
            return 10
        qc = cast(CircuitStateFn, self.reduce()).to_circuit()
        prim_str = str(qc.draw(output='text'))
        if self.coeff == 1.0:
            return '{}(\n{}\n)'.format('CircuitStateFn' if not self.is_measurement else 'CircuitMeasurement', prim_str)
        else:
            return '{}(\n{}\n) * {}'.format('CircuitStateFn' if not self.is_measurement else 'CircuitMeasurement', prim_str, self.coeff)

    def assign_parameters(self, param_dict: dict) -> Union['CircuitStateFn', ListOp]:
        if False:
            for i in range(10):
                print('nop')
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.parameters:
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if isinstance(self.coeff, ParameterExpression) and self.coeff.parameters <= set(unrolled_dict.keys()):
                param_instersection = set(unrolled_dict.keys()) & self.coeff.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                param_value = float(self.coeff.bind(binds))
            if set(unrolled_dict.keys()) & self.primitive.parameters:
                param_instersection = set(unrolled_dict.keys()) & self.primitive.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                qc = self.to_circuit().assign_parameters(binds)
        return self.__class__(qc, coeff=param_value, is_measurement=self.is_measurement)

    def eval(self, front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            return 10
        if front is None:
            vector_state_fn = self.to_matrix_op().eval()
            return vector_state_fn
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError('Cannot compute overlap with StateFn or Operator if not Measurement. Try taking sf.adjoint() first to convert to measurement.')
        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem) for front_elem in front.oplist])
        if isinstance(front, (PauliOp, CircuitOp, MatrixOp, CircuitStateFn)):
            new_front = self.compose(front)
            return new_front.eval()
        return self.to_matrix_op().eval(front)

    def to_circuit(self, meas: bool=False) -> QuantumCircuit:
        if False:
            while True:
                i = 10
        'Return QuantumCircuit representing StateFn'
        if meas:
            meas_qc = self.primitive.copy()
            meas_qc.add_register(ClassicalRegister(self.num_qubits))
            meas_qc.measure(qubit=range(self.num_qubits), cbit=range(self.num_qubits))
            return meas_qc
        else:
            return self.primitive

    def to_circuit_op(self) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Return ``StateFnCircuit`` corresponding to this StateFn.'
        return self

    def to_instruction(self):
        if False:
            for i in range(10):
                print('nop')
        'Return Instruction corresponding to primitive.'
        return self.primitive.to_instruction()

    def sample(self, shots: int=1024, massive: bool=False, reverse_endianness: bool=False) -> dict:
        if False:
            while True:
                i = 10
        '\n        Sample the state function as a normalized probability distribution. Returns dict of\n        bitstrings in order of probability, with values being probability.\n        '
        OperatorBase._check_massive('sample', False, self.num_qubits, massive)
        qc = self.to_circuit(meas=True)
        qasm_backend = BasicAer.get_backend('qasm_simulator')
        transpiled = transpile(qc, qasm_backend, optimization_level=0)
        counts = qasm_backend.run(transpiled, shots=shots).result().get_counts()
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: prob / shots for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: prob / shots for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))

    def reduce(self) -> 'CircuitStateFn':
        if False:
            while True:
                i = 10
        if self.primitive.data is not None:
            for i in reversed(range(len(self.primitive.data))):
                gate = self.primitive.data[i].operation
                if isinstance(gate, IGate) or (type(gate) == Instruction and gate.definition.data == []):
                    del self.primitive.data[i]
        return self

    def _expand_dim(self, num_qubits: int) -> 'CircuitStateFn':
        if False:
            while True:
                i = 10
        return self.permute(list(range(num_qubits, num_qubits + self.num_qubits)))

    def permute(self, permutation: List[int]) -> 'CircuitStateFn':
        if False:
            for i in range(10):
                print('nop')
        '\n        Permute the qubits of the circuit.\n\n        Args:\n            permutation: A list defining where each qubit should be permuted. The qubit at index\n                j of the circuit should be permuted to position permutation[j].\n\n        Returns:\n            A new CircuitStateFn containing the permuted circuit.\n        '
        new_qc = QuantumCircuit(max(permutation) + 1).compose(self.primitive, qubits=permutation)
        return CircuitStateFn(new_qc, coeff=self.coeff, is_measurement=self.is_measurement)