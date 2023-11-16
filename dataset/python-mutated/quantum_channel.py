"""
Abstract base class for Quantum Channels.
"""
from __future__ import annotations
import copy
import sys
from abc import abstractmethod
from numbers import Number, Integral
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _transform_rep
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

class QuantumChannel(LinearOp):
    """Quantum channel representation base class."""

    def __init__(self, data: list | np.ndarray, num_qubits: int | None=None, op_shape: OpShape | None=None):
        if False:
            return 10
        'Initialize a quantum channel Superoperator operator.\n\n        Args:\n            data (array or list): quantum channel data array.\n            op_shape (OpShape): the operator shape of the channel.\n            num_qubits (int): the number of qubits if the channel is N-qubit.\n\n        Raises:\n            QiskitError: if arguments are invalid.\n        '
        self._data = data
        super().__init__(num_qubits=num_qubits, op_shape=op_shape)

    def __repr__(self):
        if False:
            return 10
        prefix = f'{self._channel_rep}('
        pad = len(prefix) * ' '
        return '{}{},\n{}input_dims={}, output_dims={})'.format(prefix, np.array2string(np.asarray(self.data), separator=', ', prefix=prefix), pad, self.input_dims(), self.output_dims())

    def __eq__(self, other: Self):
        if False:
            i = 10
            return i + 15
        'Test if two QuantumChannels are equal.'
        if not super().__eq__(other):
            return False
        return np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        'Return data.'
        return self._data

    @property
    def _channel_rep(self):
        if False:
            while True:
                i = 10
        'Return channel representation string'
        return type(self).__name__

    @property
    def settings(self):
        if False:
            for i in range(10):
                print('nop')
        'Return settings.'
        return {'data': self.data, 'input_dims': self.input_dims(), 'output_dims': self.output_dims()}

    @abstractmethod
    def conjugate(self):
        if False:
            return 10
        'Return the conjugate quantum channel.\n\n        .. note::\n            This is equivalent to the matrix complex conjugate in the\n            :class:`~qiskit.quantum_info.SuperOp` representation\n            ie. for a channel :math:`\\mathcal{E}`, the SuperOp of\n            the conjugate channel :math:`\\overline{{\\mathcal{{E}}}}` is\n            :math:`S_{\\overline{\\mathcal{E}^\\dagger}} = \\overline{S_{\\mathcal{E}}}`.\n        '

    @abstractmethod
    def transpose(self) -> Self:
        if False:
            return 10
        'Return the transpose quantum channel.\n\n        .. note::\n            This is equivalent to the matrix transpose in the\n            :class:`~qiskit.quantum_info.SuperOp` representation,\n            ie. for a channel :math:`\\mathcal{E}`, the SuperOp of\n            the transpose channel :math:`\\mathcal{{E}}^T` is\n            :math:`S_{mathcal{E}^T} = S_{\\mathcal{E}}^T`.\n        '

    def adjoint(self) -> Self:
        if False:
            while True:
                i = 10
        'Return the adjoint quantum channel.\n\n        .. note::\n            This is equivalent to the matrix Hermitian conjugate in the\n            :class:`~qiskit.quantum_info.SuperOp` representation\n            ie. for a channel :math:`\\mathcal{E}`, the SuperOp of\n            the adjoint channel :math:`\\mathcal{{E}}^\\dagger` is\n            :math:`S_{\\mathcal{E}^\\dagger} = S_{\\mathcal{E}}^\\dagger`.\n        '
        return self.conjugate().transpose()

    def power(self, n: float) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return the power of the quantum channel.\n\n        Args:\n            n (float): the power exponent.\n\n        Returns:\n            CLASS: the channel :math:`\\mathcal{{E}} ^n`.\n\n        Raises:\n            QiskitError: if the input and output dimensions of the\n                         CLASS are not equal.\n\n        .. note::\n            For non-positive or non-integer exponents the power is\n            defined as the matrix power of the\n            :class:`~qiskit.quantum_info.SuperOp` representation\n            ie. for a channel :math:`\\mathcal{{E}}`, the SuperOp of\n            the powered channel :math:`\\mathcal{{E}}^\\n` is\n            :math:`S_{{\\mathcal{{E}}^n}} = S_{{\\mathcal{{E}}}}^n`.\n        '
        if n > 0 and isinstance(n, Integral):
            return super().power(n)
        if self._input_dim != self._output_dim:
            raise QiskitError('Can only take power with input_dim = output_dim.')
        rep = self._channel_rep
        (input_dim, output_dim) = self.dim
        superop = _transform_rep(rep, 'SuperOp', self._data, input_dim, output_dim)
        superop = np.linalg.matrix_power(superop, n)
        ret = copy.copy(self)
        ret._data = _transform_rep('SuperOp', rep, superop, input_dim, output_dim)
        return ret

    def __sub__(self, other) -> Self:
        if False:
            i = 10
            return i + 15
        qargs = getattr(other, 'qargs', None)
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return self._add(-other, qargs=qargs)

    def _add(self, other, qargs=None):
        if False:
            while True:
                i = 10
        if not isinstance(other, type(self)):
            other = type(self)(other)
        self._op_shape._validate_add(other._op_shape, qargs)
        other = ScalarOp._pad_with_identity(self, other, qargs)
        ret = copy.copy(self)
        ret._data = self._data + other._data
        return ret

    def _multiply(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Number):
            raise QiskitError('other is not a number')
        ret = copy.copy(self)
        ret._data = other * self._data
        return ret

    def is_cptp(self, atol: float | None=None, rtol: float | None=None) -> bool:
        if False:
            print('Hello World!')
        'Return True if completely-positive trace-preserving (CPTP).'
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol) and self._is_tp_helper(choi, atol, rtol)

    def is_tp(self, atol: float | None=None, rtol: float | None=None) -> bool:
        if False:
            while True:
                i = 10
        'Test if a channel is trace-preserving (TP)'
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_tp_helper(choi, atol, rtol)

    def is_cp(self, atol: float | None=None, rtol: float | None=None) -> bool:
        if False:
            print('Hello World!')
        'Test if Choi-matrix is completely-positive (CP)'
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol)

    def is_unitary(self, atol: float | None=None, rtol: float | None=None) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if QuantumChannel is a unitary channel.'
        try:
            op = self.to_operator()
            return op.is_unitary(atol=atol, rtol=rtol)
        except QiskitError:
            return False

    def to_operator(self) -> Operator:
        if False:
            return 10
        'Try to convert channel to a unitary representation Operator.'
        mat = _to_operator(self._channel_rep, self._data, *self.dim)
        return Operator(mat, self.input_dims(), self.output_dims())

    def to_instruction(self) -> Instruction:
        if False:
            for i in range(10):
                print('nop')
        'Convert to a Kraus or UnitaryGate circuit instruction.\n\n        If the channel is unitary it will be added as a unitary gate,\n        otherwise it will be added as a kraus simulator instruction.\n\n        Returns:\n            qiskit.circuit.Instruction: A kraus instruction for the channel.\n\n        Raises:\n            QiskitError: if input data is not an N-qubit CPTP quantum channel.\n        '
        num_qubits = int(np.log2(self._input_dim))
        if self._input_dim != self._output_dim or 2 ** num_qubits != self._input_dim:
            raise QiskitError('Cannot convert QuantumChannel to Instruction: channel is not an N-qubit channel.')
        if not self.is_cptp():
            raise QiskitError('Cannot convert QuantumChannel to Instruction: channel is not CPTP.')
        (kraus, _) = _to_kraus(self._channel_rep, self._data, *self.dim)
        if len(kraus) == 1:
            return Operator(kraus[0]).to_instruction()
        return Instruction('kraus', num_qubits, 0, kraus)

    def _is_cp_helper(self, choi, atol, rtol):
        if False:
            return 10
        'Test if a channel is completely-positive (CP)'
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return is_positive_semidefinite_matrix(choi, rtol=rtol, atol=atol)

    def _is_tp_helper(self, choi, atol, rtol):
        if False:
            i = 10
            return i + 15
        'Test if Choi-matrix is trace-preserving (TP)'
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        (d_in, d_out) = self.dim
        mat = np.trace(np.reshape(choi, (d_in, d_out, d_in, d_out)), axis1=1, axis2=3)
        tp_cond = np.linalg.eigvalsh(mat - np.eye(len(mat)))
        zero = np.isclose(tp_cond, 0, atol=atol, rtol=rtol)
        return np.all(zero)

    def _format_state(self, state, density_matrix=False):
        if False:
            for i in range(10):
                print('nop')
        'Format input state so it is statevector or density matrix'
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError('Input state is not a vector or matrix.')
        if ndim == 2:
            if shape[1] != 1 and shape[1] != shape[0]:
                raise QiskitError('Input state is not a vector or matrix.')
            if shape[1] == 1:
                state = np.reshape(state, shape[0])
        if density_matrix and ndim == 1:
            state = np.outer(state, np.transpose(np.conj(state)))
        return state

    @abstractmethod
    def _evolve(self, state, qargs=None):
        if False:
            return 10
        'Evolve a quantum state by the quantum channel.\n\n        Args:\n            state (DensityMatrix or Statevector): The input state.\n            qargs (list): a list of quantum state subsystem positions to apply\n                           the quantum channel on.\n\n        Returns:\n            DensityMatrix: the output quantum state as a density matrix.\n\n        Raises:\n            QiskitError: if the quantum channel dimension does not match the\n                         specified quantum state subsystem dimensions.\n        '
        pass

    @classmethod
    def _init_transformer(cls, data):
        if False:
            i = 10
            return i + 15
        'Convert input into a QuantumChannel subclass object or Operator object'
        if isinstance(data, QuantumChannel):
            return data
        if hasattr(data, 'to_quantumchannel'):
            return data.to_quantumchannel()
        if hasattr(data, 'to_channel'):
            return data.to_channel()
        return Operator(data)