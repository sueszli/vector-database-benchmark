"""
Superoperator representation of a Quantum Channel."""
from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor, _to_superop
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
if TYPE_CHECKING:
    from qiskit.quantum_info.states.densitymatrix import DensityMatrix
    from qiskit.quantum_info.states.statevector import Statevector

class SuperOp(QuantumChannel):
    """Superoperator representation of a quantum channel.

    The Superoperator representation of a quantum channel :math:`\\mathcal{E}`
    is a matrix :math:`S` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\\rho` is given by

    .. math::

        |\\mathcal{E}(\\rho)\\rangle\\!\\rangle = S |\\rho\\rangle\\!\\rangle

    where the double-ket notation :math:`|A\\rangle\\!\\rangle` denotes a vector
    formed by stacking the columns of the matrix :math:`A`
    *(column-vectorization)*.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data: QuantumCircuit | Instruction | BaseOperator | np.ndarray, input_dims: tuple | None=None, output_dims: tuple | None=None):
        if False:
            print('Hello World!')
        'Initialize a quantum channel Superoperator operator.\n\n        Args:\n            data (QuantumCircuit or\n                  Instruction or\n                  BaseOperator or\n                  matrix): data to initialize superoperator.\n            input_dims (tuple): the input subsystem dimensions.\n                                [Default: None]\n            output_dims (tuple): the output subsystem dimensions.\n                                 [Default: None]\n\n        Raises:\n            QiskitError: if input data cannot be initialized as a\n                         superoperator.\n\n        Additional Information:\n            If the input or output dimensions are None, they will be\n            automatically determined from the input data. If the input data is\n            a Numpy array of shape (4**N, 4**N) qubit systems will be used. If\n            the input operator is not an N-qubit operator, it will assign a\n            single subsystem with dimension specified by the shape of the input.\n        '
        if isinstance(data, (list, np.ndarray)):
            super_mat = np.asarray(data, dtype=complex)
            (dout, din) = super_mat.shape
            input_dim = int(np.sqrt(din))
            output_dim = int(np.sqrt(dout))
            if output_dim ** 2 != dout or input_dim ** 2 != din:
                raise QiskitError('Invalid shape for SuperOp matrix.')
            op_shape = OpShape.auto(dims_l=output_dims, dims_r=input_dims, shape=(output_dim, input_dim))
        else:
            if isinstance(data, (QuantumCircuit, Instruction)):
                data = self._init_instruction(data)
            else:
                data = self._init_transformer(data)
            op_shape = data._op_shape
            (input_dim, output_dim) = data.dim
            rep = getattr(data, '_channel_rep', 'Operator')
            super_mat = _to_superop(rep, data._data, input_dim, output_dim)
        super().__init__(super_mat, op_shape=op_shape)

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        if dtype:
            return np.asarray(self.data, dtype=dtype)
        return self.data

    @property
    def _tensor_shape(self):
        if False:
            return 10
        'Return the tensor shape of the superoperator matrix'
        return 2 * tuple(reversed(self._op_shape.dims_l())) + 2 * tuple(reversed(self._op_shape.dims_r()))

    @property
    def _bipartite_shape(self):
        if False:
            return 10
        'Return the shape for bipartite matrix'
        return (self._output_dim, self._output_dim, self._input_dim, self._input_dim)

    def conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        ret = copy.copy(self)
        ret._data = np.conj(self._data)
        return ret

    def transpose(self):
        if False:
            while True:
                i = 10
        ret = copy.copy(self)
        ret._data = np.transpose(self._data)
        ret._op_shape = self._op_shape.transpose()
        return ret

    def adjoint(self):
        if False:
            while True:
                i = 10
        ret = copy.copy(self)
        ret._data = np.conj(np.transpose(self._data))
        ret._op_shape = self._op_shape.transpose()
        return ret

    def tensor(self, other: SuperOp) -> SuperOp:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        return self._tensor(self, other)

    def expand(self, other: SuperOp) -> SuperOp:
        if False:
            while True:
                i = 10
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        if False:
            return 10
        ret = copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = _bipartite_tensor(a._data, b.data, shape1=a._bipartite_shape, shape2=b._bipartite_shape)
        return ret

    def compose(self, other: SuperOp, qargs: list | None=None, front: bool=False) -> SuperOp:
        if False:
            return 10
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()
        if qargs is None:
            if front:
                data = np.dot(self._data, other.data)
            else:
                data = np.dot(other.data, self._data)
            ret = SuperOp(data, input_dims, output_dims)
            ret._op_shape = new_shape
            return ret
        (num_qargs_l, num_qargs_r) = self._op_shape.num_qargs
        if front:
            num_indices = num_qargs_r
            shift = 2 * num_qargs_l
            right_mul = True
        else:
            num_indices = num_qargs_l
            shift = 0
            right_mul = False
        tensor = np.reshape(self.data, self._tensor_shape)
        mat = np.reshape(other.data, other._tensor_shape)
        indices = [2 * num_indices - 1 - qubit for qubit in qargs] + [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.prod(output_dims) ** 2, np.prod(input_dims) ** 2]
        data = np.reshape(Operator._einsum_matmul(tensor, mat, indices, shift, right_mul), final_shape)
        ret = SuperOp(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def _evolve(self, state, qargs=None):
        if False:
            while True:
                i = 10
        'Evolve a quantum state by the quantum channel.\n\n        Args:\n            state (DensityMatrix or Statevector): The input state.\n            qargs (list): a list of quantum state subsystem positions to apply\n                           the quantum channel on.\n\n        Returns:\n            DensityMatrix: the output quantum state as a density matrix.\n\n        Raises:\n            QiskitError: if the quantum channel dimension does not match the\n                         specified quantum state subsystem dimensions.\n        '
        from qiskit.quantum_info.states.densitymatrix import DensityMatrix
        if not isinstance(state, DensityMatrix):
            state = DensityMatrix(state)
        if qargs is None:
            if state._op_shape.shape[0] != self._op_shape.shape[1]:
                raise QiskitError('Operator input dimension is not equal to density matrix dimension.')
            vec = np.ravel(state.data, order='F')
            mat = np.reshape(np.dot(self.data, vec), (self._output_dim, self._output_dim), order='F')
            return DensityMatrix(mat, dims=self.output_dims())
        if state.dims(qargs) != self.input_dims():
            raise QiskitError('Operator input dimensions are not equal to statevector subsystem dimensions.')
        tensor = np.reshape(state.data, state._op_shape.tensor_shape)
        mat = np.reshape(self.data, self._tensor_shape)
        num_indices = len(state.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs] + [2 * num_indices - 1 - qubit for qubit in qargs]
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        new_dims = list(state.dims())
        output_dims = self.output_dims()
        for (i, qubit) in enumerate(qargs):
            new_dims[qubit] = output_dims[i]
        new_dim = np.prod(new_dims)
        tensor = np.reshape(tensor, (new_dim, new_dim))
        return DensityMatrix(tensor, dims=new_dims)

    @classmethod
    def _init_instruction(cls, instruction):
        if False:
            i = 10
            return i + 15
        'Convert a QuantumCircuit or Instruction to a SuperOp.'
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        op = SuperOp(np.eye(4 ** instruction.num_qubits))
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_superop(cls, obj):
        if False:
            return 10
        'Return superop for instruction if defined or None otherwise.'
        if not isinstance(obj, Instruction):
            raise QiskitError('Input is not an instruction.')
        chan = None
        if obj.name == 'reset':
            chan = SuperOp(np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        if obj.name == 'kraus':
            kraus = obj.params
            dim = len(kraus[0])
            chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
        elif hasattr(obj, 'to_matrix'):
            try:
                kraus = [obj.to_matrix()]
                dim = len(kraus[0])
                chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
            except QiskitError:
                pass
        return chan

    def _append_instruction(self, obj, qargs=None):
        if False:
            i = 10
            return i + 15
        'Update the current Operator by apply an instruction.'
        from qiskit.circuit.barrier import Barrier
        chan = self._instruction_to_superop(obj)
        if chan is not None:
            op = self.compose(chan, qargs=qargs)
            self._data = op.data
        elif isinstance(obj, Barrier):
            return
        else:
            if obj.definition is None:
                raise QiskitError(f'Cannot apply Instruction: {obj.name}')
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(obj.name, type(obj.definition)))
            qubit_indices = {bit: idx for (idx, bit) in enumerate(obj.definition.qubits)}
            for instruction in obj.definition.data:
                if instruction.clbits:
                    raise QiskitError(f'Cannot apply instruction with classical bits: {instruction.operation.name}')
                if qargs is None:
                    new_qargs = [qubit_indices[tup] for tup in instruction.qubits]
                else:
                    new_qargs = [qargs[qubit_indices[tup]] for tup in instruction.qubits]
                self._append_instruction(instruction.operation, qargs=new_qargs)
generate_apidocs(SuperOp)