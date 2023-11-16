"""
Chi-matrix representation of a Quantum Channel.
"""
from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_chi
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator

class Chi(QuantumChannel):
    """Pauli basis Chi-matrix representation of a quantum channel.

    The Chi-matrix representation of an :math:`n`-qubit quantum channel
    :math:`\\mathcal{E}` is a matrix :math:`\\chi` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\\rho` is given by

    .. math::

        \\mathcal{E}(ρ) = \\frac{1}{2^n} \\sum_{i, j} \\chi_{i,j} P_i ρ P_j

    where :math:`[P_0, P_1, ..., P_{4^{n}-1}]` is the :math:`n`-qubit Pauli basis in
    lexicographic order. It is related to the :class:`Choi` representation by a change
    of basis of the Choi-matrix into the Pauli basis. The :math:`\\frac{1}{2^n}`
    in the definition above is a normalization factor that arises from scaling the
    Pauli basis to make it orthonormal.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data: QuantumCircuit | Instruction | BaseOperator | np.ndarray, input_dims: int | tuple | None=None, output_dims: int | tuple | None=None):
        if False:
            return 10
        'Initialize a quantum channel Chi-matrix operator.\n\n        Args:\n            data (QuantumCircuit or\n                  Instruction or\n                  BaseOperator or\n                  matrix): data to initialize superoperator.\n            input_dims (tuple): the input subsystem dimensions.\n                                [Default: None]\n            output_dims (tuple): the output subsystem dimensions.\n                                 [Default: None]\n\n        Raises:\n            QiskitError: if input data is not an N-qubit channel or\n                         cannot be initialized as a Chi-matrix.\n\n        Additional Information:\n            If the input or output dimensions are None, they will be\n            automatically determined from the input data. The Chi matrix\n            representation is only valid for N-qubit channels.\n        '
        if isinstance(data, (list, np.ndarray)):
            chi_mat = np.asarray(data, dtype=complex)
            (dim_l, dim_r) = chi_mat.shape
            if dim_l != dim_r:
                raise QiskitError('Invalid Chi-matrix input.')
            if input_dims:
                input_dim = np.prod(input_dims)
            if output_dims:
                output_dim = np.prod(input_dims)
            if output_dims is None and input_dims is None:
                output_dim = int(np.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dims is None:
                input_dim = dim_l // output_dim
            elif output_dims is None:
                output_dim = dim_l // input_dim
            if input_dim * output_dim != dim_l:
                raise QiskitError('Invalid shape for Chi-matrix input.')
        else:
            if isinstance(data, (QuantumCircuit, Instruction)):
                data = SuperOp._init_instruction(data)
            else:
                data = self._init_transformer(data)
            (input_dim, output_dim) = data.dim
            rep = getattr(data, '_channel_rep', 'Operator')
            chi_mat = _to_chi(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        num_qubits = int(np.log2(input_dim))
        if 2 ** num_qubits != input_dim or input_dim != output_dim:
            raise QiskitError('Input is not an n-qubit Chi matrix.')
        super().__init__(chi_mat, num_qubits=num_qubits)

    def __array__(self, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        if dtype:
            return np.asarray(self.data, dtype=dtype)
        return self.data

    @property
    def _bipartite_shape(self):
        if False:
            i = 10
            return i + 15
        'Return the shape for bipartite matrix'
        return (self._input_dim, self._output_dim, self._input_dim, self._output_dim)

    def _evolve(self, state, qargs=None):
        if False:
            return 10
        return SuperOp(self)._evolve(state, qargs)

    def conjugate(self):
        if False:
            return 10
        return Chi(Choi(self).conjugate())

    def transpose(self):
        if False:
            return 10
        return Chi(Choi(self).transpose())

    def adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        return Chi(Choi(self).adjoint())

    def compose(self, other: Chi, qargs: list | None=None, front: bool=False) -> Chi:
        if False:
            return 10
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if qargs is not None:
            return Chi(SuperOp(self).compose(other, qargs=qargs, front=front))
        return Chi(Choi(self).compose(other, front=front))

    def tensor(self, other: Chi) -> Chi:
        if False:
            while True:
                i = 10
        if not isinstance(other, Chi):
            other = Chi(other)
        return self._tensor(self, other)

    def expand(self, other: Chi) -> Chi:
        if False:
            while True:
                i = 10
        if not isinstance(other, Chi):
            other = Chi(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        if False:
            i = 10
            return i + 15
        ret = copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = np.kron(a._data, b.data)
        return ret
generate_apidocs(Chi)