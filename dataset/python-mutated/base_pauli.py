"""
Optimized list of Pauli operators
"""
from __future__ import annotations
import copy
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
if TYPE_CHECKING:
    from qiskit.quantum_info.operators.symplectic.clifford import Clifford
_PARITY = np.array([-1 if bin(i).count('1') % 2 else 1 for i in range(256)], dtype=complex)

class BasePauli(BaseOperator, AdjointMixin, MultiplyMixin):
    """Symplectic representation of a list of N-qubit Paulis.

    Base class for Pauli and PauliList.
    """

    def __init__(self, z: np.ndarray, x: np.ndarray, phase: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the BasePauli.\n\n        This is an array of M N-qubit Paulis defined as\n        P = (-i)^phase Z^z X^x.\n\n        Args:\n            z (np.ndarray): input z matrix.\n            x (np.ndarray): input x matrix.\n            phase (np.ndarray): input phase vector.\n        '
        self._z = z
        self._x = x
        self._phase = phase
        (self._num_paulis, num_qubits) = self._z.shape
        super().__init__(num_qubits=num_qubits)

    def copy(self):
        if False:
            print('Hello World!')
        'Make a deep copy of current operator.'
        ret = copy.copy(self)
        ret._z = self._z.copy()
        ret._x = self._x.copy()
        ret._phase = self._phase.copy()
        return ret

    def tensor(self, other):
        if False:
            print('Hello World!')
        return self._tensor(self, other)

    def expand(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        if False:
            while True:
                i = 10
        x1 = cls._stack(a._x, b._num_paulis, False)
        x2 = cls._stack(b._x, a._num_paulis)
        z1 = cls._stack(a._z, b._num_paulis, False)
        z2 = cls._stack(b._z, a._num_paulis)
        phase1 = np.vstack(b._num_paulis * [a._phase]).transpose(1, 0).reshape(a._num_paulis * b._num_paulis)
        phase2 = cls._stack(b._phase, a._num_paulis)
        z = np.hstack([z2, z1])
        x = np.hstack([x2, x1])
        phase = np.mod(phase1 + phase2, 4)
        return BasePauli(z, x, phase)

    def compose(self, other, qargs: list | None=None, front: bool=False, inplace=False):
        if False:
            print('Hello World!')
        'Return the composition of Paulis.\n\n        Args:\n            a ({cls}): an operator object.\n            b ({cls}): an operator object.\n            qargs (list or None): Optional, qubits to apply dot product\n                                  on (default: None).\n            inplace (bool): If True update in-place (default: False).\n\n        Returns:\n            {cls}: The operator a.compose(b)\n\n        Raises:\n            QiskitError: if number of qubits of other does not match qargs.\n        '.format(cls=type(self).__name__)
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError(f'other {type(self).__name__} must be on the same number of qubits.')
        if qargs and other.num_qubits != len(qargs):
            raise QiskitError(f'Number of qubits of the other {type(self).__name__} does not match qargs.')
        if other._num_paulis not in [1, self._num_paulis]:
            raise QiskitError('Incompatible BasePaulis. Second list must either have 1 or the same number of Paulis.')
        if qargs is not None:
            (x1, z1) = (self._x[:, qargs], self._z[:, qargs])
        else:
            (x1, z1) = (self._x, self._z)
        (x2, z2) = (other._x, other._z)
        phase = self._phase + other._phase
        if front:
            phase += 2 * _count_y(x1, z2, dtype=phase.dtype)
        else:
            phase += 2 * _count_y(x2, z1, dtype=phase.dtype)
        x = np.logical_xor(x1, x2)
        z = np.logical_xor(z1, z2)
        if qargs is None:
            if not inplace:
                return BasePauli(z, x, phase)
            self._x = x
            self._z = z
            self._phase = phase
            return self
        ret = self if inplace else self.copy()
        ret._x[:, qargs] = x
        ret._z[:, qargs] = z
        ret._phase = np.mod(phase, 4)
        return ret

    def _multiply(self, other):
        if False:
            print('Hello World!')
        'Return the {cls} other * self.\n\n        Args:\n            other (complex): a complex number in ``[1, -1j, -1, 1j]``.\n\n        Returns:\n            {cls}: the {cls} other * self.\n\n        Raises:\n            QiskitError: if the phase is not in the set ``[1, -1j, -1, 1j]``.\n        '.format(cls=type(self).__name__)
        if isinstance(other, (np.ndarray, list, tuple)):
            phase = np.array([self._phase_from_complex(phase) for phase in other])
        else:
            phase = self._phase_from_complex(other)
        return BasePauli(self._z, self._x, np.mod(self._phase + phase, 4))

    def conjugate(self):
        if False:
            return 10
        'Return the conjugate of each Pauli in the list.'
        complex_phase = np.mod(self._phase, 2)
        if np.all(complex_phase == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * complex_phase, 4))

    def transpose(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the transpose of each Pauli in the list.'
        parity_y = self._count_y(dtype=self._phase.dtype) % 2
        if np.all(parity_y == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * parity_y, 4))

    def commutes(self, other: BasePauli, qargs: list | None=None) -> np.ndarray:
        if False:
            print('Hello World!')
        'Return ``True`` if Pauli commutes with ``other``.\n\n        Args:\n            other (BasePauli): another BasePauli operator.\n            qargs (list): qubits to apply dot product on (default: ``None``).\n\n        Returns:\n            np.array: Boolean array of ``True`` if Paulis commute, ``False`` if\n                      they anti-commute.\n\n        Raises:\n            QiskitError: if number of qubits of ``other`` does not match ``qargs``.\n        '
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError('Number of qubits of other Pauli does not match number of qargs ({} != {}).'.format(other.num_qubits, len(qargs)))
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError('Number of qubits of other Pauli does not match the current Pauli ({} != {}).'.format(other.num_qubits, self.num_qubits))
        if qargs is not None:
            inds = list(qargs)
            (x1, z1) = (self._x[:, inds], self._z[:, inds])
        else:
            (x1, z1) = (self._x, self._z)
        a_dot_b = np.mod(_count_y(x1, other._z), 2)
        b_dot_a = np.mod(_count_y(other._x, z1), 2)
        return a_dot_b == b_dot_a

    def evolve(self, other: BasePauli | QuantumCircuit | Clifford, qargs: list | None=None, frame: Literal['h', 's']='h') -> BasePauli:
        if False:
            print('Hello World!')
        "Performs either Heisenberg (default) or Schrödinger picture\n        evolution of the Pauli by a Clifford and returns the evolved Pauli.\n\n        Schrödinger picture evolution can be chosen by passing parameter ``frame='s'``.\n        This option yields a faster calculation.\n\n        Heisenberg picture evolves the Pauli as :math:`P^\\prime = C^\\dagger.P.C`.\n\n        Schrödinger picture evolves the Pauli as :math:`P^\\prime = C.P.C^\\dagger`.\n\n        Args:\n            other (BasePauli or QuantumCircuit): The Clifford circuit to evolve by.\n            qargs (list): a list of qubits to apply the Clifford to.\n            frame (string): ``'h'`` for Heisenberg or ``'s'`` for Schrödinger framework.\n\n        Returns:\n            BasePauli: the Pauli :math:`C^\\dagger.P.C` (Heisenberg picture)\n            or the Pauli :math:`C.P.C^\\dagger` (Schrödinger picture).\n\n        Raises:\n            QiskitError: if the Clifford number of qubits and ``qargs`` don't match.\n        "
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError('Incorrect number of qubits for Clifford circuit ({} != {}).'.format(other.num_qubits, len(qargs)))
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError('Incorrect number of qubits for Clifford circuit ({} != {}).'.format(other.num_qubits, self.num_qubits))
        if isinstance(other, BasePauli):
            if frame == 's':
                ret = self.compose(other, qargs=qargs)
                ret = ret.compose(other.adjoint(), front=True, qargs=qargs)
            else:
                ret = self.compose(other.adjoint(), qargs=qargs)
                ret = ret.compose(other, front=True, qargs=qargs)
            return ret
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford
        if isinstance(other, Clifford):
            return self._evolve_clifford(other, qargs=qargs, frame=frame)
        if frame == 's':
            return self.copy()._append_circuit(other, qargs=qargs)
        return self.copy()._append_circuit(other.inverse(), qargs=qargs)

    def _evolve_clifford(self, other, qargs=None, frame='h'):
        if False:
            i = 10
            return i + 15
        'Heisenberg picture evolution of a Pauli by a Clifford.'
        if frame == 's':
            adj = other
        else:
            adj = other.adjoint()
        if qargs is None:
            qargs_ = slice(None)
        else:
            qargs_ = list(qargs)
        from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
        num_paulis = self._x.shape[0]
        ret = self.copy()
        ret._x[:, qargs_] = False
        ret._z[:, qargs_] = False
        idx = np.concatenate((self._x[:, qargs_], self._z[:, qargs_]), axis=1)
        for (idx_, row) in zip(idx.T, PauliList.from_symplectic(z=adj.z, x=adj.x, phase=2 * adj.phase)):
            if idx_.any():
                if np.sum(idx_) == num_paulis:
                    ret.compose(row, qargs=qargs, inplace=True)
                else:
                    ret[idx_] = ret[idx_].compose(row, qargs=qargs)
        return ret

    def _eq(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Entrywise comparison of Pauli equality.'
        return self.num_qubits == other.num_qubits and np.all(np.mod(self._phase, 4) == np.mod(other._phase, 4)) and np.all(self._z == other._z) and np.all(self._x == other._x)

    def __imul__(self, other):
        if False:
            print('Hello World!')
        return self.compose(other, front=True, inplace=True)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        ret = copy.copy(self)
        ret._phase = np.mod(self._phase + 2, 4)
        return ret

    def _count_y(self, dtype=None):
        if False:
            return 10
        'Count the number of I Paulis'
        return _count_y(self._x, self._z, dtype=dtype)

    @staticmethod
    def _stack(array, size, vertical=True):
        if False:
            for i in range(10):
                print('nop')
        'Stack array.'
        if size == 1:
            return array
        if vertical:
            return np.vstack(size * [array]).reshape((size * len(array),) + array.shape[1:])
        return np.hstack(size * [array]).reshape((size * len(array),) + array.shape[1:])

    @staticmethod
    def _phase_from_complex(coeff):
        if False:
            i = 10
            return i + 15
        'Return the phase from a label'
        if np.isclose(coeff, 1):
            return 0
        if np.isclose(coeff, -1j):
            return 1
        if np.isclose(coeff, -1):
            return 2
        if np.isclose(coeff, 1j):
            return 3
        raise QiskitError('Pauli can only be multiplied by 1, -1j, -1, 1j.')

    @staticmethod
    def _from_array(z, x, phase=0):
        if False:
            return 10
        'Convert array data to BasePauli data.'
        if isinstance(z, np.ndarray) and z.dtype == bool:
            base_z = z
        else:
            base_z = np.asarray(z, dtype=bool)
        if base_z.ndim == 1:
            base_z = base_z.reshape((1, base_z.size))
        elif base_z.ndim != 2:
            raise QiskitError('Invalid Pauli z vector shape.')
        if isinstance(x, np.ndarray) and x.dtype == bool:
            base_x = x
        else:
            base_x = np.asarray(x, dtype=bool)
        if base_x.ndim == 1:
            base_x = base_x.reshape((1, base_x.size))
        elif base_x.ndim != 2:
            raise QiskitError('Invalid Pauli x vector shape.')
        if base_z.shape != base_x.shape:
            raise QiskitError('z and x vectors are different size.')
        dtype = getattr(phase, 'dtype', None)
        base_phase = np.mod(_count_y(base_x, base_z, dtype=dtype) + phase, 4)
        return (base_z, base_x, base_phase)

    @staticmethod
    def _to_matrix(z, x, phase=0, group_phase=False, sparse=False):
        if False:
            while True:
                i = 10
        'Return the matrix from symplectic representation.\n\n        The Pauli is defined as :math:`P = (-i)^{phase + z.x} * Z^z.x^x`\n        where ``array = [x, z]``.\n\n        Args:\n            z (array): The symplectic representation z vector.\n            x (array): The symplectic representation x vector.\n            phase (int): Pauli phase.\n            group_phase (bool): Optional. If ``True`` use group-phase convention\n                                instead of BasePauli ZX-phase convention.\n                                (default: ``False``).\n            sparse (bool): Optional. Of ``True`` return a sparse CSR matrix,\n                           otherwise return a dense Numpy array\n                           (default: ``False``).\n\n        Returns:\n            array: if ``sparse=False``.\n            csr_matrix: if ``sparse=True``.\n        '
        num_qubits = z.size
        if group_phase:
            phase += np.sum(x & z)
            phase %= 4
        dim = 2 ** num_qubits
        twos_array = 1 << np.arange(num_qubits, dtype=np.uint)
        x_indices = np.asarray(x).dot(twos_array)
        z_indices = np.asarray(z).dot(twos_array)
        indptr = np.arange(dim + 1, dtype=np.uint)
        indices = indptr ^ x_indices
        if phase:
            coeff = (-1j) ** phase
        else:
            coeff = 1
        vec_u64 = z_indices & indptr
        mat_u8 = np.zeros((vec_u64.size, 8), dtype=np.uint8)
        for i in range(8):
            mat_u8[:, i] = vec_u64 & 255
            vec_u64 >>= 8
            if np.all(vec_u64 == 0):
                break
        parity = _PARITY[np.bitwise_xor.reduce(mat_u8, axis=1)]
        data = coeff * parity
        if sparse:
            from scipy.sparse import csr_matrix
            return csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=complex)
        mat = np.zeros((dim, dim), dtype=complex)
        mat[range(dim), indices[:dim]] = data[:dim]
        return mat

    @staticmethod
    def _to_label(z, x, phase, group_phase=False, full_group=True, return_phase=False):
        if False:
            while True:
                i = 10
        'Return the label string for a Pauli.\n\n        Args:\n            z (array): The symplectic representation z vector.\n            x (array): The symplectic representation x vector.\n            phase (int): Pauli phase.\n            group_phase (bool): Optional. If ``True`` use group-phase convention\n                                instead of BasePauli ZX-phase convention.\n                                (default: ``False``).\n            full_group (bool): If True return the Pauli label from the full Pauli group\n                including complex coefficient from [1, -1, 1j, -1j]. If\n                ``False`` return the unsigned Pauli label with coefficient 1\n                (default: ``True``).\n            return_phase (bool): If ``True`` return the adjusted phase for the coefficient\n                of the returned Pauli label. This can be used even if\n                ``full_group=False``.\n\n        Returns:\n            str: the Pauli label from the full Pauli group (if ``full_group=True``) or\n                from the unsigned Pauli group (if ``full_group=False``).\n            tuple[str, int]: if ``return_phase=True`` returns a tuple of the Pauli\n                            label (from either the full or unsigned Pauli group) and\n                            the phase ``q`` for the coefficient :math:`(-i)^(q + x.z)`\n                            for the label from the full Pauli group.\n        '
        num_qubits = z.size
        phase = int(phase)
        coeff_labels = {0: '', 1: '-i', 2: '-', 3: 'i'}
        label = ''
        for i in range(num_qubits):
            if not z[num_qubits - 1 - i]:
                if not x[num_qubits - 1 - i]:
                    label += 'I'
                else:
                    label += 'X'
            elif not x[num_qubits - 1 - i]:
                label += 'Z'
            else:
                label += 'Y'
                if not group_phase:
                    phase -= 1
        phase %= 4
        if phase and full_group:
            label = coeff_labels[phase] + label
        if return_phase:
            return (label, phase)
        return label

    def _append_circuit(self, circuit, qargs=None):
        if False:
            return 10
        'Update BasePauli inplace by applying a Clifford circuit.\n\n        Args:\n            circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.\n            qargs (list or None): The qubits to apply gate to.\n\n        Returns:\n            BasePauli: the updated Pauli.\n\n        Raises:\n            QiskitError: if input gate cannot be decomposed into Clifford gates.\n        '
        if isinstance(circuit, (Barrier, Delay)):
            return self
        if qargs is None:
            qargs = list(range(self.num_qubits))
        if isinstance(circuit, QuantumCircuit):
            gate = circuit.to_instruction()
        else:
            gate = circuit
        basis_1q = {'i': _evolve_i, 'id': _evolve_i, 'iden': _evolve_i, 'x': _evolve_x, 'y': _evolve_y, 'z': _evolve_z, 'h': _evolve_h, 's': _evolve_s, 'sdg': _evolve_sdg, 'sinv': _evolve_sdg}
        basis_2q = {'cx': _evolve_cx, 'cz': _evolve_cz, 'cy': _evolve_cy, 'swap': _evolve_swap}
        non_clifford = ['t', 'tdg', 'ccx', 'ccz']
        if isinstance(gate, str):
            if gate not in basis_1q and gate not in basis_2q:
                raise QiskitError(f'Invalid Clifford gate name string {gate}')
            name = gate
        else:
            name = gate.name
        if name in non_clifford:
            raise QiskitError(f'Cannot update Pauli with non-Clifford gate {name}')
        if name in basis_1q:
            if len(qargs) != 1:
                raise QiskitError('Invalid qubits for 1-qubit gate.')
            return basis_1q[name](self, qargs[0])
        if name in basis_2q:
            if len(qargs) != 2:
                raise QiskitError('Invalid qubits for 2-qubit gate.')
            return basis_2q[name](self, qargs[0], qargs[1])
        if gate.definition is None:
            raise QiskitError(f'Cannot apply Instruction: {gate.name}')
        if not isinstance(gate.definition, QuantumCircuit):
            raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(gate.name, type(gate.definition)))
        flat_instr = gate.definition
        bit_indices = {bit: index for bits in [flat_instr.qubits, flat_instr.clbits] for (index, bit) in enumerate(bits)}
        for instruction in flat_instr:
            if instruction.clbits:
                raise QiskitError(f'Cannot apply Instruction with classical bits: {instruction.operation.name}')
            new_qubits = [qargs[bit_indices[tup]] for tup in instruction.qubits]
            self._append_circuit(instruction.operation, new_qubits)
        self._phase %= 4
        return self

def _evolve_h(base_pauli, qubit):
    if False:
        while True:
            i = 10
    'Update P -> H.P.H'
    x = base_pauli._x[:, qubit].copy()
    z = base_pauli._z[:, qubit].copy()
    base_pauli._x[:, qubit] = z
    base_pauli._z[:, qubit] = x
    base_pauli._phase += 2 * np.logical_and(x, z).T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_s(base_pauli, qubit):
    if False:
        return 10
    'Update P -> S.P.Sdg'
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase += x.T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_sdg(base_pauli, qubit):
    if False:
        for i in range(10):
            print('nop')
    'Update P -> Sdg.P.S'
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase -= x.T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_i(base_pauli, qubit):
    if False:
        while True:
            i = 10
    'Update P -> P'
    return base_pauli

def _evolve_x(base_pauli, qubit):
    if False:
        i = 10
        return i + 15
    'Update P -> X.P.X'
    base_pauli._phase += 2 * base_pauli._z[:, qubit].T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_y(base_pauli, qubit):
    if False:
        return 10
    'Update P -> Y.P.Y'
    xp = base_pauli._x[:, qubit].T.astype(base_pauli._phase.dtype)
    zp = base_pauli._z[:, qubit].T.astype(base_pauli._phase.dtype)
    base_pauli._phase += 2 * (xp + zp)
    return base_pauli

def _evolve_z(base_pauli, qubit):
    if False:
        while True:
            i = 10
    'Update P -> Z.P.Z'
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_cx(base_pauli, qctrl, qtrgt):
    if False:
        i = 10
        return i + 15
    'Update P -> CX.P.CX'
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    return base_pauli

def _evolve_cz(base_pauli, q1, q2):
    if False:
        return 10
    'Update P -> CZ.P.CZ'
    x1 = base_pauli._x[:, q1].copy()
    x2 = base_pauli._x[:, q2].copy()
    base_pauli._z[:, q1] ^= x2
    base_pauli._z[:, q2] ^= x1
    base_pauli._phase += 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_cy(base_pauli, qctrl, qtrgt):
    if False:
        while True:
            i = 10
    'Update P -> CY.P.CY'
    x1 = base_pauli._x[:, qctrl].copy()
    x2 = base_pauli._x[:, qtrgt].copy()
    z2 = base_pauli._z[:, qtrgt].copy()
    base_pauli._x[:, qtrgt] ^= x1
    base_pauli._z[:, qtrgt] ^= x1
    base_pauli._z[:, qctrl] ^= np.logical_xor(x2, z2)
    base_pauli._phase += x1 + 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli

def _evolve_swap(base_pauli, q1, q2):
    if False:
        while True:
            i = 10
    'Update P -> SWAP.P.SWAP'
    x1 = base_pauli._x[:, q1].copy()
    z1 = base_pauli._z[:, q1].copy()
    base_pauli._x[:, q1] = base_pauli._x[:, q2]
    base_pauli._z[:, q1] = base_pauli._z[:, q2]
    base_pauli._x[:, q2] = x1
    base_pauli._z[:, q2] = z1
    return base_pauli

def _count_y(x, z, dtype=None):
    if False:
        print('Hello World!')
    'Count the number of I Paulis'
    return (x & z).sum(axis=1, dtype=dtype)