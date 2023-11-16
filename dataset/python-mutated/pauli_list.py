"""
Optimized list of Pauli operators
"""
from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli

class PauliList(BasePauli, LinearMixin, GroupMixin):
    """List of N-qubit Pauli operators.

    This class is an efficient representation of a list of
    :class:`Pauli` operators. It supports 1D numpy array indexing
    returning a :class:`Pauli` for integer indexes or a
    :class:`PauliList` for slice or list indices.

    **Initialization**

    A PauliList object can be initialized in several ways.

        ``PauliList(list[str])``
            where strings are same representation with :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList(Pauli) and PauliList(list[Pauli])``
            where Pauli is :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList.from_symplectic(z, x, phase)``
            where ``z`` and ``x`` are 2 dimensional boolean ``numpy.ndarrays`` and ``phase`` is
            an integer in ``[0, 1, 2, 3]``.

    For example,

    .. code-block::

        import numpy as np

        from qiskit.quantum_info import Pauli, PauliList

        # 1. init from list[str]
        pauli_list = PauliList(["II", "+ZI", "-iYY"])
        print("1. ", pauli_list)

        pauli1 = Pauli("iXI")
        pauli2 = Pauli("iZZ")

        # 2. init from Pauli
        print("2. ", PauliList(pauli1))

        # 3. init from list[Pauli]
        print("3. ", PauliList([pauli1, pauli2]))

        # 4. init from np.ndarray
        z = np.array([[True, True], [False, False]])
        x = np.array([[False, True], [True, False]])
        phase = np.array([0, 1])
        pauli_list = PauliList.from_symplectic(z, x, phase)
        print("4. ", pauli_list)

    .. parsed-literal::

        1.  ['II', 'ZI', '-iYY']
        2.  ['iXI']
        3.  ['iXI', 'iZZ']
        4.  ['YZ', '-iIX']

    **Data Access**

    The individual Paulis can be accessed and updated using the ``[]``
    operator which accepts integer, lists, or slices for selecting subsets
    of PauliList. If integer is given, it returns Pauli not PauliList.

    .. code-block::

        pauli_list = PauliList(["XX", "ZZ", "IZ"])
        print("Integer: ", repr(pauli_list[1]))
        print("List: ", repr(pauli_list[[0, 2]]))
        print("Slice: ", repr(pauli_list[0:2]))

    .. parsed-literal::

        Integer:  Pauli('ZZ')
        List:  PauliList(['XX', 'IZ'])
        Slice:  PauliList(['XX', 'ZZ'])

    **Iteration**

    Rows in the Pauli table can be iterated over like a list. Iteration can
    also be done using the label or matrix representation of each row using the
    :meth:`label_iter` and :meth:`matrix_iter` methods.
    """
    __truncate__ = 2000

    def __init__(self, data: Pauli | list):
        if False:
            while True:
                i = 10
        'Initialize the PauliList.\n\n        Args:\n            data (Pauli or list): input data for Paulis. If input is a list each item in the list\n                                  must be a Pauli object or Pauli str.\n\n        Raises:\n            QiskitError: if input array is invalid shape.\n\n        Additional Information:\n            The input array is not copied so multiple Pauli tables\n            can share the same underlying array.\n        '
        if isinstance(data, BasePauli):
            (base_z, base_x, base_phase) = (data._z, data._x, data._phase)
        else:
            (base_z, base_x, base_phase) = self._from_paulis(data)
        super().__init__(base_z, base_x, base_phase)

    @property
    def settings(self):
        if False:
            return 10
        'Return settings.'
        return {'data': self.to_labels()}

    def __array__(self, dtype=None):
        if False:
            return 10
        'Convert to numpy array'
        shape = (len(self),) + 2 * (2 ** self.num_qubits,)
        ret = np.zeros(shape, dtype=complex)
        for (i, mat) in enumerate(self.matrix_iter()):
            ret[i] = mat
        return ret

    @staticmethod
    def _from_paulis(data):
        if False:
            print('Hello World!')
        'Construct a PauliList from a list of Pauli data.\n\n        Args:\n            data (iterable): list of Pauli data.\n\n        Returns:\n            PauliList: the constructed PauliList.\n\n        Raises:\n            QiskitError: If the input list is empty or contains invalid\n            Pauli strings.\n        '
        if not isinstance(data, (list, tuple, set, np.ndarray)):
            data = [data]
        num_paulis = len(data)
        if num_paulis == 0:
            raise QiskitError('Input Pauli list is empty.')
        paulis = []
        for i in data:
            if not isinstance(i, Pauli):
                paulis.append(Pauli(i))
            else:
                paulis.append(i)
        num_qubits = paulis[0].num_qubits
        base_z = np.zeros((num_paulis, num_qubits), dtype=bool)
        base_x = np.zeros((num_paulis, num_qubits), dtype=bool)
        base_phase = np.zeros(num_paulis, dtype=int)
        for (i, pauli) in enumerate(paulis):
            if pauli.num_qubits != num_qubits:
                raise ValueError(f'The {i}th Pauli is defined over {pauli.num_qubits} qubits, but num_qubits == {num_qubits} was expected.')
            base_z[i] = pauli._z
            base_x[i] = pauli._x
            base_phase[i] = pauli._phase.item()
        return (base_z, base_x, base_phase)

    def __repr__(self):
        if False:
            return 10
        'Display representation.'
        return self._truncated_str(True)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Print representation.'
        return self._truncated_str(False)

    def _truncated_str(self, show_class):
        if False:
            while True:
                i = 10
        stop = self._num_paulis
        if self.__truncate__ and self.num_qubits > 0:
            max_paulis = self.__truncate__ // self.num_qubits
            if self._num_paulis > max_paulis:
                stop = max_paulis
        labels = [str(self[i]) for i in range(stop)]
        prefix = 'PauliList(' if show_class else ''
        tail = ')' if show_class else ''
        if stop != self._num_paulis:
            suffix = ', ...]' + tail
        else:
            suffix = ']' + tail
        list_str = np.array2string(np.array(labels), threshold=stop + 1, separator=', ', prefix=prefix, suffix=suffix)
        return prefix + list_str[:-1] + suffix

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Entrywise comparison of Pauli equality.'
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if not isinstance(other, BasePauli):
            return False
        return self._eq(other)

    def equiv(self, other: PauliList | Pauli) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Entrywise comparison of Pauli equivalence up to global phase.\n\n        Args:\n            other (PauliList or Pauli): a comparison object.\n\n        Returns:\n            np.ndarray: An array of ``True`` or ``False`` for entrywise equivalence\n                        of the current table.\n        '
        if not isinstance(other, PauliList):
            other = PauliList(other)
        return np.all(self.z == other.z, axis=1) & np.all(self.x == other.x, axis=1)

    @property
    def phase(self):
        if False:
            while True:
                i = 10
        'Return the phase exponent of the PauliList.'
        return np.mod(self._phase - self._count_y(dtype=self._phase.dtype), 4)

    @phase.setter
    def phase(self, value):
        if False:
            return 10
        self._phase[:] = np.mod(value + self._count_y(dtype=self._phase.dtype), 4)

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        'The x array for the symplectic representation.'
        return self._x

    @x.setter
    def x(self, val):
        if False:
            return 10
        self._x[:] = val

    @property
    def z(self):
        if False:
            for i in range(10):
                print('nop')
        'The z array for the symplectic representation.'
        return self._z

    @z.setter
    def z(self, val):
        if False:
            i = 10
            return i + 15
        self._z[:] = val

    @property
    def shape(self):
        if False:
            print('Hello World!')
        'The full shape of the :meth:`array`'
        return (self._num_paulis, self.num_qubits)

    @property
    def size(self):
        if False:
            print('Hello World!')
        'The number of Pauli rows in the table.'
        return self._num_paulis

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return the number of Pauli rows in the table.'
        return self._num_paulis

    def __getitem__(self, index):
        if False:
            return 10
        'Return a view of the PauliList.'
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            elif len(index) > 2:
                raise IndexError(f'Invalid PauliList index {index}')
        if isinstance(index, (int, np.integer)):
            return Pauli(BasePauli(self._z[np.newaxis, index], self._x[np.newaxis, index], self._phase[np.newaxis, index]))
        elif isinstance(index, (slice, list, np.ndarray)):
            return PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))
        return PauliList((self._z[index], self._x[index], 0))

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        'Update PauliList.'
        if isinstance(index, tuple):
            if len(index) == 1:
                (row, qubit) = (index[0], None)
            elif len(index) > 2:
                raise IndexError(f'Invalid PauliList index {index}')
            else:
                (row, qubit) = index
        else:
            (row, qubit) = (index, None)
        if not isinstance(value, PauliList):
            value = PauliList(value)
        phase = value._phase.item() if isinstance(row, (int, np.integer)) else value._phase
        if qubit is None:
            self._z[row] = value._z
            self._x[row] = value._x
            self._phase[row] = phase
        else:
            self._z[row, qubit] = value._z
            self._x[row, qubit] = value._x
            self._phase[row] += phase
            self._phase %= 4

    def delete(self, ind: int | list, qubit: bool=False) -> PauliList:
        if False:
            while True:
                i = 10
        'Return a copy with Pauli rows deleted from table.\n\n        When deleting qubits the qubit index is the same as the\n        column index of the underlying :attr:`X` and :attr:`Z` arrays.\n\n        Args:\n            ind (int or list): index(es) to delete.\n            qubit (bool): if ``True`` delete qubit columns, otherwise delete\n                          Pauli rows (Default: ``False``).\n\n        Returns:\n            PauliList: the resulting table with the entries removed.\n\n        Raises:\n            QiskitError: if ``ind`` is out of bounds for the array size or\n                         number of qubits.\n        '
        if isinstance(ind, int):
            ind = [ind]
        if len(ind) == 0:
            return PauliList.from_symplectic(self._z, self._x, self.phase)
        if not qubit:
            if max(ind) >= len(self):
                raise QiskitError('Indices {} are not all less than the size of the PauliList ({})'.format(ind, len(self)))
            z = np.delete(self._z, ind, axis=0)
            x = np.delete(self._x, ind, axis=0)
            phase = np.delete(self._phase, ind)
            return PauliList(BasePauli(z, x, phase))
        if max(ind) >= self.num_qubits:
            raise QiskitError('Indices {} are not all less than the number of qubits in the PauliList ({})'.format(ind, self.num_qubits))
        z = np.delete(self._z, ind, axis=1)
        x = np.delete(self._x, ind, axis=1)
        return PauliList.from_symplectic(z, x, self.phase)

    def insert(self, ind: int, value: PauliList, qubit: bool=False) -> PauliList:
        if False:
            print('Hello World!')
        'Insert Paulis into the table.\n\n        When inserting qubits the qubit index is the same as the\n        column index of the underlying :attr:`X` and :attr:`Z` arrays.\n\n        Args:\n            ind (int): index to insert at.\n            value (PauliList): values to insert.\n            qubit (bool): if ``True`` insert qubit columns, otherwise insert\n                          Pauli rows (Default: ``False``).\n\n        Returns:\n            PauliList: the resulting table with the entries inserted.\n\n        Raises:\n            QiskitError: if the insertion index is invalid.\n        '
        if not isinstance(ind, int):
            raise QiskitError('Insert index must be an integer.')
        if not isinstance(value, PauliList):
            value = PauliList(value)
        size = self._num_paulis
        if not qubit:
            if ind > size:
                raise QiskitError('Index {} is larger than the number of rows in the PauliList ({}).'.format(ind, size))
            base_z = np.insert(self._z, ind, value._z, axis=0)
            base_x = np.insert(self._x, ind, value._x, axis=0)
            base_phase = np.insert(self._phase, ind, value._phase)
            return PauliList(BasePauli(base_z, base_x, base_phase))
        if ind > self.num_qubits:
            raise QiskitError('Index {} is greater than number of qubits in the PauliList ({})'.format(ind, self.num_qubits))
        if len(value) == 1:
            value_x = np.vstack(size * [value.x])
            value_z = np.vstack(size * [value.z])
            value_phase = np.vstack(size * [value.phase])
        elif len(value) == size:
            value_x = value.x
            value_z = value.z
            value_phase = value.phase
        else:
            raise QiskitError('Input PauliList must have a single row, or the same number of rows as the Pauli Table ({}).'.format(size))
        z = np.hstack([self.z[:, :ind], value_z, self.z[:, ind:]])
        x = np.hstack([self.x[:, :ind], value_x, self.x[:, ind:]])
        phase = self.phase + value_phase
        return PauliList.from_symplectic(z, x, phase)

    def argsort(self, weight: bool=False, phase: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Return indices for sorting the rows of the table.\n\n        The default sort method is lexicographic sorting by qubit number.\n        By using the `weight` kwarg the output can additionally be sorted\n        by the number of non-identity terms in the Pauli, where the set of\n        all Paulis of a given weight are still ordered lexicographically.\n\n        Args:\n            weight (bool): Optionally sort by weight if ``True`` (Default: ``False``).\n            phase (bool): Optionally sort by phase before weight or order\n                          (Default: ``False``).\n\n        Returns:\n            array: the indices for sorting the table.\n        '
        x = self.x
        z = self.z
        order = 1 * (x & ~z) + 2 * (x & z) + 3 * (~x & z)
        phases = self.phase
        if weight:
            weights = np.sum(x | z, axis=1)
        indices = np.arange(self._num_paulis)
        sort_inds = phases.argsort(kind='stable')
        indices = indices[sort_inds]
        order = order[sort_inds]
        if phase:
            phases = phases[sort_inds]
        if weight:
            weights = weights[sort_inds]
        for i in range(self.num_qubits):
            sort_inds = order[:, i].argsort(kind='stable')
            order = order[sort_inds]
            indices = indices[sort_inds]
            if weight:
                weights = weights[sort_inds]
            if phase:
                phases = phases[sort_inds]
        if weight:
            sort_inds = weights.argsort(kind='stable')
            indices = indices[sort_inds]
            phases = phases[sort_inds]
        if phase:
            indices = indices[phases.argsort(kind='stable')]
        return indices

    def sort(self, weight: bool=False, phase: bool=False) -> PauliList:
        if False:
            print('Hello World!')
        "Sort the rows of the table.\n\n        The default sort method is lexicographic sorting by qubit number.\n        By using the `weight` kwarg the output can additionally be sorted\n        by the number of non-identity terms in the Pauli, where the set of\n        all Paulis of a given weight are still ordered lexicographically.\n\n        **Example**\n\n        Consider sorting all a random ordering of all 2-qubit Paulis\n\n        .. code-block::\n\n            from numpy.random import shuffle\n            from qiskit.quantum_info.operators import PauliList\n\n            # 2-qubit labels\n            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',\n                      'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']\n            # Shuffle Labels\n            shuffle(labels)\n            pt = PauliList(labels)\n            print('Initial Ordering')\n            print(pt)\n\n            # Lexicographic Ordering\n            srt = pt.sort()\n            print('Lexicographically sorted')\n            print(srt)\n\n            # Weight Ordering\n            srt = pt.sort(weight=True)\n            print('Weight sorted')\n            print(srt)\n\n        .. parsed-literal::\n\n            Initial Ordering\n            ['YX', 'ZZ', 'XZ', 'YI', 'YZ', 'II', 'XX', 'XI', 'XY', 'YY', 'IX', 'IZ',\n             'ZY', 'ZI', 'ZX', 'IY']\n            Lexicographically sorted\n            ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ',\n             'ZI', 'ZX', 'ZY', 'ZZ']\n            Weight sorted\n            ['II', 'IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX', 'XY', 'XZ', 'YX', 'YY',\n             'YZ', 'ZX', 'ZY', 'ZZ']\n\n        Args:\n            weight (bool): optionally sort by weight if ``True`` (Default: ``False``).\n            phase (bool): Optionally sort by phase before weight or order\n                          (Default: ``False``).\n\n        Returns:\n            PauliList: a sorted copy of the original table.\n        "
        return self[self.argsort(weight=weight, phase=phase)]

    def unique(self, return_index: bool=False, return_counts: bool=False) -> PauliList:
        if False:
            for i in range(10):
                print('nop')
        "Return unique Paulis from the table.\n\n        **Example**\n\n        .. code-block::\n\n            from qiskit.quantum_info.operators import PauliList\n\n            pt = PauliList(['X', 'Y', '-X', 'I', 'I', 'Z', 'X', 'iZ'])\n            unique = pt.unique()\n            print(unique)\n\n        .. parsed-literal::\n\n            ['X', 'Y', '-X', 'I', 'Z', 'iZ']\n\n        Args:\n            return_index (bool): If ``True``, also return the indices that\n                                 result in the unique array.\n                                 (Default: ``False``)\n            return_counts (bool): If ``True``, also return the number of times\n                                  each unique item appears in the table.\n\n        Returns:\n            PauliList: unique\n                the table of the unique rows.\n\n            unique_indices: np.ndarray, optional\n                The indices of the first occurrences of the unique values in\n                the original array. Only provided if ``return_index`` is ``True``.\n\n            unique_counts: np.array, optional\n                The number of times each of the unique values comes up in the\n                original array. Only provided if ``return_counts`` is ``True``.\n        "
        if np.any(self._phase != self._phase[0]):
            array = np.hstack([self._z, self._x, self.phase.reshape((self.phase.shape[0], 1))])
        else:
            array = np.hstack([self._z, self._x])
        if return_counts:
            (_, index, counts) = np.unique(array, return_index=True, return_counts=True, axis=0)
        else:
            (_, index) = np.unique(array, return_index=True, axis=0)
        sort_inds = index.argsort()
        index = index[sort_inds]
        unique = PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))
        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_counts:
            ret += (counts[sort_inds],)
        if len(ret) == 1:
            return ret[0]
        return ret

    def tensor(self, other: PauliList) -> PauliList:
        if False:
            while True:
                i = 10
        'Return the tensor product with each Pauli in the list.\n\n        Args:\n            other (PauliList): another PauliList.\n\n        Returns:\n            PauliList: the list of tensor product Paulis.\n\n        Raises:\n            QiskitError: if other cannot be converted to a PauliList, does\n                         not have either 1 or the same number of Paulis as\n                         the current list.\n        '
        if not isinstance(other, PauliList):
            other = PauliList(other)
        return PauliList(super().tensor(other))

    def expand(self, other: PauliList) -> PauliList:
        if False:
            print('Hello World!')
        'Return the expand product of each Pauli in the list.\n\n        Args:\n            other (PauliList): another PauliList.\n\n        Returns:\n            PauliList: the list of tensor product Paulis.\n\n        Raises:\n            QiskitError: if other cannot be converted to a PauliList, does\n                         not have either 1 or the same number of Paulis as\n                         the current list.\n        '
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError('Incompatible PauliLists. Other list must have either 1 or the same number of Paulis.')
        return PauliList(super().expand(other))

    def compose(self, other: PauliList, qargs: None | list=None, front: bool=False, inplace: bool=False) -> PauliList:
        if False:
            return 10
        'Return the composition self∘other for each Pauli in the list.\n\n        Args:\n            other (PauliList): another PauliList.\n            qargs (None or list): qubits to apply dot product on (Default: ``None``).\n            front (bool): If True use `dot` composition method [default: ``False``].\n            inplace (bool): If ``True`` update in-place (default: ``False``).\n\n        Returns:\n            PauliList: the list of composed Paulis.\n\n        Raises:\n            QiskitError: if other cannot be converted to a PauliList, does\n                         not have either 1 or the same number of Paulis as\n                         the current list, or has the wrong number of qubits\n                         for the specified ``qargs``.\n        '
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError('Incompatible PauliLists. Other list must have either 1 or the same number of Paulis.')
        return PauliList(super().compose(other, qargs=qargs, front=front, inplace=inplace))

    def dot(self, other: PauliList, qargs: None | list=None, inplace: bool=False) -> PauliList:
        if False:
            while True:
                i = 10
        'Return the composition other∘self for each Pauli in the list.\n\n        Args:\n            other (PauliList): another PauliList.\n            qargs (None or list): qubits to apply dot product on (Default: ``None``).\n            inplace (bool): If True update in-place (default: ``False``).\n\n        Returns:\n            PauliList: the list of composed Paulis.\n\n        Raises:\n            QiskitError: if other cannot be converted to a PauliList, does\n                         not have either 1 or the same number of Paulis as\n                         the current list, or has the wrong number of qubits\n                         for the specified ``qargs``.\n        '
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def _add(self, other, qargs=None):
        if False:
            print('Hello World!')
        'Append two PauliLists.\n\n        If ``qargs`` are specified the other operator will be added\n        assuming it is identity on all other subsystems.\n\n        Args:\n            other (PauliList): another table.\n            qargs (None or list): optional subsystems to add on\n                                  (Default: ``None``)\n\n        Returns:\n            PauliList: the concatenated list ``self`` + ``other``.\n        '
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, PauliList):
            other = PauliList(other)
        self._op_shape._validate_add(other._op_shape, qargs)
        base_phase = np.hstack((self._phase, other._phase))
        if qargs is None or (sorted(qargs) == qargs and len(qargs) == self.num_qubits):
            base_z = np.vstack([self._z, other._z])
            base_x = np.vstack([self._x, other._x])
        else:
            padded = BasePauli(np.zeros((other.size, self.num_qubits), dtype=bool), np.zeros((other.size, self.num_qubits), dtype=bool), np.zeros(other.size, dtype=int))
            padded = padded.compose(other, qargs=qargs, inplace=True)
            base_z = np.vstack([self._z, padded._z])
            base_x = np.vstack([self._x, padded._x])
        return PauliList(BasePauli(base_z, base_x, base_phase))

    def _multiply(self, other):
        if False:
            print('Hello World!')
        'Multiply each Pauli in the list by a phase.\n\n        Args:\n            other (complex or array): a complex number in [1, -1j, -1, 1j]\n\n        Returns:\n            PauliList: the list of Paulis other * self.\n\n        Raises:\n            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].\n        '
        return PauliList(super()._multiply(other))

    def conjugate(self):
        if False:
            while True:
                i = 10
        'Return the conjugate of each Pauli in the list.'
        return PauliList(super().conjugate())

    def transpose(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the transpose of each Pauli in the list.'
        return PauliList(super().transpose())

    def adjoint(self):
        if False:
            while True:
                i = 10
        'Return the adjoint of each Pauli in the list.'
        return PauliList(super().adjoint())

    def inverse(self):
        if False:
            while True:
                i = 10
        'Return the inverse of each Pauli in the list.'
        return PauliList(super().adjoint())

    def commutes(self, other: BasePauli, qargs: list | None=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True for each Pauli that commutes with other.\n\n        Args:\n            other (PauliList): another PauliList operator.\n            qargs (list): qubits to apply dot product on (default: ``None``).\n\n        Returns:\n            bool: ``True`` if Paulis commute, ``False`` if they anti-commute.\n        '
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, BasePauli):
            other = PauliList(other)
        return super().commutes(other, qargs=qargs)

    def anticommutes(self, other: BasePauli, qargs: list | None=None) -> bool:
        if False:
            while True:
                i = 10
        'Return ``True`` if other Pauli that anticommutes with other.\n\n        Args:\n            other (PauliList): another PauliList operator.\n            qargs (list): qubits to apply dot product on (default: ``None``).\n\n        Returns:\n            bool: ``True`` if Paulis anticommute, ``False`` if they commute.\n        '
        return np.logical_not(self.commutes(other, qargs=qargs))

    def commutes_with_all(self, other: PauliList) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Return indexes of rows that commute ``other``.\n\n        If ``other`` is a multi-row Pauli list the returned vector indexes rows\n        of the current PauliList that commute with *all* Paulis in other.\n        If no rows satisfy the condition the returned array will be empty.\n\n        Args:\n            other (PauliList): a single Pauli or multi-row PauliList.\n\n        Returns:\n            array: index array of the commuting rows.\n        '
        return self._commutes_with_all(other)

    def anticommutes_with_all(self, other: PauliList) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Return indexes of rows that commute other.\n\n        If ``other`` is a multi-row Pauli list the returned vector indexes rows\n        of the current PauliList that anti-commute with *all* Paulis in other.\n        If no rows satisfy the condition the returned array will be empty.\n\n        Args:\n            other (PauliList): a single Pauli or multi-row PauliList.\n\n        Returns:\n            array: index array of the anti-commuting rows.\n        '
        return self._commutes_with_all(other, anti=True)

    def _commutes_with_all(self, other, anti=False):
        if False:
            print('Hello World!')
        'Return row indexes that commute with all rows in another PauliList.\n\n        Args:\n            other (PauliList): a PauliList.\n            anti (bool): if ``True`` return rows that anti-commute, otherwise\n                         return rows that commute (Default: ``False``).\n\n        Returns:\n            array: index array of commuting or anti-commuting row.\n        '
        if not isinstance(other, PauliList):
            other = PauliList(other)
        comms = self.commutes(other[0])
        (inds,) = np.where(comms == int(not anti))
        for pauli in other[1:]:
            comms = self[inds].commutes(pauli)
            (new_inds,) = np.where(comms == int(not anti))
            if new_inds.size == 0:
                return new_inds
            inds = inds[new_inds]
        return inds

    def evolve(self, other: Pauli | Clifford | QuantumCircuit, qargs: list | None=None, frame: Literal['h', 's']='h') -> Pauli:
        if False:
            for i in range(10):
                print('nop')
        "Performs either Heisenberg (default) or Schrödinger picture\n        evolution of the Pauli by a Clifford and returns the evolved Pauli.\n\n        Schrödinger picture evolution can be chosen by passing parameter ``frame='s'``.\n        This option yields a faster calculation.\n\n        Heisenberg picture evolves the Pauli as :math:`P^\\prime = C^\\dagger.P.C`.\n\n        Schrödinger picture evolves the Pauli as :math:`P^\\prime = C.P.C^\\dagger`.\n\n        Args:\n            other (Pauli or Clifford or QuantumCircuit): The Clifford operator to evolve by.\n            qargs (list): a list of qubits to apply the Clifford to.\n            frame (string): ``'h'`` for Heisenberg (default) or ``'s'`` for Schrödinger framework.\n\n        Returns:\n            PauliList: the Pauli :math:`C^\\dagger.P.C` (Heisenberg picture)\n            or the Pauli :math:`C.P.C^\\dagger` (Schrödinger picture).\n\n        Raises:\n            QiskitError: if the Clifford number of qubits and qargs don't match.\n        "
        from qiskit.circuit import Instruction
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, (BasePauli, Instruction, QuantumCircuit, Clifford)):
            other = PauliList(other)
        return PauliList(super().evolve(other, qargs=qargs, frame=frame))

    def to_labels(self, array: bool=False):
        if False:
            print('Hello World!')
        'Convert a PauliList to a list Pauli string labels.\n\n        For large PauliLists converting using the ``array=True``\n        kwarg will be more efficient since it allocates memory for\n        the full Numpy array of labels in advance.\n\n        .. list-table:: Pauli Representations\n            :header-rows: 1\n\n            * - Label\n              - Symplectic\n              - Matrix\n            * - ``"I"``\n              - :math:`[0, 0]`\n              - :math:`\\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\end{bmatrix}`\n            * - ``"X"``\n              - :math:`[1, 0]`\n              - :math:`\\begin{bmatrix} 0 & 1 \\\\ 1 & 0  \\end{bmatrix}`\n            * - ``"Y"``\n              - :math:`[1, 1]`\n              - :math:`\\begin{bmatrix} 0 & -i \\\\ i & 0  \\end{bmatrix}`\n            * - ``"Z"``\n              - :math:`[0, 1]`\n              - :math:`\\begin{bmatrix} 1 & 0 \\\\ 0 & -1  \\end{bmatrix}`\n\n        Args:\n            array (bool): return a Numpy array if ``True``, otherwise\n                          return a list (Default: ``False``).\n\n        Returns:\n            list or array: The rows of the PauliList in label form.\n        '
        if (self.phase == 1).any():
            prefix_len = 2
        elif (self.phase > 0).any():
            prefix_len = 1
        else:
            prefix_len = 0
        str_len = self.num_qubits + prefix_len
        ret = np.zeros(self.size, dtype=f'<U{str_len}')
        iterator = self.label_iter()
        for i in range(self.size):
            ret[i] = next(iterator)
        if array:
            return ret
        return ret.tolist()

    def to_matrix(self, sparse: bool=False, array: bool=False) -> list:
        if False:
            i = 10
            return i + 15
        'Convert to a list or array of Pauli matrices.\n\n        For large PauliLists converting using the ``array=True``\n        kwarg will be more efficient since it allocates memory a full\n        rank-3 Numpy array of matrices in advance.\n\n        .. list-table:: Pauli Representations\n            :header-rows: 1\n\n            * - Label\n              - Symplectic\n              - Matrix\n            * - ``"I"``\n              - :math:`[0, 0]`\n              - :math:`\\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\end{bmatrix}`\n            * - ``"X"``\n              - :math:`[1, 0]`\n              - :math:`\\begin{bmatrix} 0 & 1 \\\\ 1 & 0  \\end{bmatrix}`\n            * - ``"Y"``\n              - :math:`[1, 1]`\n              - :math:`\\begin{bmatrix} 0 & -i \\\\ i & 0  \\end{bmatrix}`\n            * - ``"Z"``\n              - :math:`[0, 1]`\n              - :math:`\\begin{bmatrix} 1 & 0 \\\\ 0 & -1  \\end{bmatrix}`\n\n        Args:\n            sparse (bool): if ``True`` return sparse CSR matrices, otherwise\n                           return dense Numpy arrays (Default: ``False``).\n            array (bool): return as rank-3 numpy array if ``True``, otherwise\n                          return a list of Numpy arrays (Default: ``False``).\n\n        Returns:\n            list: A list of dense Pauli matrices if ``array=False` and ``sparse=False`.\n            list: A list of sparse Pauli matrices if ``array=False`` and ``sparse=True``.\n            array: A dense rank-3 array of Pauli matrices if ``array=True``.\n        '
        if not array:
            return list(self.matrix_iter(sparse=sparse))
        dim = 2 ** self.num_qubits
        ret = np.zeros((self.size, dim, dim), dtype=complex)
        iterator = self.matrix_iter(sparse=sparse)
        for i in range(self.size):
            ret[i] = next(iterator)
        return ret

    def label_iter(self):
        if False:
            return 10
        'Return a label representation iterator.\n\n        This is a lazy iterator that converts each row into the string\n        label only as it is used. To convert the entire table to labels use\n        the :meth:`to_labels` method.\n\n        Returns:\n            LabelIterator: label iterator object for the PauliList.\n        '

        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""

            def __repr__(self):
                if False:
                    print('Hello World!')
                return f'<PauliList_label_iterator at {hex(id(self))}>'

            def __getitem__(self, key):
                if False:
                    return 10
                return self.obj._to_label(self.obj._z[key], self.obj._x[key], self.obj._phase[key])
        return LabelIterator(self)

    def matrix_iter(self, sparse: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a matrix representation iterator.\n\n        This is a lazy iterator that converts each row into the Pauli matrix\n        representation only as it is used. To convert the entire table to\n        matrices use the :meth:`to_matrix` method.\n\n        Args:\n            sparse (bool): optionally return sparse CSR matrices if ``True``,\n                           otherwise return Numpy array matrices\n                           (Default: ``False``)\n\n        Returns:\n            MatrixIterator: matrix iterator object for the PauliList.\n        '

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return f'<PauliList_matrix_iterator at {hex(id(self))}>'

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return self.obj._to_matrix(self.obj._z[key], self.obj._x[key], self.obj._phase[key], sparse=sparse)
        return MatrixIterator(self)

    @classmethod
    def from_symplectic(cls, z: np.ndarray, x: np.ndarray, phase: np.ndarray | None=0) -> PauliList:
        if False:
            for i in range(10):
                print('nop')
        'Construct a PauliList from a symplectic data.\n\n        Args:\n            z (np.ndarray): 2D boolean Numpy array.\n            x (np.ndarray): 2D boolean Numpy array.\n            phase (np.ndarray or None): Optional, 1D integer array from Z_4.\n\n        Returns:\n            PauliList: the constructed PauliList.\n        '
        (base_z, base_x, base_phase) = cls._from_array(z, x, phase)
        return cls(BasePauli(base_z, base_x, base_phase))

    def _noncommutation_graph(self, qubit_wise):
        if False:
            for i in range(10):
                print('nop')
        'Create an edge list representing the non-commutation graph (Pauli Graph).\n\n        An edge (i, j) is present if i and j are not commutable.\n\n        Args:\n            qubit_wise (bool): whether the commutation rule is applied to the whole operator,\n                or on a per-qubit basis.\n\n        Returns:\n            list[tuple[int,int]]: A list of pairs of indices of the PauliList that are not commutable.\n        '
        mat1 = np.array([op.z + 2 * op.x for op in self], dtype=np.int8)
        mat2 = mat1[:, None]
        qubit_anticommutation_mat = mat1 * mat2 * (mat1 - mat2)
        if qubit_wise:
            adjacency_mat = np.logical_or.reduce(qubit_anticommutation_mat, axis=2)
        else:
            adjacency_mat = np.logical_xor.reduce(qubit_anticommutation_mat, axis=2)
        return list(zip(*np.where(np.triu(adjacency_mat, k=1))))

    def _create_graph(self, qubit_wise):
        if False:
            return 10
        'Transform measurement operator grouping problem into graph coloring problem\n\n        Args:\n            qubit_wise (bool): whether the commutation rule is applied to the whole operator,\n                or on a per-qubit basis.\n\n        Returns:\n            rustworkx.PyGraph: A class of undirected graphs\n        '
        edges = self._noncommutation_graph(qubit_wise)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.size))
        graph.add_edges_from_no_data(edges)
        return graph

    def group_qubit_wise_commuting(self) -> list[PauliList]:
        if False:
            while True:
                i = 10
        'Partition a PauliList into sets of mutually qubit-wise commuting Pauli strings.\n\n        Returns:\n            list[PauliList]: List of PauliLists where each PauliList contains commutable Pauli operators.\n        '
        return self.group_commuting(qubit_wise=True)

    def group_commuting(self, qubit_wise: bool=False) -> list[PauliList]:
        if False:
            for i in range(10):
                print('nop')
        'Partition a PauliList into sets of commuting Pauli strings.\n\n        Args:\n            qubit_wise (bool): whether the commutation rule is applied to the whole operator,\n                or on a per-qubit basis.  For example:\n\n                .. code-block:: python\n\n                    >>> from qiskit.quantum_info import PauliList\n                    >>> op = PauliList(["XX", "YY", "IZ", "ZZ"])\n                    >>> op.group_commuting()\n                    [PauliList([\'XX\', \'YY\']), PauliList([\'IZ\', \'ZZ\'])]\n                    >>> op.group_commuting(qubit_wise=True)\n                    [PauliList([\'XX\']), PauliList([\'YY\']), PauliList([\'IZ\', \'ZZ\'])]\n\n        Returns:\n            list[PauliList]: List of PauliLists where each PauliList contains commuting Pauli operators.\n        '
        graph = self._create_graph(qubit_wise)
        coloring_dict = rx.graph_greedy_color(graph)
        groups = defaultdict(list)
        for (idx, color) in coloring_dict.items():
            groups[color].append(idx)
        return [self[group] for group in groups.values()]