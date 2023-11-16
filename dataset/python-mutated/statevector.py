"""
Statevector quantum state class.
"""
from __future__ import annotations
import copy
import re
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit._accelerate.pauli_expval import expval_pauli_no_x, expval_pauli_with_x

class Statevector(QuantumState, TolerancesMixin):
    """Statevector class"""

    def __init__(self, data: np.ndarray | list | Statevector | Operator | QuantumCircuit | Instruction, dims: int | tuple | list | None=None):
        if False:
            print('Hello World!')
        'Initialize a statevector object.\n\n        Args:\n            data (np.array or list or Statevector or Operator or QuantumCircuit or\n                  qiskit.circuit.Instruction):\n                Data from which the statevector can be constructed. This can be either a complex\n                vector, another statevector, a ``Operator`` with only one column or a\n                ``QuantumCircuit`` or ``Instruction``.  If the data is a circuit or instruction,\n                the statevector is constructed by assuming that all qubits are initialized to the\n                zero state.\n            dims (int or tuple or list): Optional. The subsystem dimension of\n                                         the state (See additional information).\n\n        Raises:\n            QiskitError: if input data is not valid.\n\n        Additional Information:\n            The ``dims`` kwarg can be None, an integer, or an iterable of\n            integers.\n\n            * ``Iterable`` -- the subsystem dimensions are the values in the list\n              with the total number of subsystems given by the length of the list.\n\n            * ``Int`` or ``None`` -- the length of the input vector\n              specifies the total dimension of the density matrix. If it is a\n              power of two the state will be initialized as an N-qubit state.\n              If it is not a power of two the state will have a single\n              d-dimensional subsystem.\n        '
        if isinstance(data, (list, np.ndarray)):
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, Statevector):
            self._data = data._data
            if dims is None:
                dims = data._op_shape._dims_l
        elif isinstance(data, Operator):
            (input_dim, _) = data.dim
            if input_dim != 1:
                raise QiskitError('Input Operator is not a column-vector.')
            self._data = np.ravel(data.data)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = Statevector.from_instruction(data).data
        else:
            raise QiskitError('Invalid input data format for Statevector')
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim != 1:
            if ndim == 2 and shape[1] == 1:
                self._data = np.reshape(self._data, shape[0])
                shape = self._data.shape
            elif ndim != 2 or shape[1] != 1:
                raise QiskitError('Invalid input: not a vector or column-vector.')
        super().__init__(op_shape=OpShape.auto(shape=shape, dims_l=dims, num_qubits_r=0))

    def __array__(self, dtype=None):
        if False:
            while True:
                i = 10
        if dtype:
            return np.asarray(self.data, dtype=dtype)
        return self.data

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return super().__eq__(other) and np.allclose(self._data, other._data, rtol=self.rtol, atol=self.atol)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = 'Statevector('
        pad = len(prefix) * ' '
        return '{}{},\n{}dims={})'.format(prefix, np.array2string(self._data, separator=', ', prefix=prefix), pad, self._op_shape.dims_l())

    @property
    def settings(self) -> dict:
        if False:
            i = 10
            return i + 15
        'Return settings.'
        return {'data': self._data, 'dims': self._op_shape.dims_l()}

    def draw(self, output: str | None=None, **drawer_args):
        if False:
            return 10
        "Return a visualization of the Statevector.\n\n        **repr**: ASCII TextMatrix of the state's ``__repr__``.\n\n        **text**: ASCII TextMatrix that can be printed in the console.\n\n        **latex**: An IPython Latex object for displaying in Jupyter Notebooks.\n\n        **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.\n\n        **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.\n\n        **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.\n\n        **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.\n\n        **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.\n\n        **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.\n\n        Args:\n            output (str): Select the output method to use for drawing the\n                state. Valid choices are `repr`, `text`, `latex`, `latex_source`,\n                `qsphere`, `hinton`, `bloch`, `city`, or `paulivec`. Default is `repr`.\n                Default can be changed by adding the line ``state_drawer = <default>`` to\n                ``~/.qiskit/settings.conf`` under ``[default]``.\n            drawer_args: Arguments to be passed directly to the relevant drawing\n                function or constructor (`TextMatrix()`, `array_to_latex()`,\n                `plot_state_qsphere()`, `plot_state_hinton()` or `plot_bloch_multivector()`).\n                See the relevant function under `qiskit.visualization` for that function's\n                documentation.\n\n        Returns:\n            :class:`matplotlib.Figure` or :class:`str` or\n            :class:`TextMatrix` or :class:`IPython.display.Latex`:\n            Drawing of the Statevector.\n\n        Raises:\n            ValueError: when an invalid output method is selected.\n\n        Examples:\n\n            Plot one of the Bell states\n\n            .. plot::\n               :include-source:\n\n                from numpy import sqrt\n                from qiskit.quantum_info import Statevector\n                sv=Statevector([1/sqrt(2), 0, 0, -1/sqrt(2)])\n                sv.draw(output='hinton')\n\n        "
        from qiskit.visualization.state_visualization import state_drawer
        return state_drawer(self, output=output, **drawer_args)

    def _ipython_display_(self):
        if False:
            i = 10
            return i + 15
        out = self.draw()
        if isinstance(out, str):
            print(out)
        else:
            from IPython.display import display
            display(out)

    def __getitem__(self, key: int | str) -> np.complex128:
        if False:
            i = 10
            return i + 15
        "Return Statevector item either by index or binary label\n        Args:\n            key (int or str): index or corresponding binary label, e.g. '01' = 1.\n\n        Returns:\n            numpy.complex128: Statevector item.\n\n        Raises:\n            QiskitError: if key is not valid.\n        "
        if isinstance(key, str):
            try:
                key = int(key, 2)
            except ValueError:
                raise QiskitError(f"Key '{key}' is not a valid binary string.") from None
        if isinstance(key, int):
            if key >= self.dim:
                raise QiskitError(f'Key {key} is greater than Statevector dimension {self.dim}.')
            if key < 0:
                raise QiskitError(f'Key {key} is not a valid positive value.')
            return self._data[key]
        else:
            raise QiskitError('Key must be int or a valid binary string.')

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield from self._data

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._data)

    @property
    def data(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Return data.'
        return self._data

    def is_valid(self, atol: float | None=None, rtol: float | None=None) -> bool:
        if False:
            while True:
                i = 10
        'Return True if a Statevector has norm 1.'
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        norm = np.linalg.norm(self.data)
        return np.allclose(norm, 1, rtol=rtol, atol=atol)

    def to_operator(self) -> Operator:
        if False:
            while True:
                i = 10
        'Convert state to a rank-1 projector operator'
        mat = np.outer(self.data, np.conj(self.data))
        return Operator(mat, input_dims=self.dims(), output_dims=self.dims())

    def conjugate(self) -> Statevector:
        if False:
            while True:
                i = 10
        'Return the conjugate of the operator.'
        return Statevector(np.conj(self.data), dims=self.dims())

    def trace(self) -> np.float64:
        if False:
            print('Hello World!')
        'Return the trace of the quantum state as a density matrix.'
        return np.sum(np.abs(self.data) ** 2)

    def purity(self) -> np.float64:
        if False:
            return 10
        'Return the purity of the quantum state.'
        return self.trace() ** 2

    def tensor(self, other: Statevector) -> Statevector:
        if False:
            print('Hello World!')
        'Return the tensor product state self ⊗ other.\n\n        Args:\n            other (Statevector): a quantum state object.\n\n        Returns:\n            Statevector: the tensor product operator self ⊗ other.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        if not isinstance(other, Statevector):
            other = Statevector(other)
        ret = copy.copy(self)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        ret._data = np.kron(self._data, other._data)
        return ret

    def inner(self, other: Statevector) -> np.complex128:
        if False:
            for i in range(10):
                print('nop')
        'Return the inner product of self and other as\n        :math:`\\langle self| other \\rangle`.\n\n        Args:\n            other (Statevector): a quantum state object.\n\n        Returns:\n            np.complex128: the inner product of self and other, :math:`\\langle self| other \\rangle`.\n\n        Raises:\n            QiskitError: if other is not a quantum state or has different dimension.\n        '
        if not isinstance(other, Statevector):
            other = Statevector(other)
        if self.dims() != other.dims():
            raise QiskitError(f'Statevector dimensions do not match: {self.dims()} and {other.dims()}.')
        inner = np.vdot(self.data, other.data)
        return inner

    def expand(self, other: Statevector) -> Statevector:
        if False:
            return 10
        'Return the tensor product state other ⊗ self.\n\n        Args:\n            other (Statevector): a quantum state object.\n\n        Returns:\n            Statevector: the tensor product state other ⊗ self.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        if not isinstance(other, Statevector):
            other = Statevector(other)
        ret = copy.copy(self)
        ret._op_shape = self._op_shape.expand(other._op_shape)
        ret._data = np.kron(other._data, self._data)
        return ret

    def _add(self, other):
        if False:
            return 10
        'Return the linear combination self + other.\n\n        Args:\n            other (Statevector): a quantum state object.\n\n        Returns:\n            Statevector: the linear combination self + other.\n\n        Raises:\n            QiskitError: if other is not a quantum state, or has\n                         incompatible dimensions.\n        '
        if not isinstance(other, Statevector):
            other = Statevector(other)
        self._op_shape._validate_add(other._op_shape)
        ret = copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return the scalar multiplied state self * other.\n\n        Args:\n            other (complex): a complex number.\n\n        Returns:\n            Statevector: the scalar multiplied state other * self.\n\n        Raises:\n            QiskitError: if other is not a valid complex number.\n        '
        if not isinstance(other, Number):
            raise QiskitError('other is not a number')
        ret = copy.copy(self)
        ret._data = other * self.data
        return ret

    def evolve(self, other: Operator | QuantumCircuit | Instruction, qargs: list[int] | None=None) -> Statevector:
        if False:
            while True:
                i = 10
        'Evolve a quantum state by the operator.\n\n        Args:\n            other (Operator | QuantumCircuit | circuit.Instruction): The operator to evolve by.\n            qargs (list): a list of Statevector subsystem positions to apply\n                           the operator on.\n\n        Returns:\n            Statevector: the output quantum state.\n\n        Raises:\n            QiskitError: if the operator dimension does not match the\n                         specified Statevector subsystem dimensions.\n        '
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        ret = copy.copy(self)
        if isinstance(other, QuantumCircuit):
            other = other.to_instruction()
        if isinstance(other, Instruction):
            if self.num_qubits is None:
                raise QiskitError('Cannot apply QuantumCircuit to non-qubit Statevector.')
            return self._evolve_instruction(ret, other, qargs=qargs)
        if not isinstance(other, Operator):
            dims = self.dims(qargs=qargs)
            other = Operator(other, input_dims=dims, output_dims=dims)
        if self.dims(qargs) != other.input_dims():
            raise QiskitError('Operator input dimensions are not equal to statevector subsystem dimensions.')
        return Statevector._evolve_operator(ret, other, qargs=qargs)

    def equiv(self, other: Statevector, rtol: float | None=None, atol: float | None=None) -> bool:
        if False:
            return 10
        'Return True if other is equivalent as a statevector up to global phase.\n\n        .. note::\n\n            If other is not a Statevector, but can be used to initialize a statevector object,\n            this will check that Statevector(other) is equivalent to the current statevector up\n            to global phase.\n\n        Args:\n            other (Statevector): an object from which a ``Statevector`` can be constructed.\n            rtol (float): relative tolerance value for comparison.\n            atol (float): absolute tolerance value for comparison.\n\n        Returns:\n            bool: True if statevectors are equivalent up to global phase.\n        '
        if not isinstance(other, Statevector):
            try:
                other = Statevector(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.data, other.data, ignore_phase=True, rtol=rtol, atol=atol)

    def reverse_qargs(self) -> Statevector:
        if False:
            return 10
        'Return a Statevector with reversed subsystem ordering.\n\n        For a tensor product state this is equivalent to reversing the order\n        of tensor product subsystems. For a statevector\n        :math:`|\\psi \\rangle = |\\psi_{n-1} \\rangle \\otimes ... \\otimes |\\psi_0 \\rangle`\n        the returned statevector will be\n        :math:`|\\psi_{0} \\rangle \\otimes ... \\otimes |\\psi_{n-1} \\rangle`.\n\n        Returns:\n            Statevector: the Statevector with reversed subsystem order.\n        '
        ret = copy.copy(self)
        axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
        ret._data = np.reshape(np.transpose(np.reshape(self.data, self._op_shape.tensor_shape), axes), self._op_shape.shape)
        ret._op_shape = self._op_shape.reverse()
        return ret

    def _expectation_value_pauli(self, pauli, qargs=None):
        if False:
            i = 10
            return i + 15
        'Compute the expectation value of a Pauli.\n\n        Args:\n            pauli (Pauli): a Pauli operator to evaluate expval of.\n            qargs (None or list): subsystems to apply operator on.\n\n        Returns:\n            complex: the expectation value.\n        '
        n_pauli = len(pauli)
        if qargs is None:
            qubits = np.arange(n_pauli)
        else:
            qubits = np.array(qargs)
        x_mask = np.dot(1 << qubits, pauli.x)
        z_mask = np.dot(1 << qubits, pauli.z)
        pauli_phase = (-1j) ** pauli.phase if pauli.phase else 1
        if x_mask + z_mask == 0:
            return pauli_phase * np.linalg.norm(self.data)
        if x_mask == 0:
            return pauli_phase * expval_pauli_no_x(self.data, self.num_qubits, z_mask)
        x_max = qubits[pauli.x][-1]
        y_phase = (-1j) ** pauli._count_y()
        y_phase = y_phase[0]
        return pauli_phase * expval_pauli_with_x(self.data, self.num_qubits, z_mask, x_mask, y_phase, x_max)

    def expectation_value(self, oper: BaseOperator | QuantumCircuit | Instruction, qargs: None | list[int]=None) -> complex:
        if False:
            i = 10
            return i + 15
        'Compute the expectation value of an operator.\n\n        Args:\n            oper (Operator): an operator to evaluate expval of.\n            qargs (None or list): subsystems to apply operator on.\n\n        Returns:\n            complex: the expectation value.\n        '
        if isinstance(oper, Pauli):
            return self._expectation_value_pauli(oper, qargs)
        if isinstance(oper, SparsePauliOp):
            return sum((coeff * self._expectation_value_pauli(Pauli((z, x)), qargs) for (z, x, coeff) in zip(oper.paulis.z, oper.paulis.x, oper.coeffs)))
        val = self.evolve(oper, qargs=qargs)
        conj = self.conjugate()
        return np.dot(conj.data, val.data)

    def probabilities(self, qargs: None | list[int]=None, decimals: None | int=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "Return the subsystem measurement probability vector.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            np.array: The Numpy vector array of probabilities.\n\n        Examples:\n\n            Consider a 2-qubit product state\n            :math:`|\\psi\\rangle=|+\\rangle\\otimes|0\\rangle`.\n\n            .. code-block::\n\n                from qiskit.quantum_info import Statevector\n\n                psi = Statevector.from_label('+0')\n\n                # Probabilities for measuring both qubits\n                probs = psi.probabilities()\n                print('probs: {}'.format(probs))\n\n                # Probabilities for measuring only qubit-0\n                probs_qubit_0 = psi.probabilities([0])\n                print('Qubit-0 probs: {}'.format(probs_qubit_0))\n\n                # Probabilities for measuring only qubit-1\n                probs_qubit_1 = psi.probabilities([1])\n                print('Qubit-1 probs: {}'.format(probs_qubit_1))\n\n            .. parsed-literal::\n\n                probs: [0.5 0.  0.5 0. ]\n                Qubit-0 probs: [1. 0.]\n                Qubit-1 probs: [0.5 0.5]\n\n            We can also permute the order of qubits in the ``qargs`` list\n            to change the qubit position in the probabilities output\n\n            .. code-block::\n\n                from qiskit.quantum_info import Statevector\n\n                psi = Statevector.from_label('+0')\n\n                # Probabilities for measuring both qubits\n                probs = psi.probabilities([0, 1])\n                print('probs: {}'.format(probs))\n\n                # Probabilities for measuring both qubits\n                # but swapping qubits 0 and 1 in output\n                probs_swapped = psi.probabilities([1, 0])\n                print('Swapped probs: {}'.format(probs_swapped))\n\n            .. parsed-literal::\n\n                probs: [0.5 0.  0.5 0. ]\n                Swapped probs: [0.5 0.5 0.  0. ]\n\n        "
        probs = self._subsystem_probabilities(np.abs(self.data) ** 2, self._op_shape.dims_l(), qargs=qargs)
        probs = np.clip(probs, a_min=0, a_max=1)
        if decimals is not None:
            probs = probs.round(decimals=decimals)
        return probs

    def reset(self, qargs: list[int] | None=None) -> Statevector:
        if False:
            return 10
        'Reset state or subsystems to the 0-state.\n\n        Args:\n            qargs (list or None): subsystems to reset, if None all\n                                  subsystems will be reset to their 0-state\n                                  (Default: None).\n\n        Returns:\n            Statevector: the reset state.\n\n        Additional Information:\n            If all subsystems are reset this will return the ground state\n            on all subsystems. If only a some subsystems are reset this\n            function will perform a measurement on those subsystems and\n            evolve the subsystems so that the collapsed post-measurement\n            states are rotated to the 0-state. The RNG seed for this\n            sampling can be set using the :meth:`seed` method.\n        '
        if qargs is None:
            ret = copy.copy(self)
            state = np.zeros(self._op_shape.shape, dtype=complex)
            state[0] = 1
            ret._data = state
            return ret
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])
        reset = np.eye(len(probs))
        reset[0, 0] = 0
        reset[sample, sample] = 0
        reset[0, sample] = 1
        reset = np.dot(reset, np.diag(proj))
        return self.evolve(Operator(reset, input_dims=dims, output_dims=dims), qargs=qargs)

    @classmethod
    def from_label(cls, label: str) -> Statevector:
        if False:
            print('Hello World!')
        'Return a tensor product of Pauli X,Y,Z eigenstates.\n\n        .. list-table:: Single-qubit state labels\n           :header-rows: 1\n\n           * - Label\n             - Statevector\n           * - ``"0"``\n             - :math:`[1, 0]`\n           * - ``"1"``\n             - :math:`[0, 1]`\n           * - ``"+"``\n             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`\n           * - ``"-"``\n             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`\n           * - ``"r"``\n             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`\n           * - ``"l"``\n             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`\n\n        Args:\n            label (string): a eigenstate string ket label (see table for\n                            allowed values).\n\n        Returns:\n            Statevector: The N-qubit basis state density matrix.\n\n        Raises:\n            QiskitError: if the label contains invalid characters, or the\n                         length of the label is larger than an explicitly\n                         specified num_qubits.\n        '
        if re.match('^[01rl\\-+]+$', label) is None:
            raise QiskitError('Label contains invalid characters.')
        z_label = label
        xy_states = False
        if re.match('^[01]+$', label) is None:
            xy_states = True
            z_label = z_label.replace('+', '0')
            z_label = z_label.replace('r', '0')
            z_label = z_label.replace('-', '1')
            z_label = z_label.replace('l', '1')
        num_qubits = len(label)
        data = np.zeros(1 << num_qubits, dtype=complex)
        pos = int(z_label, 2)
        data[pos] = 1
        state = Statevector(data)
        if xy_states:
            x_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            y_mat = np.dot(np.diag([1, 1j]), x_mat)
            for (qubit, char) in enumerate(reversed(label)):
                if char in ['+', '-']:
                    state = state.evolve(x_mat, qargs=[qubit])
                elif char in ['r', 'l']:
                    state = state.evolve(y_mat, qargs=[qubit])
        return state

    @staticmethod
    def from_int(i: int, dims: int | tuple | list) -> Statevector:
        if False:
            print('Hello World!')
        'Return a computational basis statevector.\n\n        Args:\n            i (int): the basis state element.\n            dims (int or tuple or list): The subsystem dimensions of the statevector\n                                         (See additional information).\n\n        Returns:\n            Statevector: The computational basis state :math:`|i\\rangle`.\n\n        Additional Information:\n            The ``dims`` kwarg can be an integer or an iterable of integers.\n\n            * ``Iterable`` -- the subsystem dimensions are the values in the list\n              with the total number of subsystems given by the length of the list.\n\n            * ``Int`` -- the integer specifies the total dimension of the\n              state. If it is a power of two the state will be initialized\n              as an N-qubit state. If it is not a power of  two the state\n              will have a single d-dimensional subsystem.\n        '
        size = np.prod(dims)
        state = np.zeros(size, dtype=complex)
        state[i] = 1.0
        return Statevector(state, dims=dims)

    @classmethod
    def from_instruction(cls, instruction: Instruction | QuantumCircuit) -> Statevector:
        if False:
            print('Hello World!')
        'Return the output statevector of an instruction.\n\n        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of the\n        same number of qubits as the input instruction or circuit, evolved\n        by the input instruction, and the output statevector returned.\n\n        Args:\n            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit\n\n        Returns:\n            Statevector: The final statevector.\n\n        Raises:\n            QiskitError: if the instruction contains invalid instructions for\n                         the statevector simulation.\n        '
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        init = np.zeros(2 ** instruction.num_qubits, dtype=complex)
        init[0] = 1.0
        vec = Statevector(init, dims=instruction.num_qubits * (2,))
        return Statevector._evolve_instruction(vec, instruction)

    def to_dict(self, decimals: None | int=None) -> dict:
        if False:
            while True:
                i = 10
        "Convert the statevector to dictionary form.\n\n        This dictionary representation uses a Ket-like notation where the\n        dictionary keys are qudit strings for the subsystem basis vectors.\n        If any subsystem has a dimension greater than 10 comma delimiters are\n        inserted between integers so that subsystems can be distinguished.\n\n        Args:\n            decimals (None or int): the number of decimal places to round\n                                    values. If None no rounding is done\n                                    (Default: None).\n\n        Returns:\n            dict: the dictionary form of the Statevector.\n\n        Example:\n\n            The ket-form of a 2-qubit statevector\n            :math:`|\\psi\\rangle = |-\\rangle\\otimes |0\\rangle`\n\n            .. code-block::\n\n                from qiskit.quantum_info import Statevector\n\n                psi = Statevector.from_label('-0')\n                print(psi.to_dict())\n\n            .. parsed-literal::\n\n                {'00': (0.7071067811865475+0j), '10': (-0.7071067811865475+0j)}\n\n            For non-qubit subsystems the integer range can go from 0 to 9. For\n            example in a qutrit system\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit.quantum_info import Statevector\n\n                vec = np.zeros(9)\n                vec[0] = 1 / np.sqrt(2)\n                vec[-1] = 1 / np.sqrt(2)\n                psi = Statevector(vec, dims=(3, 3))\n                print(psi.to_dict())\n\n            .. parsed-literal::\n\n                {'00': (0.7071067811865475+0j), '22': (0.7071067811865475+0j)}\n\n            For large subsystem dimensions delimiters are required. The\n            following example is for a 20-dimensional system consisting of\n            a qubit and 10-dimensional qudit.\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit.quantum_info import Statevector\n\n                vec = np.zeros(2 * 10)\n                vec[0] = 1 / np.sqrt(2)\n                vec[-1] = 1 / np.sqrt(2)\n                psi = Statevector(vec, dims=(2, 10))\n                print(psi.to_dict())\n\n            .. parsed-literal::\n\n                {'00': (0.7071067811865475+0j), '91': (0.7071067811865475+0j)}\n\n        "
        return self._vector_to_dict(self.data, self._op_shape.dims_l(), decimals=decimals, string_labels=True)

    @staticmethod
    def _evolve_operator(statevec, oper, qargs=None):
        if False:
            print('Hello World!')
        'Evolve a qudit statevector'
        new_shape = statevec._op_shape.compose(oper._op_shape, qargs=qargs)
        if qargs is None:
            statevec._data = np.dot(oper._data, statevec._data)
            statevec._op_shape = new_shape
            return statevec
        num_qargs = statevec._op_shape.num_qargs[0]
        indices = [num_qargs - 1 - i for i in reversed(qargs)]
        axes = indices + [i for i in range(num_qargs) if i not in indices]
        axes_inv = np.argsort(axes).tolist()
        contract_dim = oper._op_shape.shape[1]
        contract_shape = (contract_dim, statevec._op_shape.shape[0] // contract_dim)
        tensor = np.transpose(np.reshape(statevec.data, statevec._op_shape.tensor_shape), axes)
        tensor_shape = tensor.shape
        tensor = np.reshape(np.dot(oper.data, np.reshape(tensor, contract_shape)), tensor_shape)
        statevec._data = np.reshape(np.transpose(tensor, axes_inv), new_shape.shape[0])
        statevec._op_shape = new_shape
        return statevec

    @staticmethod
    def _evolve_instruction(statevec, obj, qargs=None):
        if False:
            while True:
                i = 10
        'Update the current Statevector by applying an instruction.'
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier
        from qiskit.circuit.library.data_preparation.initializer import Initialize
        mat = Operator._instruction_to_matrix(obj)
        if mat is not None:
            return Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)
        if isinstance(obj, Reset):
            statevec._data = statevec.reset(qargs)._data
            return statevec
        if isinstance(obj, Barrier):
            return statevec
        if isinstance(obj, Initialize):
            if all((isinstance(param, str) for param in obj.params)):
                initialization = Statevector.from_label(''.join(obj.params))._data
            elif len(obj.params) == 1:
                state = int(np.real(obj.params[0]))
                initialization = Statevector.from_int(state, (2,) * obj.num_qubits)._data
            else:
                initialization = np.asarray(obj.params, dtype=complex)
            if qargs is None:
                statevec._data = initialization
            else:
                statevec._data = statevec.reset(qargs)._data
                mat = np.zeros((2 ** len(qargs), 2 ** len(qargs)), dtype=complex)
                mat[:, 0] = initialization
                statevec = Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)
            return statevec
        if obj.definition is None:
            raise QiskitError(f'Cannot apply Instruction: {obj.name}')
        if not isinstance(obj.definition, QuantumCircuit):
            raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(obj.name, type(obj.definition)))
        if obj.definition.global_phase:
            statevec._data *= np.exp(1j * float(obj.definition.global_phase))
        qubits = {qubit: i for (i, qubit) in enumerate(obj.definition.qubits)}
        for instruction in obj.definition:
            if instruction.clbits:
                raise QiskitError(f'Cannot apply instruction with classical bits: {instruction.operation.name}')
            if qargs is None:
                new_qargs = [qubits[tup] for tup in instruction.qubits]
            else:
                new_qargs = [qargs[qubits[tup]] for tup in instruction.qubits]
            Statevector._evolve_instruction(statevec, instruction.operation, qargs=new_qargs)
        return statevec