"""
DensityMatrix quantum state class.
"""
from __future__ import annotations
import copy
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit._accelerate.pauli_expval import density_expval_pauli_no_x, density_expval_pauli_with_x
from qiskit.quantum_info.states.statevector import Statevector

class DensityMatrix(QuantumState, TolerancesMixin):
    """DensityMatrix class"""

    def __init__(self, data: np.ndarray | list | QuantumCircuit | Instruction | QuantumState, dims: int | tuple | list | None=None):
        if False:
            while True:
                i = 10
        'Initialize a density matrix object.\n\n        Args:\n            data (np.ndarray or list or matrix_like or QuantumCircuit or\n                  qiskit.circuit.Instruction):\n                A statevector, quantum instruction or an object with a ``to_operator`` or\n                ``to_matrix`` method from which the density matrix can be constructed.\n                If a vector the density matrix is constructed as the projector of that vector.\n                If a quantum instruction, the density matrix is constructed by assuming all\n                qubits are initialized in the zero state.\n            dims (int or tuple or list): Optional. The subsystem dimension\n                    of the state (See additional information).\n\n        Raises:\n            QiskitError: if input data is not valid.\n\n        Additional Information:\n            The ``dims`` kwarg can be None, an integer, or an iterable of\n            integers.\n\n            * ``Iterable`` -- the subsystem dimensions are the values in the list\n              with the total number of subsystems given by the length of the list.\n\n            * ``Int`` or ``None`` -- the leading dimension of the input matrix\n              specifies the total dimension of the density matrix. If it is a\n              power of two the state will be initialized as an N-qubit state.\n              If it is not a power of two the state will have a single\n              d-dimensional subsystem.\n        '
        if isinstance(data, (list, np.ndarray)):
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = DensityMatrix.from_instruction(data)._data
        elif hasattr(data, 'to_operator'):
            op = data.to_operator()
            self._data = op.data
            if dims is None:
                dims = op.output_dims()
        elif hasattr(data, 'to_matrix'):
            self._data = np.asarray(data.to_matrix(), dtype=complex)
        else:
            raise QiskitError('Invalid input data format for DensityMatrix')
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim == 2 and shape[0] == shape[1]:
            pass
        elif ndim == 1:
            self._data = np.outer(self._data, np.conj(self._data))
        elif ndim == 2 and shape[1] == 1:
            self._data = np.reshape(self._data, shape[0])
        else:
            raise QiskitError('Invalid DensityMatrix input: not a square matrix.')
        super().__init__(op_shape=OpShape.auto(shape=self._data.shape, dims_l=dims, dims_r=dims))

    def __array__(self, dtype=None):
        if False:
            return 10
        if dtype:
            return np.asarray(self.data, dtype=dtype)
        return self.data

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return super().__eq__(other) and np.allclose(self._data, other._data, rtol=self.rtol, atol=self.atol)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = 'DensityMatrix('
        pad = len(prefix) * ' '
        return '{}{},\n{}dims={})'.format(prefix, np.array2string(self._data, separator=', ', prefix=prefix), pad, self._op_shape.dims_l())

    @property
    def settings(self):
        if False:
            print('Hello World!')
        'Return settings.'
        return {'data': self.data, 'dims': self._op_shape.dims_l()}

    def draw(self, output: str | None=None, **drawer_args):
        if False:
            i = 10
            return i + 15
        "Return a visualization of the Statevector.\n\n        **repr**: ASCII TextMatrix of the state's ``__repr__``.\n\n        **text**: ASCII TextMatrix that can be printed in the console.\n\n        **latex**: An IPython Latex object for displaying in Jupyter Notebooks.\n\n        **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.\n\n        **qsphere**: Matplotlib figure, rendering of density matrix using `plot_state_qsphere()`.\n\n        **hinton**: Matplotlib figure, rendering of density matrix using `plot_state_hinton()`.\n\n        **bloch**: Matplotlib figure, rendering of density matrix using `plot_bloch_multivector()`.\n\n        Args:\n            output (str): Select the output method to use for drawing the\n                state. Valid choices are `repr`, `text`, `latex`, `latex_source`,\n                `qsphere`, `hinton`, or `bloch`. Default is `repr`. Default can\n                be changed by adding the line ``state_drawer = <default>`` to\n                ``~/.qiskit/settings.conf`` under ``[default]``.\n            drawer_args: Arguments to be passed directly to the relevant drawing\n                function or constructor (`TextMatrix()`, `array_to_latex()`,\n                `plot_state_qsphere()`, `plot_state_hinton()` or `plot_bloch_multivector()`).\n                See the relevant function under `qiskit.visualization` for that function's\n                documentation.\n\n        Returns:\n            :class:`matplotlib.Figure` or :class:`str` or\n            :class:`TextMatrix` or :class:`IPython.display.Latex`:\n            Drawing of the Statevector.\n\n        Raises:\n            ValueError: when an invalid output method is selected.\n        "
        from qiskit.visualization.state_visualization import state_drawer
        return state_drawer(self, output=output, **drawer_args)

    def _ipython_display_(self):
        if False:
            print('Hello World!')
        out = self.draw()
        if isinstance(out, str):
            print(out)
        else:
            from IPython.display import display
            display(out)

    @property
    def data(self):
        if False:
            print('Hello World!')
        'Return data.'
        return self._data

    def is_valid(self, atol=None, rtol=None):
        if False:
            return 10
        'Return True if trace 1 and positive semidefinite.'
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        if not np.allclose(self.trace(), 1, rtol=rtol, atol=atol):
            return False
        if not is_hermitian_matrix(self.data, rtol=rtol, atol=atol):
            return False
        return is_positive_semidefinite_matrix(self.data, rtol=rtol, atol=atol)

    def to_operator(self) -> Operator:
        if False:
            while True:
                i = 10
        'Convert to Operator'
        dims = self.dims()
        return Operator(self.data, input_dims=dims, output_dims=dims)

    def conjugate(self):
        if False:
            return 10
        'Return the conjugate of the density matrix.'
        return DensityMatrix(np.conj(self.data), dims=self.dims())

    def trace(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the trace of the density matrix.'
        return np.trace(self.data)

    def purity(self):
        if False:
            print('Hello World!')
        'Return the purity of the quantum state.'
        return np.trace(np.dot(self.data, self.data))

    def tensor(self, other: DensityMatrix) -> DensityMatrix:
        if False:
            print('Hello World!')
        'Return the tensor product state self ⊗ other.\n\n        Args:\n            other (DensityMatrix): a quantum state object.\n\n        Returns:\n            DensityMatrix: the tensor product operator self ⊗ other.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        ret = copy.copy(self)
        ret._data = np.kron(self._data, other._data)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        return ret

    def expand(self, other: DensityMatrix) -> DensityMatrix:
        if False:
            return 10
        'Return the tensor product state other ⊗ self.\n\n        Args:\n            other (DensityMatrix): a quantum state object.\n\n        Returns:\n            DensityMatrix: the tensor product state other ⊗ self.\n\n        Raises:\n            QiskitError: if other is not a quantum state.\n        '
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        ret = copy.copy(self)
        ret._data = np.kron(other._data, self._data)
        ret._op_shape = self._op_shape.expand(other._op_shape)
        return ret

    def _add(self, other):
        if False:
            print('Hello World!')
        'Return the linear combination self + other.\n\n        Args:\n            other (DensityMatrix): a quantum state object.\n\n        Returns:\n            DensityMatrix: the linear combination self + other.\n\n        Raises:\n            QiskitError: if other is not a quantum state, or has\n                         incompatible dimensions.\n        '
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self._op_shape._validate_add(other._op_shape)
        ret = copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        if False:
            i = 10
            return i + 15
        'Return the scalar multiplied state other * self.\n\n        Args:\n            other (complex): a complex number.\n\n        Returns:\n            DensityMatrix: the scalar multiplied state other * self.\n\n        Raises:\n            QiskitError: if other is not a valid complex number.\n        '
        if not isinstance(other, Number):
            raise QiskitError('other is not a number')
        ret = copy.copy(self)
        ret._data = other * self.data
        return ret

    def evolve(self, other: Operator | QuantumChannel | Instruction | QuantumCircuit, qargs: list[int] | None=None) -> DensityMatrix:
        if False:
            i = 10
            return i + 15
        'Evolve a quantum state by an operator.\n\n        Args:\n            other (Operator or QuantumChannel\n                   or Instruction or Circuit): The operator to evolve by.\n            qargs (list): a list of QuantumState subsystem positions to apply\n                           the operator on.\n\n        Returns:\n            DensityMatrix: the output density matrix.\n\n        Raises:\n            QiskitError: if the operator dimension does not match the\n                         specified QuantumState subsystem dimensions.\n        '
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)
        if hasattr(other, 'to_quantumchannel'):
            return other.to_quantumchannel()._evolve(self, qargs=qargs)
        if isinstance(other, QuantumChannel):
            return other._evolve(self, qargs=qargs)
        if not isinstance(other, Operator):
            dims = self.dims(qargs=qargs)
            other = Operator(other, input_dims=dims, output_dims=dims)
        return self._evolve_operator(other, qargs=qargs)

    def reverse_qargs(self) -> DensityMatrix:
        if False:
            return 10
        'Return a DensityMatrix with reversed subsystem ordering.\n\n        For a tensor product state this is equivalent to reversing the order\n        of tensor product subsystems. For a density matrix\n        :math:`\\rho = \\rho_{n-1} \\otimes ... \\otimes \\rho_0`\n        the returned state will be\n        :math:`\\rho_0 \\otimes ... \\otimes \\rho_{n-1}`.\n\n        Returns:\n            DensityMatrix: the state with reversed subsystem order.\n        '
        ret = copy.copy(self)
        axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
        axes = axes + tuple((len(axes) + i for i in axes))
        ret._data = np.reshape(np.transpose(np.reshape(self.data, self._op_shape.tensor_shape), axes), self._op_shape.shape)
        ret._op_shape = self._op_shape.reverse()
        return ret

    def _expectation_value_pauli(self, pauli, qargs=None):
        if False:
            print('Hello World!')
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
            return pauli_phase * self.trace()
        data = np.ravel(self.data, order='F')
        if x_mask == 0:
            return pauli_phase * density_expval_pauli_no_x(data, self.num_qubits, z_mask)
        x_max = qubits[pauli.x][-1]
        y_phase = (-1j) ** pauli._count_y()
        y_phase = y_phase[0]
        return pauli_phase * density_expval_pauli_with_x(data, self.num_qubits, z_mask, x_mask, y_phase, x_max)

    def expectation_value(self, oper: Operator, qargs: None | list[int]=None) -> complex:
        if False:
            for i in range(10):
                print('nop')
        'Compute the expectation value of an operator.\n\n        Args:\n            oper (Operator): an operator to evaluate expval.\n            qargs (None or list): subsystems to apply the operator on.\n\n        Returns:\n            complex: the expectation value.\n        '
        if isinstance(oper, Pauli):
            return self._expectation_value_pauli(oper, qargs)
        if isinstance(oper, SparsePauliOp):
            return sum((coeff * self._expectation_value_pauli(Pauli((z, x)), qargs) for (z, x, coeff) in zip(oper.paulis.z, oper.paulis.x, oper.coeffs)))
        if not isinstance(oper, Operator):
            oper = Operator(oper)
        return np.trace(Operator(self).dot(oper, qargs=qargs).data)

    def probabilities(self, qargs: None | list[int]=None, decimals: None | int=None) -> np.ndarray:
        if False:
            return 10
        "Return the subsystem measurement probability vector.\n\n        Measurement probabilities are with respect to measurement in the\n        computation (diagonal) basis.\n\n        Args:\n            qargs (None or list): subsystems to return probabilities for,\n                if None return for all subsystems (Default: None).\n            decimals (None or int): the number of decimal places to round\n                values. If None no rounding is done (Default: None).\n\n        Returns:\n            np.array: The Numpy vector array of probabilities.\n\n        Examples:\n\n            Consider a 2-qubit product state :math:`\\rho=\\rho_1\\otimes\\rho_0`\n            with :math:`\\rho_1=|+\\rangle\\!\\langle+|`,\n            :math:`\\rho_0=|0\\rangle\\!\\langle0|`.\n\n            .. code-block::\n\n                from qiskit.quantum_info import DensityMatrix\n\n                rho = DensityMatrix.from_label('+0')\n\n                # Probabilities for measuring both qubits\n                probs = rho.probabilities()\n                print('probs: {}'.format(probs))\n\n                # Probabilities for measuring only qubit-0\n                probs_qubit_0 = rho.probabilities([0])\n                print('Qubit-0 probs: {}'.format(probs_qubit_0))\n\n                # Probabilities for measuring only qubit-1\n                probs_qubit_1 = rho.probabilities([1])\n                print('Qubit-1 probs: {}'.format(probs_qubit_1))\n\n            .. parsed-literal::\n\n                probs: [0.5 0.  0.5 0. ]\n                Qubit-0 probs: [1. 0.]\n                Qubit-1 probs: [0.5 0.5]\n\n            We can also permute the order of qubits in the ``qargs`` list\n            to change the qubit position in the probabilities output\n\n            .. code-block::\n\n                from qiskit.quantum_info import DensityMatrix\n\n                rho = DensityMatrix.from_label('+0')\n\n                # Probabilities for measuring both qubits\n                probs = rho.probabilities([0, 1])\n                print('probs: {}'.format(probs))\n\n                # Probabilities for measuring both qubits\n                # but swapping qubits 0 and 1 in output\n                probs_swapped = rho.probabilities([1, 0])\n                print('Swapped probs: {}'.format(probs_swapped))\n\n            .. parsed-literal::\n\n                probs: [0.5 0.  0.5 0. ]\n                Swapped probs: [0.5 0.5 0.  0. ]\n        "
        probs = self._subsystem_probabilities(np.abs(self.data.diagonal()), self._op_shape.dims_l(), qargs=qargs)
        probs = np.clip(probs, a_min=0, a_max=1)
        if decimals is not None:
            probs = probs.round(decimals=decimals)
        return probs

    def reset(self, qargs: list[int] | None=None) -> DensityMatrix:
        if False:
            return 10
        'Reset state or subsystems to the 0-state.\n\n        Args:\n            qargs (list or None): subsystems to reset, if None all\n                                  subsystems will be reset to their 0-state\n                                  (Default: None).\n\n        Returns:\n            DensityMatrix: the reset state.\n\n        Additional Information:\n            If all subsystems are reset this will return the ground state\n            on all subsystems. If only a some subsystems are reset this\n            function will perform evolution by the reset\n            :class:`~qiskit.quantum_info.SuperOp` of the reset subsystems.\n        '
        if qargs is None:
            ret = copy.copy(self)
            state = np.zeros(self._op_shape.shape, dtype=complex)
            state[0, 0] = 1
            ret._data = state
            return ret
        dims = self.dims(qargs)
        reset_superop = SuperOp(ScalarOp(dims, coeff=0))
        reset_superop.data[0] = Operator(ScalarOp(dims)).data.ravel()
        return self.evolve(reset_superop, qargs=qargs)

    @classmethod
    def from_label(cls, label: str) -> DensityMatrix:
        if False:
            i = 10
            return i + 15
        'Return a tensor product of Pauli X,Y,Z eigenstates.\n\n        .. list-table:: Single-qubit state labels\n           :header-rows: 1\n\n           * - Label\n             - Statevector\n           * - ``"0"``\n             - :math:`\\begin{pmatrix} 1 & 0 \\\\ 0 & 0 \\end{pmatrix}`\n           * - ``"1"``\n             - :math:`\\begin{pmatrix} 0 & 0 \\\\ 0 & 1 \\end{pmatrix}`\n           * - ``"+"``\n             - :math:`\\frac{1}{2}\\begin{pmatrix} 1 & 1 \\\\ 1 & 1 \\end{pmatrix}`\n           * - ``"-"``\n             - :math:`\\frac{1}{2}\\begin{pmatrix} 1 & -1 \\\\ -1 & 1 \\end{pmatrix}`\n           * - ``"r"``\n             - :math:`\\frac{1}{2}\\begin{pmatrix} 1 & -i \\\\ i & 1 \\end{pmatrix}`\n           * - ``"l"``\n             - :math:`\\frac{1}{2}\\begin{pmatrix} 1 & i \\\\ -i & 1 \\end{pmatrix}`\n\n        Args:\n            label (string): a eigenstate string ket label (see table for\n                            allowed values).\n\n        Returns:\n            DensityMatrix: The N-qubit basis state density matrix.\n\n        Raises:\n            QiskitError: if the label contains invalid characters, or the length\n                         of the label is larger than an explicitly specified num_qubits.\n        '
        return DensityMatrix(Statevector.from_label(label))

    @staticmethod
    def from_int(i: int, dims: int | tuple | list) -> DensityMatrix:
        if False:
            for i in range(10):
                print('nop')
        'Return a computational basis state density matrix.\n\n        Args:\n            i (int): the basis state element.\n            dims (int or tuple or list): The subsystem dimensions of the statevector\n                                         (See additional information).\n\n        Returns:\n            DensityMatrix: The computational basis state :math:`|i\\rangle\\!\\langle i|`.\n\n        Additional Information:\n            The ``dims`` kwarg can be an integer or an iterable of integers.\n\n            * ``Iterable`` -- the subsystem dimensions are the values in the list\n              with the total number of subsystems given by the length of the list.\n\n            * ``Int`` -- the integer specifies the total dimension of the\n              state. If it is a power of two the state will be initialized\n              as an N-qubit state. If it is not a power of  two the state\n              will have a single d-dimensional subsystem.\n        '
        size = np.prod(dims)
        state = np.zeros((size, size), dtype=complex)
        state[i, i] = 1.0
        return DensityMatrix(state, dims=dims)

    @classmethod
    def from_instruction(cls, instruction: Instruction | QuantumCircuit) -> DensityMatrix:
        if False:
            for i in range(10):
                print('nop')
        'Return the output density matrix of an instruction.\n\n        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of\n        the same number of qubits as the input instruction or circuit, evolved\n        by the input instruction, and the output statevector returned.\n\n        Args:\n            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit\n\n        Returns:\n            DensityMatrix: the final density matrix.\n\n        Raises:\n            QiskitError: if the instruction contains invalid instructions for\n                         density matrix simulation.\n        '
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        num_qubits = instruction.num_qubits
        init = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
        init[0, 0] = 1
        vec = DensityMatrix(init, dims=num_qubits * (2,))
        vec._append_instruction(instruction)
        return vec

    def to_dict(self, decimals: None | int=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "Convert the density matrix to dictionary form.\n\n        This dictionary representation uses a Ket-like notation where the\n        dictionary keys are qudit strings for the subsystem basis vectors.\n        If any subsystem has a dimension greater than 10 comma delimiters are\n        inserted between integers so that subsystems can be distinguished.\n\n        Args:\n            decimals (None or int): the number of decimal places to round\n                                    values. If None no rounding is done\n                                    (Default: None).\n\n        Returns:\n            dict: the dictionary form of the DensityMatrix.\n\n        Examples:\n\n            The ket-form of a 2-qubit density matrix\n            :math:`rho = |-\\rangle\\!\\langle -|\\otimes |0\\rangle\\!\\langle 0|`\n\n            .. code-block::\n\n                from qiskit.quantum_info import DensityMatrix\n\n                rho = DensityMatrix.from_label('-0')\n                print(rho.to_dict())\n\n            .. parsed-literal::\n\n               {\n                   '00|00': (0.4999999999999999+0j),\n                   '10|00': (-0.4999999999999999-0j),\n                   '00|10': (-0.4999999999999999+0j),\n                   '10|10': (0.4999999999999999+0j)\n               }\n\n            For non-qubit subsystems the integer range can go from 0 to 9. For\n            example in a qutrit system\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit.quantum_info import DensityMatrix\n\n                mat = np.zeros((9, 9))\n                mat[0, 0] = 0.25\n                mat[3, 3] = 0.25\n                mat[6, 6] = 0.25\n                mat[-1, -1] = 0.25\n                rho = DensityMatrix(mat, dims=(3, 3))\n                print(rho.to_dict())\n\n            .. parsed-literal::\n\n                {'00|00': (0.25+0j), '10|10': (0.25+0j), '20|20': (0.25+0j), '22|22': (0.25+0j)}\n\n            For large subsystem dimensions delimiters are required. The\n            following example is for a 20-dimensional system consisting of\n            a qubit and 10-dimensional qudit.\n\n            .. code-block::\n\n                import numpy as np\n                from qiskit.quantum_info import DensityMatrix\n\n                mat = np.zeros((2 * 10, 2 * 10))\n                mat[0, 0] = 0.5\n                mat[-1, -1] = 0.5\n                rho = DensityMatrix(mat, dims=(2, 10))\n                print(rho.to_dict())\n\n            .. parsed-literal::\n\n                {'00|00': (0.5+0j), '91|91': (0.5+0j)}\n        "
        return self._matrix_to_dict(self.data, self._op_shape.dims_l(), decimals=decimals, string_labels=True)

    def _evolve_operator(self, other, qargs=None):
        if False:
            print('Hello World!')
        'Evolve density matrix by an operator'
        new_shape = self._op_shape.compose(other._op_shape, qargs=qargs)
        new_shape._dims_r = new_shape._dims_l
        new_shape._num_qargs_r = new_shape._num_qargs_l
        ret = copy.copy(self)
        if qargs is None:
            op_mat = other.data
            ret._data = np.dot(op_mat, self.data).dot(op_mat.T.conj())
            ret._op_shape = new_shape
            return ret
        tensor = np.reshape(self.data, self._op_shape.tensor_shape)
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        mat = np.reshape(other.data, other._op_shape.tensor_shape)
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        adj = other.adjoint()
        mat_adj = np.reshape(adj.data, adj._op_shape.tensor_shape)
        tensor = Operator._einsum_matmul(tensor, mat_adj, indices, num_indices, True)
        ret._data = np.reshape(tensor, new_shape.shape)
        ret._op_shape = new_shape
        return ret

    def _append_instruction(self, other, qargs=None):
        if False:
            while True:
                i = 10
        'Update the current Statevector by applying an instruction.'
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier
        mat = Operator._instruction_to_matrix(other)
        if mat is not None:
            self._data = self._evolve_operator(Operator(mat), qargs=qargs).data
            return
        if isinstance(other, Reset):
            self._data = self.reset(qargs)._data
            return
        if isinstance(other, Barrier):
            return
        chan = SuperOp._instruction_to_superop(other)
        if chan is not None:
            self._data = chan._evolve(self, qargs=qargs).data
            return
        if other.definition is None:
            raise QiskitError(f'Cannot apply Instruction: {other.name}')
        if not isinstance(other.definition, QuantumCircuit):
            raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(other.name, type(other.definition)))
        qubit_indices = {bit: idx for (idx, bit) in enumerate(other.definition.qubits)}
        for instruction in other.definition:
            if instruction.clbits:
                raise QiskitError(f'Cannot apply instruction with classical bits: {instruction.operation.name}')
            if qargs is None:
                new_qargs = [qubit_indices[tup] for tup in instruction.qubits]
            else:
                new_qargs = [qargs[qubit_indices[tup]] for tup in instruction.qubits]
            self._append_instruction(instruction.operation, qargs=new_qargs)

    def _evolve_instruction(self, obj, qargs=None):
        if False:
            print('Hello World!')
        'Return a new statevector by applying an instruction.'
        if isinstance(obj, QuantumCircuit):
            obj = obj.to_instruction()
        vec = copy.copy(self)
        vec._append_instruction(obj, qargs=qargs)
        return vec

    def to_statevector(self, atol: float | None=None, rtol: float | None=None) -> Statevector:
        if False:
            return 10
        "Return a statevector from a pure density matrix.\n\n        Args:\n            atol (float): Absolute tolerance for checking operation validity.\n            rtol (float): Relative tolerance for checking operation validity.\n\n        Returns:\n            Statevector: The pure density matrix's corresponding statevector.\n                Corresponds to the eigenvector of the only non-zero eigenvalue.\n\n        Raises:\n            QiskitError: if the state is not pure.\n        "
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        if not is_hermitian_matrix(self._data, atol=atol, rtol=rtol):
            raise QiskitError('Not a valid density matrix (non-hermitian).')
        (evals, evecs) = np.linalg.eig(self._data)
        nonzero_evals = evals[abs(evals) > atol]
        if len(nonzero_evals) != 1 or not np.isclose(nonzero_evals[0], 1, atol=atol, rtol=rtol):
            raise QiskitError('Density matrix is not a pure state')
        psi = evecs[:, np.argmax(evals)]
        return Statevector(psi)

    def partial_transpose(self, qargs: list[int]) -> DensityMatrix:
        if False:
            while True:
                i = 10
        'Return partially transposed density matrix.\n\n        Args:\n            qargs (list): The subsystems to be transposed.\n\n        Returns:\n            DensityMatrix: The partially transposed density matrix.\n        '
        arr = self._data.reshape(self._op_shape.tensor_shape)
        qargs = len(self._op_shape.dims_l()) - 1 - np.array(qargs)
        n = len(self.dims())
        lst = list(range(2 * n))
        for i in qargs:
            (lst[i], lst[i + n]) = (lst[i + n], lst[i])
        rho = np.transpose(arr, lst)
        rho = np.reshape(rho, self._op_shape.shape)
        return DensityMatrix(rho, dims=self.dims())