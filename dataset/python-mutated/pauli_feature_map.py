"""The Pauli expansion circuit module."""
from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal

class PauliFeatureMap(NLocal):
    """The Pauli Expansion circuit.

    The Pauli Expansion circuit is a data encoding circuit that transforms input data
    :math:`\\vec{x} \\in \\mathbb{R}^n`, where `n` is the ``feature_dimension``, as

    .. math::

        U_{\\Phi(\\vec{x})}=\\exp\\left(i\\sum_{S \\in \\mathcal{I}}
        \\phi_S(\\vec{x})\\prod_{i\\in S} P_i\\right).

    Here, :math:`S` is a set of qubit indices that describes the connections in the feature map,
    :math:`\\mathcal{I}` is a set containing all these index sets, and
    :math:`P_i \\in \\{I, X, Y, Z\\}`. Per default the data-mapping
    :math:`\\phi_S` is

    .. math::

        \\phi_S(\\vec{x}) = \\begin{cases}
            x_i \\text{ if } S = \\{i\\} \\\\
            \\prod_{j \\in S} (\\pi - x_j) \\text{ if } |S| > 1
            \\end{cases}.

    The possible connections can be set using the ``entanglement`` and ``paulis`` arguments.
    For example, for single-qubit :math:`Z` rotations and two-qubit :math:`YY` interactions
    between all qubit pairs, we can set::


        feature_map = PauliFeatureMap(..., paulis=["Z", "YY"], entanglement="full")

    which will produce blocks of the form

    .. parsed-literal::

        ┌───┐┌──────────────┐┌──────────┐                                             ┌───────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ RX(pi/2) ├──■───────────────────────────────────────■──┤ RX(-pi/2) ├
        ├───┤├──────────────┤├──────────┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐├───────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ RX(pi/2) ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ RX(-pi/2) ├
        └───┘└──────────────┘└──────────┘└───┘└─────────────────────────────────┘└───┘└───────────┘

    The circuit contains ``reps`` repetitions of this transformation.

    Please refer to :class:`.ZFeatureMap` for the case of single-qubit Pauli-:math:`Z` rotations
    and to :class:`.ZZFeatureMap` for the single- and two-qubit Pauli-:math:`Z` rotations.

    Examples:

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZZ'])
        >>> print(prep)
             ┌───┐
        q_0: ┤ H ├──■───────────────────────────────────────■──
             ├───┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└───┘└─────────────────────────────────┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['Z', 'XX'])
        >>> print(prep)
             ┌───┐┌──────────────┐┌───┐                                             ┌───┐
        q_0: ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├──■───────────────────────────────────────■──┤ H ├
             ├───┤├──────────────┤├───┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐├───┤
        q_1: ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ H ├
             └───┘└──────────────┘└───┘└───┘└─────────────────────────────────┘└───┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZY'])
        >>> print(prep)
             ┌───┐┌──────────┐                                             ┌───────────┐
        q_0: ┤ H ├┤ RX(pi/2) ├──■───────────────────────────────────────■──┤ RX(-pi/2) ├
             ├───┤└──────────┘┌─┴─┐┌─────────────────────────────────┐┌─┴─┐└───────────┘
        q_1: ┤ H ├────────────┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├─────────────
             └───┘            └───┘└─────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import EfficientSU2
        >>> prep = PauliFeatureMap(3, reps=3, paulis=['Z', 'YY', 'ZXZ'])
        >>> wavefunction = EfficientSU2(3)
        >>> classifier = prep.compose(wavefunction
        >>> classifier.num_parameters
        27
        >>> classifier.count_ops()
        OrderedDict([('cx', 39), ('rx', 36), ('u1', 21), ('h', 15), ('ry', 12), ('rz', 12)])

    References:



    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.

    """

    def __init__(self, feature_dimension: Optional[int]=None, reps: int=2, entanglement: Union[str, List[List[int]], Callable[[int], List[int]]]='full', alpha: float=2.0, paulis: Optional[List[str]]=None, data_map_func: Optional[Callable[[np.ndarray], float]]=None, parameter_prefix: str='x', insert_barriers: bool=False, name: str='PauliFeatureMap') -> None:
        if False:
            return 10
        "Create a new Pauli expansion circuit.\n\n        Args:\n            feature_dimension: Number of qubits in the circuit.\n            reps: The number of repeated circuits.\n            entanglement: Specifies the entanglement structure. Refer to\n                :class:`~qiskit.circuit.library.NLocal` for detail.\n            alpha: The Pauli rotation factor, multiplicative to the pauli rotations\n            paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``\n                will be used.\n            data_map_func: A mapping function for data x which can be supplied to override the\n                default mapping from :meth:`self_product`.\n            parameter_prefix: The prefix used if default parameters are generated.\n            insert_barriers: If True, barriers are inserted in between the evolution instructions\n                and hadamard layers.\n\n        "
        super().__init__(num_qubits=feature_dimension, reps=reps, rotation_blocks=HGate(), entanglement=entanglement, parameter_prefix=parameter_prefix, insert_barriers=insert_barriers, skip_final_rotation_layer=True, name=name)
        self._data_map_func = data_map_func or self_product
        self._paulis = paulis or ['Z', 'ZZ']
        self._alpha = alpha

    def _parameter_generator(self, rep: int, block: int, indices: List[int]) -> Optional[List[Parameter]]:
        if False:
            for i in range(10):
                print('nop')
        'If certain blocks should use certain parameters this method can be overridden.'
        params = [self.ordered_parameters[i] for i in indices]
        return params

    @property
    def num_parameters_settable(self):
        if False:
            return 10
        'The number of distinct parameters.'
        return self.feature_dimension

    @property
    def paulis(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'The Pauli strings used in the entanglement of the qubits.\n\n        Returns:\n            The Pauli strings as list.\n        '
        return self._paulis

    @paulis.setter
    def paulis(self, paulis: List[str]) -> None:
        if False:
            print('Hello World!')
        'Set the pauli strings.\n\n        Args:\n            paulis: The new pauli strings.\n        '
        self._invalidate()
        self._paulis = paulis

    @property
    def alpha(self) -> float:
        if False:
            return 10
        'The Pauli rotation factor (alpha).\n\n        Returns:\n            The Pauli rotation factor.\n        '
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if False:
            return 10
        'Set the Pauli rotation factor (alpha).\n\n        Args:\n            alpha: Pauli rotation factor\n        '
        self._invalidate()
        self._alpha = alpha

    @property
    def entanglement_blocks(self):
        if False:
            return 10
        return [self.pauli_block(pauli) for pauli in self._paulis]

    @entanglement_blocks.setter
    def entanglement_blocks(self, entanglement_blocks):
        if False:
            return 10
        self._entanglement_blocks = entanglement_blocks

    @property
    def feature_dimension(self) -> int:
        if False:
            print('Hello World!')
        'Returns the feature dimension (which is equal to the number of qubits).\n\n        Returns:\n            The feature dimension of this feature map.\n        '
        return self.num_qubits

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        if False:
            while True:
                i = 10
        'Set the feature dimension.\n\n        Args:\n            feature_dimension: The new feature dimension.\n        '
        self.num_qubits = feature_dimension

    def _extract_data_for_rotation(self, pauli, x):
        if False:
            i = 10
            return i + 15
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]

    def pauli_block(self, pauli_string):
        if False:
            for i in range(10):
                print('nop')
        'Get the Pauli block for the feature map circuit.'
        params = ParameterVector('_', length=len(pauli_string))
        time = self._data_map_func(np.asarray(params))
        return self.pauli_evolution(pauli_string, time)

    def pauli_evolution(self, pauli_string, time):
        if False:
            i = 10
            return i + 15
        'Get the evolution block for the given pauli string.'
        pauli_string = pauli_string[::-1]
        trimmed = []
        indices = []
        for (i, pauli) in enumerate(pauli_string):
            if pauli != 'I':
                trimmed += [pauli]
                indices += [i]
        evo = QuantumCircuit(len(pauli_string))
        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            if False:
                print('Hello World!')
            for (i, pauli) in enumerate(pauli_string):
                if pauli == 'X':
                    circuit.h(i)
                elif pauli == 'Y':
                    circuit.rx(-np.pi / 2 if inverse else np.pi / 2, i)

        def cx_chain(circuit, inverse=False):
            if False:
                return 10
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])
        basis_change(evo)
        cx_chain(evo)
        evo.p(self.alpha * time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo

def self_product(x: np.ndarray) -> float:
    if False:
        while True:
            i = 10
    '\n    Define a function map from R^n to R.\n\n    Args:\n        x: data\n\n    Returns:\n        float: the mapped value\n    '
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
    return coeff