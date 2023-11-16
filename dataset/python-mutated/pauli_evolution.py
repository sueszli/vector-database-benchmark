"""A gate to implement time-evolution of operators."""
from __future__ import annotations
from typing import Union, Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp

class PauliEvolutionGate(Gate):
    """Time-evolution of an operator consisting of Paulis.

    For an operator :math:`H` consisting of Pauli terms and (real) evolution time :math:`t`
    this gate implements

    .. math::

        U(t) = e^{-itH}.

    This gate serves as a high-level definition of the evolution and can be synthesized into
    a circuit using different algorithms.

    The evolution gates are related to the Pauli rotation gates by a factor of 2. For example
    the time evolution of the Pauli :math:`X` operator is connected to the Pauli :math:`X` rotation
    :math:`R_X` by

    .. math::

        U(t) = e^{-itX} = R_X(2t).

    **Examples:**

    .. code-block:: python

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.opflow import I, Z, X

        # build the evolution gate
        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=0.2)

        # plug it into a circuit
        circuit = QuantumCircuit(2)
        circuit.append(evo, range(2))
        print(circuit.draw())

    The above will print (note that the ``-0.1`` coefficient is not printed!)::

             ┌──────────────────────────┐
        q_0: ┤0                         ├
             │  exp(-it (ZZ + XI))(0.2) │
        q_1: ┤1                         ├
             └──────────────────────────┘


    **References:**

    [1] G. Li et al. Paulihedral: A Generalized Block-Wise Compiler Optimization
    Framework For Quantum Simulation Kernels (2021).
    [`arXiv:2109.03371 <https://arxiv.org/abs/2109.03371>`_]
    """

    def __init__(self, operator, time: Union[int, float, ParameterExpression]=1.0, label: Optional[str]=None, synthesis: Optional[EvolutionSynthesis]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            operator (Pauli | PauliOp | SparsePauliOp | PauliSumOp | list):\n                The operator to evolve. Can also be provided as list of non-commuting\n                operators where the elements are sums of commuting operators.\n                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.\n            time: The evolution time.\n            label: A label for the gate to display in visualizations. Per default, the label is\n                set to ``exp(-it <operators>)`` where ``<operators>`` is the sum of the Paulis.\n                Note that the label does not include any coefficients of the Paulis. See the\n                class docstring for an example.\n            synthesis: A synthesis strategy. If None, the default synthesis is the Lie-Trotter\n                product formula with a single repetition.\n        '
        if isinstance(operator, list):
            operator = [_to_sparse_pauli_op(op) for op in operator]
        else:
            operator = _to_sparse_pauli_op(operator)
        if synthesis is None:
            synthesis = LieTrotter()
        if label is None:
            label = _get_default_label(operator)
        num_qubits = operator[0].num_qubits if isinstance(operator, list) else operator.num_qubits
        super().__init__(name='PauliEvolution', num_qubits=num_qubits, params=[time], label=label)
        self.operator = operator
        self.synthesis = synthesis

    @property
    def time(self) -> Union[float, ParameterExpression]:
        if False:
            print('Hello World!')
        'Return the evolution time as stored in the gate parameters.\n\n        Returns:\n            The evolution time.\n        '
        return self.params[0]

    @time.setter
    def time(self, time: Union[float, ParameterExpression]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the evolution time.\n\n        Args:\n            time: The evolution time.\n        '
        self.params = [time]

    def _define(self):
        if False:
            while True:
                i = 10
        'Unroll, where the default synthesis is matrix based.'
        self.definition = self.synthesis.synthesize(self)

    def validate_parameter(self, parameter: Union[int, float, ParameterExpression]) -> Union[float, ParameterExpression]:
        if False:
            return 10
        'Gate parameters should be int, float, or ParameterExpression'
        if isinstance(parameter, int):
            parameter = float(parameter)
        return super().validate_parameter(parameter)

def _to_sparse_pauli_op(operator):
    if False:
        while True:
            i = 10
    'Cast the operator to a SparsePauliOp.\n\n    For Opflow objects, return a global coefficient that must be multiplied to the evolution time.\n    Since this coefficient might contain unbound parameters it cannot be absorbed into the\n    coefficients of the SparsePauliOp.\n    '
    from qiskit.opflow import PauliSumOp, PauliOp
    if isinstance(operator, PauliSumOp):
        sparse_pauli = operator.primitive
        sparse_pauli._coeffs *= operator.coeff
    elif isinstance(operator, PauliOp):
        sparse_pauli = SparsePauliOp(operator.primitive)
        sparse_pauli._coeffs *= operator.coeff
    elif isinstance(operator, Pauli):
        sparse_pauli = SparsePauliOp(operator)
    elif isinstance(operator, SparsePauliOp):
        sparse_pauli = operator
    else:
        raise ValueError(f'Unsupported operator type for evolution: {type(operator)}.')
    if any(np.iscomplex(sparse_pauli.coeffs)):
        raise ValueError('Operator contains complex coefficients, which are not supported.')
    if any((isinstance(coeff, ParameterExpression) for coeff in sparse_pauli.coeffs)):
        raise ValueError('Operator contains ParameterExpression, which are not supported.')
    return sparse_pauli

def _get_default_label(operator):
    if False:
        return 10
    if isinstance(operator, list):
        label = f"exp(-it ({[' + '.join(op.paulis.to_labels()) for op in operator]}))"
    elif len(operator.paulis) == 1:
        label = f'exp(-it {operator.paulis.to_labels()[0]})'
    else:
        label = f"exp(-it ({' + '.join(operator.paulis.to_labels())}))"
    return label