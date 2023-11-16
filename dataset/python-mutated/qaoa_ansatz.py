"""A generalized QAOA quantum circuit with a support of custom initial states and mixers."""
from __future__ import annotations
import numpy as np
from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp

class QAOAAnsatz(EvolvedOperatorAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(self, cost_operator=None, reps: int=1, initial_state: QuantumCircuit | None=None, mixer_operator=None, name: str='QAOA', flatten: bool | None=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            cost_operator (BaseOperator or OperatorBase, optional): The operator\n                representing the cost of the optimization problem, denoted as :math:`U(C, \\gamma)`\n                in the original paper. Must be set either in the constructor or via property setter.\n            reps (int): The integer parameter p, which determines the depth of the circuit,\n                as specified in the original paper, default is 1.\n            initial_state (QuantumCircuit, optional): An optional initial state to use.\n                If `None` is passed then a set of Hadamard gates is applied as an initial state\n                to all qubits.\n            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): An optional\n                custom mixer to use instead of the global X-rotations, denoted as :math:`U(B, \\beta)`\n                in the original paper. Can be an operator or an optionally parameterized quantum\n                circuit.\n            name (str): A name of the circuit, default 'qaoa'\n            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple\n                layers of gate objects. By default currently the contents of\n                the output circuit will be wrapped in nested objects for\n                cleaner visualization. However, if you're using this circuit\n                for anything besides visualization its **strongly** recommended\n                to set this flag to ``True`` to avoid a large performance\n                overhead for parameter binding.\n        "
        super().__init__(reps=reps, name=name, flatten=flatten)
        self._cost_operator = None
        self._reps = reps
        self._initial_state: QuantumCircuit | None = initial_state
        self._mixer = mixer_operator
        self._bounds: list[tuple[float | None, float | None]] | None = None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            while True:
                i = 10
        'Check if the current configuration is valid.'
        valid = True
        if not super()._check_configuration(raise_on_failure):
            return False
        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError('The operator representing the cost of the optimization problem is not set')
        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits of the initial state {} does not match the number of qubits of the cost operator {}'.format(self.initial_state.num_qubits, self.num_qubits))
        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits of the mixer {} does not match the number of qubits of the cost operator {}'.format(self.mixer_operator.num_qubits, self.num_qubits))
        return valid

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        if False:
            while True:
                i = 10
        'The parameter bounds for the unbound parameters in the circuit.\n\n        Returns:\n            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded\n            parameter in the corresponding direction. If None is returned, problem is fully\n            unbounded.\n        '
        if self._bounds is not None:
            return self._bounds
        if isinstance(self.mixer_operator, QuantumCircuit):
            return None
        beta_bounds = (0, 2 * np.pi)
        gamma_bounds = (None, None)
        bounds: list[tuple[float | None, float | None]] = []
        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]
        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]
        return bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float | None, float | None]] | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the parameter bounds.\n\n        Args:\n            bounds: The new parameter bounds.\n        '
        self._bounds = bounds

    @property
    def operators(self) -> list:
        if False:
            return 10
        'The operators that are evolved in this circuit.\n\n        Returns:\n             List[Union[BaseOperator, OperatorBase, QuantumCircuit]]: The operators to be evolved\n                (and circuits) in this ansatz.\n        '
        return [self.cost_operator, self.mixer_operator]

    @property
    def cost_operator(self):
        if False:
            print('Hello World!')
        'Returns an operator representing the cost of the optimization problem.\n\n        Returns:\n            BaseOperator or OperatorBase: cost operator.\n        '
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        if False:
            while True:
                i = 10
        'Sets cost operator.\n\n        Args:\n            cost_operator (BaseOperator or OperatorBase, optional): cost operator to set.\n        '
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name='q')]
        self._invalidate()

    @property
    def reps(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the `reps` parameter, which determines the depth of the circuit.'
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        if False:
            print('Hello World!')
        'Sets the `reps` parameter.'
        self._reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> QuantumCircuit | None:
        if False:
            i = 10
            return i + 15
        'Returns an optional initial state as a circuit'
        if self._initial_state is not None:
            return self._initial_state
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state
        return None

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        if False:
            return 10
        'Sets initial state.'
        self._initial_state = initial_state
        self._invalidate()

    @property
    def mixer_operator(self):
        if False:
            i = 10
            return i + 15
        'Returns an optional mixer operator expressed as an operator or a quantum circuit.\n\n        Returns:\n            BaseOperator or OperatorBase or QuantumCircuit, optional: mixer operator or circuit.\n        '
        if self._mixer is not None:
            return self._mixer
        if self.cost_operator is not None:
            num_qubits = self.cost_operator.num_qubits
            mixer_terms = [('I' * left + 'X' + 'I' * (num_qubits - left - 1), 1) for left in range(num_qubits)]
            mixer = SparsePauliOp.from_list(mixer_terms)
            return mixer
        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        if False:
            print('Hello World!')
        'Sets mixer operator.\n\n        Args:\n            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): mixer\n                operator or circuit to set.\n        '
        self._mixer = mixer_operator
        self._invalidate()

    @property
    def num_qubits(self) -> int:
        if False:
            return 10
        if self._cost_operator is None:
            return 0
        return self._cost_operator.num_qubits

    def _build(self):
        if False:
            i = 10
            return i + 15
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 0 if _is_pauli_identity(self.mixer_operator) else 1
        betas = ParameterVector('β', self.reps * num_mixer)
        gammas = ParameterVector('γ', self.reps * num_cost)
        reordered = []
        for rep in range(self.reps):
            reordered.extend(gammas[rep * num_cost:(rep + 1) * num_cost])
            reordered.extend(betas[rep * num_mixer:(rep + 1) * num_mixer])
        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)