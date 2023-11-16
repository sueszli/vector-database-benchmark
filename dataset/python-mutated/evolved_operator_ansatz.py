"""The evolved operator ansatz."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.synthesis.evolution import LieTrotter
from .pauli_evolution import PauliEvolutionGate
from .n_local.n_local import NLocal

class EvolvedOperatorAnsatz(NLocal):
    """The evolved operator ansatz."""

    def __init__(self, operators=None, reps: int=1, evolution=None, insert_barriers: bool=False, name: str='EvolvedOps', parameter_prefix: str | Sequence[str]='t', initial_state: QuantumCircuit | None=None, flatten: bool | None=None):
        if False:
            while True:
                i = 10
        "\n        Args:\n            operators (BaseOperator | OperatorBase | QuantumCircuit | list | None): The operators\n                to evolve. If a circuit is passed, we assume it implements an already evolved\n                operator and thus the circuit is not evolved again. Can be a single operator\n                (circuit) or a list of operators (and circuits).\n            reps: The number of times to repeat the evolved operators.\n            evolution (EvolutionBase | EvolutionSynthesis | None):\n                A specification of which evolution synthesis to use for the\n                :class:`.PauliEvolutionGate`, if the operator is from :mod:`qiskit.quantum_info`\n                or an opflow converter object if the operator is from :mod:`qiskit.opflow`.\n                Defaults to first order Trotterization.\n            insert_barriers: Whether to insert barriers in between each evolution.\n            name: The name of the circuit.\n            parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix\n                will be used for each parameters. Can also be a list to specify a prefix per\n                operator.\n            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.\n            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple\n                layers of gate objects. By default currently the contents of\n                the output circuit will be wrapped in nested objects for\n                cleaner visualization. However, if you're using this circuit\n                for anything besides visualization its **strongly** recommended\n                to set this flag to ``True`` to avoid a large performance\n                overhead for parameter binding.\n        "
        super().__init__(initial_state=initial_state, parameter_prefix=parameter_prefix, reps=reps, insert_barriers=insert_barriers, name=name, flatten=flatten)
        self._operators = None
        if operators is not None:
            self.operators = operators
        self._evolution = evolution
        self._ops_are_parameterized = None

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if the current configuration is valid.'
        if not super()._check_configuration(raise_on_failure):
            return False
        if self.operators is None:
            if raise_on_failure:
                raise ValueError('The operators are not set.')
            return False
        return True

    @property
    def num_qubits(self) -> int:
        if False:
            return 10
        if self.operators is None:
            return 0
        if isinstance(self.operators, list) and len(self.operators) > 0:
            return self.operators[0].num_qubits
        return self.operators.num_qubits

    @property
    def evolution(self):
        if False:
            i = 10
            return i + 15
        'The evolution converter used to compute the evolution.\n\n        Returns:\n            EvolutionBase or EvolutionSynthesis: The evolution converter used to compute the evolution.\n        '
        if self._evolution is None:
            from qiskit.opflow import PauliTrotterEvolution
            return PauliTrotterEvolution()
        return self._evolution

    @evolution.setter
    def evolution(self, evol) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the evolution converter used to compute the evolution.\n\n        Args:\n            evol (EvolutionBase | EvolutionSynthesis): An evolution synthesis object or\n                opflow converter object to construct the evolution.\n        '
        self._invalidate()
        self._evolution = evol

    @property
    def operators(self):
        if False:
            while True:
                i = 10
        'The operators that are evolved in this circuit.\n\n        Returns:\n            list: The operators to be evolved (and circuits) contained in this ansatz.\n        '
        return self._operators

    @operators.setter
    def operators(self, operators=None) -> None:
        if False:
            while True:
                i = 10
        'Set the operators to be evolved.\n\n        operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.\n            If a circuit is passed, we assume it implements an already evolved operator and thus\n            the circuit is not evolved again. Can be a single operator (circuit) or a list of\n            operators (and circuits).\n        '
        operators = _validate_operators(operators)
        self._invalidate()
        self._operators = operators
        self.qregs = [QuantumRegister(self.num_qubits, name='q')]

    @property
    def preferred_init_points(self):
        if False:
            for i in range(10):
                print('nop')
        'Getter of preferred initial points based on the given initial state.'
        if self._initial_state is None:
            return None
        else:
            self._build()
            return np.zeros(self.reps * len(self.operators), dtype=float)

    def _evolve_operator(self, operator, time):
        if False:
            for i in range(10):
                print('nop')
        from qiskit.opflow import OperatorBase, EvolutionBase
        from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
        if isinstance(operator, OperatorBase):
            if not isinstance(self.evolution, EvolutionBase):
                raise QiskitError(f'If qiskit.opflow operators are evolved the evolution must be a qiskit.opflow.EvolutionBase, not a {type(self.evolution)}.')
            evolved = self.evolution.convert((time * operator).exp_i())
            return evolved.reduce().to_circuit()
        if isinstance(operator, Operator):
            gate = HamiltonianGate(operator, time)
        else:
            evolution = LieTrotter() if self._evolution is None else self._evolution
            gate = PauliEvolutionGate(operator, time, synthesis=evolution)
        evolved = QuantumCircuit(operator.num_qubits)
        if not self.flatten:
            evolved.append(gate, evolved.qubits)
        else:
            evolved.compose(gate.definition, evolved.qubits, inplace=True)
        return evolved

    def _build(self):
        if False:
            return 10
        if self._is_built:
            return
        self._check_configuration()
        coeff = Parameter('c')
        circuits = []
        for op in self.operators:
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
            else:
                if _is_pauli_identity(op):
                    continue
                evolved = self._evolve_operator(op, coeff)
                circuits.append(evolved)
        self.rotation_blocks = []
        self.entanglement_blocks = circuits
        super()._build()

def _validate_operators(operators):
    if False:
        while True:
            i = 10
    if not isinstance(operators, list):
        operators = [operators]
    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any((operators[i].num_qubits != num_qubits for i in range(1, len(operators)))):
            raise ValueError('All operators must act on the same number of qubits.')
    return operators

def _validate_prefix(parameter_prefix, operators):
    if False:
        i = 10
        return i + 15
    if isinstance(parameter_prefix, str):
        return len(operators) * [parameter_prefix]
    if len(parameter_prefix) != len(operators):
        raise ValueError('The number of parameter prefixes must match the operators.')
    return parameter_prefix

def _is_pauli_identity(operator):
    if False:
        return 10
    from qiskit.opflow import PauliOp, PauliSumOp
    if isinstance(operator, PauliSumOp):
        operator = operator.to_pauli_op()
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]
        else:
            return False
    if isinstance(operator, PauliOp):
        operator = operator.primitive
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False