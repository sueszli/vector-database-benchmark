"""A product formula base for decomposing non-commuting operator exponentials."""
from typing import Callable, Optional, Union, Any, Dict
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from .evolution_synthesis import EvolutionSynthesis

class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    :obj:`.LieTrotter` and :obj:`.SuzukiTrotter` inherit from this class.
    """

    def __init__(self, order: int, reps: int=1, insert_barriers: bool=False, cx_structure: str='chain', atomic_evolution: Optional[Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            order: The order of the product formula.\n            reps: The number of time steps.\n            insert_barriers: Whether to insert barriers between the atomic evolutions.\n            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be\n                "chain", where next neighbor connections are used, or "fountain", where all\n                qubits are connected to one.\n            atomic_evolution: A function to construct the circuit for the evolution of single\n                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain\n                and a single qubit Z rotation.\n        '
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers
        self._atomic_evolution = atomic_evolution
        self._cx_structure = cx_structure
        if atomic_evolution is None:
            atomic_evolution = partial(_default_atomic_evolution, cx_structure=cx_structure)
        self.atomic_evolution = atomic_evolution

    @property
    def settings(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return the settings in a dictionary, which can be used to reconstruct the object.\n\n        Returns:\n            A dictionary containing the settings of this product formula.\n\n        Raises:\n            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.\n        '
        if self._atomic_evolution is not None:
            raise NotImplementedError('Cannot serialize a product formula with a custom atomic evolution.')
        return {'order': self.order, 'reps': self.reps, 'insert_barriers': self.insert_barriers, 'cx_structure': self._cx_structure}

def evolve_pauli(pauli: Pauli, time: Union[float, ParameterExpression]=1.0, cx_structure: str='chain', label: Optional[str]=None) -> QuantumCircuit:
    if False:
        while True:
            i = 10
    'Construct a circuit implementing the time evolution of a single Pauli string.\n\n    For a Pauli string :math:`P = \\{I, X, Y, Z\\}^{\\otimes n}` on :math:`n` qubits and an\n    evolution time :math:`t`, the returned circuit implements the unitary operation\n\n    .. math::\n\n        U(t) = e^{-itP}.\n\n    Since only a single Pauli string is evolved the circuit decomposition is exact.\n\n    Args:\n        pauli: The Pauli to evolve.\n        time: The evolution time.\n        cx_structure: Determine the structure of CX gates, can be either "chain" for\n            next-neighbor connections or "fountain" to connect directly to the top qubit.\n        label: A label for the gate.\n\n    Returns:\n        A quantum circuit implementing the time evolution of the Pauli.\n    '
    num_non_identity = len([label for label in pauli.to_label() if label != 'I'])
    if num_non_identity == 0:
        definition = QuantumCircuit(pauli.num_qubits, global_phase=-time)
    elif num_non_identity == 1:
        definition = _single_qubit_evolution(pauli, time)
    elif num_non_identity == 2:
        definition = _two_qubit_evolution(pauli, time, cx_structure)
    else:
        definition = _multi_qubit_evolution(pauli, time, cx_structure)
    definition.name = f'exp(it {pauli.to_label()})'
    return definition

def _single_qubit_evolution(pauli, time):
    if False:
        return 10
    definition = QuantumCircuit(pauli.num_qubits)
    for (i, pauli_i) in enumerate(reversed(pauli.to_label())):
        if pauli_i == 'X':
            definition.rx(2 * time, i)
        elif pauli_i == 'Y':
            definition.ry(2 * time, i)
        elif pauli_i == 'Z':
            definition.rz(2 * time, i)
    return definition

def _two_qubit_evolution(pauli, time, cx_structure):
    if False:
        for i in range(10):
            print('nop')
    labels_as_array = np.array(list(reversed(pauli.to_label())))
    qubits = np.where(labels_as_array != 'I')[0]
    labels = np.array([labels_as_array[idx] for idx in qubits])
    definition = QuantumCircuit(pauli.num_qubits)
    if all(labels == 'X'):
        definition.rxx(2 * time, qubits[0], qubits[1])
    elif all(labels == 'Y'):
        definition.ryy(2 * time, qubits[0], qubits[1])
    elif all(labels == 'Z'):
        definition.rzz(2 * time, qubits[0], qubits[1])
    elif labels[0] == 'Z' and labels[1] == 'X':
        definition.rzx(2 * time, qubits[0], qubits[1])
    elif labels[0] == 'X' and labels[1] == 'Z':
        definition.rzx(2 * time, qubits[1], qubits[0])
    else:
        definition = _multi_qubit_evolution(pauli, time, cx_structure)
    return definition

def _multi_qubit_evolution(pauli, time, cx_structure):
    if False:
        for i in range(10):
            print('nop')
    cliff = diagonalizing_clifford(pauli)
    if cx_structure == 'chain':
        chain = cnot_chain(pauli)
    else:
        chain = cnot_fountain(pauli)
    target = None
    for (i, pauli_i) in enumerate(reversed(pauli.to_label())):
        if pauli_i != 'I':
            target = i
            break
    definition = QuantumCircuit(pauli.num_qubits)
    definition.compose(cliff, inplace=True)
    definition.compose(chain, inplace=True)
    definition.rz(2 * time, target)
    definition.compose(chain.inverse(), inplace=True)
    definition.compose(cliff.inverse(), inplace=True)
    return definition

def diagonalizing_clifford(pauli: Pauli) -> QuantumCircuit:
    if False:
        while True:
            i = 10
    'Get the clifford circuit to diagonalize the Pauli operator.\n\n    Args:\n        pauli: The Pauli to diagonalize.\n\n    Returns:\n        A circuit to diagonalize.\n    '
    cliff = QuantumCircuit(pauli.num_qubits)
    for (i, pauli_i) in enumerate(reversed(pauli.to_label())):
        if pauli_i == 'Y':
            cliff.sdg(i)
        if pauli_i in ['X', 'Y']:
            cliff.h(i)
    return cliff

def cnot_chain(pauli: Pauli) -> QuantumCircuit:
    if False:
        print('Hello World!')
    "CX chain.\n\n    For example, for the Pauli with the label 'XYZIX'.\n\n    .. parsed-literal::\n\n                       ┌───┐\n        q_0: ──────────┤ X ├\n                       └─┬─┘\n        q_1: ────────────┼──\n                  ┌───┐  │\n        q_2: ─────┤ X ├──■──\n             ┌───┐└─┬─┘\n        q_3: ┤ X ├──■───────\n             └─┬─┘\n        q_4: ──■────────────\n\n    Args:\n        pauli: The Pauli for which to construct the CX chain.\n\n    Returns:\n        A circuit implementing the CX chain.\n    "
    chain = QuantumCircuit(pauli.num_qubits)
    (control, target) = (None, None)
    for (i, pauli_i) in enumerate(pauli.to_label()):
        i = pauli.num_qubits - i - 1
        if pauli_i != 'I':
            if control is None:
                control = i
            else:
                target = i
        if control is not None and target is not None:
            chain.cx(control, target)
            control = i
            target = None
    return chain

def cnot_fountain(pauli: Pauli) -> QuantumCircuit:
    if False:
        print('Hello World!')
    "CX chain in the fountain shape.\n\n    For example, for the Pauli with the label 'XYZIX'.\n\n    .. parsed-literal::\n\n             ┌───┐┌───┐┌───┐\n        q_0: ┤ X ├┤ X ├┤ X ├\n             └─┬─┘└─┬─┘└─┬─┘\n        q_1: ──┼────┼────┼──\n               │    │    │\n        q_2: ──■────┼────┼──\n                    │    │\n        q_3: ───────■────┼──\n                         │\n        q_4: ────────────■──\n\n    Args:\n        pauli: The Pauli for which to construct the CX chain.\n\n    Returns:\n        A circuit implementing the CX chain.\n    "
    chain = QuantumCircuit(pauli.num_qubits)
    (control, target) = (None, None)
    for (i, pauli_i) in enumerate(reversed(pauli.to_label())):
        if pauli_i != 'I':
            if target is None:
                target = i
            else:
                control = i
        if control is not None and target is not None:
            chain.cx(control, target)
            control = None
    return chain

def _default_atomic_evolution(operator, time, cx_structure):
    if False:
        i = 10
        return i + 15
    if isinstance(operator, Pauli):
        evolution_circuit = evolve_pauli(operator, time, cx_structure)
    else:
        pauli_list = [(Pauli(op), np.real(coeff)) for (op, coeff) in operator.to_list()]
        name = f'exp(it {[pauli.to_label() for (pauli, _) in pauli_list]})'
        evolution_circuit = QuantumCircuit(operator.num_qubits, name=name)
        for (pauli, coeff) in pauli_list:
            evolution_circuit.compose(evolve_pauli(pauli, coeff * time, cx_structure), inplace=True)
    return evolution_circuit