"""QDrift Class"""
from typing import Union, Optional, Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli
from .product_formula import ProductFormula
from .lie_trotter import LieTrotter

class QDrift(ProductFormula):
    """The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in Ref. [1].

    References:
        [1]: E. Campbell, "A random compiler for fast Hamiltonian simulation" (2018).
        `arXiv:quant-ph/1811.08017 <https://arxiv.org/abs/1811.08017>`_
    """

    def __init__(self, reps: int=1, insert_barriers: bool=False, cx_structure: str='chain', atomic_evolution: Optional[Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]]=None, seed: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            reps: The number of times to repeat the Trotterization circuit.\n            insert_barriers: Whether to insert barriers between the atomic evolutions.\n            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be\n                "chain", where next neighbor connections are used, or "fountain", where all\n                qubits are connected to one.\n            atomic_evolution: A function to construct the circuit for the evolution of single\n                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain\n                and a single qubit Z rotation.\n            seed: An optional seed for reproducibility of the random sampling process.\n        '
        super().__init__(1, reps, insert_barriers, cx_structure, atomic_evolution)
        self.sampled_ops = None
        self.rng = np.random.default_rng(seed)

    def synthesize(self, evolution):
        if False:
            while True:
                i = 10
        operators = evolution.operator
        time = evolution.time
        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), coeff) for (op, coeff) in operators.to_list()]
            coeffs = [np.real(coeff) for (op, coeff) in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]
            coeffs = [1 for op in operators]
        weights = np.abs(coeffs)
        lambd = np.sum(weights)
        num_gates = int(np.ceil(2 * lambd ** 2 * time ** 2 * self.reps))
        evolution_time = lambd * time / num_gates
        self.sampled_ops = self.rng.choice(np.array(pauli_list, dtype=object), size=(num_gates,), p=weights / lambd)
        self.sampled_ops = [(op, evolution_time) for (op, coeff) in self.sampled_ops]
        from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
        lie_trotter = LieTrotter(insert_barriers=self.insert_barriers, atomic_evolution=self.atomic_evolution)
        evolution_circuit = PauliEvolutionGate(sum((SparsePauliOp(op) for (op, coeff) in self.sampled_ops)), time=evolution_time, synthesis=lie_trotter).definition
        return evolution_circuit