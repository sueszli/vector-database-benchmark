"""Exact synthesis of operator evolution via (exponentially expensive) matrix exponentiation."""
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .evolution_synthesis import EvolutionSynthesis

class MatrixExponential(EvolutionSynthesis):
    """Exact operator evolution via matrix exponentiation and unitary synthesis.

    This class synthesis the exponential of operators by calculating their exponentially-sized
    matrix representation and using exact matrix exponentiation followed by unitary synthesis
    to obtain a circuit. This process is not scalable and serves as comparison or benchmark
    for small systems.
    """

    def synthesize(self, evolution):
        if False:
            print('Hello World!')
        from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
        operators = evolution.operator
        time = evolution.time
        if not isinstance(operators, list):
            matrix = operators.to_matrix()
        else:
            matrix = sum((op.to_matrix() for op in operators))
        evolution_circuit = QuantumCircuit(operators[0].num_qubits)
        gate = HamiltonianGate(matrix, time)
        evolution_circuit.append(gate, evolution_circuit.qubits)
        return evolution_circuit