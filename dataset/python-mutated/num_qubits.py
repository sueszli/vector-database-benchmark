"""Calculate the number of qubits of a DAG circuit."""
from qiskit.transpiler.basepasses import AnalysisPass

class NumQubits(AnalysisPass):
    """Calculate the number of qubits of a DAG circuit.

    The result is saved in ``property_set['num_qubits']`` as an integer.
    """

    def run(self, dag):
        if False:
            while True:
                i = 10
        'Run the NumQubits pass on `dag`.'
        self.property_set['num_qubits'] = dag.num_qubits()