"""Collect sequences of uninterrupted gates acting on 1 qubit."""
from qiskit.transpiler.basepasses import AnalysisPass

class Collect1qRuns(AnalysisPass):
    """Collect one-qubit subcircuits."""

    def run(self, dag):
        if False:
            return 10
        'Run the Collect1qBlocks pass on `dag`.\n\n        The blocks contain "op" nodes in topological order such that all gates\n        in a block act on the same qubits and are adjacent in the circuit.\n\n        After the execution, ``property_set[\'run_list\']`` is set to a list of\n        tuples of "op" node.\n        '
        self.property_set['run_list'] = dag.collect_1q_runs()
        return dag