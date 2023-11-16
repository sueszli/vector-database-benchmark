"""Collect sequences of uninterrupted gates acting on 2 qubits."""
from collections import defaultdict
from qiskit.transpiler.basepasses import AnalysisPass

class Collect2qBlocks(AnalysisPass):
    """Collect two-qubit subcircuits."""

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the Collect2qBlocks pass on `dag`.\n\n        The blocks contain "op" nodes in topological order such that all gates\n        in a block act on the same qubits and are adjacent in the circuit.\n\n        After the execution, ``property_set[\'block_list\']`` is set to a list of\n        tuples of "op" node.\n        '
        self.property_set['commutation_set'] = defaultdict(list)
        self.property_set['block_list'] = dag.collect_2q_runs()
        return dag