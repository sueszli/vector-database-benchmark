"""Return the longest path in a :class:`.DAGCircuit` as a list of DAGNodes."""
from qiskit.transpiler.basepasses import AnalysisPass

class DAGLongestPath(AnalysisPass):
    """Return the longest path in a :class:`.DAGCircuit` as a list of
    :class:`.DAGOpNode`\\ s, :class:`.DAGInNode`\\ s, and :class:`.DAGOutNode`\\ s."""

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the DAGLongestPath pass on `dag`.'
        self.property_set['dag_longest_path'] = dag.longest_path()