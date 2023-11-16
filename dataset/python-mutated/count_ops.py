"""Count the operations in a DAG circuit."""
from qiskit.transpiler.basepasses import AnalysisPass

class CountOps(AnalysisPass):
    """Count the operations in a DAG circuit.

    The result is saved in ``property_set['count_ops']`` as an integer.
    """

    def __init__(self, *, recurse=True):
        if False:
            print('Hello World!')
        super().__init__()
        self.recurse = recurse

    def run(self, dag):
        if False:
            return 10
        'Run the CountOps pass on `dag`.'
        self.property_set['count_ops'] = dag.count_ops(recurse=self.recurse)