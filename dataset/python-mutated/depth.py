"""Calculate the depth of a DAG circuit."""
from qiskit.transpiler.basepasses import AnalysisPass

class Depth(AnalysisPass):
    """Calculate the depth of a DAG circuit."""

    def __init__(self, *, recurse=False):
        if False:
            while True:
                i = 10
        '\n        Args:\n            recurse: whether to allow recursion into control flow.  If this is ``False`` (default),\n                the pass will throw an error when control flow is present, to avoid returning a\n                number with little meaning.\n        '
        super().__init__()
        self.recurse = recurse

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the Depth pass on `dag`.'
        self.property_set['depth'] = dag.depth(recurse=self.recurse)