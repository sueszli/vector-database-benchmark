"""Set the ``layout`` property to the given layout."""
from qiskit.transpiler import Layout, TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass

class SetLayout(AnalysisPass):
    """Set the ``layout`` property to the given layout.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.
    """

    def __init__(self, layout):
        if False:
            print('Hello World!')
        'SetLayout initializer.\n\n        Args:\n            layout (Layout or List[int]): the layout to set. It can be:\n\n                * a :class:`Layout` instance: sets that layout.\n                * a list of integers: takes the index in the list as the physical position in which the\n                  virtual qubit is going to be mapped.\n\n        '
        super().__init__()
        self.layout = layout

    def run(self, dag):
        if False:
            while True:
                i = 10
        'Run the SetLayout pass on ``dag``.\n\n        Args:\n            dag (DAGCircuit): DAG to map.\n\n        Returns:\n            DAGCircuit: the original DAG.\n        '
        if isinstance(self.layout, list):
            if len(self.layout) != len(dag.qubits):
                raise TranspilerError(f'The length of the layout is different than the size of the circuit: {len(self.layout)} <> {len(dag.qubits)}')
            layout = Layout({phys: dag.qubits[i] for (i, phys) in enumerate(self.layout)})
        elif isinstance(self.layout, Layout):
            layout = self.layout.copy()
        elif self.layout is None:
            layout = None
        else:
            raise TranspilerError(f'SetLayout was intialized with the layout type: {type(self.layout)}')
        self.property_set['layout'] = layout
        return dag