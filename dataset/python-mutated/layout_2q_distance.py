"""Evaluate how good the layout selection was.

No CX direction is considered.
Saves in `property_set['layout_score']` the sum of distances for each circuit CX.
The lower the number, the better the selection.
Therefore, 0 is a perfect layout selection.
"""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target

class Layout2qDistance(AnalysisPass):
    """Evaluate how good the layout selection was.

    Saves in ``property_set['layout_score']`` (or the property name in property_name)
    the sum of distances for each circuit CX.
    The lower the number, the better the selection. Therefore, 0 is a perfect layout selection.
    No CX direction is considered.
    """

    def __init__(self, coupling_map, property_name='layout_score'):
        if False:
            i = 10
            return i + 15
        'Layout2qDistance initializer.\n\n        Args:\n            coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.\n            property_name (str): The property name to save the score. Default: layout_score\n        '
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.property_name = property_name

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run the Layout2qDistance pass on `dag`.\n        Args:\n            dag (DAGCircuit): DAG to evaluate.\n        '
        layout = self.property_set['layout']
        if layout is None:
            return
        if self.coupling_map is None or len(self.coupling_map.graph) == 0:
            self.property_set[self.property_name] = 0
            return
        self.coupling_map.compute_distance_matrix()
        sum_distance = 0
        virtual_physical_map = layout.get_virtual_bits()
        dist_matrix = self.coupling_map.distance_matrix
        for gate in dag.two_qubit_ops():
            physical_q0 = virtual_physical_map[gate.qargs[0]]
            physical_q1 = virtual_physical_map[gate.qargs[1]]
            sum_distance += dist_matrix[physical_q0, physical_q1] - 1
        self.property_set[self.property_name] = sum_distance