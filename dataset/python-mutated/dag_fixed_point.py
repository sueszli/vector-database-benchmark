"""Check if the DAG has reached a fixed point."""
from copy import deepcopy
from qiskit.transpiler.basepasses import AnalysisPass

class DAGFixedPoint(AnalysisPass):
    """Check if the DAG has reached a fixed point.

    A dummy analysis pass that checks if the DAG a fixed point (the DAG is not
    modified anymore). The result is saved in
    ``property_set['dag_fixed_point']`` as a boolean.
    """

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the DAGFixedPoint pass on `dag`.'
        if self.property_set['_dag_fixed_point_previous_dag'] is None:
            self.property_set['dag_fixed_point'] = False
        else:
            fixed_point_reached = self.property_set['_dag_fixed_point_previous_dag'] == dag
            self.property_set['dag_fixed_point'] = fixed_point_reached
        self.property_set['_dag_fixed_point_previous_dag'] = deepcopy(dag)