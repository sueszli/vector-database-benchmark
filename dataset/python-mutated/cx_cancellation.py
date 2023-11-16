"""Cancel back-to-back ``cx`` gates in dag."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

class CXCancellation(TransformationPass):
    """Cancel back-to-back ``cx`` gates in dag."""

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the CXCancellation pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the directed acyclic graph to run on.\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        cx_runs = dag.collect_runs(['cx'])
        for cx_run in cx_runs:
            partitions = []
            chunk = []
            for i in range(len(cx_run) - 1):
                chunk.append(cx_run[i])
                if cx_run[i].qargs != cx_run[i + 1].qargs:
                    partitions.append(chunk)
                    chunk = []
            chunk.append(cx_run[-1])
            partitions.append(chunk)
            for chunk in partitions:
                if len(chunk) % 2 == 0:
                    dag.remove_op_node(chunk[0])
                for node in chunk[1:]:
                    dag.remove_op_node(node)
        return dag