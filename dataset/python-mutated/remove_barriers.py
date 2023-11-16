"""Remove all barriers in a circuit"""
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

class RemoveBarriers(TransformationPass):
    """Return a circuit with any barrier removed.

    This transformation is not semantics preserving.

    Example:

        .. plot::
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import RemoveBarriers

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.barrier()
            circuit.h(0)

            circuit = RemoveBarriers()(circuit)
            circuit.draw('mpl')

    """

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            print('Hello World!')
        'Run the RemoveBarriers pass on `dag`.'
        dag.remove_all_ops_named('barrier')
        return dag