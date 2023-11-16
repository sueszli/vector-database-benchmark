"""Remove final measurements and barriers at the end of a circuit."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGOpNode

class RemoveFinalMeasurements(TransformationPass):
    """Remove final measurements and barriers at the end of a circuit.

    This pass removes final barriers and final measurements, as well as all
    unused classical registers and bits they are connected to.
    Measurements and barriers are considered final if they are
    followed by no other operations (aside from other measurements or barriers.)

    Classical registers are removed iff they reference at least one bit
    that has become unused by the circuit as a result of the operation, and all
    of their other bits are also unused. Separately, classical bits are removed
    iff they have become unused by the circuit as a result of the operation,
    or they appear in a removed classical register, but do not appear
    in a classical register that will remain.
    """

    def _calc_final_ops(self, dag):
        if False:
            for i in range(10):
                print('nop')
        final_op_types = {'measure', 'barrier'}
        final_ops = []
        to_visit = [next(dag.predecessors(dag.output_map[qubit])) for qubit in dag.qubits]
        barrier_encounters_remaining = {}
        while to_visit:
            node = to_visit.pop()
            if not isinstance(node, DAGOpNode):
                continue
            if node.op.name == 'barrier':
                if node not in barrier_encounters_remaining:
                    barrier_encounters_remaining[node] = sum((1 for _ in dag.quantum_successors(node)))
                if barrier_encounters_remaining[node] - 1 > 0:
                    barrier_encounters_remaining[node] -= 1
                    continue
            if node.name in final_op_types:
                final_ops.append(node)
                to_visit.extend(dag.quantum_predecessors(node))
        return final_ops

    def run(self, dag):
        if False:
            while True:
                i = 10
        'Run the RemoveFinalMeasurements pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        final_ops = self._calc_final_ops(dag)
        if not final_ops:
            return dag
        clbits_with_final_measures = set()
        for node in final_ops:
            for carg in node.cargs:
                clbits_with_final_measures.add(carg)
            dag.remove_op_node(node)
        idle_wires = set(dag.idle_wires())
        clbits_with_final_measures &= idle_wires
        if not clbits_with_final_measures:
            return dag
        idle_register_bits = set()
        busy_register_bits = set()
        for creg in dag.cregs.values():
            clbits = set(creg)
            if not clbits.isdisjoint(clbits_with_final_measures) and clbits.issubset(idle_wires):
                idle_register_bits |= clbits
            else:
                busy_register_bits |= clbits
        bits_to_remove = (clbits_with_final_measures | idle_register_bits) - busy_register_bits
        dag.remove_clbits(*bits_to_remove)
        return dag