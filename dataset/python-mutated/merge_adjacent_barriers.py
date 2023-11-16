"""Return a circuit with any adjacent barriers merged together."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.barrier import Barrier

class MergeAdjacentBarriers(TransformationPass):
    """Return a circuit with any adjacent barriers merged together.

    Only barriers which can be merged without affecting the barrier structure
    of the DAG will be merged.

    Not all redundant barriers will necessarily be merged, only adjacent
    barriers are merged.

    For example, the circuit::

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])
        circuit.barrier(qr)

    Will be transformed into a circuit corresponding to::

        circuit.barrier(qr[0])
        circuit.barrier(qr)

    i.e,

    .. parsed-literal::
              ░  ░             ░  ░
        q_0: ─░──░─      q_0: ─░──░─
              ░  ░             ░  ░
        q_1: ─░──░─  =>  q_1: ────░─
              ░  ░                ░
        q_2: ────░─      q_2: ────░─
                 ░

    after one iteration of the pass. These two barriers were not merged by the
    first pass as they are not adjacent in the initial circuit.

    The pass then can be reapplied to merge the newly adjacent barriers.
    """

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the MergeAdjacentBarriers pass on `dag`.'
        indices = {qubit: index for (index, qubit) in enumerate(dag.qubits)}
        barriers = [nd for nd in dag.topological_op_nodes() if nd.name == 'barrier']
        node_to_barrier_qubits = MergeAdjacentBarriers._collect_potential_merges(dag, barriers)
        if not node_to_barrier_qubits:
            return dag
        for barrier in barriers:
            if barrier in node_to_barrier_qubits:
                (barrier_to_add, nodes) = node_to_barrier_qubits[barrier]
                dag.replace_block_with_op(nodes, barrier_to_add, wire_pos_map=indices, cycle_check=False)
        return dag

    @staticmethod
    def _collect_potential_merges(dag, barriers):
        if False:
            i = 10
            return i + 15
        'Return the potential merges.\n\n        Returns a dict of DAGOpNode: (Barrier, [DAGOpNode]) objects, where the barrier needs to be\n        inserted where the corresponding index DAGOpNode appears in the main DAG, in replacement of\n        the listed DAGOpNodes.\n        '
        if len(barriers) < 2:
            return None
        node_to_barrier_qubits = {}
        current_barrier = barriers[0]
        end_of_barrier = current_barrier
        current_barrier_nodes = [current_barrier]
        current_qubits = set(current_barrier.qargs)
        current_ancestors = dag.ancestors(current_barrier)
        current_descendants = dag.descendants(current_barrier)
        barrier_to_add = Barrier(len(current_qubits))
        for next_barrier in barriers[1:]:
            if dag._multi_graph.has_edge(end_of_barrier._node_id, next_barrier._node_id):
                next_ancestors = {nd for nd in dag.ancestors(next_barrier) if nd not in current_barrier_nodes}
                next_descendants = {nd for nd in dag.descendants(next_barrier) if nd not in current_barrier_nodes}
                next_qubits = set(next_barrier.qargs)
                if not current_qubits.isdisjoint(next_qubits) and current_ancestors.isdisjoint(next_descendants) and current_descendants.isdisjoint(next_ancestors):
                    current_ancestors = current_ancestors | next_ancestors
                    current_descendants = current_descendants | next_descendants
                    current_qubits = current_qubits | next_qubits
                    barrier_to_add = Barrier(len(current_qubits))
                    end_of_barrier = next_barrier
                    current_barrier_nodes.append(end_of_barrier)
                    continue
            if barrier_to_add:
                node_to_barrier_qubits[end_of_barrier] = (barrier_to_add, current_barrier_nodes)
            current_qubits = set(next_barrier.qargs)
            current_ancestors = dag.ancestors(next_barrier)
            current_descendants = dag.descendants(next_barrier)
            barrier_to_add = Barrier(len(current_qubits))
            current_barrier_nodes = []
            end_of_barrier = next_barrier
            current_barrier_nodes.append(end_of_barrier)
        if barrier_to_add:
            node_to_barrier_qubits[end_of_barrier] = (barrier_to_add, current_barrier_nodes)
        return node_to_barrier_qubits