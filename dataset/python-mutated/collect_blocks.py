"""Various ways to divide a DAG into blocks of nodes, to split blocks of nodes
into smaller sub-blocks, and to consolidate blocks."""
from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError

class BlockCollector:
    """This class implements various strategies of dividing a DAG (direct acyclic graph)
    into blocks of nodes that satisfy certain criteria. It works both with the
    :class:`~qiskit.dagcircuit.DAGCircuit` and
    :class:`~qiskit.dagcircuit.DAGDependency` representations of a DAG, where
    DagDependency takes into account commutativity between nodes.

    Collecting nodes from DAGDependency generally leads to more optimal results, but is
    slower, as it requires to construct a DAGDependency beforehand. Thus, DAGCircuit should
    be used with lower transpiler settings, and DAGDependency should be used with higher
    transpiler settings.

    In general, there are multiple ways to collect maximal blocks. The approaches used
    here are of the form 'starting from the input nodes of a DAG, greedily collect
    the largest block of nodes that match certain criteria'. For additional details,
    see https://github.com/Qiskit/qiskit-terra/issues/5775.
    """

    def __init__(self, dag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            dag (Union[DAGCircuit, DAGDependency]): The input DAG.\n\n        Raises:\n            DAGCircuitError: the input object is not a DAG.\n        '
        self.dag = dag
        self._pending_nodes = None
        self._in_degree = None
        self._collect_from_back = False
        if isinstance(dag, DAGCircuit):
            self.is_dag_dependency = False
        elif isinstance(dag, DAGDependency):
            self.is_dag_dependency = True
        else:
            raise DAGCircuitError('not a DAG.')

    def _setup_in_degrees(self):
        if False:
            for i in range(10):
                print('nop')
        'For an efficient implementation, for every node we keep the number of its\n        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``\n        is set up at the start and updated throughout the algorithm.\n        A node is leaf (or input) node iff its ``_in_degree`` is 0.\n        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate\n        successor is updated by subtracting 1.\n        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose\n        ``_in_degree`` is 0.\n        '
        self._pending_nodes = []
        self._in_degree = {}
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self):
        if False:
            i = 10
            return i + 15
        'Returns DAG nodes.'
        if not self.is_dag_dependency:
            return self.dag.op_nodes()
        else:
            return self.dag.get_nodes()

    def _direct_preds(self, node):
        if False:
            print('Hello World!')
        "Returns direct predecessors of a node. This function takes into account the\n        direction of collecting blocks, that is node's predecessors when collecting\n        backwards are the direct successors of a node in the DAG.\n        "
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [pred for pred in self.dag.successors(node) if isinstance(pred, DAGOpNode)]
            else:
                return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        elif self._collect_from_back:
            return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_successors(node.node_id)]
        else:
            return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_predecessors(node.node_id)]

    def _direct_succs(self, node):
        if False:
            i = 10
            return i + 15
        "Returns direct successors of a node. This function takes into account the\n        direction of collecting blocks, that is node's successors when collecting\n        backwards are the direct predecessors of a node in the DAG.\n        "
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [succ for succ in self.dag.predecessors(node) if isinstance(succ, DAGOpNode)]
            else:
                return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
        elif self._collect_from_back:
            return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_predecessors(node.node_id)]
        else:
            return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_successors(node.node_id)]

    def _have_uncollected_nodes(self):
        if False:
            print('Hello World!')
        'Returns whether there are uncollected (pending) nodes'
        return len(self._pending_nodes) > 0

    def collect_matching_block(self, filter_fn):
        if False:
            i = 10
            return i + 15
        "Iteratively collects the largest block of input nodes (that is, nodes with\n        ``_in_degree`` equal to 0) that match a given filtering function.\n        Examples of this include collecting blocks of swap gates,\n        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,\n        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,\n        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes\n        to become input and to be eligible for collecting into the current block.\n        Returns the block of collected nodes.\n        "
        current_block = []
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                if filter_fn(node):
                    current_block.append(node)
                    for suc in self._direct_succs(node):
                        self._in_degree[suc] -= 1
                        if self._in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self._pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes
        return current_block

    def collect_all_matching_blocks(self, filter_fn, split_blocks=True, min_block_size=2, split_layers=False, collect_from_back=False):
        if False:
            while True:
                i = 10
        'Collects all blocks that match a given filtering function filter_fn.\n        This iteratively finds the largest block that does not match filter_fn,\n        then the largest block that matches filter_fn, and so on, until no more uncollected\n        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to\n        find larger blocks of matching nodes later on.\n\n        After the blocks are collected, they can be optionally refined. The option\n        ``split_blocks`` allows to split collected blocks into sub-blocks over disjoint\n        qubit subsets. The option ``split_layers`` allows to split collected blocks\n        into layers of non-overlapping instructions. The option ``min_block_size``\n        specifies the minimum number of gates in the block for the block to be collected.\n\n        By default, blocks are collected in the direction from the inputs towards the outputs\n        of the circuit. The option ``collect_from_back`` allows to change this direction,\n        that is collect blocks from the outputs towards the inputs of the circuit.\n\n        Returns the list of matching blocks only.\n        '

        def not_filter_fn(node):
            if False:
                return 10
            'Returns the opposite of filter_fn.'
            return not filter_fn(node)
        self._collect_from_back = collect_from_back
        self._setup_in_degrees()
        matching_blocks = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(not_filter_fn)
            matching_block = self.collect_matching_block(filter_fn)
            if matching_block:
                matching_blocks.append(matching_block)
        if split_layers:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(split_block_into_layers(block))
            matching_blocks = tmp_blocks
        if split_blocks:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(BlockSplitter().run(block))
            matching_blocks = tmp_blocks
        if self._collect_from_back:
            matching_blocks = [block[::-1] for block in matching_blocks[::-1]]
        matching_blocks = [block for block in matching_blocks if len(block) >= min_block_size]
        return matching_blocks

class BlockSplitter:
    """Splits a block of nodes into sub-blocks over disjoint qubits.
    The implementation is based on the Disjoint Set Union data structure."""

    def __init__(self):
        if False:
            return 10
        self.leader = {}
        self.group = {}

    def find_leader(self, index):
        if False:
            return 10
        'Find in DSU.'
        if index not in self.leader:
            self.leader[index] = index
            self.group[index] = []
            return index
        if self.leader[index] == index:
            return index
        self.leader[index] = self.find_leader(self.leader[index])
        return self.leader[index]

    def union_leaders(self, index1, index2):
        if False:
            for i in range(10):
                print('nop')
        'Union in DSU.'
        leader1 = self.find_leader(index1)
        leader2 = self.find_leader(index2)
        if leader1 == leader2:
            return
        if len(self.group[leader1]) < len(self.group[leader2]):
            (leader1, leader2) = (leader2, leader1)
        self.leader[leader2] = leader1
        self.group[leader1].extend(self.group[leader2])
        self.group[leader2].clear()

    def run(self, block):
        if False:
            print('Hello World!')
        'Splits block of nodes into sub-blocks over disjoint qubits.'
        for node in block:
            indices = node.qargs
            if not indices:
                continue
            first = indices[0]
            for index in indices[1:]:
                self.union_leaders(first, index)
            self.group[self.find_leader(first)].append(node)
        blocks = []
        for index in self.leader:
            if self.leader[index] == index:
                blocks.append(self.group[index])
        return blocks

def split_block_into_layers(block):
    if False:
        return 10
    'Splits a block of nodes into sub-blocks of non-overlapping instructions\n    (or, in other words, into depth-1 sub-blocks).\n    '
    bit_depths = {}
    layers = []
    for node in block:
        cur_bits = set(node.qargs)
        cur_bits.update(node.cargs)
        cond = getattr(node.op, 'condition', None)
        if cond is not None:
            cur_bits.update(condition_resources(cond).clbits)
        cur_depth = max((bit_depths.get(bit, 0) for bit in cur_bits))
        while len(layers) <= cur_depth:
            layers.append([])
        for bit in cur_bits:
            bit_depths[bit] = cur_depth + 1
        layers[cur_depth].append(node)
    return layers

class BlockCollapser:
    """This class implements various strategies of consolidating blocks of nodes
     in a DAG (direct acyclic graph). It works both with the
    :class:`~qiskit.dagcircuit.DAGCircuit` and
    :class:`~qiskit.dagcircuit.DAGDependency` DAG representations.
    """

    def __init__(self, dag):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            dag (Union[DAGCircuit, DAGDependency]): The input DAG.\n        '
        self.dag = dag

    def collapse_to_operation(self, blocks, collapse_fn):
        if False:
            print('Hello World!')
        'For each block, constructs a quantum circuit containing instructions in the block,\n        then uses collapse_fn to collapse this circuit into a single operation.\n        '
        global_index_map = {wire: idx for (idx, wire) in enumerate(self.dag.qubits)}
        global_index_map.update({wire: idx for (idx, wire) in enumerate(self.dag.clbits)})
        for block in blocks:
            cur_qubits = set()
            cur_clbits = set()
            cur_clregs = set()
            for node in block:
                cur_qubits.update(node.qargs)
                cur_clbits.update(node.cargs)
                cond = getattr(node.op, 'condition', None)
                if cond is not None:
                    cur_clbits.update(condition_resources(cond).clbits)
                    if isinstance(cond[0], ClassicalRegister):
                        cur_clregs.add(cond[0])
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            sorted_clbits = sorted(cur_clbits, key=lambda x: global_index_map[x])
            qc = QuantumCircuit(sorted_qubits, sorted_clbits)
            for reg in cur_clregs:
                qc.add_register(reg)
            wire_pos_map = {qb: ix for (ix, qb) in enumerate(sorted_qubits)}
            wire_pos_map.update({qb: ix for (ix, qb) in enumerate(sorted_clbits)})
            for node in block:
                instructions = qc.append(CircuitInstruction(node.op, node.qargs, node.cargs))
                cond = getattr(node.op, 'condition', None)
                if cond is not None:
                    instructions.c_if(*cond)
            op = collapse_fn(qc)
            self.dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)
        return self.dag