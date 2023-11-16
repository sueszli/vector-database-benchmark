"""DAGDependency class for representing non-commutativity in a circuit.
"""
import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
from qiskit.circuit.commutation_checker import CommutationChecker

class DAGDependency:
    """Object to represent a quantum circuit as a Directed Acyclic Graph (DAG)
    via operation dependencies (i.e. lack of commutation).

    The nodes in the graph are operations represented by quantum gates.
    The edges correspond to non-commutation between two operations
    (i.e. a dependency). A directed edge from node A to node B means that
    operation A does not commute with operation B.
    The object's methods allow circuits to be constructed.

    The nodes in the graph have the following attributes:
    'operation', 'successors', 'predecessors'.

    **Example:**

    Bell circuit with no measurement.

    .. parsed-literal::

              ┌───┐
        qr_0: ┤ H ├──■──
              └───┘┌─┴─┐
        qr_1: ─────┤ X ├
                   └───┘

    The dependency DAG for the above circuit is represented by two nodes.
    The first one corresponds to Hadamard gate, the second one to the CNOT gate
    as the gates do not commute there is an edge between the two nodes.

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Create an empty DAGDependency.\n        '
        self.name = None
        self.metadata = {}
        self._multi_graph = rx.PyDAG()
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()
        self.qubits = []
        self.clbits = []
        self._global_phase = 0
        self._calibrations = defaultdict(dict)
        self.duration = None
        self.unit = 'dt'
        self.comm_checker = CommutationChecker()

    @property
    def global_phase(self):
        if False:
            while True:
                i = 10
        'Return the global phase of the circuit.'
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        if False:
            print('Hello World!')
        'Set the global phase of the circuit.\n\n        Args:\n            angle (float, ParameterExpression)\n        '
        from qiskit.circuit.parameterexpression import ParameterExpression
        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    @property
    def calibrations(self):
        if False:
            while True:
                i = 10
        "Return calibration dictionary.\n\n        The custom pulse definition of a given gate is of the form\n        ``{'gate_name': {(qubits, params): schedule}}``.\n        "
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        if False:
            return 10
        "Set the circuit calibration data from a dictionary of calibration definition.\n\n        Args:\n            calibrations (dict): A dictionary of input in the format\n                {'gate_name': {(qubits, gate_params): schedule}}\n        "
        self._calibrations = defaultdict(dict, calibrations)

    def to_retworkx(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the DAGDependency in retworkx format.'
        return self._multi_graph

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of gates in the circuit'
        return len(self._multi_graph)

    def depth(self):
        if False:
            i = 10
            return i + 15
        'Return the circuit depth.\n        Returns:\n            int: the circuit depth\n        '
        depth = rx.dag_longest_path_length(self._multi_graph)
        return depth if depth >= 0 else 0

    def add_qubits(self, qubits):
        if False:
            i = 10
            return i + 15
        'Add individual qubit wires.'
        if any((not isinstance(qubit, Qubit) for qubit in qubits)):
            raise DAGDependencyError('not a Qubit instance.')
        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGDependencyError('duplicate qubits %s' % duplicate_qubits)
        self.qubits.extend(qubits)

    def add_clbits(self, clbits):
        if False:
            for i in range(10):
                print('nop')
        'Add individual clbit wires.'
        if any((not isinstance(clbit, Clbit) for clbit in clbits)):
            raise DAGDependencyError('not a Clbit instance.')
        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGDependencyError('duplicate clbits %s' % duplicate_clbits)
        self.clbits.extend(clbits)

    def add_qreg(self, qreg):
        if False:
            while True:
                i = 10
        'Add qubits in a quantum register.'
        if not isinstance(qreg, QuantumRegister):
            raise DAGDependencyError('not a QuantumRegister instance.')
        if qreg.name in self.qregs:
            raise DAGDependencyError('duplicate register %s' % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])

    def add_creg(self, creg):
        if False:
            return 10
        'Add clbits in a classical register.'
        if not isinstance(creg, ClassicalRegister):
            raise DAGDependencyError('not a ClassicalRegister instance.')
        if creg.name in self.cregs:
            raise DAGDependencyError('duplicate register %s' % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] not in existing_clbits:
                self.clbits.append(creg[j])

    def _add_multi_graph_node(self, node):
        if False:
            print('Hello World!')
        '\n        Args:\n            node (DAGDepNode): considered node.\n\n        Returns:\n            node_id(int): corresponding label to the added node.\n        '
        node_id = self._multi_graph.add_node(node)
        node.node_id = node_id
        return node_id

    def get_nodes(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            generator(dict): iterator over all the nodes.\n        '
        return iter(self._multi_graph.nodes())

    def get_node(self, node_id):
        if False:
            print('Hello World!')
        '\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            node: corresponding to the label.\n        '
        return self._multi_graph.get_node_data(node_id)

    def _add_multi_graph_edge(self, src_id, dest_id, data):
        if False:
            while True:
                i = 10
        '\n        Function to add an edge from given data (dict) between two nodes.\n\n        Args:\n            src_id (int): label of the first node.\n            dest_id (int): label of the second node.\n            data (dict): data contained on the edge.\n\n        '
        self._multi_graph.add_edge(src_id, dest_id, data)

    def get_edges(self, src_id, dest_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Edge enumeration between two nodes through method get_all_edge_data.\n\n        Args:\n            src_id (int): label of the first node.\n            dest_id (int): label of the second node.\n\n        Returns:\n            List: corresponding to all edges between the two nodes.\n        '
        return self._multi_graph.get_all_edge_data(src_id, dest_id)

    def get_all_edges(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enumeration of all edges.\n\n        Returns:\n            List: corresponding to the label.\n        '
        return [(src, dest, data) for src_node in self._multi_graph.nodes() for (src, dest, data) in self._multi_graph.out_edges(src_node.node_id)]

    def get_in_edges(self, node_id):
        if False:
            print('Hello World!')
        '\n        Enumeration of all incoming edges for a given node.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: corresponding incoming edges data.\n        '
        return self._multi_graph.in_edges(node_id)

    def get_out_edges(self, node_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enumeration of all outgoing edges for a given node.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: corresponding outgoing edges data.\n        '
        return self._multi_graph.out_edges(node_id)

    def direct_successors(self, node_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Direct successors id of a given node as sorted list.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: direct successors id as a sorted list\n        '
        return sorted(self._multi_graph.adj_direction(node_id, False).keys())

    def direct_predecessors(self, node_id):
        if False:
            print('Hello World!')
        '\n        Direct predecessors id of a given node as sorted list.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: direct predecessors id as a sorted list\n        '
        return sorted(self._multi_graph.adj_direction(node_id, True).keys())

    def successors(self, node_id):
        if False:
            while True:
                i = 10
        '\n        Successors id of a given node as sorted list.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: all successors id as a sorted list\n        '
        return self._multi_graph.get_node_data(node_id).successors

    def predecessors(self, node_id):
        if False:
            while True:
                i = 10
        '\n        Predecessors id of a given node as sorted list.\n\n        Args:\n            node_id (int): label of considered node.\n\n        Returns:\n            List: all predecessors id as a sorted list\n        '
        return self._multi_graph.get_node_data(node_id).predecessors

    def topological_nodes(self):
        if False:
            print('Hello World!')
        '\n        Yield nodes in topological order.\n\n        Returns:\n            generator(DAGNode): node in topological order.\n        '

        def _key(x):
            if False:
                print('Hello World!')
            return x.sort_key
        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=_key))

    def _create_op_node(self, operation, qargs, cargs):
        if False:
            print('Hello World!')
        'Creates a DAGDepNode to the graph and update the edges.\n\n        Args:\n            operation (qiskit.circuit.Operation): operation\n            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts\n            cargs (list[Clbit]): list of classical wires to attach to\n\n        Returns:\n            DAGDepNode: the newly added node.\n        '
        directives = ['measure']
        if not getattr(operation, '_directive', False) and operation.name not in directives:
            qindices_list = []
            for elem in qargs:
                qindices_list.append(self.qubits.index(elem))
            if getattr(operation, 'condition', None):
                cond_bits = condition_resources(operation.condition).clbits
                cindices_list = [self.clbits.index(clbit) for clbit in cond_bits]
            else:
                cindices_list = []
        else:
            qindices_list = []
            cindices_list = []
        new_node = DAGDepNode(type='op', op=operation, name=operation.name, qargs=qargs, cargs=cargs, successors=[], predecessors=[], qindices=qindices_list, cindices=cindices_list)
        return new_node

    def add_op_node(self, operation, qargs, cargs):
        if False:
            i = 10
            return i + 15
        'Add a DAGDepNode to the graph and update the edges.\n\n        Args:\n            operation (qiskit.circuit.Operation): operation as a quantum gate\n            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts\n            cargs (list[Clbit]): list of classical wires to attach to\n        '
        new_node = self._create_op_node(operation, qargs, cargs)
        self._add_multi_graph_node(new_node)
        self._update_edges()

    def _update_edges(self):
        if False:
            i = 10
            return i + 15
        '\n        Updates DagDependency by adding edges to the newly added node (max_node)\n        from the previously added nodes.\n        For each previously added node (prev_node), an edge from prev_node to max_node\n        is added if max_node is "reachable" from prev_node (this means that the two\n        nodes can be made adjacent by commuting them with other nodes), but the two nodes\n        themselves do not commute.\n\n        Currently. this function is only used when creating a new DAGDependency from another\n        representation of a circuit, and hence there are no removed nodes (this is why\n        iterating over all nodes is fine).\n        '
        max_node_id = len(self._multi_graph) - 1
        max_node = self.get_node(max_node_id)
        reachable = [True] * max_node_id
        for prev_node_id in range(max_node_id - 1, -1, -1):
            if reachable[prev_node_id]:
                prev_node = self.get_node(prev_node_id)
                if not self.comm_checker.commute(prev_node.op, prev_node.qargs, prev_node.cargs, max_node.op, max_node.qargs, max_node.cargs):
                    self._multi_graph.add_edge(prev_node_id, max_node_id, {'commute': False})
                    predecessor_ids = self._multi_graph.predecessor_indices(prev_node_id)
                    for predecessor_id in predecessor_ids:
                        reachable[predecessor_id] = False
            else:
                predecessor_ids = self._multi_graph.predecessor_indices(prev_node_id)
                for predecessor_id in predecessor_ids:
                    reachable[predecessor_id] = False

    def _add_successors(self):
        if False:
            print('Hello World!')
        "\n        Create the list of successors. Update DAGDependency 'successors' attribute. It has to\n        be used when the DAGDependency() object is complete (i.e. converters).\n        "
        for node_id in range(len(self._multi_graph) - 1, -1, -1):
            self._multi_graph.get_node_data(node_id).successors = list(rx.descendants(self._multi_graph, node_id))

    def _add_predecessors(self):
        if False:
            i = 10
            return i + 15
        "\n        Create the list of predecessors for each node. Update DAGDependency\n        'predecessors' attribute. It has to be used when the DAGDependency() object\n        is complete (i.e. converters).\n        "
        for node_id in range(0, len(self._multi_graph)):
            self._multi_graph.get_node_data(node_id).predecessors = list(rx.ancestors(self._multi_graph, node_id))

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Function to copy a DAGDependency object.\n        Returns:\n            DAGDependency: a copy of a DAGDependency object.\n        '
        dag = DAGDependency()
        dag.name = self.name
        dag.cregs = self.cregs.copy()
        dag.qregs = self.qregs.copy()
        for node in self.get_nodes():
            dag._multi_graph.add_node(node.copy())
        for edges in self.get_all_edges():
            dag._multi_graph.add_edge(edges[0], edges[1], edges[2])
        return dag

    def draw(self, scale=0.7, filename=None, style='color'):
        if False:
            return 10
        "\n        Draws the DAGDependency graph.\n\n        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs\n        Graphviz <https://www.graphviz.org/>` to be installed.\n\n        Args:\n            scale (float): scaling factor\n            filename (str): file path to save image to (format inferred from name)\n            style (str): 'plain': B&W graph\n                         'color' (default): color input/output/op nodes\n\n        Returns:\n            Ipython.display.Image: if in Jupyter notebook and not saving to file, otherwise None.\n        "
        from qiskit.visualization.dag_visualization import dag_drawer
        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)

    def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
        if False:
            for i in range(10):
                print('nop')
        "Replace a block of nodes with a single node.\n\n        This is used to consolidate a block of DAGDepNodes into a single\n        operation. A typical example is a block of CX and SWAP gates consolidated\n        into a LinearFunction. This function is an adaptation of a similar\n        function from DAGCircuit.\n\n        It is important that such consolidation preserves commutativity assumptions\n        present in DAGDependency. As an example, suppose that every node in a\n        block [A, B, C, D] commutes with another node E. Let F be the consolidated\n        node, F = A o B o C o D. Then F also commutes with E, and thus the result of\n        replacing [A, B, C, D] by F results in a valid DAGDependency. That is, any\n        deduction about commutativity in consolidated DAGDependency is correct.\n        On the other hand, suppose that at least one of the nodes, say B, does not commute\n        with E. Then the consolidated DAGDependency would imply that F does not commute\n        with E. Even though F and E may actually commute, it is still safe to assume that\n        they do not. That is, the current implementation of consolidation may lead to\n        suboptimal but not to incorrect results.\n\n        Args:\n            node_block (List[DAGDepNode]): A list of dag nodes that represents the\n                node block to be replaced\n            op (qiskit.circuit.Operation): The operation to replace the\n                block with\n            wire_pos_map (Dict[~qiskit.circuit.Qubit, int]): The dictionary mapping the qarg to\n                the position. This is necessary to reconstruct the qarg order\n                over multiple gates in the combined single op node.\n            cycle_check (bool): When set to True this method will check that\n                replacing the provided ``node_block`` with a single node\n                would introduce a cycle (which would invalidate the\n                ``DAGDependency``) and will raise a ``DAGDependencyError`` if a cycle\n                would be introduced. This checking comes with a run time\n                penalty. If you can guarantee that your input ``node_block`` is\n                a contiguous block and won't introduce a cycle when it's\n                contracted to a single node, this can be set to ``False`` to\n                improve the runtime performance of this method.\n        Raises:\n            DAGDependencyError: if ``cycle_check`` is set to ``True`` and replacing\n                the specified block introduces a cycle or if ``node_block`` is\n                empty.\n        "
        block_qargs = set()
        block_cargs = set()
        block_ids = [x.node_id for x in node_block]
        if not node_block:
            raise DAGDependencyError("Can't replace an empty node_block")
        for nd in node_block:
            block_qargs |= set(nd.qargs)
            block_cargs |= set(nd.cargs)
            cond = getattr(nd.op, 'condition', None)
            if cond is not None:
                block_cargs.update(condition_resources(cond).clbits)
        new_node = self._create_op_node(op, qargs=sorted(block_qargs, key=lambda x: wire_pos_map[x]), cargs=sorted(block_cargs, key=lambda x: wire_pos_map[x]))
        try:
            new_node.node_id = self._multi_graph.contract_nodes(block_ids, new_node, check_cycle=cycle_check)
        except rx.DAGWouldCycle as ex:
            raise DAGDependencyError('Replacing the specified node block would introduce a cycle') from ex

def merge_no_duplicates(*iterables):
    if False:
        print('Hello World!')
    'Merge K list without duplicate using python heapq ordered merging\n\n    Args:\n        *iterables: A list of k sorted lists\n\n    Yields:\n        Iterator: List from the merging of the k ones (without duplicates\n    '
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val