"""
Directed graph object for representing coupling between physical qubits.

The nodes of the graph correspond to physical qubits (represented as integers) and the
directed edges indicate which physical qubits are coupled and the permitted direction of
CNOT gates. The object has a distance function that can be used to map quantum circuits
onto a device with this coupling.
"""
import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError

class CouplingMap:
    """
    Directed graph specifying fixed coupling.

    Nodes correspond to physical qubits (integers) and directed edges correspond
    to permitted CNOT gates, with source and destination corresponding to control
    and target qubits, respectively.
    """
    __slots__ = ('description', 'graph', '_dist_matrix', '_qubit_list', '_size', '_is_symmetric')

    def __init__(self, couplinglist=None, description=None):
        if False:
            return 10
        '\n        Create coupling graph. By default, the generated coupling has no nodes.\n\n        Args:\n            couplinglist (list or None): An initial coupling graph, specified as\n                an adjacency list containing couplings, e.g. [[0,1], [0,2], [1,2]].\n                It is required that nodes are contiguously indexed starting at 0.\n                Missed nodes will be added as isolated nodes in the coupling map.\n            description (str): A string to describe the coupling map.\n        '
        self.description = description
        self.graph = rx.PyDiGraph()
        self._dist_matrix = None
        self._qubit_list = None
        self._size = None
        self._is_symmetric = None
        if couplinglist is not None:
            self.graph.extend_from_edge_list([tuple(x) for x in couplinglist])

    def size(self):
        if False:
            while True:
                i = 10
        'Return the number of physical qubits in this graph.'
        if self._size is None:
            self._size = len(self.graph)
        return self._size

    def get_edges(self):
        if False:
            print('Hello World!')
        '\n        Gets the list of edges in the coupling graph.\n\n        Returns:\n            Tuple(int,int): Each edge is a pair of physical qubits.\n        '
        return self.graph.edge_list()

    def __iter__(self):
        if False:
            return 10
        return iter(self.graph.edge_list())

    def add_physical_qubit(self, physical_qubit):
        if False:
            return 10
        'Add a physical qubit to the coupling graph as a node.\n\n        physical_qubit (int): An integer representing a physical qubit.\n\n        Raises:\n            CouplingError: if trying to add duplicate qubit\n        '
        if not isinstance(physical_qubit, int):
            raise CouplingError('Physical qubits should be integers.')
        if physical_qubit in self.physical_qubits:
            raise CouplingError('The physical qubit %s is already in the coupling graph' % physical_qubit)
        self.graph.add_node(physical_qubit)
        self._dist_matrix = None
        self._qubit_list = None
        self._size = None

    def add_edge(self, src, dst):
        if False:
            return 10
        '\n        Add directed edge to coupling graph.\n\n        src (int): source physical qubit\n        dst (int): destination physical qubit\n        '
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        self.graph.add_edge(src, dst, None)
        self._dist_matrix = None
        self._is_symmetric = None

    @property
    def physical_qubits(self):
        if False:
            while True:
                i = 10
        'Returns a sorted list of physical_qubits'
        if self._qubit_list is None:
            self._qubit_list = self.graph.node_indexes()
        return self._qubit_list

    def is_connected(self):
        if False:
            while True:
                i = 10
        '\n        Test if the graph is connected.\n\n        Return True if connected, False otherwise\n        '
        try:
            return rx.is_weakly_connected(self.graph)
        except rx.NullGraph:
            return False

    def neighbors(self, physical_qubit):
        if False:
            i = 10
            return i + 15
        'Return the nearest neighbors of a physical qubit.\n\n        Directionality matters, i.e. a neighbor must be reachable\n        by going one hop in the direction of an edge.\n        '
        return self.graph.neighbors(physical_qubit)

    @property
    def distance_matrix(self):
        if False:
            return 10
        "Return the distance matrix for the coupling map.\n\n        For any qubits where there isn't a path available between them the value\n        in this position of the distance matrix will be ``math.inf``.\n        "
        self.compute_distance_matrix()
        return self._dist_matrix

    def compute_distance_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        "Compute the full distance matrix on pairs of nodes.\n\n        The distance map self._dist_matrix is computed from the graph using\n        all_pairs_shortest_path_length. This is normally handled internally\n        by the :attr:`~qiskit.transpiler.CouplingMap.distance_matrix`\n        attribute or the :meth:`~qiskit.transpiler.CouplingMap.distance` method\n        but can be called if you're accessing the distance matrix outside of\n        those or want to pre-generate it.\n        "
        if self._dist_matrix is None:
            self._dist_matrix = rx.digraph_distance_matrix(self.graph, as_undirected=True, null_value=math.inf)

    def distance(self, physical_qubit1, physical_qubit2):
        if False:
            i = 10
            return i + 15
        'Returns the undirected distance between physical_qubit1 and physical_qubit2.\n\n        Args:\n            physical_qubit1 (int): A physical qubit\n            physical_qubit2 (int): Another physical qubit\n\n        Returns:\n            int: The undirected distance\n\n        Raises:\n            CouplingError: if the qubits do not exist in the CouplingMap\n        '
        if physical_qubit1 >= self.size():
            raise CouplingError('%s not in coupling graph' % physical_qubit1)
        if physical_qubit2 >= self.size():
            raise CouplingError('%s not in coupling graph' % physical_qubit2)
        self.compute_distance_matrix()
        res = self._dist_matrix[physical_qubit1, physical_qubit2]
        if res == math.inf:
            raise CouplingError(f'No path from {physical_qubit1} to {physical_qubit2}')
        return int(res)

    def shortest_undirected_path(self, physical_qubit1, physical_qubit2):
        if False:
            return 10
        'Returns the shortest undirected path between physical_qubit1 and physical_qubit2.\n\n        Args:\n            physical_qubit1 (int): A physical qubit\n            physical_qubit2 (int): Another physical qubit\n        Returns:\n            List: The shortest undirected path\n        Raises:\n            CouplingError: When there is no path between physical_qubit1, physical_qubit2.\n        '
        paths = rx.digraph_dijkstra_shortest_paths(self.graph, source=physical_qubit1, target=physical_qubit2, as_undirected=True)
        if not paths:
            raise CouplingError(f'Nodes {str(physical_qubit1)} and {str(physical_qubit2)} are not connected')
        return paths[physical_qubit2]

    @property
    def is_symmetric(self):
        if False:
            print('Hello World!')
        '\n        Test if the graph is symmetric.\n\n        Return True if symmetric, False otherwise\n        '
        if self._is_symmetric is None:
            self._is_symmetric = self._check_symmetry()
        return self._is_symmetric

    def make_symmetric(self):
        if False:
            print('Hello World!')
        '\n        Convert uni-directional edges into bi-directional.\n        '
        edges = self.get_edges()
        edge_set = set(edges)
        for (src, dest) in edges:
            if (dest, src) not in edge_set:
                self.graph.add_edge(dest, src, None)
        self._dist_matrix = None
        self._is_symmetric = None

    def _check_symmetry(self):
        if False:
            print('Hello World!')
        '\n        Calculates symmetry\n\n        Returns:\n            Bool: True if symmetric, False otherwise\n        '
        return self.graph.is_symmetric()

    def reduce(self, mapping, check_if_connected=True):
        if False:
            for i in range(10):
                print('nop')
        'Returns a reduced coupling map that\n        corresponds to the subgraph of qubits\n        selected in the mapping.\n\n        Args:\n            mapping (list): A mapping of reduced qubits to device\n                qubits.\n            check_if_connected (bool): if True, checks that the reduced\n                coupling map is connected.\n\n        Returns:\n            CouplingMap: A reduced coupling_map for the selected qubits.\n\n        Raises:\n            CouplingError: Reduced coupling map must be connected.\n        '
        inv_map = [None] * (max(mapping) + 1)
        for (idx, val) in enumerate(mapping):
            inv_map[val] = idx
        reduced_cmap = []
        for edge in self.get_edges():
            if edge[0] in mapping and edge[1] in mapping:
                reduced_cmap.append([inv_map[edge[0]], inv_map[edge[1]]])
        reduced_coupling_map = CouplingMap()
        for node in range(len(mapping)):
            reduced_coupling_map.graph.add_node(node)
        reduced_coupling_map.graph.extend_from_edge_list([tuple(x) for x in reduced_cmap])
        if check_if_connected and (not reduced_coupling_map.is_connected()):
            raise CouplingError('coupling_map must be connected.')
        return reduced_coupling_map

    @classmethod
    def from_full(cls, num_qubits, bidirectional=True) -> 'CouplingMap':
        if False:
            for i in range(10):
                print('nop')
        'Return a fully connected coupling map on n qubits.'
        cmap = cls(description='full')
        if bidirectional:
            cmap.graph = rx.generators.directed_mesh_graph(num_qubits)
        else:
            edge_list = []
            for i in range(num_qubits):
                for j in range(i):
                    edge_list.append((j, i))
            cmap.graph.extend_from_edge_list(edge_list)
        return cmap

    @classmethod
    def from_line(cls, num_qubits, bidirectional=True) -> 'CouplingMap':
        if False:
            for i in range(10):
                print('nop')
        'Return a coupling map of n qubits connected in a line.'
        cmap = cls(description='line')
        cmap.graph = rx.generators.directed_path_graph(num_qubits, bidirectional=bidirectional)
        return cmap

    @classmethod
    def from_ring(cls, num_qubits, bidirectional=True) -> 'CouplingMap':
        if False:
            return 10
        'Return a coupling map of n qubits connected to each of their neighbors in a ring.'
        cmap = cls(description='ring')
        cmap.graph = rx.generators.directed_cycle_graph(num_qubits, bidirectional=bidirectional)
        return cmap

    @classmethod
    def from_grid(cls, num_rows, num_columns, bidirectional=True) -> 'CouplingMap':
        if False:
            while True:
                i = 10
        'Return a coupling map of qubits connected on a grid of num_rows x num_columns.'
        cmap = cls(description='grid')
        cmap.graph = rx.generators.directed_grid_graph(num_rows, num_columns, bidirectional=bidirectional)
        return cmap

    @classmethod
    def from_heavy_hex(cls, distance, bidirectional=True) -> 'CouplingMap':
        if False:
            return 10
        'Return a heavy hexagon graph coupling map.\n\n        A heavy hexagon graph is described in:\n\n        https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011022\n\n        Args:\n            distance (int): The code distance for the generated heavy hex\n                graph. The value for distance can be any odd positive integer.\n                The distance relates to the number of qubits by:\n                :math:`n = \\frac{5d^2 - 2d - 1}{2}` where :math:`n` is the\n                number of qubits and :math:`d` is the ``distance`` parameter.\n            bidirectional (bool): Whether the edges in the output coupling\n                graph are bidirectional or not. By default this is set to\n                ``True``\n        Returns:\n            CouplingMap: A heavy hex coupling graph\n        '
        cmap = cls(description='heavy-hex')
        cmap.graph = rx.generators.directed_heavy_hex_graph(distance, bidirectional=bidirectional)
        return cmap

    @classmethod
    def from_heavy_square(cls, distance, bidirectional=True) -> 'CouplingMap':
        if False:
            print('Hello World!')
        'Return a heavy square graph coupling map.\n\n        A heavy square graph is described in:\n\n        https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011022\n\n        Args:\n            distance (int): The code distance for the generated heavy square\n                graph. The value for distance can be any odd positive integer.\n                The distance relates to the number of qubits by:\n                :math:`n = 3d^2 - 2d` where :math:`n` is the\n                number of qubits and :math:`d` is the ``distance`` parameter.\n            bidirectional (bool): Whether the edges in the output coupling\n                graph are bidirectional or not. By default this is set to\n                ``True``\n        Returns:\n            CouplingMap: A heavy square coupling graph\n        '
        cmap = cls(description='heavy-square')
        cmap.graph = rx.generators.directed_heavy_square_graph(distance, bidirectional=bidirectional)
        return cmap

    @classmethod
    def from_hexagonal_lattice(cls, rows, cols, bidirectional=True) -> 'CouplingMap':
        if False:
            i = 10
            return i + 15
        'Return a hexagonal lattice graph coupling map.\n\n        Args:\n            rows (int): The number of rows to generate the graph with.\n            cols (int): The number of columns to generate the graph with.\n            bidirectional (bool): Whether the edges in the output coupling\n                graph are bidirectional or not. By default this is set to\n                ``True``\n        Returns:\n            CouplingMap: A hexagonal lattice coupling graph\n        '
        cmap = cls(description='hexagonal-lattice')
        cmap.graph = rx.generators.directed_hexagonal_lattice_graph(rows, cols, bidirectional=bidirectional)
        return cmap

    def largest_connected_component(self):
        if False:
            return 10
        'Return a set of qubits in the largest connected component.'
        return max(rx.weakly_connected_components(self.graph), key=len)

    def connected_components(self) -> List['CouplingMap']:
        if False:
            print('Hello World!')
        "Separate a :Class:`~.CouplingMap` into subgraph :class:`~.CouplingMap`\n        for each connected component.\n\n        The connected components of a :class:`~.CouplingMap` are the subgraphs\n        that are not part of any larger subgraph. For example, if you had a\n        coupling map that looked like::\n\n            0 --> 1   4 --> 5 ---> 6 --> 7\n            |     |\n            |     |\n            V     V\n            2 --> 3\n\n        then the connected components of that graph are the subgraphs::\n\n            0 --> 1\n            |     |\n            |     |\n            V     V\n            2 --> 3\n\n        and::\n\n            4 --> 5 ---> 6 --> 7\n\n        For a connected :class:`~.CouplingMap` object there is only a single connected\n        component, the entire :class:`~.CouplingMap`.\n\n        This method will return a list of :class:`~.CouplingMap` objects, one for each connected\n        component in this :class:`~.CouplingMap`. The data payload of each node in the\n        :attr:`~.CouplingMap.graph` attribute will contain the qubit number in the original\n        graph. This will enables mapping the qubit index in a component subgraph to\n        the original qubit in the combined :class:`~.CouplingMap`. For example::\n\n            from qiskit.transpiler import CouplingMap\n\n            cmap = CouplingMap([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])\n            component_cmaps = cmap.connected_components()\n            print(component_cmaps[1].graph[0])\n\n        will print ``3`` as index ``0`` in the second component is qubit 3 in the original cmap.\n\n        Returns:\n            list: A list of :class:`~.CouplingMap` objects for each connected\n                components. The order of this list is deterministic but\n                implementation specific and shouldn't be relied upon as\n                part of the API.\n        "
        for node in self.graph.node_indices():
            self.graph[node] = node
        components = rx.weakly_connected_components(self.graph)
        output_list = []
        for component in components:
            new_cmap = CouplingMap()
            new_cmap.graph = self.graph.subgraph(sorted(component))
            output_list.append(new_cmap)
        return output_list

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a string representation of the coupling graph.'
        string = ''
        if self.get_edges():
            string += '['
            string += ', '.join([f'[{src}, {dst}]' for (src, dst) in self.get_edges()])
            string += ']'
        return string

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Check if the graph in ``other`` has the same node labels and edges as the graph in\n        ``self``.\n\n        This function assumes that the graphs in :class:`.CouplingMap` instances are connected.\n\n        Args:\n            other (CouplingMap): The other coupling map.\n\n        Returns:\n            bool: Whether or not other is isomorphic to self.\n        '
        if not isinstance(other, CouplingMap):
            return False
        return set(self.graph.edge_list()) == set(other.graph.edge_list())

    def draw(self):
        if False:
            return 10
        'Draws the coupling map.\n\n        This function calls the :func:`~rustworkx.visualization.graphviz_draw` function from the\n        ``rustworkx`` package to draw the :class:`CouplingMap` object.\n\n        Returns:\n            PIL.Image: Drawn coupling map.\n\n        '
        return graphviz_draw(self.graph, method='neato')