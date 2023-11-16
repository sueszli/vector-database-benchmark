"""Gate equivalence library."""
import copy
from collections import namedtuple
from rustworkx.visualization import graphviz_draw
import rustworkx as rx
from qiskit.exceptions import InvalidFileError
from .exceptions import CircuitError
from .parameter import Parameter
from .parameterexpression import ParameterExpression
Key = namedtuple('Key', ['name', 'num_qubits'])
Equivalence = namedtuple('Equivalence', ['params', 'circuit'])
NodeData = namedtuple('NodeData', ['key', 'equivs'])
EdgeData = namedtuple('EdgeData', ['index', 'num_gates', 'rule', 'source'])

class EquivalenceLibrary:
    """A library providing a one-way mapping of Gates to their equivalent
    implementations as QuantumCircuits."""

    def __init__(self, *, base=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a new equivalence library.\n\n        Args:\n            base (Optional[EquivalenceLibrary]):  Base equivalence library to\n                be referenced if an entry is not found in this library.\n        '
        self._base = base
        if base is None:
            self._graph = rx.PyDiGraph()
            self._key_to_node_index = {}
            self._rule_count = 0
        else:
            self._graph = base._graph.copy()
            self._key_to_node_index = copy.deepcopy(base._key_to_node_index)
            self._rule_count = base._rule_count

    @property
    def graph(self) -> rx.PyDiGraph:
        if False:
            return 10
        'Return graph representing the equivalence library data.\n\n        This property should be treated as read-only as it provides\n        a reference to the internal state of the :class:`~.EquivalenceLibrary` object.\n        If the graph returned by this property is mutated it could corrupt the\n        the contents of the object. If you need to modify the output ``PyDiGraph``\n        be sure to make a copy prior to any modification.\n\n        Returns:\n            PyDiGraph: A graph object with equivalence data in each node.\n        '
        return self._graph

    def _set_default_node(self, key):
        if False:
            while True:
                i = 10
        'Create a new node if key not found'
        if key not in self._key_to_node_index:
            self._key_to_node_index[key] = self._graph.add_node(NodeData(key=key, equivs=[]))
        return self._key_to_node_index[key]

    def add_equivalence(self, gate, equivalent_circuit):
        if False:
            i = 10
            return i + 15
        'Add a new equivalence to the library. Future queries for the Gate\n        will include the given circuit, in addition to all existing equivalences\n        (including those from base).\n\n        Parameterized Gates (those including `qiskit.circuit.Parameters` in their\n        `Gate.params`) can be marked equivalent to parameterized circuits,\n        provided the parameters match.\n\n        Args:\n            gate (Gate): A Gate instance.\n            equivalent_circuit (QuantumCircuit): A circuit equivalently\n                implementing the given Gate.\n        '
        _raise_if_shape_mismatch(gate, equivalent_circuit)
        _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)
        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equiv = Equivalence(params=gate.params.copy(), circuit=equivalent_circuit.copy())
        target = self._set_default_node(key)
        self._graph[target].equivs.append(equiv)
        sources = {Key(name=instruction.operation.name, num_qubits=len(instruction.qubits)) for instruction in equivalent_circuit}
        edges = [(self._set_default_node(source), target, EdgeData(index=self._rule_count, num_gates=len(sources), rule=equiv, source=source)) for source in sources]
        self._graph.add_edges_from(edges)
        self._rule_count += 1

    def has_entry(self, gate):
        if False:
            return 10
        'Check if a library contains any decompositions for gate.\n\n        Args:\n            gate (Gate): A Gate instance.\n\n        Returns:\n            Bool: True if gate has a known decomposition in the library.\n                False otherwise.\n        '
        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        return key in self._key_to_node_index

    def set_entry(self, gate, entry):
        if False:
            i = 10
            return i + 15
        "Set the equivalence record for a Gate. Future queries for the Gate\n        will return only the circuits provided.\n\n        Parameterized Gates (those including `qiskit.circuit.Parameters` in their\n        `Gate.params`) can be marked equivalent to parameterized circuits,\n        provided the parameters match.\n\n        Args:\n            gate (Gate): A Gate instance.\n            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each\n                equivalently implementing the given Gate.\n        "
        for equiv in entry:
            _raise_if_shape_mismatch(gate, equiv)
            _raise_if_param_mismatch(gate.params, equiv.parameters)
        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equivs = [Equivalence(params=gate.params.copy(), circuit=equiv.copy()) for equiv in entry]
        self._graph[self._set_default_node(key)] = NodeData(key=key, equivs=equivs)

    def get_entry(self, gate):
        if False:
            return 10
        'Gets the set of QuantumCircuits circuits from the library which\n        equivalently implement the given Gate.\n\n        Parameterized circuits will have their parameters replaced with the\n        corresponding entries from Gate.params.\n\n        Args:\n            gate (Gate) - Gate: A Gate instance.\n\n        Returns:\n            List[QuantumCircuit]: A list of equivalent QuantumCircuits. If empty,\n                library contains no known decompositions of Gate.\n\n                Returned circuits will be ordered according to their insertion in\n                the library, from earliest to latest, from top to base. The\n                ordering of the StandardEquivalenceLibrary will not generally be\n                consistent across Qiskit versions.\n        '
        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        query_params = gate.params
        return [_rebind_equiv(equiv, query_params) for equiv in self._get_equivalences(key)]

    def keys(self):
        if False:
            return 10
        'Return list of keys to key to node index map.\n\n        Returns:\n            List: Keys to the key to node index map.\n        '
        return self._key_to_node_index.keys()

    def node_index(self, key):
        if False:
            print('Hello World!')
        'Return node index for a given key.\n\n        Args:\n            key (Key): Key to an equivalence.\n\n        Returns:\n            Int: Index to the node in the graph for the given key.\n        '
        return self._key_to_node_index[key]

    def draw(self, filename=None):
        if False:
            return 10
        'Draws the equivalence relations available in the library.\n\n        Args:\n            filename (str): An optional path to write the output image to\n                if specified this method will return None.\n\n        Returns:\n            PIL.Image or IPython.display.SVG: Drawn equivalence library as an\n                IPython SVG if in a jupyter notebook, or as a PIL.Image otherwise.\n\n        Raises:\n            InvalidFileError: if filename is not valid.\n        '
        image_type = None
        if filename:
            if '.' not in filename:
                raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
            image_type = filename.split('.')[-1]
        return graphviz_draw(self._build_basis_graph(), lambda node: {'label': node['label']}, lambda edge: edge, filename=filename, image_type=image_type)

    def _build_basis_graph(self):
        if False:
            while True:
                i = 10
        graph = rx.PyDiGraph()
        node_map = {}
        for key in self._key_to_node_index:
            (name, num_qubits) = key
            equivalences = self._get_equivalences(key)
            basis = frozenset([f'{name}/{num_qubits}'])
            for (params, decomp) in equivalences:
                decomp_basis = frozenset((f'{name}/{num_qubits}' for (name, num_qubits) in {(instruction.operation.name, instruction.operation.num_qubits) for instruction in decomp.data}))
                if basis not in node_map:
                    basis_node = graph.add_node({'basis': basis, 'label': str(set(basis))})
                    node_map[basis] = basis_node
                if decomp_basis not in node_map:
                    decomp_basis_node = graph.add_node({'basis': decomp_basis, 'label': str(set(decomp_basis))})
                    node_map[decomp_basis] = decomp_basis_node
                label = '{}\n{}'.format(str(params), str(decomp) if num_qubits <= 5 else '...')
                graph.add_edge(node_map[basis], node_map[decomp_basis], {'label': label, 'fontname': 'Courier', 'fontsize': str(8)})
        return graph

    def _get_equivalences(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Get all the equivalences for the given key'
        return self._graph[self._key_to_node_index[key]].equivs if key in self._key_to_node_index else []

def _raise_if_param_mismatch(gate_params, circuit_parameters):
    if False:
        for i in range(10):
            print('nop')
    gate_parameters = [p for p in gate_params if isinstance(p, ParameterExpression)]
    if set(gate_parameters) != circuit_parameters:
        raise CircuitError('Cannot add equivalence between circuit and gate of different parameters. Gate params: {}. Circuit params: {}.'.format(gate_parameters, circuit_parameters))

def _raise_if_shape_mismatch(gate, circuit):
    if False:
        for i in range(10):
            print('nop')
    if gate.num_qubits != circuit.num_qubits or gate.num_clbits != circuit.num_clbits:
        raise CircuitError('Cannot add equivalence between circuit and gate of different shapes. Gate: {} qubits and {} clbits. Circuit: {} qubits and {} clbits.'.format(gate.num_qubits, gate.num_clbits, circuit.num_qubits, circuit.num_clbits))

def _rebind_equiv(equiv, query_params):
    if False:
        for i in range(10):
            print('nop')
    (equiv_params, equiv_circuit) = equiv
    param_map = {x: y for (x, y) in zip(equiv_params, query_params) if isinstance(x, Parameter)}
    equiv = equiv_circuit.assign_parameters(param_map, inplace=False, flat_input=True)
    return equiv