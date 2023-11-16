"""A swap strategy pass for blocks of commuting gates."""
from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import Commuting2qBlock

class Commuting2qGateRouter(TransformationPass):
    """A class to swap route one or more commuting gates to the coupling map.

    This pass routes blocks of commuting two-qubit gates encapsulated as
    :class:`.Commuting2qBlock` instructions. This pass will not apply to other instructions.
    The mapping to the coupling map is done using swap strategies, see :class:`.SwapStrategy`.
    The swap strategy should suit the problem and the coupling map. This transpiler pass
    should ideally be executed before the quantum circuit is enlarged with any idle ancilla
    qubits. Otherwise, we may swap qubits outside the portion of the chip we want to use.
    Therefore, the swap strategy and its associated coupling map do not represent physical
    qubits. Instead, they represent an intermediate mapping that corresponds to the physical
    qubits once the initial layout is applied. The example below shows how to map a four
    qubit :class:`.PauliEvolutionGate` to qubits 0, 1, 3, and 4 of the five qubit device with
    the coupling map

    .. parsed-literal::

        0 -- 1 -- 2
             |
             3
             |
             4

    To do this we use a line swap strategy for qubits 0, 1, 3, and 4 defined it in terms
    of virtual qubits 0, 1, 2, and 3.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.opflow import PauliSumOp
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.transpiler import Layout, CouplingMap, PassManager
        from qiskit.transpiler.passes import FullAncillaAllocation
        from qiskit.transpiler.passes import EnlargeWithAncilla
        from qiskit.transpiler.passes import ApplyLayout
        from qiskit.transpiler.passes import SetLayout

        from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
            SwapStrategy,
            FindCommutingPauliEvolutions,
            Commuting2qGateRouter,
        )

        # Define the circuit on virtual qubits
        op = PauliSumOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Define the swap strategy on qubits before the initial_layout is applied.
        swap_strat = SwapStrategy.from_line([0, 1, 2, 3])

        # Chose qubits 0, 1, 3, and 4 from the backend coupling map shown above.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])
        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        # Insert swap gates, map to initial_layout and finally enlarge with ancilla.
        pm_pre.run(circ).draw("mpl")

    This pass manager relies on the ``current_layout`` which corresponds to the qubit layout as
    swap gates are applied. The pass will traverse all nodes in the dag. If a node should be
    routed using a swap strategy then it will be decomposed into sub-instructions with swap
    layers in between and the ``current_layout`` will be modified. Nodes that should not be
    routed using swap strategies will be added back to the dag taking the ``current_layout``
    into account.
    """

    def __init__(self, swap_strategy: SwapStrategy | None=None, edge_coloring: dict[tuple[int, int], int] | None=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            swap_strategy: An instance of a :class:`.SwapStrategy` that holds the swap layers\n                that are used, and the order in which to apply them, to map the instruction to\n                the hardware. If this field is not given, it should be contained in the\n                property set of the pass. This allows other passes to determine the most\n                appropriate swap strategy at run-time.\n            edge_coloring: An optional edge coloring of the coupling map (I.e. no two edges that\n                share a node have the same color). If the edge coloring is given then the commuting\n                gates that can be simultaneously applied given the current qubit permutation are\n                grouped according to the edge coloring and applied according to this edge\n                coloring. Here, a color is an int which is used as the index to define and\n                access the groups of commuting gates that can be applied simultaneously.\n                If the edge coloring is not given then the sets will be built-up using a\n                greedy algorithm. The edge coloring is useful to position gates such as\n                ``RZZGate``\\s next to swap gates to exploit CX cancellations.\n        '
        super().__init__()
        self._swap_strategy = swap_strategy
        self._bit_indices: dict[Qubit, int] | None = None
        self._edge_coloring = edge_coloring

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            print('Hello World!')
        'Run the pass by decomposing the nodes it applies on.\n\n        Args:\n            dag: The dag to which we will add swaps.\n\n        Returns:\n            A dag where swaps have been added for the intended gate type.\n\n        Raises:\n            TranspilerError: If the swap strategy was not given at init time and there is\n                no swap strategy in the property set.\n            TranspilerError: If the quantum circuit contains more than one qubit register.\n            TranspilerError: If there are qubits that are not contained in the quantum register.\n        '
        if self._swap_strategy is None:
            swap_strategy = self.property_set['swap_strategy']
            if swap_strategy is None:
                raise TranspilerError('No swap strategy given at init or in the property set.')
        else:
            swap_strategy = self._swap_strategy
        if len(dag.qregs) != 1:
            raise TranspilerError(f'{self.__class__.__name__} runs on circuits with one quantum register.')
        if len(dag.qubits) != next(iter(dag.qregs.values())).size:
            raise TranspilerError('Circuit has qubits not contained in the qubit register.')
        new_dag = dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        accumulator = new_dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if isinstance(node.op, Commuting2qBlock):
                self._check_edges(dag, node, swap_strategy)
                accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag)
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)
        self._compose_non_swap_nodes(accumulator, current_layout, new_dag)
        return new_dag

    def _compose_non_swap_nodes(self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit) -> DAGCircuit:
        if False:
            for i in range(10):
                print('nop')
        'Add all the non-swap strategy nodes that we have accumulated up to now.\n\n        This method also resets the node accumulator to an empty dag.\n\n        Args:\n            layout: The current layout that keeps track of the swaps.\n            new_dag: The new dag that we are building up.\n            accumulator: A DAG to keep track of nodes that do not decompose\n                using swap strategies.\n\n        Returns:\n            A new accumulator with the same registers as ``new_dag``.\n        '
        order = layout.reorder_bits(new_dag.qubits)
        order_bits: list[int | None] = [None] * len(layout)
        for (idx, val) in enumerate(order):
            order_bits[val] = idx
        new_dag.compose(accumulator, qubits=order_bits)
        return new_dag.copy_empty_like()

    def _position_in_cmap(self, dag: DAGCircuit, j: int, k: int, layout: Layout) -> tuple[int, ...]:
        if False:
            return 10
        'A helper function to track the movement of virtual qubits through the swaps.\n\n        Args:\n            j: The index of decision variable j (i.e. virtual qubit).\n            k: The index of decision variable k (i.e. virtual qubit).\n            layout: The current layout that takes into account previous swap gates.\n\n        Returns:\n            The position in the coupling map of the virtual qubits j and k as a tuple.\n        '
        bit0 = dag.find_bit(layout.get_physical_bits()[j]).index
        bit1 = dag.find_bit(layout.get_physical_bits()[k]).index
        return (bit0, bit1)

    def _build_sub_layers(self, current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
        if False:
            for i in range(10):
                print('nop')
        'A helper method to build-up sets of gates to simultaneously apply.\n\n        This is done with an edge coloring if the ``edge_coloring`` init argument was given or with\n        a greedy algorithm if not. With an edge coloring all gates on edges with the same color\n        will be applied simultaneously. These sublayers are applied in the order of their color,\n        which is an int, in increasing color order.\n\n        Args:\n            current_layer: All gates in the current layer can be applied given the qubit ordering\n                of the current layout. However, not all gates in the current layer can be applied\n                simultaneously. This function creates sub-layers by building up sub-layers\n                of gates. All gates in a sub-layer can simultaneously be applied given the coupling\n                map and current qubit configuration.\n\n        Returns:\n             A list of gate dicts that can be applied. The gates a position 0 are applied first.\n             A gate dict has the qubit tuple as key and the gate to apply as value.\n        '
        if self._edge_coloring is not None:
            return self._edge_coloring_build_sub_layers(current_layer)
        else:
            return self._greedy_build_sub_layers(current_layer)

    def _edge_coloring_build_sub_layers(self, current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
        if False:
            return 10
        'The edge coloring method of building sub-layers of commuting gates.'
        sub_layers: list[dict[tuple[int, int], Gate]] = [{} for _ in set(self._edge_coloring.values())]
        for (edge, gate) in current_layer.items():
            color = self._edge_coloring[edge]
            sub_layers[color][edge] = gate
        return sub_layers

    @staticmethod
    def _greedy_build_sub_layers(current_layer: dict[tuple[int, int], Gate]) -> list[dict[tuple[int, int], Gate]]:
        if False:
            while True:
                i = 10
        'The greedy method of building sub-layers of commuting gates.'
        sub_layers = []
        while len(current_layer) > 0:
            (current_sub_layer, remaining_gates) = ({}, {})
            blocked_vertices: set[tuple] = set()
            for (edge, evo_gate) in current_layer.items():
                if blocked_vertices.isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate
                    blocked_vertices = blocked_vertices.union(edge)
                else:
                    remaining_gates[edge] = evo_gate
            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)
        return sub_layers

    def swap_decompose(self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy) -> DAGCircuit:
        if False:
            i = 10
            return i + 15
        'Take an instance of :class:`.Commuting2qBlock` and map it to the coupling map.\n\n        The mapping is done with the swap strategy.\n\n        Args:\n            dag: The dag which contains the :class:`.Commuting2qBlock` we route.\n            node: A node whose operation is a :class:`.Commuting2qBlock`.\n            current_layout: The layout before the swaps are applied. This function will\n                modify the layout so that subsequent gates can be properly composed on the dag.\n            swap_strategy: The swap strategy used to decompose the node.\n\n        Returns:\n            A dag that is compatible with the coupling map where swap gates have been added\n            to map the gates in the :class:`.Commuting2qBlock` to the hardware.\n        '
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        gate_layers = self._make_op_layers(dag, node.op, current_layout, swap_strategy)
        max_distance = max(gate_layers.keys())
        circuit_with_swap = QuantumCircuit(len(dag.qubits))
        for i in range(max_distance + 1):
            current_layer = {}
            for ((j, k), local_gate) in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, j, k, current_layout)] = local_gate
            sub_layers = self._build_sub_layers(current_layer)
            for sublayer in sub_layers:
                for (edge, local_gate) in sublayer.items():
                    circuit_with_swap.append(local_gate, edge)
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]
                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)
        return circuit_to_dag(circuit_with_swap)

    def _make_op_layers(self, dag: DAGCircuit, op: Commuting2qBlock, layout: Layout, swap_strategy: SwapStrategy) -> dict[int, dict[tuple, Gate]]:
        if False:
            return 10
        'Creates layers of two-qubit gates based on the distance in the swap strategy.'
        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)
        for node in op.node_block:
            edge = (dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index)
            bit0 = layout.get_virtual_bits()[dag.qubits[edge[0]]]
            bit1 = layout.get_virtual_bits()[dag.qubits[edge[1]]]
            distance = swap_strategy.distance_matrix[bit0, bit1]
            gate_layers[distance][edge] = node.op
        return gate_layers

    def _check_edges(self, dag: DAGCircuit, node: DAGOpNode, swap_strategy: SwapStrategy):
        if False:
            for i in range(10):
                print('nop')
        'Check if the swap strategy can create the required connectivity.\n\n        Args:\n            node: The dag node for which to check if the swap strategy provides enough connectivity.\n            swap_strategy: The swap strategy that is being used.\n\n        Raises:\n            TranspilerError: If there is an edge that the swap strategy cannot accommodate\n                and if the pass has been configured to raise on such issues.\n        '
        required_edges = set()
        for sub_node in node.op:
            edge = (dag.find_bit(sub_node.qargs[0]).index, dag.find_bit(sub_node.qargs[1]).index)
            required_edges.add(edge)
        if not required_edges.issubset(swap_strategy.possible_edges):
            raise TranspilerError(f'{swap_strategy} cannot implement all edges in {required_edges}.')