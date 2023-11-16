"""Replace each sequence of Clifford gates by a single Clifford gate."""
from functools import partial
from qiskit.transpiler.passes.optimization.collect_and_collapse import CollectAndCollapse, collect_using_filter_function, collapse_to_operation
from qiskit.quantum_info.operators import Clifford

class CollectCliffords(CollectAndCollapse):
    """Collects blocks of Clifford gates and replaces them by a :class:`~qiskit.quantum_info.Clifford`
    object.
    """

    def __init__(self, do_commutative_analysis=False, split_blocks=True, min_block_size=2, split_layers=False, collect_from_back=False):
        if False:
            while True:
                i = 10
        'CollectCliffords initializer.\n\n        Args:\n            do_commutative_analysis (bool): if True, exploits commutativity relations\n                between nodes.\n            split_blocks (bool): if True, splits collected blocks into sub-blocks\n                over disjoint qubit subsets.\n            min_block_size (int): specifies the minimum number of gates in the block\n                for the block to be collected.\n            split_layers (bool): if True, splits collected blocks into sub-blocks\n                over disjoint qubit subsets.\n            collect_from_back (bool): specifies if blocks should be collected started\n                from the end of the circuit.\n        '
        collect_function = partial(collect_using_filter_function, filter_function=_is_clifford_gate, split_blocks=split_blocks, min_block_size=min_block_size, split_layers=split_layers, collect_from_back=collect_from_back)
        collapse_function = partial(collapse_to_operation, collapse_function=_collapse_to_clifford)
        super().__init__(collect_function=collect_function, collapse_function=collapse_function, do_commutative_analysis=do_commutative_analysis)
clifford_gate_names = ['x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cy', 'cz', 'swap', 'clifford', 'linear_function', 'pauli']

def _is_clifford_gate(node):
    if False:
        i = 10
        return i + 15
    'Specifies whether a node holds a clifford gate.'
    return node.op.name in clifford_gate_names and getattr(node.op, 'condition', None) is None

def _collapse_to_clifford(circuit):
    if False:
        return 10
    'Specifies how to construct a ``Clifford`` from a quantum circuit (that must\n    consist of Clifford gates only).'
    return Clifford(circuit)