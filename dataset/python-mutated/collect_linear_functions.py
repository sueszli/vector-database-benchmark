"""Replace each sequence of CX and SWAP gates by a single LinearFunction gate."""
from functools import partial
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.passes.optimization.collect_and_collapse import CollectAndCollapse, collect_using_filter_function, collapse_to_operation

class CollectLinearFunctions(CollectAndCollapse):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    def __init__(self, do_commutative_analysis=False, split_blocks=True, min_block_size=2, split_layers=False, collect_from_back=False):
        if False:
            for i in range(10):
                print('nop')
        'CollectLinearFunctions initializer.\n\n        Args:\n            do_commutative_analysis (bool): if True, exploits commutativity relations\n                between nodes.\n            split_blocks (bool): if True, splits collected blocks into sub-blocks\n                over disjoint qubit subsets.\n            min_block_size (int): specifies the minimum number of gates in the block\n                for the block to be collected.\n            split_layers (bool): if True, splits collected blocks into sub-blocks\n                over disjoint qubit subsets.\n            collect_from_back (bool): specifies if blocks should be collected started\n                from the end of the circuit.\n        '
        collect_function = partial(collect_using_filter_function, filter_function=_is_linear_gate, split_blocks=split_blocks, min_block_size=min_block_size, split_layers=split_layers, collect_from_back=collect_from_back)
        collapse_function = partial(collapse_to_operation, collapse_function=_collapse_to_linear_function)
        super().__init__(collect_function=collect_function, collapse_function=collapse_function, do_commutative_analysis=do_commutative_analysis)

def _is_linear_gate(node):
    if False:
        i = 10
        return i + 15
    'Specifies whether a node holds a linear gate.'
    return node.op.name in ('cx', 'swap') and getattr(node.op, 'condition', None) is None

def _collapse_to_linear_function(circuit):
    if False:
        print('Hello World!')
    'Specifies how to construct a ``LinearFunction`` from a quantum circuit (that must\n    consist of linear gates only).'
    return LinearFunction(circuit)