"""Provides a general transpiler pass for collecting and consolidating blocks of nodes
in a circuit."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockCollapser
from qiskit.transpiler.passes.utils import control_flow

class CollectAndCollapse(TransformationPass):
    """A general transpiler pass to collect and to consolidate blocks of nodes
    in a circuit.

    This transpiler pass depends on two functions: the collection function and the
    collapsing function. The collection function ``collect_function`` takes a DAG
    and returns a list of blocks. The collapsing function ``collapse_function``
    takes a DAG and a list of blocks, consolidates each block, and returns the modified
    DAG.

    The input and the output DAGs are of type :class:`~qiskit.dagcircuit.DAGCircuit`,
    however when exploiting commutativity analysis to collect blocks, the
    :class:`~qiskit.dagcircuit.DAGDependency` representation is used internally.
    To support this, the ``collect_function`` and ``collapse_function`` should work
    with both types of DAGs and DAG nodes.

    Other collection and consolidation transpiler passes, for instance
    :class:`~.CollectLinearFunctions`, may derive from this pass, fixing
    ``collect_function`` and ``collapse_function`` to specific functions.
    """

    def __init__(self, collect_function, collapse_function, do_commutative_analysis=False):
        if False:
            return 10
        '\n        Args:\n            collect_function (callable): a function that takes a DAG and returns a list\n                of "collected" blocks of nodes\n            collapse_function (callable): a function that takes a DAG and a list of\n                "collected" blocks, and consolidates each block.\n            do_commutative_analysis (bool): if True, exploits commutativity relations\n                between nodes.\n        '
        self.collect_function = collect_function
        self.collapse_function = collapse_function
        self.do_commutative_analysis = do_commutative_analysis
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the CollectLinearFunctions pass on `dag`.\n        Args:\n            dag (DAGCircuit): the DAG to be optimized.\n        Returns:\n            DAGCircuit: the optimized DAG.\n        '
        if self.do_commutative_analysis:
            dag = dag_to_dagdependency(dag)
        blocks = self.collect_function(dag)
        self.collapse_function(dag, blocks)
        if self.do_commutative_analysis:
            dag = dagdependency_to_dag(dag)
        return dag

def collect_using_filter_function(dag, filter_function, split_blocks, min_block_size, split_layers=False, collect_from_back=False):
    if False:
        while True:
            i = 10
    'Corresponds to an important block collection strategy that greedily collects\n    maximal blocks of nodes matching a given ``filter_function``.\n    '
    return BlockCollector(dag).collect_all_matching_blocks(filter_fn=filter_function, split_blocks=split_blocks, min_block_size=min_block_size, split_layers=split_layers, collect_from_back=collect_from_back)

def collapse_to_operation(dag, blocks, collapse_function):
    if False:
        for i in range(10):
            print('nop')
    'Corresponds to an important block collapsing strategy that collapses every block\n    to a specific object as specified by ``collapse_function``.\n    '
    return BlockCollapser(dag).collapse_to_operation(blocks, collapse_function)