"""Synthesize LinearFunctions."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.utils.deprecation import deprecate_func

class LinearFunctionsSynthesis(HighLevelSynthesis):
    """DEPRECATED: Synthesize linear functions.

    Under the hood, this runs the default high-level synthesis plugin for linear functions.
    """

    @deprecate_func(additional_msg='Instead, use :class:`~.HighLevelSynthesis`.', since='0.23.0', package_name='qiskit-terra')
    def __init__(self):
        if False:
            while True:
                i = 10
        default_linear_config = HLSConfig(linear_function=[('default', {})], use_default_on_unspecified=False)
        super().__init__(hls_config=default_linear_config)

class LinearFunctionsToPermutations(TransformationPass):
    """Promotes linear functions to permutations when possible."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            for i in range(10):
                print('nop')
        'Run the LinearFunctionsToPermutations pass on `dag`.\n        Args:\n            dag: input dag.\n        Returns:\n            Output dag with LinearFunctions synthesized.\n        '
        for node in dag.named_nodes('linear_function'):
            try:
                pattern = node.op.permutation_pattern()
            except CircuitError:
                continue
            permutation = PermutationGate(pattern)
            dag.substitute_node(node, permutation)
        return dag