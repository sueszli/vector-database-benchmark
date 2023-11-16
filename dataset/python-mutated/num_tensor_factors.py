"""Calculate the number of tensor factors of a DAG circuit."""
from qiskit.transpiler.basepasses import AnalysisPass

class NumTensorFactors(AnalysisPass):
    """Calculate the number of tensor factors of a DAG circuit.

    The result is saved in ``property_set['num_tensor_factors']`` as an integer.
    """

    def run(self, dag):
        if False:
            print('Hello World!')
        'Run the NumTensorFactors pass on `dag`.'
        self.property_set['num_tensor_factors'] = dag.num_tensor_factors()