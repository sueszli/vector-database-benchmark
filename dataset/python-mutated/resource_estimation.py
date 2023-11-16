"""Automatically require analysis passes for resource estimation."""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.analysis.depth import Depth
from qiskit.transpiler.passes.analysis.width import Width
from qiskit.transpiler.passes.analysis.size import Size
from qiskit.transpiler.passes.analysis.count_ops import CountOps
from qiskit.transpiler.passes.analysis.num_tensor_factors import NumTensorFactors
from qiskit.transpiler.passes.analysis.num_qubits import NumQubits

class ResourceEstimation(AnalysisPass):
    """Automatically require analysis passes for resource estimation.

    An analysis pass for automatically running:
    * Depth()
    * Width()
    * Size()
    * CountOps()
    * NumTensorFactors()
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.requires += [Depth(), Width(), Size(), CountOps(), NumTensorFactors(), NumQubits()]

    def run(self, _):
        if False:
            print('Hello World!')
        'Run the ResourceEstimation pass on `dag`.'
        pass