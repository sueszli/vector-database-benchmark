"""Set classical IO latency information to circuit."""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.dagcircuit import DAGCircuit

class SetIOLatency(AnalysisPass):
    """Set IOLatency information to the input circuit.

    The ``clbit_write_latency`` and ``conditional_latency`` are added to
    the property set of pass manager. This information can be shared among the passes
    that perform scheduling on instructions acting on classical registers.

    Once these latencies are added to the property set, this information
    is also copied to the output circuit object as protected attributes,
    so that it can be utilized outside the transpilation,
    for example, the timeline visualization can use latency to accurately show
    time occupation by instructions on the classical registers.
    """

    def __init__(self, clbit_write_latency: int=0, conditional_latency: int=0):
        if False:
            i = 10
            return i + 15
        'Create pass with latency information.\n\n        Args:\n            clbit_write_latency: A control flow constraints. Because standard superconducting\n                quantum processor implement dispersive QND readout, the actual data transfer\n                to the clbit happens after the round-trip stimulus signal is buffered\n                and discriminated into quantum state.\n                The interval ``[t0, t0 + clbit_write_latency]`` is regarded as idle time\n                for clbits associated with the measure instruction.\n                This defaults to 0 dt which is identical to Qiskit Pulse scheduler.\n            conditional_latency: A control flow constraints. This value represents\n                a latency of reading a classical register for the conditional operation.\n                The gate operation occurs after this latency. This appears as a delay\n                in front of the DAGOpNode of the gate.\n                This defaults to 0 dt.\n        '
        super().__init__()
        self._conditional_latency = conditional_latency
        self._clbit_write_latency = clbit_write_latency

    def run(self, dag: DAGCircuit):
        if False:
            return 10
        'Add IO latency information.\n\n        Args:\n            dag: Input DAG circuit.\n        '
        self.property_set['conditional_latency'] = self._conditional_latency
        self.property_set['clbit_write_latency'] = self._clbit_write_latency