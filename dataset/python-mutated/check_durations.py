"""A pass to check if input circuit requires reschedule."""
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass

class InstructionDurationCheck(AnalysisPass):
    """Duration validation pass for reschedule.

    This pass investigates the input quantum circuit and checks if the circuit requires
    rescheduling for execution. Note that this pass can be triggered without scheduling.
    This pass only checks the duration of delay instructions and user defined pulse gates,
    which report duration values without pre-scheduling.

    This pass assumes backend supported instructions, i.e. basis gates, have no violation
    of the hardware alignment constraints, which is true in general.
    """

    def __init__(self, acquire_alignment: int=1, pulse_alignment: int=1):
        if False:
            i = 10
            return i + 15
        'Create new duration validation pass.\n\n        The alignment values depend on the control electronics of your quantum processor.\n\n        Args:\n            acquire_alignment: Integer number representing the minimum time resolution to\n                trigger acquisition instruction in units of ``dt``.\n            pulse_alignment: Integer number representing the minimum time resolution to\n                trigger gate instruction in units of ``dt``.\n        '
        super().__init__()
        self.acquire_align = acquire_alignment
        self.pulse_align = pulse_alignment

    def run(self, dag: DAGCircuit):
        if False:
            i = 10
            return i + 15
        'Run duration validation passes.\n\n        Args:\n            dag: DAG circuit to check instruction durations.\n        '
        self.property_set['reschedule_required'] = False
        if self.acquire_align == 1 and self.pulse_align == 1:
            return
        for delay_node in dag.op_nodes(Delay):
            dur = delay_node.op.duration
            if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                self.property_set['reschedule_required'] = True
                return
        for inst_defs in dag.calibrations.values():
            for caldef in inst_defs.values():
                dur = caldef.duration
                if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                    self.property_set['reschedule_required'] = True
                    return