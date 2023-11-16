"""ASAP Scheduling."""
from qiskit.circuit import Delay, Qubit, Measure
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecation import deprecate_func
from .base_scheduler import BaseSchedulerTransform

class ASAPSchedule(BaseSchedulerTransform):
    """ASAP Scheduling pass, which schedules the start time of instructions as early as possible..

    See :class:`~qiskit.transpiler.passes.scheduling.base_scheduler.BaseSchedulerTransform` for the
    detailed behavior of the control flow operation, i.e. ``c_if``.

    .. note::

        This base class has been superseded by :class:`~.ASAPScheduleAnalysis` and
        the new scheduling workflow. It will be deprecated and subsequently
        removed in a future release.
    """

    @deprecate_func(additional_msg='Instead, use :class:`~.ASAPScheduleAnalysis`, which is an analysis pass that requires a padding pass to later modify the circuit.', since='0.21.0', pending=True)
    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the ASAPSchedule pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): DAG to schedule.\n\n        Returns:\n            DAGCircuit: A scheduled DAG.\n\n        Raises:\n            TranspilerError: if the circuit is not mapped on physical qubits.\n            TranspilerError: if conditional bit is added to non-supported instruction.\n        '
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('ASAP schedule runs on physical circuits only')
        time_unit = self.property_set['time_unit']
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        idle_after = {q: 0 for q in dag.qubits + dag.clbits}
        for node in dag.topological_op_nodes():
            op_duration = self._get_node_duration(node, dag)
            if isinstance(node.op, self.CONDITIONAL_SUPPORTED):
                t0q = max((idle_after[q] for q in node.qargs))
                if node.op.condition_bits:
                    t0c = max((idle_after[bit] for bit in node.op.condition_bits))
                    if t0q > t0c:
                        t0c = max(t0q - self.conditional_latency, t0c)
                    t1c = t0c + self.conditional_latency
                    for bit in node.op.condition_bits:
                        idle_after[bit] = t1c
                    t0 = max(t0q, t1c)
                else:
                    t0 = t0q
                t1 = t0 + op_duration
            else:
                if node.op.condition_bits:
                    raise TranspilerError(f'Conditional instruction {node.op.name} is not supported in ASAP scheduler.')
                if isinstance(node.op, Measure):
                    t0q = max((idle_after[q] for q in node.qargs))
                    t0c = max((idle_after[c] for c in node.cargs))
                    t0 = max(t0q, t0c - self.clbit_write_latency)
                    t1 = t0 + op_duration
                    for clbit in node.cargs:
                        idle_after[clbit] = t1
                else:
                    t0 = max((idle_after[bit] for bit in node.qargs + node.cargs))
                    t1 = t0 + op_duration
            for bit in node.qargs:
                delta = t0 - idle_after[bit]
                if delta > 0 and isinstance(bit, Qubit) and self._delay_supported(dag.find_bit(bit).index):
                    new_dag.apply_operation_back(Delay(delta, time_unit), [bit], [])
                idle_after[bit] = t1
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        circuit_duration = max(idle_after.values())
        for (bit, after) in idle_after.items():
            delta = circuit_duration - after
            if not (delta > 0 and isinstance(bit, Qubit)):
                continue
            if self._delay_supported(dag.find_bit(bit).index):
                new_dag.apply_operation_back(Delay(delta, time_unit), [bit], [])
        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.calibrations = dag.calibrations
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit
        return new_dag