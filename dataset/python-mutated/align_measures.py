"""Align measurement instructions."""
from __future__ import annotations
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QubitSpecifier
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecation import deprecate_func

class AlignMeasures(TransformationPass):
    """Measurement alignment.

    This is a control electronics aware optimization pass.

    In many quantum computing architectures gates (instructions) are implemented with
    shaped analog stimulus signals. These signals are digitally stored in the
    waveform memory of the control electronics and converted into analog voltage signals
    by electronic components called digital to analog converters (DAC).

    In a typical hardware implementation of superconducting quantum processors,
    a single qubit instruction is implemented by a
    microwave signal with the duration of around several tens of ns with a per-sample
    time resolution of ~0.1-10ns, as reported by ``backend.configuration().dt``.
    In such systems requiring higher DAC bandwidth, control electronics often
    defines a `pulse granularity`, in other words a data chunk, to allow the DAC to
    perform the signal conversion in parallel to gain the bandwidth.

    Measurement alignment is required if a backend only allows triggering ``measure``
    instructions at a certain multiple value of this pulse granularity.
    This value is usually provided by ``backend.configuration().timing_constraints``.

    In Qiskit SDK, the duration of delay can take arbitrary value in units of ``dt``,
    thus circuits involving delays may violate the above alignment constraint (i.e. misalignment).
    This pass shifts measurement instructions to a new time position to fix the misalignment,
    by inserting extra delay right before the measure instructions.
    The input of this pass should be scheduled :class:`~qiskit.dagcircuit.DAGCircuit`,
    thus one should select one of the scheduling passes
    (:class:`~qiskit.transpiler.passes.ALAPSchedule` or
    :class:`~qiskit.trasnpiler.passes.ASAPSchedule`) before calling this.

    Examples:
        We assume executing the following circuit on a backend with ``alignment=16``.

        .. parsed-literal::

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

        Note that delay of 100 dt induces a misalignment of 4 dt at the measurement.
        This pass appends an extra 12 dt time shift to the input circuit.

        .. parsed-literal::

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(112[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

        This pass always inserts a positive delay before measurements
        rather than reducing other delays.

    Notes:
        The Backend may allow users to execute circuits violating the alignment constraint.
        However, it may return meaningless measurement data mainly due to the phase error.
    """

    @deprecate_func(additional_msg='Instead, use :class:`~.ConstrainedReschedule`, which performs the same function but also supports aligning to additional timing constraints.', since='0.21.0', pending=True)
    def __init__(self, alignment: int=1):
        if False:
            while True:
                i = 10
        'Create new pass.\n\n        Args:\n            alignment: Integer number representing the minimum time resolution to\n                trigger measure instruction in units of ``dt``. This value depends on\n                the control electronics of your quantum processor.\n        '
        super().__init__()
        self.alignment = alignment

    def run(self, dag: DAGCircuit):
        if False:
            print('Hello World!')
        'Run the measurement alignment pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): DAG to be checked.\n\n        Returns:\n            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.\n\n        Raises:\n            TranspilerError: If circuit is not scheduled.\n        '
        time_unit = self.property_set['time_unit']
        if not _check_alignment_required(dag, self.alignment, Measure):
            return dag
        if dag.duration is None:
            raise TranspilerError(f"This circuit {dag.name} may involve a delay instruction violating the pulse controller alignment. To adjust instructions to right timing, you should call one of scheduling passes first. This is usually done by calling transpiler with scheduling_method='alap'.")
        new_dag = dag.copy_empty_like()
        qubit_time_available: dict[QubitSpecifier, int] = defaultdict(int)
        qubit_stop_times: dict[QubitSpecifier, int] = defaultdict(int)
        clbit_readable: dict[ClbitSpecifier, int] = defaultdict(int)
        clbit_writeable: dict[ClbitSpecifier, int] = defaultdict(int)

        def pad_with_delays(qubits: Iterable[QubitSpecifier], until, unit) -> None:
            if False:
                print('Hello World!')
            'Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``.'
            for q in qubits:
                if qubit_stop_times[q] < until:
                    idle_duration = until - qubit_stop_times[q]
                    new_dag.apply_operation_back(Delay(idle_duration, unit), (q,), check=False)
        for node in dag.topological_op_nodes():
            clbit_time_available = clbit_writeable if isinstance(node.op, Measure) else clbit_readable
            delta = node.op.duration if isinstance(node.op, Measure) else 0
            start_time = max(itertools.chain((qubit_time_available[q] for q in node.qargs), (clbit_time_available[c] - delta for c in node.cargs + tuple(node.op.condition_bits))))
            if isinstance(node.op, Measure):
                if start_time % self.alignment != 0:
                    start_time = (start_time // self.alignment + 1) * self.alignment
            if not isinstance(node.op, Delay):
                pad_with_delays(node.qargs, until=start_time, unit=time_unit)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
            stop_time = start_time + node.op.duration
            for q in node.qargs:
                qubit_time_available[q] = stop_time
                if not isinstance(node.op, Delay):
                    qubit_stop_times[q] = stop_time
            for c in node.cargs:
                clbit_writeable[c] = clbit_readable[c] = stop_time
            for c in node.op.condition_bits:
                clbit_writeable[c] = max(start_time, clbit_writeable[c])
        working_qubits = qubit_time_available.keys()
        circuit_duration = max((qubit_time_available[q] for q in working_qubits))
        pad_with_delays(new_dag.qubits, until=circuit_duration, unit=time_unit)
        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit
        return new_dag

def _check_alignment_required(dag: DAGCircuit, alignment: int, instructions: Type | list[Type]) -> bool:
    if False:
        while True:
            i = 10
    'Check DAG nodes and return a boolean representing if instruction scheduling is necessary.\n\n    Args:\n        dag: DAG circuit to check.\n        alignment: Instruction alignment condition.\n        instructions: Target instructions.\n\n    Returns:\n        If instruction scheduling is necessary.\n    '
    if not isinstance(instructions, list):
        instructions = [instructions]
    if alignment == 1:
        return False
    if all((len(dag.op_nodes(inst)) == 0 for inst in instructions)):
        return False
    for delay_node in dag.op_nodes(Delay):
        duration = delay_node.op.duration
        if isinstance(duration, ParameterExpression):
            warnings.warn(f'Parametrized delay with {repr(duration)} is found in circuit {dag.name}. This backend requires alignment={alignment}. Please make sure all assigned values are multiple values of the alignment.', UserWarning)
        elif duration % alignment != 0:
            return True
    return False