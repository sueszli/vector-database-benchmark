"""Rescheduler pass to adjust node start times."""
from __future__ import annotations
from collections.abc import Generator
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError

class ConstrainedReschedule(AnalysisPass):
    """Rescheduler pass that updates node start times to conform to the hardware alignments.

    This pass shifts DAG node start times previously scheduled with one of
    the scheduling passes, e.g. :class:`ASAPScheduleAnalysis` or :class:`ALAPScheduleAnalysis`,
    so that every instruction start time satisfies alignment constraints.

    Examples:

        We assume executing the following circuit on a backend with 16 dt of acquire alignment.

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

    Notes:

        Your backend may execute circuits violating these alignment constraints.
        However, you may obtain erroneous measurement result because of the
        untracked phase originating in the instruction misalignment.
    """

    def __init__(self, acquire_alignment: int=1, pulse_alignment: int=1):
        if False:
            print('Hello World!')
        'Create new rescheduler pass.\n\n        The alignment values depend on the control electronics of your quantum processor.\n\n        Args:\n            acquire_alignment: Integer number representing the minimum time resolution to\n                trigger acquisition instruction in units of ``dt``.\n            pulse_alignment: Integer number representing the minimum time resolution to\n                trigger gate instruction in units of ``dt``.\n        '
        super().__init__()
        self.acquire_align = acquire_alignment
        self.pulse_align = pulse_alignment

    @classmethod
    def _get_next_gate(cls, dag: DAGCircuit, node: DAGOpNode) -> Generator[DAGOpNode, None, None]:
        if False:
            print('Hello World!')
        'Get next non-delay nodes.\n\n        Args:\n            dag: DAG circuit to be rescheduled with constraints.\n            node: Current node.\n\n        Returns:\n            A list of non-delay successors.\n        '
        for next_node in dag.successors(node):
            if not isinstance(next_node, DAGOutNode):
                yield next_node

    def _push_node_back(self, dag: DAGCircuit, node: DAGOpNode):
        if False:
            i = 10
            return i + 15
        'Update the start time of the current node to satisfy alignment constraints.\n        Immediate successors are pushed back to avoid overlap and will be processed later.\n\n        .. note::\n\n            This logic assumes the all bits in the qregs and cregs synchronously start and end,\n            i.e. occupy the same time slot, but qregs and cregs can take\n            different time slot due to classical I/O latencies.\n\n        Args:\n            dag: DAG circuit to be rescheduled with constraints.\n            node: Current node.\n        '
        node_start_time = self.property_set['node_start_time']
        conditional_latency = self.property_set.get('conditional_latency', 0)
        clbit_write_latency = self.property_set.get('clbit_write_latency', 0)
        if isinstance(node.op, Gate):
            alignment = self.pulse_align
        elif isinstance(node.op, Measure):
            alignment = self.acquire_align
        elif isinstance(node.op, Delay) or getattr(node.op, '_directive', False):
            alignment = None
        else:
            raise TranspilerError(f'Unknown operation type for {repr(node)}.')
        this_t0 = node_start_time[node]
        if alignment is not None:
            misalignment = node_start_time[node] % alignment
            if misalignment != 0:
                shift = max(0, alignment - misalignment)
            else:
                shift = 0
            this_t0 += shift
            node_start_time[node] = this_t0
        new_t1q = this_t0 + node.op.duration
        this_qubits = set(node.qargs)
        if isinstance(node.op, Measure):
            new_t1c = new_t1q
            this_clbits = set(node.cargs)
        elif node.op.condition_bits:
            new_t1c = this_t0
            this_clbits = set(node.op.condition_bits)
        else:
            new_t1c = None
            this_clbits = set()
        for next_node in self._get_next_gate(dag, node):
            next_t0q = node_start_time[next_node]
            next_qubits = set(next_node.qargs)
            if isinstance(next_node.op, Measure):
                next_t0c = next_t0q + clbit_write_latency
                next_clbits = set(next_node.cargs)
            elif next_node.op.condition_bits:
                next_t0c = next_t0q - conditional_latency
                next_clbits = set(next_node.op.condition_bits)
            else:
                next_t0c = None
                next_clbits = set()
            if any(this_qubits & next_qubits):
                qreg_overlap = new_t1q - next_t0q
            else:
                qreg_overlap = 0
            if any(this_clbits & next_clbits):
                creg_overlap = new_t1c - next_t0c
            else:
                creg_overlap = 0
            overlap = max(qreg_overlap, creg_overlap)
            node_start_time[next_node] = node_start_time[next_node] + overlap

    def run(self, dag: DAGCircuit):
        if False:
            for i in range(10):
                print('nop')
        "Run rescheduler.\n\n        This pass should perform rescheduling to satisfy:\n\n            - All DAGOpNode nodes (except for compiler directives) are placed at start time\n              satisfying hardware alignment constraints.\n            - The end time of a node does not overlap with the start time of successor nodes.\n\n        Assumptions:\n\n            - Topological order and absolute time order of DAGOpNode are consistent.\n            - All bits in either qargs or cargs associated with node synchronously start.\n            - Start time of qargs and cargs may different due to I/O latency.\n\n        Based on the configurations above, the rescheduler pass takes the following strategy:\n\n        1. The nodes are processed in the topological order, from the beginning of\n            the circuit (i.e. from left to right). For every node (including compiler\n            directives), the function ``_push_node_back`` performs steps 2 and 3.\n        2. If the start time of the node violates the alignment constraint,\n            the start time is increased to satisfy the constraint.\n        3. Each immediate successor whose start_time overlaps the node's end_time is\n            pushed backwards (towards the end of the wire). Note that at this point\n            the shifted successor does not need to satisfy the constraints, but this\n            will be taken care of when that successor node itself is processed.\n        4. After every node is processed, all misalignment constraints will be resolved,\n            and there will be no overlap between the nodes.\n\n        Args:\n            dag: DAG circuit to be rescheduled with constraints.\n\n        Raises:\n            TranspilerError: If circuit is not scheduled.\n        "
        if 'node_start_time' not in self.property_set:
            raise TranspilerError(f'The input circuit {dag.name} is not scheduled. Call one of scheduling passes before running the {self.__class__.__name__} pass.')
        node_start_time = self.property_set['node_start_time']
        for node in dag.topological_op_nodes():
            start_time = node_start_time.get(node)
            if start_time is None:
                raise TranspilerError(f'Start time of {repr(node)} is not found. This node is likely added after this circuit is scheduled. Run scheduler again.')
            if start_time == 0:
                continue
            self._push_node_back(dag, node)