"""Padding pass to insert Delay to the empty slots."""
from qiskit.circuit import Qubit
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOutNode
from qiskit.transpiler.target import Target
from .base_padding import BasePadding

class PadDelay(BasePadding):
    """Padding idle time with Delay instructions.

    Consecutive delays will be merged in the output of this pass.

    .. code-block:: python

        durations = InstructionDurations([("x", None, 160), ("cx", None, 800)])

        qc = QuantumCircuit(2)
        qc.delay(100, 0)
        qc.x(1)
        qc.cx(0, 1)

    The ASAP-scheduled circuit output may become

    .. parsed-literal::

             ┌────────────────┐
        q_0: ┤ Delay(160[dt]) ├──■──
             └─────┬───┬──────┘┌─┴─┐
        q_1: ──────┤ X ├───────┤ X ├
                   └───┘       └───┘

    Note that the additional idle time of 60dt on the ``q_0`` wire coming from the duration difference
    between ``Delay`` of 100dt (``q_0``) and ``XGate`` of 160 dt (``q_1``) is absorbed in
    the delay instruction on the ``q_0`` wire, i.e. in total 160 dt.

    See :class:`BasePadding` pass for details.
    """

    def __init__(self, fill_very_end: bool=True, target: Target=None):
        if False:
            i = 10
            return i + 15
        'Create new padding delay pass.\n\n        Args:\n            fill_very_end: Set ``True`` to fill the end of circuit with delay.\n            target: The :class:`~.Target` representing the target backend.\n                If it is supplied and does not support delay instruction on a qubit,\n                padding passes do not pad any idle time of the qubit.\n        '
        super().__init__(target=target)
        self.fill_very_end = fill_very_end

    def _pad(self, dag: DAGCircuit, qubit: Qubit, t_start: int, t_end: int, next_node: DAGNode, prev_node: DAGNode):
        if False:
            i = 10
            return i + 15
        if not self.fill_very_end and isinstance(next_node, DAGOutNode):
            return
        time_interval = t_end - t_start
        self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag.unit), qubit)