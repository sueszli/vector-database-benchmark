"""Unify time unit in circuit for scheduling and following passes."""
from typing import Set
from qiskit.circuit import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.target import Target

class TimeUnitConversion(TransformationPass):
    """Choose a time unit to be used in the following time-aware passes,
    and make all circuit time units consistent with that.

    This pass will add a :attr:`.Instruction.duration` metadata to each op whose duration is known
    which will be used by subsequent scheduling passes for scheduling.

    If ``dt`` (in seconds) is known to transpiler, the unit ``'dt'`` is chosen. Otherwise,
    the unit to be selected depends on what units are used in delays and instruction durations:

    * ``'s'``: if they are all in SI units.
    * ``'dt'``: if they are all in the unit ``'dt'``.
    * raise error: if they are a mix of SI units and ``'dt'``.
    """

    def __init__(self, inst_durations: InstructionDurations=None, target: Target=None):
        if False:
            return 10
        'TimeUnitAnalysis initializer.\n\n        Args:\n            inst_durations (InstructionDurations): A dictionary of durations of instructions.\n            target: The :class:`~.Target` representing the target backend, if both\n                  ``inst_durations`` and ``target`` are specified then this argument will take\n                  precedence and ``inst_durations`` will be ignored.\n\n\n        '
        super().__init__()
        self.inst_durations = inst_durations or InstructionDurations()
        if target is not None:
            self.inst_durations = target.durations()

    def run(self, dag: DAGCircuit):
        if False:
            i = 10
            return i + 15
        'Run the TimeUnitAnalysis pass on `dag`.\n\n        Args:\n            dag (DAGCircuit): DAG to be checked.\n\n        Returns:\n            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.\n\n        Raises:\n            TranspilerError: if the units are not unifiable\n        '
        if self.inst_durations.dt is not None:
            time_unit = 'dt'
        else:
            units_delay = self._units_used_in_delays(dag)
            if self._unified(units_delay) == 'mixed':
                raise TranspilerError('Fail to unify time units in delays. SI units and dt unit must not be mixed when dt is not supplied.')
            units_other = self.inst_durations.units_used()
            if self._unified(units_other) == 'mixed':
                raise TranspilerError('Fail to unify time units in instruction_durations. SI units and dt unit must not be mixed when dt is not supplied.')
            unified_unit = self._unified(units_delay | units_other)
            if unified_unit == 'SI':
                time_unit = 's'
            elif unified_unit == 'dt':
                time_unit = 'dt'
            else:
                raise TranspilerError('Fail to unify time units. SI units and dt unit must not be mixed when dt is not supplied.')
        for node in dag.op_nodes():
            try:
                duration = self.inst_durations.get(node.op, [dag.find_bit(qarg).index for qarg in node.qargs], unit=time_unit)
            except TranspilerError:
                continue
            node.op = node.op.to_mutable()
            node.op.duration = duration
            node.op.unit = time_unit
        self.property_set['time_unit'] = time_unit
        return dag

    @staticmethod
    def _units_used_in_delays(dag: DAGCircuit) -> Set[str]:
        if False:
            i = 10
            return i + 15
        units_used = set()
        for node in dag.op_nodes(op=Delay):
            units_used.add(node.op.unit)
        return units_used

    @staticmethod
    def _unified(unit_set: Set[str]) -> str:
        if False:
            print('Hello World!')
        if not unit_set:
            return 'dt'
        if len(unit_set) == 1 and 'dt' in unit_set:
            return 'dt'
        all_si = True
        for unit in unit_set:
            if not unit.endswith('s'):
                all_si = False
                break
        if all_si:
            return 'SI'
        return 'mixed'