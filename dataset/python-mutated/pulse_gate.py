"""Instruction schedule map reference pass."""
from typing import List, Union
from qiskit.circuit import Instruction as CircuitInst
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from .base_builder import CalibrationBuilder

class PulseGates(CalibrationBuilder):
    """Pulse gate adding pass.

    This pass adds gate calibrations from the supplied ``InstructionScheduleMap``
    to a quantum circuit.

    This pass checks each DAG circuit node and acquires a corresponding schedule from
    the instruction schedule map object that may be provided by the target backend.
    Because this map is a mutable object, the end-user can provide a configured backend to
    execute the circuit with customized gate implementations.

    This mapping object returns a schedule with "publisher" metadata which is an integer Enum
    value representing who created the gate schedule.
    If the gate schedule is provided by end-users, this pass attaches the schedule to
    the DAG circuit as a calibration.

    This pass allows users to easily override quantum circuit with custom gate definitions
    without directly dealing with those schedules.

    References
        * [1] OpenQASM 3: A broader and deeper quantum assembly language
          https://arxiv.org/abs/2104.14722
    """

    def __init__(self, inst_map: InstructionScheduleMap=None, target: Target=None):
        if False:
            for i in range(10):
                print('nop')
        'Create new pass.\n\n        Args:\n            inst_map: Instruction schedule map that user may override.\n            target: The :class:`~.Target` representing the target backend, if both\n                ``inst_map`` and ``target`` are specified then it updates instructions\n                in the ``target`` with ``inst_map``.\n        '
        super().__init__()
        if inst_map is None and target is None:
            raise TranspilerError('inst_map and target cannot be None simulataneously.')
        if target is None:
            target = Target()
            target.update_from_instruction_schedule_map(inst_map)
        self.target = target

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Determine if a given node supports the calibration.\n\n        Args:\n            node_op: Target instruction object.\n            qubits: Integer qubit indices to check.\n\n        Returns:\n            Return ``True`` is calibration can be provided.\n        '
        return self.target.has_calibration(node_op.name, tuple(qubits))

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        if False:
            while True:
                i = 10
        'Gets the calibrated schedule for the given instruction and qubits.\n\n        Args:\n            node_op: Target instruction object.\n            qubits: Integer qubit indices to check.\n\n        Returns:\n            Return Schedule of target gate instruction.\n\n        Raises:\n            TranspilerError: When node is parameterized and calibration is raw schedule object.\n        '
        return self.target.get_calibration(node_op.name, tuple(qubits), *node_op.params)