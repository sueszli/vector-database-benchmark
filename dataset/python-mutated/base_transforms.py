"""A collection of set of transforms."""
from typing import Union, Iterable, Tuple
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.schedule import ScheduleBlock, Schedule
from qiskit.pulse.transforms import canonicalization
InstructionSched = Union[Tuple[int, Instruction], Instruction]

def target_qobj_transform(sched: Union[ScheduleBlock, Schedule, InstructionSched, Iterable[InstructionSched]], remove_directives: bool=True) -> Schedule:
    if False:
        print('Hello World!')
    'A basic pulse program transformation for OpenPulse API execution.\n\n    Args:\n        sched: Input program to transform.\n        remove_directives: Set `True` to remove compiler directives.\n\n    Returns:\n        Transformed program for execution.\n    '
    if not isinstance(sched, Schedule):
        if isinstance(sched, ScheduleBlock):
            sched = canonicalization.block_to_schedule(sched)
        else:
            sched = Schedule(*_format_schedule_component(sched))
    sched = canonicalization.inline_subroutines(sched)
    sched = canonicalization.flatten(sched)
    if remove_directives:
        sched = canonicalization.remove_directives(sched)
    return sched

def _format_schedule_component(sched: Union[InstructionSched, Iterable[InstructionSched]]):
    if False:
        print('Hello World!')
    'A helper function to convert instructions into list of instructions.'
    try:
        sched = list(sched)
        if isinstance(sched[0], int):
            return [tuple(sched)]
        else:
            return sched
    except TypeError:
        return [sched]