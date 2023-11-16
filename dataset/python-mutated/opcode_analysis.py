from __future__ import annotations
import dataclasses
from enum import Enum
from ...utils import InnerError, OrderedSet
from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL, UNCONDITIONAL_JUMP

@dataclasses.dataclass
class State:
    reads: OrderedSet[str]
    writes: OrderedSet[str]
    visited: OrderedSet[int]

def is_read_opcode(opname):
    if False:
        return 10
    if opname in ['LOAD_FAST', 'LOAD_DEREF', 'LOAD_NAME', 'LOAD_GLOBAL', 'LOAD_CLOSURE']:
        return True
    if opname in ('DELETE_FAST', 'DELETE_DEREF', 'DELETE_NAME', 'DELETE_GLOBAL'):
        return True
    return False

def is_write_opcode(opname):
    if False:
        print('Hello World!')
    if opname in ['STORE_FAST', 'STORE_NAME', 'STORE_DEREF', 'STORE_GLOBAL']:
        return True
    if opname in ('DELETE_FAST', 'DELETE_DEREF', 'DELETE_NAME', 'DELETE_GLOBAL'):
        return True
    return False

def analysis_inputs(instructions: list[Instruction], current_instr_idx: int, stop_instr_idx: int | None=None) -> OrderedSet[str]:
    if False:
        print('Hello World!')
    '\n    Analyze the inputs of the instructions from current_instr_idx to stop_instr_idx.\n\n    Args:\n        instructions (list[Instruction]): The instructions to analyze.\n        current_instr_idx (int): The index of the current instruction.\n        stop_instr_idx (int | None, optional): The index of the instruction to stop. Defaults to None.\n            If None, the analysis will stop at the end of the instructions.\n\n    Returns:\n        set[str]: The analysis result.\n    '
    root_state = State(OrderedSet(), OrderedSet(), OrderedSet())

    def fork(state: State, start: int, jump: bool, jump_target: int) -> OrderedSet[str]:
        if False:
            i = 10
            return i + 15
        new_start = start + 1 if not jump else jump_target
        new_state = State(OrderedSet(state.reads), OrderedSet(state.writes), OrderedSet(state.visited))
        return walk(new_state, new_start)

    def walk(state: State, start: int) -> OrderedSet[str]:
        if False:
            return 10
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state.reads
            state.visited.add(i)
            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in state.writes:
                    state.reads.add(instr.argval)
                elif is_write_opcode(instr.opname):
                    state.writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                jump_branch = fork(state, i, True, target_idx)
                not_jump_branch = fork(state, i, False, target_idx) if instr.opname not in UNCONDITIONAL_JUMP else OrderedSet()
                return jump_branch | not_jump_branch
            elif instr.opname == 'RETURN_VALUE':
                return state.reads
        return state.reads
    return walk(root_state, current_instr_idx)

@dataclasses.dataclass
class SpaceState:
    reads: dict[str, Space]
    writes: dict[str, Space]
    visited: OrderedSet[int]

    def __or__(self, other):
        if False:
            print('Hello World!')
        reads = {}
        reads.update(other.reads)
        reads.update(self.reads)
        writes = {}
        writes.update(other.writes)
        writes.update(self.writes)
        return SpaceState(reads, writes, OrderedSet())

class Space(Enum):
    locals = 1
    globals = 2
    cells = 3
    all = 4

def get_space(opname: str):
    if False:
        for i in range(10):
            print('nop')
    if 'FAST' in opname:
        return Space.locals
    elif 'GLOBAL' in opname:
        return Space.globals
    elif 'DEREF' in opname or 'CLOSURE' in opname:
        return Space.cells
    elif 'NAME' in opname:
        return Space.all
    else:
        raise InnerError(f'Unknown space for {opname}')

def analysis_used_names_with_space(instructions: list[Instruction], start_instr_idx: int, stop_instr_idx: int | None=None):
    if False:
        print('Hello World!')
    root_state = SpaceState({}, {}, OrderedSet())

    def fork(state: SpaceState, start: int, jump: bool, jump_target: int) -> SpaceState:
        if False:
            i = 10
            return i + 15
        new_start = start + 1 if not jump else jump_target
        new_state = SpaceState(dict(state.reads), dict(state.writes), OrderedSet(state.visited))
        return walk(new_state, new_start)

    def walk(state: SpaceState, start: int) -> SpaceState:
        if False:
            while True:
                i = 10
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state
            state.visited.add(i)
            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in state.writes:
                    space = get_space(instr.opname)
                    state.reads[instr.argval] = space
                elif is_write_opcode(instr.opname):
                    space = get_space(instr.opname)
                    state.writes[instr.argval] = space
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                jump_branch = fork(state, i, True, target_idx)
                not_jump_branch = fork(state, i, False, target_idx) if instr.opname not in UNCONDITIONAL_JUMP else SpaceState({}, {}, OrderedSet())
                return jump_branch | not_jump_branch
            elif instr.opname == 'RETURN_VALUE':
                return state
        return state
    state = walk(root_state, start_instr_idx)
    all_used_vars = {}
    all_used_vars.update(state.writes)
    all_used_vars.update(state.reads)
    return all_used_vars