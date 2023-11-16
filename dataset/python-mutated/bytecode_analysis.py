import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
TERMINAL_OPCODES = {dis.opmap['RETURN_VALUE'], dis.opmap['JUMP_FORWARD'], dis.opmap['RAISE_VARARGS']}
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap['RERAISE'])
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap['JUMP_BACKWARD'])
    TERMINAL_OPCODES.add(dis.opmap['JUMP_FORWARD'])
else:
    TERMINAL_OPCODES.add(dis.opmap['JUMP_ABSOLUTE'])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)
stack_effect = dis.stack_effect

def get_indexof(insts):
    if False:
        return 10
    '\n    Get a mapping from instruction memory address to index in instruction list.\n    Additionally checks that each instruction only appears once in the list.\n    '
    indexof = {}
    for (i, inst) in enumerate(insts):
        assert inst not in indexof
        indexof[inst] = i
    return indexof

def remove_dead_code(instructions):
    if False:
        while True:
            i = 10
    'Dead code elimination'
    indexof = get_indexof(instructions)
    live_code = set()

    def find_live_code(start):
        if False:
            print('Hello World!')
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.exn_tab_entry:
                find_live_code(indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[inst.target])
            if inst.opcode in TERMINAL_OPCODES:
                return
    find_live_code(0)
    if sys.version_info >= (3, 11):
        live_idx = sorted(live_code)
        for (i, inst) in enumerate(instructions):
            if i in live_code and inst.exn_tab_entry:
                start_idx = bisect.bisect_left(live_idx, indexof[inst.exn_tab_entry.start])
                assert start_idx < len(live_idx)
                end_idx = bisect.bisect_right(live_idx, indexof[inst.exn_tab_entry.end]) - 1
                assert end_idx >= 0
                assert live_idx[start_idx] <= i <= live_idx[end_idx]
                inst.exn_tab_entry.start = instructions[live_idx[start_idx]]
                inst.exn_tab_entry.end = instructions[live_idx[end_idx]]
    return [inst for (i, inst) in enumerate(instructions) if i in live_code]

def remove_pointless_jumps(instructions):
    if False:
        for i in range(10):
            print('nop')
    'Eliminate jumps to the next instruction'
    pointless_jumps = {id(a) for (a, b) in zip(instructions, instructions[1:]) if a.opname == 'JUMP_ABSOLUTE' and a.target is b}
    return [inst for inst in instructions if id(inst) not in pointless_jumps]

def propagate_line_nums(instructions):
    if False:
        i = 10
        return i + 15
    'Ensure every instruction has line number set in case some are removed'
    cur_line_no = None

    def populate_line_num(inst):
        if False:
            while True:
                i = 10
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line
        inst.starts_line = cur_line_no
    for inst in instructions:
        populate_line_num(inst)

def remove_extra_line_nums(instructions):
    if False:
        i = 10
        return i + 15
    'Remove extra starts line properties before packing bytecode'
    cur_line_no = None

    def remove_line_num(inst):
        if False:
            for i in range(10):
                print('nop')
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line
    for inst in instructions:
        remove_line_num(inst)

@dataclasses.dataclass
class ReadsWrites:
    reads: Set[Any]
    writes: Set[Any]
    visited: Set[Any]

def livevars_analysis(instructions, instruction):
    if False:
        return 10
    indexof = get_indexof(instructions)
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state, start):
        if False:
            return 10
        if start in state.visited:
            return
        state.visited.add(start)
        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if 'LOAD' in inst.opname or 'DELETE' in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif 'STORE' in inst.opname:
                    state.writes.add(inst.argval)
                elif inst.opname == 'MAKE_CELL':
                    pass
                else:
                    raise NotImplementedError(f'unhandled {inst.opname}')
            if inst.exn_tab_entry:
                walk(may, indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                walk(may, indexof[inst.target])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return
    walk(must, indexof[instruction])
    return must.reads | may.reads

@dataclasses.dataclass
class FixedPointBox:
    value: bool = True

@dataclasses.dataclass
class StackSize:
    low: Union[int, float]
    high: Union[int, float]
    fixed_point: FixedPointBox

    def zero(self):
        if False:
            i = 10
            return i + 15
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other, n):
        if False:
            for i in range(10):
                print('nop')
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False

    def exn_tab_jump(self, depth):
        if False:
            for i in range(10):
                print('nop')
        prior = (self.low, self.high)
        self.low = min(self.low, depth)
        self.high = max(self.high, depth)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False

def stacksize_analysis(instructions) -> Union[int, float]:
    if False:
        print('Hello World!')
    assert instructions
    fixed_point = FixedPointBox()
    stack_sizes = {inst: StackSize(float('inf'), float('-inf'), fixed_point) for inst in instructions}
    stack_sizes[instructions[0]].zero()
    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True
        for (inst, next_inst) in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            is_call_finally = sys.version_info < (3, 9) and inst.opcode == dis.opmap['CALL_FINALLY']
            if inst.opcode not in TERMINAL_OPCODES:
                assert next_inst is not None, f'missing next inst: {inst}'
                stack_sizes[next_inst].offset_of(stack_size, stack_effect(inst.opcode, inst.arg, jump=is_call_finally))
            if inst.opcode in JUMP_OPCODES and (not is_call_finally):
                stack_sizes[inst.target].offset_of(stack_size, stack_effect(inst.opcode, inst.arg, jump=True))
            if inst.exn_tab_entry:
                depth = inst.exn_tab_entry.depth + int(inst.exn_tab_entry.lasti) + 1
                stack_sizes[inst.exn_tab_entry.target].exn_tab_jump(depth)
    if False:
        for inst in instructions:
            stack_size = stack_sizes[inst]
            print(stack_size.low, stack_size.high, inst)
    low = min([x.low for x in stack_sizes.values()])
    high = max([x.high for x in stack_sizes.values()])
    assert fixed_point.value, 'failed to reach fixed point'
    assert low >= 0
    return high