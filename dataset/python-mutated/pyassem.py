"""A flow graph representation for Python bytecode"""
from __future__ import annotations
import sys
from contextlib import contextmanager
from types import CodeType
from typing import ClassVar, Generator, List, Optional
from . import opcode_cinder, opcodes
from .consts import CO_ASYNC_GENERATOR, CO_COROUTINE, CO_GENERATOR, CO_NEWLOCALS, CO_OPTIMIZED, CO_SUPPRESS_JIT
from .flow_graph_optimizer import FlowGraphOptimizer
from .opcodebase import Opcode
MAX_COPY_SIZE = 4

def sign(a):
    if False:
        print('Hello World!')
    if not isinstance(a, float):
        raise TypeError(f'Must be a real number, not {type(a)}')
    if a != a:
        return 1.0
    return 1.0 if str(a)[0] != '-' else -1.0

def instrsize(oparg):
    if False:
        return 10
    if oparg <= 255:
        return 1
    elif oparg <= 65535:
        return 2
    elif oparg <= 16777215:
        return 3
    else:
        return 4

def cast_signed_byte_to_unsigned(i):
    if False:
        for i in range(10):
            print('nop')
    if i < 0:
        i = 255 + i + 1
    return i
FVC_MASK = 3
FVC_NONE = 0
FVC_STR = 1
FVC_REPR = 2
FVC_ASCII = 3
FVS_MASK = 4
FVS_HAVE_SPEC = 4

class Instruction:
    __slots__ = ('opname', 'oparg', 'target', 'ioparg', 'lineno')

    def __init__(self, opname: str, oparg: object, ioparg: int=0, lineno: int=-1, target: Optional[Block]=None):
        if False:
            for i in range(10):
                print('nop')
        self.opname = opname
        self.oparg = oparg
        self.lineno = lineno
        self.ioparg = ioparg
        self.target = target

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        args = [f'{self.opname!r}', f'{self.oparg!r}', f'{self.ioparg!r}', f'{self.lineno!r}']
        if self.target is not None:
            args.append(f'{self.target!r}')
        return f"Instruction({', '.join(args)})"

    def is_jump(self, opcode: Opcode) -> bool:
        if False:
            return 10
        op = opcode.opmap[self.opname]
        return opcode.has_jump(op)

    def copy(self) -> Instruction:
        if False:
            i = 10
            return i + 15
        return Instruction(self.opname, self.oparg, self.ioparg, self.lineno, self.target)

class CompileScope:
    START_MARKER = 'compile-scope-start-marker'
    __slots__ = 'blocks'

    def __init__(self, blocks):
        if False:
            for i in range(10):
                print('nop')
        self.blocks = blocks

class FlowGraph:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.block_count = 0
        self.ordered_blocks = []
        self.current = None
        self.entry = Block('entry')
        self.startBlock(self.entry)
        self.lineno = 0
        self.firstline = 0
        self.first_inst_lineno = 0
        self.do_not_emit_bytecode = 0

    def blocks_in_reverse_allocation_order(self):
        if False:
            return 10
        for block in sorted(self.ordered_blocks, key=lambda b: b.alloc_id, reverse=True):
            yield block

    @contextmanager
    def new_compile_scope(self) -> Generator[CompileScope, None, None]:
        if False:
            while True:
                i = 10
        prev_current = self.current
        prev_ordered_blocks = self.ordered_blocks
        prev_line_no = self.first_inst_lineno
        try:
            self.ordered_blocks = []
            self.current = self.newBlock(CompileScope.START_MARKER)
            yield CompileScope(self.ordered_blocks)
        finally:
            self.current = prev_current
            self.ordered_blocks = prev_ordered_blocks
            self.first_inst_lineno = prev_line_no

    def apply_from_scope(self, scope: CompileScope):
        if False:
            i = 10
            return i + 15
        block: Block = scope.blocks[0]
        assert block.prev is not None
        assert block.prev.label == CompileScope.START_MARKER
        block.prev = None
        self.current.addNext(block)
        self.ordered_blocks.extend(scope.blocks)
        self.current = scope.blocks[-1]

    def startBlock(self, block):
        if False:
            for i in range(10):
                print('nop')
        if self._debug:
            if self.current:
                print('end', repr(self.current))
                print('    next', self.current.next)
                print('    prev', self.current.prev)
                print('   ', self.current.get_children())
            print(repr(block))
        block.bid = self.block_count
        self.block_count += 1
        self.current = block
        if block and block not in self.ordered_blocks:
            self.ordered_blocks.append(block)

    def nextBlock(self, block=None, label=''):
        if False:
            while True:
                i = 10
        if self.do_not_emit_bytecode:
            return
        if block is None:
            block = self.newBlock(label=label)
        self.current.addNext(block)
        self.startBlock(block)

    def newBlock(self, label=''):
        if False:
            i = 10
            return i + 15
        b = Block(label)
        return b
    _debug = 0

    def _enable_debug(self):
        if False:
            while True:
                i = 10
        self._debug = 1

    def _disable_debug(self):
        if False:
            for i in range(10):
                print('nop')
        self._debug = 0

    def emit(self, opcode: str, oparg: object=0, lineno: int | None=None) -> None:
        if False:
            return 10
        if lineno is None:
            lineno = self.lineno
        if isinstance(oparg, Block):
            if not self.do_not_emit_bytecode:
                self.current.addOutEdge(oparg)
                self.current.emit(Instruction(opcode, 0, 0, lineno, target=oparg))
            return
        ioparg = self.convertArg(opcode, oparg)
        if not self.do_not_emit_bytecode:
            self.current.emit(Instruction(opcode, oparg, ioparg, lineno))

    def emit_noline(self, opcode: str, oparg: object=0):
        if False:
            i = 10
            return i + 15
        self.emit(opcode, oparg, -1)

    def emitWithBlock(self, opcode: str, oparg: object, target: Block):
        if False:
            while True:
                i = 10
        if not self.do_not_emit_bytecode:
            self.current.addOutEdge(target)
            self.current.emit(Instruction(opcode, oparg, target=target))

    def set_lineno(self, lineno: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.first_inst_lineno:
            self.first_inst_lineno = lineno
        self.lineno = lineno

    def convertArg(self, opcode: str, oparg: object) -> int:
        if False:
            print('Hello World!')
        if isinstance(oparg, int):
            return oparg
        raise ValueError(f'invalid oparg {oparg!r} for {opcode!r}')

    def getBlocksInOrder(self):
        if False:
            return 10
        'Return the blocks in the order they should be output.'
        return self.ordered_blocks

    def getBlocks(self):
        if False:
            print('Hello World!')
        return self.ordered_blocks

    def getRoot(self):
        if False:
            return 10
        'Return nodes appropriate for use with dominator'
        return self.entry

    def getContainedGraphs(self):
        if False:
            return 10
        result = []
        for b in self.getBlocks():
            result.extend(b.getContainedGraphs())
        return result

class Block:
    allocated_block_count: ClassVar[int] = 0

    def __init__(self, label=''):
        if False:
            print('Hello World!')
        self.insts: List[Instruction] = []
        self.outEdges = set()
        self.label: str = label
        self.bid: int | None = None
        self.next: Block | None = None
        self.prev: Block | None = None
        self.returns: bool = False
        self.offset: int = 0
        self.seen: bool = False
        self.startdepth: int = -1
        self.is_exit: bool = False
        self.no_fallthrough: bool = False
        self.num_predecessors: int = 0
        self.alloc_id: int = Block.allocated_block_count
        Block.allocated_block_count += 1

    def __repr__(self):
        if False:
            while True:
                i = 10
        data = []
        data.append(f'id={self.bid}')
        data.append(f'startdepth={self.startdepth}')
        if self.next:
            data.append(f'next={self.next.bid}')
        extras = ', '.join(data)
        if self.label:
            return f'<block {self.label} {extras}>'
        else:
            return f'<block {extras}>'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        insts = map(str, self.insts)
        insts = '\n'.join(insts)
        return f'<block label={self.label} bid={self.bid} startdepth={self.startdepth}: {insts}>'

    def emit(self, instr: Instruction) -> None:
        if False:
            i = 10
            return i + 15
        if instr.opname in ('RETURN_VALUE', 'RETURN_PRIMITIVE'):
            self.returns = True
        self.insts.append(instr)

    def getInstructions(self):
        if False:
            while True:
                i = 10
        return self.insts

    def addOutEdge(self, block):
        if False:
            return 10
        self.outEdges.add(block)

    def addNext(self, block):
        if False:
            for i in range(10):
                print('nop')
        assert self.next is None, self.next
        self.next = block
        assert block.prev is None, block.prev
        block.prev = self

    def removeNext(self):
        if False:
            print('Hello World!')
        assert self.next is not None
        next = self.next
        next.prev = None
        self.next = None

    def has_return(self):
        if False:
            print('Hello World!')
        return self.insts and self.insts[-1].opname in ('RETURN_VALUE', 'RETURN_PRIMITIVE')

    def get_children(self):
        if False:
            i = 10
            return i + 15
        return list(self.outEdges) + ([self.next] if self.next is not None else [])

    def getContainedGraphs(self):
        if False:
            for i in range(10):
                print('nop')
        'Return all graphs contained within this block.\n\n        For example, a MAKE_FUNCTION block will contain a reference to\n        the graph for the function body.\n        '
        contained = []
        for inst in self.insts:
            if len(inst) == 1:
                continue
            op = inst[1]
            if hasattr(op, 'graph'):
                contained.append(op.graph)
        return contained

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.no_fallthrough
        result = Block()
        result.insts = [instr.copy() for instr in self.insts]
        result.is_exit = self.is_exit
        result.no_fallthrough = True
        return result
ACTIVE = 'ACTIVE'
CLOSED = 'CLOSED'
CONSTS_CLOSED = 'CONSTS_CLOSED'
OPTIMIZED = 'OPTIMIZED'
ORDERED = 'ORDERED'
FINAL = 'FINAL'
FLAT = 'FLAT'
DONE = 'DONE'

class IndexedSet:
    """Container that behaves like a `set` that assigns stable dense indexes
    to each element. Put another way: This behaves like a `list` where you
    check `x in <list>` before doing any insertion to avoid duplicates. But
    contrary to the list this does not require an O(n) member check."""
    __delitem__ = None

    def __init__(self, iterable=()):
        if False:
            for i in range(10):
                print('nop')
        self.keys = {}
        for item in iterable:
            self.get_index(item)

    def __add__(self, iterable):
        if False:
            for i in range(10):
                print('nop')
        result = IndexedSet()
        for item in self.keys.keys():
            result.get_index(item)
        for item in iterable:
            result.get_index(item)
        return result

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item in self.keys

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.keys.keys())

    def __len__(self):
        if False:
            return 10
        return len(self.keys)

    def get_index(self, item):
        if False:
            while True:
                i = 10
        'Return index of name in collection, appending if necessary'
        assert type(item) is str
        idx = self.keys.get(item)
        if idx is not None:
            return idx
        idx = len(self.keys)
        self.keys[item] = idx
        return idx

    def index(self, item):
        if False:
            print('Hello World!')
        assert type(item) is str
        idx = self.keys.get(item)
        if idx is not None:
            return idx
        raise ValueError()

    def update(self, iterable):
        if False:
            i = 10
            return i + 15
        for item in iterable:
            self.get_index(item)

class PyFlowGraph(FlowGraph):
    super_init = FlowGraph.__init__
    opcode = opcodes.opcode

    def __init__(self, name: str, filename: str, scope, flags: int=0, args=(), kwonlyargs=(), starargs=(), optimized: int=0, klass: bool=False, docstring: Optional[str]=None, firstline: int=0, posonlyargs: int=0) -> None:
        if False:
            i = 10
            return i + 15
        self.super_init()
        self.name = name
        self.filename = filename
        self.scope = scope
        self.docstring = None
        self.args = args
        self.kwonlyargs = kwonlyargs
        self.posonlyargs = posonlyargs
        self.starargs = starargs
        self.klass = klass
        self.stacksize = 0
        self.docstring = docstring
        self.flags = flags
        if optimized:
            self.setFlag(CO_OPTIMIZED | CO_NEWLOCALS)
        self.consts = {}
        self.names = IndexedSet()
        if scope is not None:
            self.freevars = IndexedSet(scope.get_free_vars())
            self.cellvars = IndexedSet(scope.get_cell_vars())
        else:
            self.freevars = IndexedSet([])
            self.cellvars = IndexedSet([])
        self.closure = self.cellvars + self.freevars
        varnames = IndexedSet()
        varnames.update(args)
        varnames.update(kwonlyargs)
        varnames.update(starargs)
        self.varnames = varnames
        self.stage = ACTIVE
        self.firstline = firstline
        self.first_inst_lineno = 0
        self.lineno = 0
        self.extra_consts = []
        self.initializeConsts()
        self.fast_vars = set()
        self.gen_kind = None
        if flags & CO_COROUTINE:
            self.gen_kind = 1
        elif flags & CO_ASYNC_GENERATOR:
            self.gen_kind = 2
        elif flags & CO_GENERATOR:
            self.gen_kind = 0

    def emit_gen_start(self) -> None:
        if False:
            print('Hello World!')
        if self.gen_kind is not None:
            self.emit('GEN_START', self.gen_kind, -1)

    def setFlag(self, flag: int) -> None:
        if False:
            while True:
                i = 10
        self.flags |= flag

    def checkFlag(self, flag: int) -> Optional[int]:
        if False:
            while True:
                i = 10
        if self.flags & flag:
            return 1

    def initializeConsts(self) -> None:
        if False:
            while True:
                i = 10
        if self.name == '<lambda>':
            self.consts[self.get_const_key(None)] = 0
        elif not self.name.startswith('<') and (not self.klass):
            if self.docstring is not None:
                self.consts[self.get_const_key(self.docstring)] = 0
            else:
                self.consts[self.get_const_key(None)] = 0

    def convertArg(self, opcode: str, oparg: object) -> int:
        if False:
            for i in range(10):
                print('nop')
        assert self.stage in {ACTIVE, CLOSED}, self.stage
        if self.do_not_emit_bytecode and opcode in self._quiet_opcodes:
            return -1
        conv = self._converters.get(opcode)
        if conv is not None:
            return conv(self, oparg)
        return super().convertArg(opcode, oparg)

    def finalize(self) -> None:
        if False:
            return 10
        'Perform final optimizations and normalization of flow graph.'
        assert self.stage == ACTIVE, self.stage
        self.stage = CLOSED
        for block in self.ordered_blocks:
            self.normalize_basic_block(block)
        for block in self.blocks_in_reverse_allocation_order():
            self.extend_block(block)
        self.optimizeCFG()
        self.duplicate_exits_without_lineno()
        self.stage = CONSTS_CLOSED
        self.trim_unused_consts()
        self.propagate_line_numbers()
        self.firstline = self.firstline or self.first_inst_lineno or 1
        self.guarantee_lineno_for_exits()
        self.stage = ORDERED
        self.normalize_jumps()
        self.stage = FINAL

    def getCode(self):
        if False:
            while True:
                i = 10
        'Get a Python code object'
        self.finalize()
        assert self.stage == FINAL, self.stage
        self.computeStackDepth()
        self.flattenGraph()
        assert self.stage == FLAT, self.stage
        self.makeByteCode()
        assert self.stage == DONE, self.stage
        code = self.newCodeObject()
        return code

    def dump(self, io=None):
        if False:
            return 10
        if io:
            save = sys.stdout
            sys.stdout = io
        pc = 0
        for block in self.getBlocks():
            print(repr(block))
            for instr in block.getInstructions():
                opname = instr.opname
                if instr.target is None:
                    print('\t', f'{pc:3} {instr.lineno} {opname} {instr.oparg}')
                elif instr.target.label:
                    print('\t', f'{pc:3} {instr.lineno} {opname} {instr.target.bid} ({instr.target.label})')
                else:
                    print('\t', f'{pc:3} {instr.lineno} {opname} {instr.target.bid}')
                pc += self.opcode.CODEUNIT_SIZE
        if io:
            sys.stdout = save

    def push_block(self, worklist: List[Block], block: Block, depth: int):
        if False:
            i = 10
            return i + 15
        assert block.startdepth < 0 or block.startdepth >= depth, f'{block!r}: {block.startdepth} vs {depth}'
        if block.startdepth < depth:
            block.startdepth = depth
            worklist.append(block)

    def stackdepth_walk(self, block):
        if False:
            print('Hello World!')
        maxdepth = 0
        worklist = []
        self.push_block(worklist, block, 0 if self.gen_kind is None else 1)
        while worklist:
            block = worklist.pop()
            next = block.next
            depth = block.startdepth
            assert depth >= 0
            for instr in block.getInstructions():
                delta = self.opcode.stack_effect_raw(instr.opname, instr.oparg, False)
                new_depth = depth + delta
                if new_depth > maxdepth:
                    maxdepth = new_depth
                assert depth >= 0
                op = self.opcode.opmap[instr.opname]
                if self.opcode.has_jump(op):
                    delta = self.opcode.stack_effect_raw(instr.opname, instr.oparg, True)
                    target_depth = depth + delta
                    if target_depth > maxdepth:
                        maxdepth = target_depth
                    assert target_depth >= 0
                    self.push_block(worklist, instr.target, target_depth)
                depth = new_depth
                if instr.opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD', 'RETURN_VALUE', 'RETURN_PRIMITIVE', 'RAISE_VARARGS', 'RERAISE'):
                    next = None
                    break
            if next:
                self.push_block(worklist, next, depth)
        return maxdepth

    def computeStackDepth(self):
        if False:
            i = 10
            return i + 15
        'Compute the max stack depth.\n\n        Find the flow path that needs the largest stack.  We assume that\n        cycles in the flow graph have no net effect on the stack depth.\n        '
        assert self.stage == FINAL, self.stage
        for block in self.getBlocksInOrder():
            if block.getInstructions():
                self.stacksize = self.stackdepth_walk(block)
                break

    def flattenGraph(self):
        if False:
            for i in range(10):
                print('nop')
        'Arrange the blocks in order and resolve jumps'
        assert self.stage == FINAL, self.stage
        extended_arg_recompile = True
        while extended_arg_recompile:
            extended_arg_recompile = False
            self.insts = insts = []
            pc = 0
            for b in self.getBlocksInOrder():
                b.offset = pc
                for inst in b.getInstructions():
                    insts.append(inst)
                    pc += instrsize(inst.ioparg)
            pc = 0
            for inst in insts:
                pc += instrsize(inst.ioparg)
                op = self.opcode.opmap[inst.opname]
                if self.opcode.has_jump(op):
                    oparg = inst.ioparg
                    target = inst.target
                    offset = target.offset
                    if op in self.opcode.hasjrel:
                        offset -= pc
                    if instrsize(oparg) != instrsize(offset):
                        extended_arg_recompile = True
                    assert offset >= 0, 'Offset value: %d' % offset
                    inst.ioparg = offset
        self.stage = FLAT

    def sort_cellvars(self):
        if False:
            while True:
                i = 10
        self.closure = self.cellvars + self.freevars

    def _convert_LOAD_CONST(self, arg: object) -> int:
        if False:
            return 10
        getCode = getattr(arg, 'getCode', None)
        if getCode is not None:
            arg = getCode()
        key = self.get_const_key(arg)
        res = self.consts.get(key, self)
        if res is self:
            res = self.consts[key] = len(self.consts)
        return res

    def get_const_key(self, value: object):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, float):
            return (type(value), value, sign(value))
        elif isinstance(value, complex):
            return (type(value), value, sign(value.real), sign(value.imag))
        elif isinstance(value, (tuple, frozenset)):
            return (type(value), value, type(value)((self.get_const_key(const) for const in value)))
        return (type(value), value)

    def _convert_LOAD_FAST(self, arg: object) -> int:
        if False:
            return 10
        self.fast_vars.add(arg)
        return self.varnames.get_index(arg)

    def _convert_LOAD_LOCAL(self, arg: object) -> int:
        if False:
            print('Hello World!')
        self.fast_vars.add(arg)
        assert isinstance(arg, tuple), 'invalid oparg {arg!r}'
        return self._convert_LOAD_CONST((self.varnames.get_index(arg[0]), arg[1]))

    def _convert_NAME(self, arg: object) -> int:
        if False:
            return 10
        return self.names.get_index(arg)

    def _convert_LOAD_SUPER(self, arg: object) -> int:
        if False:
            i = 10
            return i + 15
        assert isinstance(arg, tuple), 'invalid oparg {arg!r}'
        return self._convert_LOAD_CONST((self._convert_NAME(arg[0]), arg[1]))

    def _convert_DEREF(self, arg: object) -> int:
        if False:
            i = 10
            return i + 15
        if arg in self.freevars:
            return self.freevars.get_index(arg) + len(self.cellvars)
        return self.closure.get_index(arg)
    _converters = {'LOAD_CLASS': _convert_LOAD_CONST, 'LOAD_CONST': _convert_LOAD_CONST, 'INVOKE_FUNCTION': _convert_LOAD_CONST, 'INVOKE_METHOD': _convert_LOAD_CONST, 'INVOKE_NATIVE': _convert_LOAD_CONST, 'LOAD_FIELD': _convert_LOAD_CONST, 'STORE_FIELD': _convert_LOAD_CONST, 'CAST': _convert_LOAD_CONST, 'TP_ALLOC': _convert_LOAD_CONST, 'BUILD_CHECKED_MAP': _convert_LOAD_CONST, 'BUILD_CHECKED_LIST': _convert_LOAD_CONST, 'PRIMITIVE_LOAD_CONST': _convert_LOAD_CONST, 'LOAD_FAST': _convert_LOAD_FAST, 'STORE_FAST': _convert_LOAD_FAST, 'DELETE_FAST': _convert_LOAD_FAST, 'LOAD_LOCAL': _convert_LOAD_LOCAL, 'STORE_LOCAL': _convert_LOAD_LOCAL, 'LOAD_NAME': _convert_NAME, 'LOAD_CLOSURE': lambda self, arg: self.closure.get_index(arg), 'COMPARE_OP': lambda self, arg: self.opcode.CMP_OP.index(arg), 'LOAD_GLOBAL': _convert_NAME, 'STORE_GLOBAL': _convert_NAME, 'DELETE_GLOBAL': _convert_NAME, 'CONVERT_NAME': _convert_NAME, 'STORE_NAME': _convert_NAME, 'STORE_ANNOTATION': _convert_NAME, 'DELETE_NAME': _convert_NAME, 'IMPORT_NAME': _convert_NAME, 'IMPORT_FROM': _convert_NAME, 'STORE_ATTR': _convert_NAME, 'LOAD_ATTR': _convert_NAME, 'DELETE_ATTR': _convert_NAME, 'LOAD_METHOD': _convert_NAME, 'LOAD_DEREF': _convert_DEREF, 'STORE_DEREF': _convert_DEREF, 'DELETE_DEREF': _convert_DEREF, 'LOAD_CLASSDEREF': _convert_DEREF, 'REFINE_TYPE': _convert_LOAD_CONST, 'LOAD_METHOD_SUPER': _convert_LOAD_SUPER, 'LOAD_ATTR_SUPER': _convert_LOAD_SUPER, 'LOAD_TYPE': _convert_LOAD_CONST}
    _const_converters = {_convert_LOAD_CONST, _convert_LOAD_LOCAL, _convert_LOAD_SUPER}
    _const_opcodes = set()
    for (op, converter) in _converters.items():
        if converter in _const_converters:
            _const_opcodes.add(op)
    _quiet_opcodes = {'LOAD_GLOBAL', 'LOAD_CONST', 'IMPORT_NAME', 'STORE_ATTR', 'LOAD_ATTR', 'DELETE_ATTR', 'LOAD_METHOD', 'STORE_FAST', 'LOAD_FAST'}

    def makeByteCode(self):
        if False:
            i = 10
            return i + 15
        assert self.stage == FLAT, self.stage
        self.lnotab = lnotab = LineAddrTable(self.opcode)
        lnotab.setFirstLine(self.firstline)
        for t in self.insts:
            if lnotab.current_line != t.lineno:
                lnotab.nextLine(t.lineno)
            oparg = t.ioparg
            assert 0 <= oparg <= 4294967295, oparg
            if oparg > 16777215:
                lnotab.addCode(self.opcode.EXTENDED_ARG, oparg >> 24 & 255)
            if oparg > 65535:
                lnotab.addCode(self.opcode.EXTENDED_ARG, oparg >> 16 & 255)
            if oparg > 255:
                lnotab.addCode(self.opcode.EXTENDED_ARG, oparg >> 8 & 255)
            lnotab.addCode(self.opcode.opmap[t.opname], oparg & 255)
        lnotab.emitCurrentLine()
        self.stage = DONE

    def newCodeObject(self):
        if False:
            return 10
        assert self.stage == DONE, self.stage
        if self.flags & CO_NEWLOCALS == 0:
            nlocals = len(self.fast_vars)
        else:
            nlocals = len(self.varnames)
        firstline = self.firstline
        if not firstline:
            firstline = self.first_inst_lineno
        if not firstline:
            firstline = 1
        consts = self.getConsts()
        code = self.lnotab.getCode()
        lnotab = self.lnotab.getTable()
        consts = consts + tuple(self.extra_consts)
        return self.make_code(nlocals, code, consts, firstline, lnotab)

    def make_code(self, nlocals, code, consts, firstline, lnotab) -> CodeType:
        if False:
            for i in range(10):
                print('nop')
        return CodeType(len(self.args), self.posonlyargs, len(self.kwonlyargs), nlocals, self.stacksize, self.flags, code, consts, tuple(self.names), tuple(self.varnames), self.filename, self.name, firstline, lnotab, tuple(self.freevars), tuple(self.cellvars))

    def getConsts(self):
        if False:
            print('Hello World!')
        'Return a tuple for the const slot of the code object'
        return tuple((const[1] for (const, idx) in sorted(self.consts.items(), key=lambda o: o[1])))

    def propagate_line_numbers(self):
        if False:
            while True:
                i = 10
        'Propagate line numbers to instructions without.'
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            prev_lineno = -1
            for instr in block.insts:
                if instr.lineno < 0:
                    instr.lineno = prev_lineno
                else:
                    prev_lineno = instr.lineno
            if not block.no_fallthrough and block.next.num_predecessors == 1:
                assert block.next.insts
                next_instr = block.next.insts[0]
                if next_instr.lineno < 0:
                    next_instr.lineno = prev_lineno
            last_instr = block.insts[-1]
            if last_instr.is_jump(self.opcode) and last_instr.opname not in {'SETUP_ASYNC_WITH', 'SETUP_WITH', 'SETUP_FINALLY'}:
                target = last_instr.target
                if target.num_predecessors == 1:
                    assert target.insts
                    next_instr = target.insts[0]
                    if next_instr.lineno < 0:
                        next_instr.lineno = prev_lineno

    def guarantee_lineno_for_exits(self):
        if False:
            return 10
        lineno = self.firstline
        assert lineno > 0
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last_instr = block.insts[-1]
            if last_instr.lineno < 0:
                if last_instr.opname in ('RETURN_VALUE', 'RETURN_PRIMITIVE'):
                    for instr in block.insts:
                        assert instr.lineno < 0
                        instr.lineno = lineno
            else:
                lineno = last_instr.lineno

    def duplicate_exits_without_lineno(self):
        if False:
            print('Hello World!')
        '\n        PEP 626 mandates that the f_lineno of a frame is correct\n        after a frame terminates. It would be prohibitively expensive\n        to continuously update the f_lineno field at runtime,\n        so we make sure that all exiting instruction (raises and returns)\n        have a valid line number, allowing us to compute f_lineno lazily.\n        We can do this by duplicating the exit blocks without line number\n        so that none have more than one predecessor. We can then safely\n        copy the line number from the sole predecessor block.\n        '
        append_after = {}
        for block in self.blocks_in_reverse_allocation_order():
            if block.insts and (last := block.insts[-1]).is_jump(self.opcode):
                if last.opname in {'SETUP_ASYNC_WITH', 'SETUP_WITH', 'SETUP_FINALLY'}:
                    continue
                target = last.target
                assert target.insts
                if target.is_exit and target.insts[0].lineno < 0 and (target.num_predecessors > 1):
                    new_target = target.copy()
                    new_target.insts[0].lineno = last.lineno
                    last.target = new_target
                    target.num_predecessors -= 1
                    new_target.num_predecessors = 1
                    new_target.next = target.next
                    target.next = new_target
                    new_target.prev = target
                    new_target.bid = self.block_count
                    self.block_count += 1
                    append_after.setdefault(target, []).append(new_target)
        for (after, to_append) in append_after.items():
            idx = self.ordered_blocks.index(after) + 1
            self.ordered_blocks[idx:idx] = reversed(to_append)

    def normalize_jumps(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.stage == ORDERED, self.stage
        seen_blocks = set()
        for block in self.ordered_blocks:
            seen_blocks.add(block.bid)
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.opname == 'JUMP_ABSOLUTE' and last.target.bid not in seen_blocks:
                last.opname = 'JUMP_FORWARD'
            elif last.opname == 'JUMP_FORWARD' and last.target.bid in seen_blocks:
                last.opname = 'JUMP_ABSOLUTE'

    def optimizeCFG(self):
        if False:
            return 10
        'Optimize a well-formed CFG.'
        assert self.stage == CLOSED, self.stage
        optimizer = FlowGraphOptimizer(self)
        for block in self.ordered_blocks:
            optimizer.optimize_basic_block(block)
            optimizer.clean_basic_block(block, -1)
        for block in self.blocks_in_reverse_allocation_order():
            self.extend_block(block)
        prev_block = None
        for block in self.ordered_blocks:
            prev_lineno = -1
            if prev_block and prev_block.insts:
                prev_lineno = prev_block.insts[-1].lineno
            optimizer.clean_basic_block(block, prev_lineno)
            prev_block = None if block.no_fallthrough else block
        self.eliminate_empty_basic_blocks()
        self.remove_unreachable_basic_blocks()
        maybe_empty_blocks = False
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.opname not in {'JUMP_ABSOLUTE', 'JUMP_FORWARD'}:
                continue
            if last.target == block.next:
                block.no_fallthrough = False
                last.opname = 'NOP'
                last.oparg = last.ioparg = 0
                last.target = None
                optimizer.clean_basic_block(block, -1)
                maybe_empty_blocks = True
        if maybe_empty_blocks:
            self.eliminate_empty_basic_blocks()
        self.stage = OPTIMIZED

    def eliminate_empty_basic_blocks(self):
        if False:
            i = 10
            return i + 15
        for block in self.ordered_blocks:
            next_block = block.next
            if next_block:
                while not next_block.insts and next_block.next:
                    next_block = next_block.next
                block.next = next_block
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.is_jump(self.opcode):
                target = last.target
                while not target.insts and target.next:
                    target = target.next
                last.target = target

    def remove_unreachable_basic_blocks(self):
        if False:
            return 10
        reachable_blocks = set()
        worklist = [self.entry]
        while worklist:
            entry = worklist.pop()
            if entry.bid in reachable_blocks:
                continue
            reachable_blocks.add(entry.bid)
            for instruction in entry.getInstructions():
                target = instruction.target
                if target is not None:
                    worklist.append(target)
                    target.num_predecessors += 1
            if not entry.no_fallthrough:
                worklist.append(entry.next)
                entry.next.num_predecessors += 1
        self.ordered_blocks = [block for block in self.ordered_blocks if block.bid in reachable_blocks]
        prev = None
        for block in self.ordered_blocks:
            block.prev = prev
            if prev is not None:
                prev.next = block
            prev = block

    def normalize_basic_block(self, block: Block) -> None:
        if False:
            return 10
        'Sets the `fallthrough` and `exit` properties of a block, and ensures that the targets of\n        any jumps point to non-empty blocks by following the next pointer of empty blocks.'
        for instr in block.getInstructions():
            if instr.opname in ('RETURN_VALUE', 'RETURN_PRIMITIVE', 'RAISE_VARARGS', 'RERAISE'):
                block.is_exit = True
                block.no_fallthrough = True
                continue
            elif instr.opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD'):
                block.no_fallthrough = True
            elif not instr.is_jump(self.opcode):
                continue
            while not instr.target.insts:
                instr.target = instr.target.next

    def extend_block(self, block: Block) -> None:
        if False:
            while True:
                i = 10
        'If this block ends with an unconditional jump to an exit block,\n        then remove the jump and extend this block with the target.\n        '
        if len(block.insts) == 0:
            return
        last = block.insts[-1]
        if last.opname not in ('JUMP_ABSOLUTE', 'JUMP_FORWARD'):
            return
        target = last.target
        assert target is not None
        if not target.is_exit:
            return
        if len(target.insts) > MAX_COPY_SIZE:
            return
        last = block.insts[-1]
        last.opname = 'NOP'
        last.oparg = last.ioparg = 0
        last.target = None
        for instr in target.insts:
            block.insts.append(instr.copy())
        block.next = None
        block.is_exit = True
        block.no_fallthrough = True

    def trim_unused_consts(self) -> None:
        if False:
            i = 10
            return i + 15
        'Remove trailing unused constants.'
        assert self.stage == CONSTS_CLOSED, self.stage
        max_const_index = 0
        for block in self.ordered_blocks:
            for instr in block.insts:
                if instr.opname in self._const_opcodes and instr.ioparg > max_const_index:
                    max_const_index = instr.ioparg
        self.consts = {key: index for (key, index) in self.consts.items() if index <= max_const_index}

class PyFlowGraphCinder(PyFlowGraph):
    opcode = opcode_cinder.opcode

    def make_code(self, nlocals, code, consts, firstline, lnotab) -> CodeType:
        if False:
            return 10
        if self.scope is not None and self.scope.suppress_jit:
            self.setFlag(CO_SUPPRESS_JIT)
        return super().make_code(nlocals, code, consts, firstline, lnotab)

class LineAddrTable:
    """linetable / lnotab

    This class builds the linetable, which is documented in
    Objects/lnotab_notes.txt. Here's a brief recap:

    For each new lineno after the first one, two bytes are added to the
    linetable.  (In some cases, multiple two-byte entries are added.)  The first
    byte is the distance in bytes between the instruction for the current lineno
    and the next lineno.  The second byte is offset in line numbers.  If either
    offset is greater than 255, multiple two-byte entries are added -- see
    lnotab_notes.txt for the delicate details.

    """

    def __init__(self, opcode):
        if False:
            while True:
                i = 10
        self.code = []
        self.current_start = 0
        self.current_end = 0
        self.current_line = 0
        self.prev_line = 0
        self.linetable = []
        self.opcode = opcode

    def setFirstLine(self, lineno):
        if False:
            i = 10
            return i + 15
        self.current_line = lineno
        self.prev_line = lineno

    def addCode(self, opcode, oparg):
        if False:
            print('Hello World!')
        self.code.append(opcode)
        self.code.append(oparg)
        self.current_end += self.opcode.CODEUNIT_SIZE

    def nextLine(self, lineno):
        if False:
            for i in range(10):
                print('nop')
        if not lineno:
            return
        self.emitCurrentLine()
        self.current_start = self.current_end
        if self.current_line >= 0:
            self.prev_line = self.current_line
        self.current_line = lineno

    def emitCurrentLine(self):
        if False:
            i = 10
            return i + 15
        addr_delta = self.current_end - self.current_start
        if not addr_delta:
            return
        if self.current_line < 0:
            line_delta = -128
        else:
            line_delta = self.current_line - self.prev_line
            while line_delta < -127 or 127 < line_delta:
                if line_delta < 0:
                    k = -127
                else:
                    k = 127
                self.push_entry(0, k)
                line_delta -= k
        while addr_delta > 254:
            self.push_entry(254, line_delta)
            line_delta = -128 if self.current_line < 0 else 0
            addr_delta -= 254
        assert -128 <= line_delta and line_delta <= 127
        self.push_entry(addr_delta, line_delta)

    def getCode(self):
        if False:
            for i in range(10):
                print('nop')
        return bytes(self.code)

    def getTable(self):
        if False:
            while True:
                i = 10
        return bytes(self.linetable)

    def push_entry(self, addr_delta, line_delta):
        if False:
            return 10
        self.linetable.append(addr_delta)
        self.linetable.append(cast_signed_byte_to_unsigned(line_delta))