import typing
import types
import logging
from dataclasses import dataclass
from .executor import Executor
from collections import deque
from math import ceil
from .types import BranchImm, BranchTableImm, CallImm, CallIndirectImm, ConcretizeStack, convert_instructions, debug, F32, F64, FuncIdx, FunctionType, GlobalIdx, GlobalType, I32, I64, Instruction, LimitType, MemIdx, MemoryType, MissingExportException, Name, NonExistentFunctionCallTrap, OutOfBoundsMemoryTrap, OverflowDivisionTrap, TableIdx, TableType, Trap, TypeIdx, TypeMismatchTrap, U32, ValType, Value, Value_t, WASMExpression, ZeroDivisionTrap
from .state import State
from ..core.smtlib import BitVec, Bool, issymbolic, Operators, Expression
from ..core.state import Concretize
from ..utils.event import Eventful
from ..utils import config
from wasm import decode_module, Section
from wasm.wasmtypes import SEC_TYPE, SEC_IMPORT, SEC_FUNCTION, SEC_TABLE, SEC_MEMORY, SEC_GLOBAL, SEC_EXPORT, SEC_START, SEC_ELEMENT, SEC_CODE, SEC_DATA, SEC_UNK
from ..core.smtlib.solver import SelectedSolver
logger = logging.getLogger(__name__)
consts = config.get_group('wasm')
consts.add('decode_names', default=False, description='Should Manticore attempt to decode custom name sections')
PAGESIZE = 2 ** 16

class Addr(int):
    pass

class FuncAddr(Addr):
    pass

class TableAddr(Addr):
    pass

class MemAddr(Addr):
    pass

class GlobalAddr(Addr):
    pass
ExternVal = typing.Union[FuncAddr, TableAddr, MemAddr, GlobalAddr]
FuncElem = typing.Optional[FuncAddr]
ExportDesc = typing.Union[FuncIdx, TableIdx, MemIdx, GlobalIdx]
ImportDesc = typing.Union[TypeIdx, TableType, MemoryType, GlobalType]

@dataclass
class Function:
    """
    A WASM Function

    https://www.w3.org/TR/wasm-core-1/#functions%E2%91%A0
    """
    type: TypeIdx
    locals: typing.List[ValType]
    body: WASMExpression

    def allocate(self, store: 'Store', module: 'ModuleInstance') -> FuncAddr:
        if False:
            print('Hello World!')
        "\n        https://www.w3.org/TR/wasm-core-1/#functions%E2%91%A5\n\n        :param store: Destination Store that we'll insert this Function into after allocation\n        :param module: The module containing the type referenced by self.type\n        :return: The address of this within `store`\n        "
        a = FuncAddr(len(store.funcs))
        store.funcs.append(FuncInst(module.types[self.type], module, self))
        return a

@dataclass
class Table:
    """
    Vector of opaque values of type self.type

    https://www.w3.org/TR/wasm-core-1/#tables%E2%91%A0
    """
    type: TableType

    def allocate(self, store: 'Store') -> TableAddr:
        if False:
            print('Hello World!')
        "\n        https://www.w3.org/TR/wasm-core-1/#tables%E2%91%A5\n\n        :param store: Destination Store that we'll insert this Table into after allocation\n        :return: The address of this within `store`\n        "
        a = TableAddr(len(store.tables))
        store.tables.append(TableInst([None for _i in range(self.type.limits.min)], self.type.limits.max))
        return a

@dataclass
class Memory:
    """
    Big chunk o' raw bytes

    https://www.w3.org/TR/wasm-core-1/#memories%E2%91%A0
    """
    type: MemoryType

    def allocate(self, store: 'Store') -> MemAddr:
        if False:
            while True:
                i = 10
        "\n        https://www.w3.org/TR/wasm-core-1/#memories%E2%91%A5\n\n        :param store: Destination Store that we'll insert this Memory into after allocation\n        :return: The address of this within `store`\n        "
        a = MemAddr(len(store.mems))
        store.mems.append(MemInst([0] * self.type.min * 64 * 1024, self.type.max))
        return a

@dataclass
class Global:
    """
    A global variable of a given type

    https://www.w3.org/TR/wasm-core-1/#globals%E2%91%A0
    """
    type: GlobalType
    init: WASMExpression

    def allocate(self, store: 'Store', val: Value) -> GlobalAddr:
        if False:
            while True:
                i = 10
        "\n        https://www.w3.org/TR/wasm-core-1/#globals%E2%91%A5\n\n        :param store: Destination Store that we'll insert this Global into after allocation\n        :param val: The initial value of the new global\n        :return: The address of this within `store`\n        "
        a = GlobalAddr(len(store.globals))
        store.globals.append(GlobalInst(val, self.type.mut))
        return a

@dataclass
class Elem:
    """
    List of functions to initialize part of a table

    https://www.w3.org/TR/wasm-core-1/#element-segments%E2%91%A0
    """
    table: TableIdx
    offset: WASMExpression
    init: typing.List[FuncIdx]

@dataclass
class Data:
    """
    Vector of bytes that initializes part of a memory

    https://www.w3.org/TR/wasm-core-1/#data-segments%E2%91%A0
    """
    data: MemIdx
    offset: WASMExpression
    init: typing.List[int]

@dataclass
class Import:
    """
    Something imported from another module (or the environment) that we need to instantiate a module

    https://www.w3.org/TR/wasm-core-1/#imports%E2%91%A0
    """
    module: Name
    name: Name
    desc: ImportDesc

@dataclass
class Export:
    """
    Something the module exposes to the outside world once it's been instantiated

    https://www.w3.org/TR/wasm-core-1/#exports%E2%91%A0
    """
    name: Name
    desc: ExportDesc

def strip_quotes(rough_name: str) -> Name:
    if False:
        for i in range(10):
            print('nop')
    '\n    For some reason, the parser returns the function names with quotes around them\n\n    :param rough_name:\n    :return:\n    '
    return Name(rough_name[1:-1])

class Module:
    """
    Internal representation of a WASM Module
    """
    __slots__ = ['types', 'funcs', 'tables', 'mems', 'globals', 'elem', 'data', 'start', 'imports', 'exports', 'function_names', 'local_names', '_raw']
    _raw: bytes

    def __init__(self):
        if False:
            while True:
                i = 10
        self.types: typing.List[FunctionType] = []
        self.funcs: typing.List[Function] = []
        self.tables: typing.List[Table] = []
        self.mems: typing.List[Memory] = []
        self.globals: typing.List[Global] = []
        self.elem: typing.List[Elem] = []
        self.data: typing.List[Data] = []
        self.start: typing.Optional[FuncIdx] = None
        self.imports: typing.List[Import] = []
        self.exports: typing.List[Export] = []
        self.function_names: typing.Dict[FuncAddr, str] = {}
        self.local_names: typing.Dict[FuncAddr, typing.Dict[int, str]] = {}

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = {'types': self.types, 'funcs': self.funcs, 'tables': self.tables, 'mems': self.mems, 'globals': self.globals, 'elem': self.elem, 'data': self.data, 'start': self.start, 'imports': self.imports, 'exports': self.exports, 'function_names': self.function_names, 'local_names': self.local_names, '_raw': self._raw}
        return state

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        self.types = state['types']
        self.funcs = state['funcs']
        self.tables = state['tables']
        self.mems = state['mems']
        self.globals = state['globals']
        self.elem = state['elem']
        self.data = state['data']
        self.start = state['start']
        self.imports = state['imports']
        self.exports = state['exports']
        self.function_names = state['function_names']
        self.local_names = state['local_names']
        self._raw = state['_raw']

    def get_funcnames(self) -> typing.List[Name]:
        if False:
            i = 10
            return i + 15
        return [e.name for e in self.exports if isinstance(e.desc, FuncIdx)]

    @classmethod
    def load(cls, filename: str):
        if False:
            print('Hello World!')
        '\n        Converts a WASM module in binary format into Python types that Manticore can understand\n\n        :param filename: name of the WASM module\n        :return: Module\n        '
        type_map = {-16: FunctionType, -4: F64, -3: F32, -2: I64, -1: I32}
        m: Module = cls()
        with open(filename, 'rb') as wasm_file:
            m._raw = wasm_file.read()
        module_iter = decode_module(m._raw, decode_name_subsections=consts.decode_names)
        _header = next(module_iter)
        section: Section
        for (section, section_data) in module_iter:
            sec_id = getattr(section_data, 'id', SEC_UNK)
            if sec_id == SEC_TYPE:
                for ft in section_data.payload.entries:
                    m.types.append(FunctionType([type_map[p_type] for p_type in ft.param_types], [type_map[ft.return_type] for _i in range(ft.return_count)]))
            elif sec_id == SEC_IMPORT:
                for i in section_data.payload.entries:
                    ty_map = i.get_decoder_meta()['types']
                    mod_name = strip_quotes(ty_map['module_str'].to_string(i.module_str))
                    field_name = strip_quotes(ty_map['field_str'].to_string(i.field_str))
                    if i.kind == 0:
                        m.imports.append(Import(mod_name, field_name, TypeIdx(i.type.type)))
                    elif i.kind == 1:
                        m.imports.append(Import(mod_name, field_name, TableType(LimitType(i.type.limits.initial, i.type.limits.maximum), type_map[i.type.element_type])))
                    elif i.kind == 2:
                        m.imports.append(Import(mod_name, field_name, MemoryType(i.type.limits.initial, i.type.limits.maximum)))
                    elif i.kind == 3:
                        m.imports.append(Import(mod_name, field_name, GlobalType(bool(i.type.mutability), type_map[i.type.content_type])))
                    else:
                        raise RuntimeError("Can't decode kind field of:", i.kind)
            elif sec_id == SEC_FUNCTION:
                for f in section_data.payload.types:
                    m.funcs.append(Function(TypeIdx(f), [], []))
            elif sec_id == SEC_TABLE:
                for t in section_data.payload.entries:
                    m.tables.append(Table(TableType(LimitType(t.limits.initial, t.limits.maximum), FunctionType)))
            elif sec_id == SEC_MEMORY:
                for mem in section_data.payload.entries:
                    m.mems.append(Memory(LimitType(mem.limits.initial, mem.limits.maximum)))
            elif sec_id == SEC_GLOBAL:
                for g in section_data.payload.globals:
                    m.globals.append(Global(GlobalType(g.type.mutability, type_map[g.type.content_type]), convert_instructions(g.init)))
            elif sec_id == SEC_EXPORT:
                mapping = (FuncIdx, TableIdx, MemIdx, GlobalIdx)
                for e in section_data.payload.entries:
                    ty = e.get_decoder_meta()['types']['field_str']
                    m.exports.append(Export(strip_quotes(ty.to_string(e.field_str)), mapping[e.kind](e.index)))
            elif sec_id == SEC_START:
                m.start = FuncIdx(section_data.payload.index)
            elif sec_id == SEC_ELEMENT:
                for e in section_data.payload.entries:
                    m.elem.append(Elem(TableIdx(e.index), convert_instructions(e.offset), [FuncIdx(i) for i in e.elems]))
            elif sec_id == SEC_CODE:
                for (idx, c) in enumerate(section_data.payload.bodies):
                    m.funcs[idx].locals = [type_map[e.type] for e in c.locals for _i in range(e.count)]
                    m.funcs[idx].body = convert_instructions(c.code)
            elif sec_id == SEC_DATA:
                for d in section_data.payload.entries:
                    m.data.append(Data(MemIdx(d.index), convert_instructions(d.offset), d.data.tolist()))
            elif sec_id == SEC_UNK:
                if hasattr(section, 'name_type') and hasattr(section, 'payload_len') and hasattr(section, 'payload'):
                    name_type = section_data.name_type
                    if name_type == 0:
                        pass
                    elif name_type == 1:
                        for n in section_data.payload.names:
                            ty = n.get_decoder_meta()['types']['name_str']
                            m.function_names[FuncAddr(n.index)] = strip_quotes(ty.to_string(n.name_str))
                    elif name_type == 2:
                        for func in section_data.payload.funcs:
                            func_idx = func.index
                            for n in func.local_map.names:
                                ty = n.get_decoder_meta()['types']['name_str']
                                m.local_names.setdefault(FuncAddr(func_idx), {})[n.index] = strip_quotes(ty.to_string(n.name_str))
                else:
                    logger.info('Encountered unknown section')
        return m

@dataclass
class ProtoFuncInst:
    """
    Groups FuncInst and HostFuncInst into the same category
    """
    type: FunctionType

@dataclass
class TableInst:
    """
    Runtime representation of a table. Remember that the Table type stores the type of the data contained in the table
    and basically nothing else, so if you're dealing with a table at runtime, it's probably a TableInst. The WASM
    spec has a lot of similar-sounding names for different versions of one thing.

    https://www.w3.org/TR/wasm-core-1/#table-instances%E2%91%A0
    """
    elem: typing.List[FuncElem]
    max: typing.Optional[U32]

class MemInst(Eventful):
    """
    Runtime representation of a memory. As with tables, if you're dealing with a memory at runtime, it's probably a
    MemInst. Currently doesn't support any sort of symbolic indexing, although you can read and write symbolic bytes
    using smtlib. There's a minor quirk where uninitialized data is stored as bytes, but smtlib tries to convert
    concrete data into ints. That can cause problems if you try to read from the memory directly (without using smtlib)
    but shouldn't break any of the built-in WASM instruction implementations.

    Memory in WASM is broken up into 65536-byte pages. All pages behave the same way, but note that operations that
    deal with memory size do so in terms of pages, not bytes.

    TODO: We should implement some kind of symbolic memory model

    https://www.w3.org/TR/wasm-core-1/#memory-instances%E2%91%A0
    """
    _published_events = {'write_memory', 'read_memory'}
    _pages: typing.Dict[int, typing.List[int]]
    max: typing.Optional[U32]
    _current_size: int

    def __init__(self, starting_data, max=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._current_size = ceil(len(starting_data) / PAGESIZE)
        self.max = max
        self._pages = {}
        chunked = [starting_data[i:i + PAGESIZE] for i in range(0, len(starting_data), PAGESIZE)]
        for (idx, page) in enumerate(chunked):
            if len(page) < PAGESIZE:
                page.extend([0] * (PAGESIZE - len(page)))
            self._pages[idx] = page

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = super().__getstate__()
        state['pages'] = self._pages
        state['max'] = self.max
        state['current'] = self._current_size
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        super().__setstate__(state)
        self._pages = state['pages']
        self.max = state['max']
        self._current_size = state['current']

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item in range(self.npages * PAGESIZE)

    def _check_initialize_index(self, memidx):
        if False:
            return 10
        page = memidx // PAGESIZE
        if page not in range(self.npages):
            raise OutOfBoundsMemoryTrap(memidx)
        if page not in self._pages:
            self._pages[page] = [0] * PAGESIZE
        return divmod(memidx, PAGESIZE)

    def _read_byte(self, addr):
        if False:
            return 10
        (page, idx) = self._check_initialize_index(addr)
        return self._pages[page][idx]

    def _write_byte(self, addr, val):
        if False:
            while True:
                i = 10
        (page, idx) = self._check_initialize_index(addr)
        self._pages[page][idx] = val

    @property
    def npages(self):
        if False:
            for i in range(10):
                print('nop')
        return self._current_size

    def grow(self, n: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds n blank pages to the current memory\n\n        See: https://www.w3.org/TR/wasm-core-1/#grow-mem\n\n        :param n: The number of pages to attempt to add\n        :return: True if the operation succeeded, otherwise False\n        '
        ln = n + self.npages
        if ln > PAGESIZE:
            return False
        if self.max is not None:
            if ln > self.max:
                return False
        self._current_size = ln
        return True

    def write_int(self, base: int, expression: typing.Union[Expression, int], size: int=32):
        if False:
            return 10
        '\n        Writes an integer into memory.\n\n        :param base: Index to write at\n        :param expression: integer to write\n        :param size: Optional size of the integer\n        '
        b = [Operators.CHR(Operators.EXTRACT(expression, offset, 8)) for offset in range(0, size, 8)]
        self.write_bytes(base, b)

    def write_bytes(self, base: int, data: typing.Union[str, typing.Sequence[int], typing.Sequence[bytes]]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Writes  a stream of bytes into memory\n\n        :param base: Index to start writing at\n        :param data: Data to write\n        '
        self._publish('will_write_memory', base, base + len(data), data)
        for (idx, v) in enumerate(data):
            self._write_byte(base + idx, v)
        self._publish('did_write_memory', base, data)

    def read_int(self, base: int, size: int=32) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reads bytes from memory and combines them into an int\n\n        :param base: Address to read the int from\n        :param size: Size of the int (in bits)\n        :return: The int in question\n        '
        return Operators.CONCAT(size, *map(Operators.ORD, reversed(self.read_bytes(base, size // 8))))

    def read_bytes(self, base: int, size: int) -> typing.List[typing.Union[int, bytes]]:
        if False:
            while True:
                i = 10
        '\n        Reads bytes from memory\n\n        :param base: Address to read from\n        :param size: number of bytes to read\n        :return: List of bytes\n        '
        self._publish('will_read_memory', base, base + size)
        d = [self._read_byte(i) for i in range(base, base + size)]
        self._publish('did_read_memory', base, d)
        return d

    def dump(self):
        if False:
            i = 10
            return i + 15
        return self.read_bytes(0, self._current_size * PAGESIZE)

@dataclass
class GlobalInst:
    """
    Instance of a global variable. Stores the value (calculated from evaluating a Global.init) and the mutable flag
    (taken from GlobalType.mut)

    https://www.w3.org/TR/wasm-core-1/#global-instances%E2%91%A0
    """
    value: Value
    mut: bool

@dataclass
class ExportInst:
    """
    Runtime representation of any thing that can be exported

    https://www.w3.org/TR/wasm-core-1/#export-instances%E2%91%A0
    """
    name: Name
    value: ExternVal

class Store:
    """
    Implementation of the WASM store. Nothing fancy here, just collects lists of functions, tables, memories, and
    globals. Because the store is not atomic, instructions SHOULD NOT make changes to the Store or any of its contents
    (including memories and global variables) before raising a Concretize exception.

    https://www.w3.org/TR/wasm-core-1/#store%E2%91%A0
    """
    __slots__ = ['funcs', 'tables', 'mems', 'globals']
    funcs: typing.List[ProtoFuncInst]
    tables: typing.List[TableInst]
    mems: typing.List[MemInst]
    globals: typing.List[GlobalInst]

    def __init__(self):
        if False:
            return 10
        self.funcs = []
        self.tables = []
        self.mems = []
        self.globals = []

    def __getstate__(self):
        if False:
            return 10
        state = {'funcs': self.funcs, 'tables': self.tables, 'mems': self.mems, 'globals': self.globals}
        return state

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        self.funcs = state['funcs']
        self.tables = state['tables']
        self.mems = state['mems']
        self.globals = state['globals']

def _eval_maybe_symbolic(state, expression) -> bool:
    if False:
        return 10
    if issymbolic(expression):
        return state.must_be_true(expression)
    return True if expression else False

class ModuleInstance(Eventful):
    """
    Runtime instance of a module. Stores function types, list of addresses within the store, and exports. In this
    implementation, it's also responsible for managing the instruction queue and executing control-flow instructions.

    https://www.w3.org/TR/wasm-core-1/#module-instances%E2%91%A0
    """
    __slots__ = ['types', 'funcaddrs', 'tableaddrs', 'memaddrs', 'globaladdrs', 'exports', 'export_map', 'executor', 'function_names', 'local_names', '_instruction_queue', '_block_depths', '_advice', '_state']
    _published_events = {'execute_instruction', 'call_hostfunc', 'exec_expression', 'raise_trap'}
    types: typing.List[FunctionType]
    funcaddrs: typing.List[FuncAddr]
    tableaddrs: typing.List[TableAddr]
    memaddrs: typing.List[MemAddr]
    globaladdrs: typing.List[GlobalAddr]
    exports: typing.List[ExportInst]
    export_map: typing.Dict[str, int]
    executor: Executor
    function_names: typing.Dict[FuncAddr, str]
    local_names: typing.Dict[FuncAddr, typing.Dict[int, str]]
    _instruction_queue: typing.Deque[Instruction]
    _block_depths: typing.List[int]
    _advice: typing.Optional[typing.List[bool]]
    instantiated: bool
    _state: State

    def __init__(self, constraints=None):
        if False:
            while True:
                i = 10
        self.types = []
        self.funcaddrs = []
        self.tableaddrs = []
        self.memaddrs = []
        self.globaladdrs = []
        self.exports = []
        self.export_map = {}
        self.executor = Executor()
        self.function_names = {}
        self.local_names = {}
        self._instruction_queue = deque()
        self._block_depths = [0]
        self._advice = None
        self._state = None
        super().__init__()

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = super().__getstate__()
        state.update({'types': self.types, 'funcaddrs': self.funcaddrs, 'tableaddrs': self.tableaddrs, 'memaddrs': self.memaddrs, 'globaladdrs': self.globaladdrs, 'exports': self.exports, 'export_map': self.export_map, 'executor': self.executor, 'function_names': self.function_names, 'local_names': self.local_names, '_instruction_queue': self._instruction_queue, '_block_depths': self._block_depths})
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.types = state['types']
        self.funcaddrs = state['funcaddrs']
        self.tableaddrs = state['tableaddrs']
        self.memaddrs = state['memaddrs']
        self.globaladdrs = state['globaladdrs']
        self.exports = state['exports']
        self.export_map = state['export_map']
        self.executor = state['executor']
        self.function_names = state['function_names']
        self.local_names = state['local_names']
        self._instruction_queue = state['_instruction_queue']
        self._block_depths = state['_block_depths']
        self._advice = None
        self._state = None
        super().__setstate__(state)

    def reset_internal(self):
        if False:
            print('Hello World!')
        '\n        Empties the instruction queue and clears the block depths\n        '
        self._instruction_queue.clear()
        self._block_depths = [0]

    def instantiate(self, store: Store, module: 'Module', extern_vals: typing.List[ExternVal], exec_start: bool=False):
        if False:
            print('Hello World!')
        '\n        Type checks the module, evaluates globals, performs allocation, and puts the element and data sections into\n        their proper places. Optionally calls the start function _outside_ of a symbolic context if exec_start is true.\n\n        https://www.w3.org/TR/wasm-core-1/#instantiation%E2%91%A1\n\n        :param store: The store to place the allocated contents in\n        :param module: The WASM Module to instantiate in this instance\n        :param extern_vals: Imports needed to instantiate the module\n        :param exec_start: whether or not to execute the start section (if present)\n        '
        assert module
        assert len(module.imports) == len(extern_vals), f'Expected {len(module.imports)} imports, got {len(extern_vals)}'
        stack = Stack()
        aux_mod = ModuleInstance()
        aux_mod.globaladdrs = [i for i in extern_vals if isinstance(i, GlobalAddr)]
        aux_frame = Frame([], aux_mod)
        stack.push(Activation(1, aux_frame))
        vals = [self.exec_expression(store, stack, gb.init) for gb in module.globals]
        last_frame = stack.pop()
        assert isinstance(last_frame, Activation)
        assert last_frame.frame == aux_frame
        self.allocate(store, module, extern_vals, vals)
        f = Frame(locals=[], module=self)
        stack.push(Activation(0, f))
        for elem in module.elem:
            eoval = self.exec_expression(store, stack, elem.offset)
            assert isinstance(eoval, I32)
            assert elem.table in range(len(self.tableaddrs))
            tableaddr: TableAddr = self.tableaddrs[elem.table]
            assert tableaddr in range(len(store.tables))
            tableinst: TableInst = store.tables[tableaddr]
            eend = eoval + len(elem.init)
            assert eend <= len(tableinst.elem)
            func_idx: FuncIdx
            for (j, func_idx) in enumerate(elem.init):
                assert func_idx in range(len(self.funcaddrs))
                funcaddr = self.funcaddrs[func_idx]
                tableinst.elem[eoval + j] = funcaddr
        for data in module.data:
            doval = self.exec_expression(store, stack, data.offset)
            assert isinstance(doval, I32), f'{type(doval)} is not an I32'
            assert data.data in range(len(self.memaddrs))
            memaddr = self.memaddrs[data.data]
            assert memaddr in range(len(store.mems))
            meminst = store.mems[memaddr]
            dend = doval + len(data.init)
            assert dend <= meminst.npages * PAGESIZE
            meminst.write_bytes(doval, data.init)
        last_frame = stack.pop()
        assert isinstance(last_frame, Activation)
        assert last_frame.frame == f
        if module.start is not None:
            assert module.start in range(len(self.funcaddrs))
            funcaddr = self.funcaddrs[module.start]
            assert funcaddr in range(len(store.funcs))
            self.invoke(stack, self.funcaddrs[module.start], store, [])
            if exec_start:
                stack.push(self.exec_expression(store, stack, []))
        logger.info('Initialization Complete')

    def allocate(self, store: Store, module: 'Module', extern_vals: typing.List[ExternVal], values: typing.List[Value]):
        if False:
            return 10
        '\n        Inserts imports into the store, then creates and inserts function instances, table instances, memory instances,\n        global instances, and export instances.\n\n        https://www.w3.org/TR/wasm-core-1/#allocation%E2%91%A0\n        https://www.w3.org/TR/wasm-core-1/#modules%E2%91%A6\n\n        :param store: The Store to put all of the allocated subcomponents in\n        :param module: Tne Module containing all the items to allocate\n        :param extern_vals: Imported values\n        :param values: precalculated global values\n        '
        self.types = module.types
        for ev in extern_vals:
            if isinstance(ev, FuncAddr):
                self.funcaddrs.append(ev)
            if isinstance(ev, TableAddr):
                self.tableaddrs.append(ev)
            if isinstance(ev, MemAddr):
                self.memaddrs.append(ev)
            if isinstance(ev, GlobalAddr):
                self.globaladdrs.append(ev)
        for func in module.funcs:
            addr = func.allocate(store, self)
            self.funcaddrs.append(addr)
            name = module.function_names.get(addr, None)
            if name:
                self.function_names[addr] = name
            local_map = module.local_names.get(addr, None)
            if local_map:
                self.local_names[addr] = local_map.copy()
        for table_i in module.tables:
            self.tableaddrs.append(table_i.allocate(store))
        for memory_i in module.mems:
            self.memaddrs.append(memory_i.allocate(store))
        for (idx, global_i) in enumerate(module.globals):
            assert isinstance(values[idx], global_i.type.value)
            self.globaladdrs.append(global_i.allocate(store, values[idx]))
        for (idx, export_i) in enumerate(module.exports):
            if isinstance(export_i.desc, FuncIdx):
                self.exports.append(ExportInst(export_i.name, self.funcaddrs[export_i.desc]))
            elif isinstance(export_i.desc, TableIdx):
                self.exports.append(ExportInst(export_i.name, self.tableaddrs[export_i.desc]))
            elif isinstance(export_i.desc, MemIdx):
                self.exports.append(ExportInst(export_i.name, self.memaddrs[export_i.desc]))
            elif isinstance(export_i.desc, GlobalIdx):
                self.exports.append(ExportInst(export_i.name, self.globaladdrs[export_i.desc]))
            else:
                raise RuntimeError("Export desc wasn't a function, table, memory, or global")
            self.export_map[export_i.name] = len(self.exports) - 1

    def invoke_by_name(self, name: str, stack, store, argv):
        if False:
            i = 10
            return i + 15
        '\n        Iterates over the exports, attempts to find the function specified by `name`. Calls `invoke` with its FuncAddr,\n        passing argv\n\n        :param name: Name of the function to look for\n        :param argv: Arguments to pass to the function. Can be BitVecs or Values\n        '
        for export in self.exports:
            if export.name == name and isinstance(export.value, FuncAddr):
                return self.invoke(stack, export.value, store, argv)
        raise RuntimeError("Can't find a function called", name)

    def invoke(self, stack: 'Stack', funcaddr: FuncAddr, store: Store, argv: typing.List[Value]):
        if False:
            return 10
        "\n        Invocation wrapper. Checks the function type, pushes the args to the stack, and calls _invoke_inner.\n        Unclear why the spec separates the two procedures, but I've tried to implement it as close to verbatim\n        as possible.\n\n        Note that this doesn't actually _run_ any code. It just sets up the instruction queue so that when you call\n        `exec_instruction, it'll actually have instructions to execute.\n\n        https://www.w3.org/TR/wasm-core-1/#invocation%E2%91%A1\n\n        :param funcaddr: Address (in Store) of the function to call\n        :param argv: Arguments to pass to the function. Can be BitVecs or Values\n        "
        assert funcaddr in range(len(store.funcs))
        funcinst = store.funcs[funcaddr]
        ty = funcinst.type
        assert len(ty.param_types) == len(argv), f'Function {funcaddr} expects {len(ty.param_types)} arguments'
        dummy_frame = Frame([], ModuleInstance())
        stack.push(dummy_frame)
        for v in argv:
            stack.push(v)
        self._invoke_inner(stack, funcaddr, store)

    def _invoke_inner(self, stack: 'Stack', funcaddr: FuncAddr, store: Store):
        if False:
            while True:
                i = 10
        "\n        Invokes the function at address funcaddr. Validates the function type, sets up the Activation with the local\n        variables, and executes the function. If the function is a HostFunc (native code), it executes it blindly and\n        pushes the return values to the stack. If it's a WASM function, it enters the outermost code block.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-invoke\n\n        :param stack: The current stack, to use for execution\n        :param funcaddr: The address of the function to invoke\n        :param store: The current store, to use for execution\n        "
        assert funcaddr in range(len(store.funcs))
        f: ProtoFuncInst = store.funcs[funcaddr]
        ty = f.type
        assert len(ty.result_types) <= 1
        local_vars: typing.List[Value] = []
        for v in [stack.pop() for _ in ty.param_types][::-1]:
            assert not isinstance(v, (Label, Activation))
            local_vars.append(v)
        name = self.function_names.get(funcaddr, f'Func{funcaddr}')
        buffer = ' | ' * (len(self._block_depths) - 1)
        logger.debug(buffer + '%s(%s)' % (name, ', '.join((str(i) for i in local_vars))))
        if isinstance(f, HostFunc):
            self._publish('will_call_hostfunc', f, local_vars)
            res = list(f.hostcode(self._state, *local_vars))
            self._publish('did_call_hostfunc', f, local_vars, res)
            logger.info('HostFunc returned: %s', res)
            assert len(res) == len(ty.result_types)
            for (r, t) in zip(res, ty.result_types):
                assert t in {I32, I64, F32, F64}
                stack.push(t.cast(r))
        else:
            assert isinstance(f, FuncInst), 'Got a non-WASM function! (Maybe cast to HostFunc?)'
            for cast in f.code.locals:
                local_vars.append(cast(0))
            frame = Frame(local_vars, f.module)
            stack.push(Activation(len(ty.result_types), frame, expected_block_depth=len(self._block_depths)))
            self._block_depths.append(0)
            self.block(store, stack, ty.result_types, f.code.body)

    def exec_expression(self, store: Store, stack: 'Stack', expr: WASMExpression):
        if False:
            while True:
                i = 10
        '\n        Pushes the given expression to the stack, calls exec_instruction until there are no more instructions to exec,\n        then returns the top value on the stack. Used during initialization to calculate global values, memory offsets,\n        element offsets, etc.\n\n        :param expr: The expression to execute\n        :return: The result of the expression\n        '
        self.push_instructions(expr)
        self._publish('will_exec_expression', expr)
        while self.exec_instruction(store, stack):
            pass
        self._publish('did_exec_expression', expr, stack.peek())
        return stack.pop()

    def enter_block(self, insts, label: 'Label', stack: 'Stack'):
        if False:
            while True:
                i = 10
        '\n        Push the instructions for the next block to the queue and bump the block depth number\n\n        https://www.w3.org/TR/wasm-core-1/#exec-instr-seq-enter\n\n        :param insts: Instructions for this block\n        :param label: Label referencing the continuation of this block\n        :param stack: The execution stack (where we push the label)\n        '
        stack.push(label)
        self._block_depths[-1] += 1
        self.push_instructions(insts)

    def exit_block(self, stack: 'Stack'):
        if False:
            print('Hello World!')
        '\n        Cleans up after execution of a code block.\n\n        https://www.w3.org/TR/wasm-core-1/#exiting--hrefsyntax-instrmathitinstrast-with-label--l\n        '
        label_idx = stack.find_type(Label)
        if label_idx is not None:
            logger.debug('EXITING BLOCK (FD: %d, BD: %d)', len(self._block_depths), self._block_depths[-1])
            vals = []
            while not isinstance(stack.peek(), Label):
                vals.append(stack.pop())
                assert isinstance(vals[-1], Value_t), f'{type(vals[-1])} is not a value or a label'
            label = stack.pop()
            assert isinstance(label, Label), f'Stack contained a {type(label)} instead of a Label'
            self._block_depths[-1] -= 1
            for v in vals[::-1]:
                stack.push(v)

    def exit_function(self, stack: 'AtomicStack'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Discards the current frame, allowing execution to return to the point after the call\n\n        https://www.w3.org/TR/wasm-core-1/#returning-from-a-function%E2%91%A0\n        '
        if len(self._block_depths) > 1:
            f = stack.get_frame()
            n = f.arity
            stack.has_type_on_top(Value_t, n)
            vals = [stack.pop() for _ in range(n)]
            logger.debug('EXITING FUNCTION (FD: %d, BD: %d) (%s)', len(self._block_depths), self._block_depths[-1], vals)
            assert isinstance(stack.peek(), Activation), f'Stack should have an activation on top, instead has {type(stack.peek())}'
            self._block_depths.pop()
            stack.pop()
            for v in vals[::-1]:
                stack.push(v)

    def push_instructions(self, insts: WASMExpression):
        if False:
            print('Hello World!')
        '\n        Pushes instructions into the instruction queue.\n        :param insts: Instructions to push\n        '
        for i in insts[::-1]:
            self._instruction_queue.appendleft(i)

    def look_forward(self, *opcodes) -> typing.List[Instruction]:
        if False:
            i = 10
            return i + 15
        '\n        Pops contents of the instruction queue until it finds an instruction with an opcode in the argument *opcodes.\n        Used to find the end of a code block in the flat instruction queue. For this reason, it calls itself\n        recursively (looking for the `end` instruction) if it encounters a `block`, `loop`, or `if` instruction.\n\n        :param opcodes: Tuple of instruction opcodes to look for\n        :return: The list of instructions popped before encountering the target instruction.\n        '
        out = []
        i = self._instruction_queue.popleft()
        while i.opcode not in opcodes:
            out.append(i)
            if i.opcode in {2, 3, 4}:
                out += self.look_forward(11)
            if len(self._instruction_queue) == 0:
                raise RuntimeError("Couldn't find an instruction with opcode " + ', '.join((hex(op) for op in opcodes)))
            i = self._instruction_queue.popleft()
        out.append(i)
        return out

    def exec_instruction(self, store: Store, stack: 'Stack', advice: typing.Optional[typing.List[bool]]=None, current_state=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        The core instruction execution function. Pops an instruction from the queue, then dispatches it to the Executor\n        if it's a numeric instruction, or executes it internally if it's a control-flow instruction.\n\n        :param store: The execution Store to use, passed in from the parent WASMWorld. This is passed to almost all\n        | instruction implementations, but for brevity's sake, it's only explicitly documented here.\n        :param stack: The execution Stack to use, likewise passed in from the parent WASMWorld and only documented here,\n        | despite being passed to all the instruction implementations.\n        :param advice: A list of concretized conditions to advice execution of the instruction.\n        :return: True if execution succeeded, False if there are no more instructions to execute\n        "
        ret_type_map = {-1: [I32], -2: [I64], -3: [F32], -4: [F64], -64: []}
        self._advice = advice
        self._state = current_state
        with AtomicStack(stack) as aStack:
            if self._instruction_queue:
                try:
                    inst = self._instruction_queue.popleft()
                    logger.info('%s: %s (%s)', hex(inst.opcode), inst.mnemonic, debug(inst.imm) if inst.imm else '')
                    self._publish('will_execute_instruction', inst)
                    if 2 <= inst.opcode <= 17:
                        self.executor.zero_div = _eval_maybe_symbolic(self._state, self.executor.zero_div)
                        if self.executor.zero_div:
                            raise ZeroDivisionTrap()
                        self.executor.overflow = _eval_maybe_symbolic(self._state, self.executor.overflow)
                        if self.executor.overflow:
                            raise OverflowDivisionTrap()
                        if inst.opcode == 2:
                            self.block(store, aStack, ret_type_map[inst.imm.sig], self.look_forward(11))
                        elif inst.opcode == 3:
                            self.loop(store, aStack, inst)
                        elif inst.opcode == 4:
                            self.if_(store, aStack, ret_type_map[inst.imm.sig])
                        elif inst.opcode == 5:
                            self.else_(store, aStack)
                        elif inst.opcode == 11:
                            self.end(store, aStack)
                        elif inst.opcode == 12:
                            self.br(store, aStack, inst.imm.relative_depth)
                        elif inst.opcode == 13:
                            assert isinstance(inst.imm, BranchImm)
                            self.br_if(store, aStack, inst.imm)
                        elif inst.opcode == 14:
                            assert isinstance(inst.imm, BranchTableImm)
                            self.br_table(store, aStack, inst.imm)
                        elif inst.opcode == 15:
                            self.return_(store, aStack)
                        elif inst.opcode == 16:
                            assert isinstance(inst.imm, CallImm)
                            self.call(store, aStack, inst.imm)
                        elif inst.opcode == 17:
                            assert isinstance(inst.imm, CallIndirectImm)
                            self.call_indirect(store, aStack, inst.imm)
                        else:
                            raise Exception('Unhandled control flow instruction')
                    else:
                        self.executor.dispatch(inst, store, aStack)
                    self._publish('did_execute_instruction', inst)
                    return True
                except Concretize as exc:
                    self._instruction_queue.appendleft(inst)
                    raise exc
                except Trap as exc:
                    self._block_depths.pop()
                    logger.info('Trap: %s', str(exc))
                    self._publish('will_raise_trap', exc)
                    raise exc
            elif aStack.find_type(Label):
                logger.info('The instruction queue is empty, but there are still labels on the stack. This should only happen when re-executing after a Trap')
        return False

    def get_export(self, name: str, store: Store) -> typing.Union[ProtoFuncInst, TableInst, MemInst, GlobalInst, typing.Callable]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves a value exported by this module instance from store\n\n        :param name: The name of the exported value to get\n        :param store: The current execution store (where the export values live)\n        :return: The value of the export\n        '
        export_addr = self.get_export_address(name)
        if isinstance(export_addr, FuncAddr):
            return store.funcs[export_addr]
        if isinstance(export_addr, TableAddr):
            return store.tables[export_addr]
        if isinstance(export_addr, MemAddr):
            return store.mems[export_addr]
        if isinstance(export_addr, GlobalAddr):
            return store.globals[export_addr]
        raise RuntimeError('Unkown export type: ' + str(type(export_addr)))

    def get_export_address(self, name: str) -> typing.Union[FuncAddr, TableAddr, MemAddr, GlobalAddr]:
        if False:
            print('Hello World!')
        '\n        Retrieves the address of a value exported by this module within the store\n\n        :param name: The name of the exported value to get\n        :return: The address of the desired export\n        '
        if name not in self.export_map:
            raise MissingExportException(name)
        export: ExportInst = self.exports[self.export_map[name]]
        assert export.name == name, f'Export name mismatch (expected {name}, got {export.name})'
        return export.value

    def block(self, store: 'Store', stack: 'Stack', ret_type: typing.List[ValType], insts: WASMExpression):
        if False:
            return 10
        '\n        Execute a block of instructions. Creates a label with an empty continuation and the proper arity, then enters\n        the block of instructions with that label.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-block\n\n        :param ret_type: List of expected return types for this block. Really only need the arity\n        :param insts: Instructions to execute\n        '
        arity = len(ret_type)
        label = Label(arity, [])
        self.enter_block(insts, label, stack)

    def loop(self, store: 'Store', stack: 'AtomicStack', loop_inst):
        if False:
            print('Hello World!')
        '\n        Enter a loop block. Creates a label with a copy of the loop as a continuation, then enters the loop instructions\n        with that label.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-loop\n\n        :param loop_inst: The current insrtuction\n        '
        insts = self.look_forward(11)
        label = Label(0, [loop_inst] + insts)
        self.enter_block(insts, label, stack)

    def extract_block(self, partial_list: typing.Deque[Instruction]) -> typing.Deque[Instruction]:
        if False:
            i = 10
            return i + 15
        '\n        Recursively extracts blocks from a list of instructions, similar to self.look_forward. The primary difference\n        is that this version takes a list of instructions to operate over, instead of popping instructions from the\n        instruction queue.\n\n        :param partial_list: List of instructions to extract the block from\n        :return: The extracted block\n        '
        out: typing.Deque[Instruction] = deque()
        i = partial_list.popleft()
        while i.opcode != 11:
            out.append(i)
            if i.opcode in {2, 3, 4}:
                out += self.extract_block(partial_list)
            if len(partial_list) == 0:
                raise RuntimeError("Couldn't find an end to this block!")
            i = partial_list.popleft()
        out.append(i)
        return out

    def _split_if_block(self, partial_list: typing.Deque[Instruction]) -> typing.Tuple[typing.Deque[Instruction], typing.Deque[Instruction]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Splits an if block into its true and false portions. Handles nested blocks in both the true and false branches,\n        and when one branch is empty and/or the else instruction is missing.\n\n        :param partial_list: Complete if block that needs to be split\n        :return: The true block and the false block\n        '
        t_block: typing.Deque[Instruction] = deque()
        assert partial_list[-1].opcode == 11, 'This block is missing an end instruction!'
        i = partial_list.popleft()
        while i.opcode not in {5, 11}:
            t_block.append(i)
            if i.opcode in {2, 3, 4}:
                t_block += self.extract_block(partial_list)
            if len(partial_list) == 0:
                raise RuntimeError("Couldn't find an end to this if statement!")
            i = partial_list.popleft()
        t_block.append(i)
        return (t_block, partial_list)

    def if_(self, store: 'Store', stack: 'AtomicStack', ret_type: typing.List[type]):
        if False:
            return 10
        '\n        Brackets two nested sequences of instructions. If the value on top of the stack is nonzero, enter the first\n        block. If not, enter the second.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-if\n        '
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if self._advice is not None:
            cond = self._advice[0]
        elif isinstance(i, Expression):
            raise ConcretizeCondition('Concretizing if_', i != 0, self._advice)
        else:
            cond = i != 0
        insn_block = self.look_forward(11)
        (t_block, f_block) = self._split_if_block(deque(insn_block))
        label = Label(len(ret_type), [])
        if cond:
            self.enter_block(list(t_block), label, stack)
        else:
            if len(f_block) == 0:
                assert t_block[-1].opcode == 11
                f_block.append(t_block[-1])
            self.enter_block(list(f_block), label, stack)

    def else_(self, store: 'Store', stack: 'AtomicStack'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Marks the end of the first block of an if statement.\n        Typically, `if` blocks look like: `if <instructions> else <instructions> end`. That's not always the case. See:\n        https://webassembly.github.io/spec/core/text/instructions.html#abbreviations\n        "
        self.exit_block(stack)

    def end(self, store: 'Store', stack: 'AtomicStack'):
        if False:
            print('Hello World!')
        '\n        Marks the end of an instruction block or function\n        '
        if self._block_depths[-1] > 0:
            self.exit_block(stack)
        if self._block_depths[-1] == 0:
            self.exit_function(stack)

    def br(self, store: 'Store', stack: 'AtomicStack', label_depth: int):
        if False:
            while True:
                i = 10
        '\n        Branch to the `label_depth`th label deep on the stack\n\n        https://www.w3.org/TR/wasm-core-1/#exec-br\n        '
        assert stack.has_at_least(Label, label_depth + 1)
        label: Label = stack.get_nth(Label, label_depth)
        stack.has_type_on_top(Value_t, label.arity)
        vals = [stack.pop() for _ in range(label.arity)]
        for _ in range(label_depth + 1):
            while isinstance(stack.peek(), Value_t):
                stack.pop()
            assert isinstance(stack.peek(), Label)
            stack.pop()
            assert self._block_depths[-1] > 0, 'Trying to break out of a function call'
            self._block_depths[-1] -= 1
        for v in vals[::-1]:
            stack.push(v)
        for _ in range(label_depth + 1):
            self.look_forward(11, 5)
        self.push_instructions(label.instr)

    def br_if(self, store: 'Store', stack: 'AtomicStack', imm: BranchImm):
        if False:
            while True:
                i = 10
        '\n        Perform a branch if the value on top of the stack is nonzero\n\n        https://www.w3.org/TR/wasm-core-1/#exec-br-if\n        '
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if self._advice is not None:
            cond = self._advice[0]
        elif isinstance(i, Expression):
            raise ConcretizeCondition('Concretizing br_if_', i != 0, self._advice)
        else:
            cond = i != 0
        if cond:
            self.br(store, stack, imm.relative_depth)

    def br_table(self, store: 'Store', stack: 'AtomicStack', imm: BranchTableImm):
        if False:
            for i in range(10):
                print('nop')
        '\n        Branch to the nth label deep on the stack where n is found by looking up a value in a table given by the\n        immediate, indexed by the value on top of the stack.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-br-table\n        '
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if self._advice is not None:
            in_range = self._advice[0]
            if not in_range:
                i = I32.cast(imm.target_count)
            elif issymbolic(i):
                raise ConcretizeStack(-1, I32, 'Concretizing br_table index', i)
        elif isinstance(i, Expression):
            raise ConcretizeCondition('Concretizing br_table range check', Operators.AND(i >= 0, i < imm.target_count), self._advice)
        if i in range(imm.target_count):
            assert isinstance(i, int)
            lab = imm.target_table[i]
        else:
            lab = imm.default_target
        self.br(store, stack, lab)

    def return_(self, store: 'Store', stack: 'AtomicStack'):
        if False:
            while True:
                i = 10
        '\n        Return from the function (ie branch to the outermost block)\n\n        https://www.w3.org/TR/wasm-core-1/#exec-return\n        '
        f = stack.get_frame()
        n = f.arity
        stack.has_type_on_top(Value_t, n)
        ret = [stack.pop() for _i in range(n)]
        while not isinstance(stack.peek(), (Activation, Frame)):
            stack.pop()
        assert stack.peek() == f
        stack.pop()
        for r in ret[::-1]:
            stack.push(r)
        while len(self._block_depths) > f.expected_block_depth:
            for i in range(self._block_depths[-1]):
                self.look_forward(11, 5)
            self._block_depths.pop()

    def call(self, store: 'Store', stack: 'AtomicStack', imm: CallImm):
        if False:
            return 10
        '\n        Invoke the function at the address in the store given by the immediate.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-call\n        '
        f = stack.get_frame()
        assert imm.function_index in range(len(f.frame.module.funcaddrs))
        a = f.frame.module.funcaddrs[imm.function_index]
        self._invoke_inner(stack, a, store)

    def call_indirect(self, store: 'Store', stack: 'AtomicStack', imm: CallIndirectImm):
        if False:
            while True:
                i = 10
        '\n        A function call, but with extra steps. Specifically, you find the index of the function to call by looking in\n        the table at the index given by the immediate.\n\n        https://www.w3.org/TR/wasm-core-1/#exec-call-indirect\n        '
        f = stack.get_frame()
        assert f.frame.module.tableaddrs
        ta = f.frame.module.tableaddrs[0]
        assert ta in range(len(store.tables))
        tab = store.tables[ta]
        assert imm.type_index in range(len(f.frame.module.types))
        ft_expect = f.frame.module.types[imm.type_index]
        stack.has_type_on_top(I32, 1)
        item = stack.pop()
        if self._advice is not None:
            in_range = self._advice[0]
            if not in_range:
                i = I32.cast(len(tab.elem))
            elif issymbolic(item):
                raise ConcretizeStack(-1, I32, 'Concretizing call_indirect operand', item)
            else:
                i = item
        elif isinstance(item, Expression):
            raise ConcretizeCondition('Concretizing call_indirect range check', (item >= 0) & (item < len(tab.elem)), self._advice)
        else:
            i = item
            assert isinstance(i, I32)
        if i not in range(len(tab.elem)):
            raise NonExistentFunctionCallTrap()
        if tab.elem[i] is None:
            raise NonExistentFunctionCallTrap()
        a = tab.elem[i]
        assert a is not None
        assert a in range(len(store.funcs))
        func = store.funcs[a]
        ft_actual = func.type
        if ft_actual != ft_expect:
            raise TypeMismatchTrap(ft_actual, ft_expect)
        self._invoke_inner(stack, a, store)

@dataclass
class Label:
    """
    A branch label that can be pushed onto the stack and then jumped to

    https://www.w3.org/TR/wasm-core-1/#labels%E2%91%A0
    """
    arity: int
    instr: typing.List[Instruction]

@dataclass
class Frame:
    """
    Holds more call data, nested inside an activation (for reasons I don't understand)

    https://www.w3.org/TR/wasm-core-1/#activations-and-frames%E2%91%A0
    """
    locals: typing.List[Value]
    module: ModuleInstance

@dataclass
class Activation:
    """
    Pushed onto the stack with each function invocation to keep track of the call stack

    https://www.w3.org/TR/wasm-core-1/#activations-and-frames%E2%91%A0
    """
    arity: int
    frame: Frame
    expected_block_depth: int

    def __init__(self, arity, frame, expected_block_depth=0):
        if False:
            i = 10
            return i + 15
        self.arity = arity
        self.frame = frame
        self.expected_block_depth = expected_block_depth
StackItem = typing.Union[Value, Label, Activation]

class Stack(Eventful):
    """
    Stores the execution stack & provides helper methods

    https://www.w3.org/TR/wasm-core-1/#stack%E2%91%A0
    """
    data: typing.Deque[StackItem]
    _published_events = {'push_item', 'pop_item'}

    def __init__(self, init_data=None):
        if False:
            return 10
        '\n        :param init_data: Optional initialization value\n        '
        self.data = init_data if init_data else deque()
        super().__init__()

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = super().__getstate__()
        state['data'] = self.data
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.data = state['data']
        super().__setstate__(state)

    def push(self, val: StackItem) -> None:
        if False:
            print('Hello World!')
        '\n        Push a value to the stack\n\n        :param val: The value to push\n        :return: None\n        '
        if isinstance(val, list):
            raise RuntimeError("Don't push lists")
        logger.debug('+%d %s (%s)', len(self.data), val, type(val))
        self._publish('will_push_item', val, len(self.data))
        self.data.append(val)
        self._publish('did_push_item', val, len(self.data))

    def pop(self) -> StackItem:
        if False:
            return 10
        '\n        Pop a value from the stack\n\n        :return: the popped value\n        '
        logger.debug('-%d %s (%s)', len(self.data) - 1, self.peek(), type(self.peek()))
        self._publish('will_pop_item', len(self.data))
        item = self.data.pop()
        self._publish('did_pop_item', item, len(self.data))
        return item

    def peek(self) -> typing.Optional[StackItem]:
        if False:
            return 10
        '\n        :return: the item on top of the stack (without removing it)\n        '
        if self.data:
            return self.data[-1]
        return None

    def empty(self) -> bool:
        if False:
            return 10
        '\n        :return: True if the stack is empty, otherwise False\n        '
        return len(self.data) == 0

    def has_type_on_top(self, t: typing.Union[type, typing.Tuple[type, ...]], n: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        *Asserts* that the stack has at least n values of type t or type BitVec on the top\n\n        :param t: type of value to look for (Bitvec is always included as an option)\n        :param n: Number of values to check\n        :return: True\n        '
        for i in range(1, n + 1):
            assert isinstance(self.data[i * -1], (t, BitVec)), f'{type(self.data[i * -1])} is not an {t}!'
        return True

    def find_type(self, t: type) -> typing.Optional[int]:
        if False:
            print('Hello World!')
        '\n        :param t: The type to look for\n        :return: The depth of the first value of type t\n        '
        for (idx, v) in enumerate(reversed(self.data)):
            if isinstance(v, t):
                return -1 * idx
        return None

    def has_at_least(self, t: type, n: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        :param t: type to look for\n        :param n: number to look for\n        :return: whether the stack contains at least n values of type t\n        '
        count = 0
        for v in reversed(self.data):
            if isinstance(v, t):
                count += 1
            if count == n:
                return True
        return False

    def get_nth(self, t: type, n: int) -> typing.Optional[StackItem]:
        if False:
            print('Hello World!')
        '\n        :param t: type to look for\n        :param n: number to look for\n        :return: the nth item of type t from the top of the stack, or None\n        '
        seen = 0
        for v in reversed(self.data):
            if isinstance(v, t):
                if seen == n:
                    return v
                seen += 1
        return None

    def get_frame(self) -> Activation:
        if False:
            i = 10
            return i + 15
        '\n        :return: the topmost frame (Activation) on the stack\n        '
        for item in reversed(self.data):
            if isinstance(item, Activation):
                return item
        raise RuntimeError("Couldn't find a frame on the stack")

class AtomicStack(Stack):
    """
    Allows for the rolling-back of the stack in the event of a concretization exception.
    Inherits from Stack so that the types will be correct, but never calls `super`.
    Provides a context manager that will intercept Concretization Exceptions before raising them.
    """

    class PushItem:
        pass

    @dataclass
    class PopItem:
        val: StackItem

    def __init__(self, parent: Stack):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.actions: typing.List[typing.Union[AtomicStack.PushItem, AtomicStack.PopItem]] = []

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = {'parent': self.parent, 'actions': self.actions}
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.parent = state['parent']
        self.actions = state['actions']

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        if isinstance(exc_val, Concretize):
            logger.info('Rolling back stack for concretization')
            self.rollback()

    def rollback(self):
        if False:
            while True:
                i = 10
        while self.actions:
            action = self.actions.pop()
            if isinstance(action, AtomicStack.PopItem):
                self.parent.push(action.val)
            elif isinstance(action, AtomicStack.PushItem):
                self.parent.pop()

    def push(self, val: StackItem) -> None:
        if False:
            i = 10
            return i + 15
        self.actions.append(AtomicStack.PushItem())
        self.parent.push(val)

    def pop(self) -> StackItem:
        if False:
            while True:
                i = 10
        val = self.parent.pop()
        self.actions.append(AtomicStack.PopItem(val))
        return val

    def peek(self):
        if False:
            print('Hello World!')
        return self.parent.peek()

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.empty()

    def has_type_on_top(self, t: typing.Union[type, typing.Tuple[type, ...]], n: int):
        if False:
            print('Hello World!')
        return self.parent.has_type_on_top(t, n)

    def find_type(self, t: type):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.find_type(t)

    def has_at_least(self, t: type, n: int):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.has_at_least(t, n)

    def get_nth(self, t: type, n: int):
        if False:
            return 10
        return self.parent.get_nth(t, n)

    def get_frame(self) -> Activation:
        if False:
            while True:
                i = 10
        return self.parent.get_frame()

@dataclass
class FuncInst(ProtoFuncInst):
    """
    Instance type for WASM functions
    """
    module: ModuleInstance
    code: 'Function'

@dataclass
class HostFunc(ProtoFuncInst):
    """
    Instance type for native functions that have been provided via import
    """
    hostcode: types.FunctionType

    def allocate(self, store: Store, functype: FunctionType, host_func: types.FunctionType) -> FuncAddr:
        if False:
            return 10
        '\n        Currently not needed.\n\n        https://www.w3.org/TR/wasm-core-1/#host-functions%E2%91%A2\n        '
        pass

class ConcretizeCondition(Concretize):
    """Tells Manticore to concretize a condition required to direct execution."""

    def __init__(self, message: str, condition: Bool, current_advice: typing.Optional[typing.List[bool]], **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :param message: Debug message describing the reason for concretization\n        :param condition: The boolean expression to concretize\n        '
        advice = current_advice if current_advice is not None else []

        def setstate(state, value: bool):
            if False:
                return 10
            state.platform.advice = advice + [value]
        super().__init__(message, condition, setstate, **kwargs)