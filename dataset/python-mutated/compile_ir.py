import copy
import functools
import math
from dataclasses import dataclass
import cbor2
from vyper.codegen.ir_node import IRnode
from vyper.compiler.settings import OptimizationLevel
from vyper.evm.opcodes import get_opcodes, version_check
from vyper.exceptions import CodegenPanic, CompilerPanic
from vyper.utils import MemoryPositions
from vyper.version import version_tuple
PUSH_OFFSET = 95
DUP_OFFSET = 127
SWAP_OFFSET = 143

def num_to_bytearray(x):
    if False:
        return 10
    o = []
    while x > 0:
        o.insert(0, x % 256)
        x //= 256
    return o

def PUSH(x):
    if False:
        i = 10
        return i + 15
    bs = num_to_bytearray(x)
    if len(bs) == 0 and (not version_check(begin='shanghai')):
        bs = [0]
    return [f'PUSH{len(bs)}'] + bs

def PUSH_N(x, n):
    if False:
        for i in range(10):
            print('nop')
    o = []
    for _i in range(n):
        o.insert(0, x % 256)
        x //= 256
    assert x == 0
    return [f'PUSH{len(o)}'] + o
_next_symbol = 0

def mksymbol(name=''):
    if False:
        print('Hello World!')
    global _next_symbol
    _next_symbol += 1
    return f'_sym_{name}{_next_symbol}'

def mkdebug(pc_debugger, source_pos):
    if False:
        i = 10
        return i + 15
    i = Instruction('DEBUG', source_pos)
    i.pc_debugger = pc_debugger
    return [i]

def is_symbol(i):
    if False:
        return 10
    return isinstance(i, str) and i.startswith('_sym_')

def is_mem_sym(i):
    if False:
        i = 10
        return i + 15
    return isinstance(i, str) and i.startswith('_mem_')

def is_ofst(sym):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(sym, str) and sym == '_OFST'

def _runtime_code_offsets(ctor_mem_size, runtime_codelen):
    if False:
        print('Hello World!')
    runtime_code_end = max(ctor_mem_size, runtime_codelen)
    runtime_code_start = runtime_code_end - runtime_codelen
    return (runtime_code_start, runtime_code_end)

def calc_mem_ofst_size(ctor_mem_size):
    if False:
        i = 10
        return i + 15
    return math.ceil(math.log(ctor_mem_size + 1, 256))

def _rewrite_return_sequences(ir_node, label_params=None):
    if False:
        print('Hello World!')
    args = ir_node.args
    if ir_node.value == 'return':
        if args[0].value == 'ret_ofst' and args[1].value == 'ret_len':
            ir_node.args[0].value = 'pass'
            ir_node.args[1].value = 'pass'
    if ir_node.value == 'exit_to':
        if args[0].value == 'return_pc':
            ir_node.value = 'jump'
            args[0].value = 'pass'
        else:
            ir_node.value = 'seq'
            _t = ['seq']
            if 'return_buffer' in label_params:
                _t.append(['pop', 'pass'])
            dest = args[0].value
            more_args = ['pass' if t.value == 'return_pc' else t for t in args[1:]]
            _t.append(['goto', dest] + more_args)
            ir_node.args = IRnode.from_list(_t, source_pos=ir_node.source_pos).args
    if ir_node.value == 'label':
        label_params = set((t.value for t in ir_node.args[1].args))
    for t in args:
        _rewrite_return_sequences(t, label_params)

def _assert_false():
    if False:
        i = 10
        return i + 15
    global _revert_label
    return [_revert_label, 'JUMPI']

def _add_postambles(asm_ops):
    if False:
        while True:
            i = 10
    to_append = []
    global _revert_label
    _revert_string = [_revert_label, 'JUMPDEST', *PUSH(0), 'DUP1', 'REVERT']
    if _revert_label in asm_ops:
        to_append.extend(_revert_string)
    if len(to_append) > 0:
        runtime = None
        if isinstance(asm_ops[-1], list) and isinstance(asm_ops[-1][0], _RuntimeHeader):
            runtime = asm_ops.pop()
        asm_ops.append('STOP')
        asm_ops.extend(to_append)
        if runtime:
            asm_ops.append(runtime)
    for t in asm_ops:
        if isinstance(t, list):
            _add_postambles(t)

class Instruction(str):

    def __new__(cls, sstr, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return super().__new__(cls, sstr)

    def __init__(self, sstr, source_pos=None, error_msg=None):
        if False:
            while True:
                i = 10
        self.error_msg = error_msg
        self.pc_debugger = False
        if source_pos is not None:
            (self.lineno, self.col_offset, self.end_lineno, self.end_col_offset) = source_pos
        else:
            (self.lineno, self.col_offset, self.end_lineno, self.end_col_offset) = [None] * 4

def apply_line_numbers(func):
    if False:
        print('Hello World!')

    @functools.wraps(func)
    def apply_line_no_wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        code = args[0]
        ret = func(*args, **kwargs)
        new_ret = [Instruction(i, code.source_pos, code.error_msg) if isinstance(i, str) and (not isinstance(i, Instruction)) else i for i in ret]
        return new_ret
    return apply_line_no_wrapper

@apply_line_numbers
def compile_to_assembly(code, optimize=OptimizationLevel.GAS):
    if False:
        return 10
    global _revert_label
    _revert_label = mksymbol('revert')
    code = copy.deepcopy(code)
    _rewrite_return_sequences(code)
    res = _compile_to_assembly(code)
    _add_postambles(res)
    _relocate_segments(res)
    if optimize != OptimizationLevel.NONE:
        _optimize_assembly(res)
    return res

@apply_line_numbers
def _compile_to_assembly(code, withargs=None, existing_labels=None, break_dest=None, height=0):
    if False:
        return 10
    if withargs is None:
        withargs = {}
    if not isinstance(withargs, dict):
        raise CompilerPanic(f'Incorrect type for withargs: {type(withargs)}')

    def _data_ofst_of(sym, ofst, height_):
        if False:
            return 10
        assert is_symbol(sym) or is_mem_sym(sym)
        if isinstance(ofst.value, int):
            return ['_OFST', sym, ofst.value]
        else:
            ofst = _compile_to_assembly(ofst, withargs, existing_labels, break_dest, height_)
            return ofst + [sym, 'ADD']

    def _height_of(witharg):
        if False:
            return 10
        ret = height - withargs[witharg]
        if ret > 16:
            raise Exception('With statement too deep')
        return ret
    if existing_labels is None:
        existing_labels = set()
    if not isinstance(existing_labels, set):
        raise CompilerPanic(f'must be set(), but got {type(existing_labels)}')
    if isinstance(code.value, str) and code.value.upper() in get_opcodes():
        o = []
        for (i, c) in enumerate(code.args[::-1]):
            o.extend(_compile_to_assembly(c, withargs, existing_labels, break_dest, height + i))
        o.append(code.value.upper())
        return o
    elif isinstance(code.value, int):
        if code.value < -2 ** 255:
            raise Exception(f'Value too low: {code.value}')
        elif code.value >= 2 ** 256:
            raise Exception(f'Value too high: {code.value}')
        return PUSH(code.value % 2 ** 256)
    elif isinstance(code.value, str) and code.value in withargs:
        return ['DUP' + str(_height_of(code.value))]
    elif code.value == 'set':
        if len(code.args) != 2 or code.args[0].value not in withargs:
            raise Exception('Set expects two arguments, the first being a stack variable')
        if height - withargs[code.args[0].value] > 16:
            raise Exception('With statement too deep')
        return _compile_to_assembly(code.args[1], withargs, existing_labels, break_dest, height) + ['SWAP' + str(height - withargs[code.args[0].value]), 'POP']
    elif code.value in ('pass', 'dummy'):
        return []
    elif code.value == 'dload':
        loc = code.args[0]
        o = []
        o.extend(PUSH(32))
        o.extend(_data_ofst_of('_sym_code_end', loc, height + 1))
        o.extend(PUSH(MemoryPositions.FREE_VAR_SPACE) + ['CODECOPY'])
        o.extend(PUSH(MemoryPositions.FREE_VAR_SPACE) + ['MLOAD'])
        return o
    elif code.value == 'dloadbytes':
        dst = code.args[0]
        src = code.args[1]
        len_ = code.args[2]
        o = []
        o.extend(_compile_to_assembly(len_, withargs, existing_labels, break_dest, height))
        o.extend(_data_ofst_of('_sym_code_end', src, height + 1))
        o.extend(_compile_to_assembly(dst, withargs, existing_labels, break_dest, height + 2))
        o.extend(['CODECOPY'])
        return o
    elif code.value == 'iload':
        loc = code.args[0]
        o = []
        o.extend(_data_ofst_of('_mem_deploy_end', loc, height))
        o.append('MLOAD')
        return o
    elif code.value == 'istore':
        loc = code.args[0]
        val = code.args[1]
        o = []
        o.extend(_compile_to_assembly(val, withargs, existing_labels, break_dest, height))
        o.extend(_data_ofst_of('_mem_deploy_end', loc, height + 1))
        o.append('MSTORE')
        return o
    elif code.value == 'istorebytes':
        raise Exception('unimplemented')
    elif code.value == 'if' and len(code.args) == 2:
        o = []
        o.extend(_compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height))
        end_symbol = mksymbol('join')
        o.extend(['ISZERO', end_symbol, 'JUMPI'])
        o.extend(_compile_to_assembly(code.args[1], withargs, existing_labels, break_dest, height))
        o.extend([end_symbol, 'JUMPDEST'])
        return o
    elif code.value == 'if' and len(code.args) == 3:
        o = []
        o.extend(_compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height))
        mid_symbol = mksymbol('else')
        end_symbol = mksymbol('join')
        o.extend(['ISZERO', mid_symbol, 'JUMPI'])
        o.extend(_compile_to_assembly(code.args[1], withargs, existing_labels, break_dest, height))
        o.extend([end_symbol, 'JUMP', mid_symbol, 'JUMPDEST'])
        o.extend(_compile_to_assembly(code.args[2], withargs, existing_labels, break_dest, height))
        o.extend([end_symbol, 'JUMPDEST'])
        return o
    elif code.value == 'repeat':
        o = []
        if len(code.args) != 5:
            raise CompilerPanic('bad number of repeat args')
        i_name = code.args[0]
        start = code.args[1]
        rounds = code.args[2]
        rounds_bound = code.args[3]
        body = code.args[4]
        (entry_dest, continue_dest, exit_dest) = (mksymbol('loop_start'), mksymbol('loop_continue'), mksymbol('loop_exit'))
        o.extend(_compile_to_assembly(start, withargs, existing_labels, break_dest, height))
        o.extend(_compile_to_assembly(rounds, withargs, existing_labels, break_dest, height + 1))
        if rounds != rounds_bound:
            o.extend(_compile_to_assembly(rounds_bound, withargs, existing_labels, break_dest, height + 2))
            o.extend(['DUP2', 'GT'] + _assert_false())
            o.extend(['DUP1', 'ISZERO', exit_dest, 'JUMPI'])
        if start.value != 0:
            o.extend(['DUP2', 'ADD'])
        o.extend(['SWAP1'])
        if i_name.value in withargs:
            raise CompilerPanic(f'shadowed loop variable {i_name}')
        withargs[i_name.value] = height + 1
        o.extend([entry_dest, 'JUMPDEST'])
        o.extend(_compile_to_assembly(body, withargs, existing_labels, (exit_dest, continue_dest, height + 2), height + 2))
        del withargs[i_name.value]
        o.extend(['POP'] * body.valency)
        o.extend([continue_dest, 'JUMPDEST', 'PUSH1', 1, 'ADD'])
        o.extend(['DUP2', 'DUP2', 'XOR', entry_dest, 'JUMPI'])
        o.extend([exit_dest, 'JUMPDEST', 'POP', 'POP'])
        return o
    elif code.value == 'continue':
        if not break_dest:
            raise CompilerPanic('Invalid break')
        (dest, continue_dest, break_height) = break_dest
        return [continue_dest, 'JUMP']
    elif code.value == 'break':
        if not break_dest:
            raise CompilerPanic('Invalid break')
        (dest, continue_dest, break_height) = break_dest
        n_local_vars = height - break_height
        cleanup_local_vars = ['POP'] * n_local_vars
        return cleanup_local_vars + [dest, 'JUMP']
    elif code.value == 'cleanup_repeat':
        if not break_dest:
            raise CompilerPanic('Invalid break')
        (_, _, break_height) = break_dest
        if 'return_buffer' in withargs:
            break_height -= 1
        if 'return_pc' in withargs:
            break_height -= 1
        return ['POP'] * break_height
    elif code.value == 'with':
        o = []
        o.extend(_compile_to_assembly(code.args[1], withargs, existing_labels, break_dest, height))
        old = withargs.get(code.args[0].value, None)
        withargs[code.args[0].value] = height
        o.extend(_compile_to_assembly(code.args[2], withargs, existing_labels, break_dest, height + 1))
        if code.args[2].valency:
            o.extend(['SWAP1', 'POP'])
        else:
            o.extend(['POP'])
        if old is not None:
            withargs[code.args[0].value] = old
        else:
            del withargs[code.args[0].value]
        return o
    elif code.value == 'deploy':
        memsize = code.args[0].value
        ir = code.args[1]
        immutables_len = code.args[2].value
        assert isinstance(memsize, int), 'non-int memsize'
        assert isinstance(immutables_len, int), 'non-int immutables_len'
        runtime_begin = mksymbol('runtime_begin')
        subcode = _compile_to_assembly(ir)
        o = []
        o.extend(['_sym_subcode_size', runtime_begin, '_mem_deploy_start', 'CODECOPY'])
        o.extend(['_OFST', '_sym_subcode_size', immutables_len])
        o.extend(['_mem_deploy_start'])
        o.extend(['RETURN'])
        subcode = [_RuntimeHeader(runtime_begin, memsize, immutables_len)] + subcode
        o.append(subcode)
        return o
    elif code.value == 'seq':
        o = []
        for arg in code.args:
            o.extend(_compile_to_assembly(arg, withargs, existing_labels, break_dest, height))
            if arg.valency == 1 and arg != code.args[-1]:
                o.append('POP')
        return o
    elif code.value == 'assert_unreachable':
        o = _compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height)
        end_symbol = mksymbol('reachable')
        o.extend([end_symbol, 'JUMPI', 'INVALID', end_symbol, 'JUMPDEST'])
        return o
    elif code.value == 'assert':
        o = _compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height)
        o.extend(['ISZERO'])
        o.extend(_assert_false())
        return o
    elif code.value == 'sha3_32':
        o = _compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height)
        o.extend([*PUSH(MemoryPositions.FREE_VAR_SPACE), 'MSTORE', *PUSH(32), *PUSH(MemoryPositions.FREE_VAR_SPACE), 'SHA3'])
        return o
    elif code.value == 'sha3_64':
        o = _compile_to_assembly(code.args[0], withargs, existing_labels, break_dest, height)
        o.extend(_compile_to_assembly(code.args[1], withargs, existing_labels, break_dest, height))
        o.extend([*PUSH(MemoryPositions.FREE_VAR_SPACE2), 'MSTORE', *PUSH(MemoryPositions.FREE_VAR_SPACE), 'MSTORE', *PUSH(64), *PUSH(MemoryPositions.FREE_VAR_SPACE), 'SHA3'])
        return o
    elif code.value == 'select':
        cond = code.args[0]
        a = code.args[1]
        b = code.args[2]
        o = []
        o.extend(_compile_to_assembly(b, withargs, existing_labels, break_dest, height))
        o.extend(_compile_to_assembly(a, withargs, existing_labels, break_dest, height + 1))
        o.extend(['DUP2', 'XOR'])
        o.extend(_compile_to_assembly(cond, withargs, existing_labels, break_dest, height + 2))
        o.extend(['MUL', 'XOR'])
        return o
    elif code.value == 'le':
        return _compile_to_assembly(IRnode.from_list(['iszero', ['gt', code.args[0], code.args[1]]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'ge':
        return _compile_to_assembly(IRnode.from_list(['iszero', ['lt', code.args[0], code.args[1]]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'sle':
        return _compile_to_assembly(IRnode.from_list(['iszero', ['sgt', code.args[0], code.args[1]]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'sge':
        return _compile_to_assembly(IRnode.from_list(['iszero', ['slt', code.args[0], code.args[1]]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'ne':
        return _compile_to_assembly(IRnode.from_list(['iszero', ['eq', code.args[0], code.args[1]]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'ceil32':
        x = code.args[0]
        return _compile_to_assembly(IRnode.from_list(['and', ['add', x, 31], ['not', 31]]), withargs, existing_labels, break_dest, height)
    elif code.value == 'data':
        data_node = [_DataHeader('_sym_' + code.args[0].value)]
        for c in code.args[1:]:
            if isinstance(c.value, int):
                assert 0 <= c < 256, f'invalid data byte {c}'
                data_node.append(c.value)
            elif isinstance(c.value, bytes):
                data_node.append(c.value)
            elif isinstance(c, IRnode):
                assert c.value == 'symbol'
                data_node.extend(_compile_to_assembly(c, withargs, existing_labels, break_dest, height))
            else:
                raise ValueError(f'Invalid data: {type(c)} {c}')
        return [data_node]
    elif code.value == 'goto':
        o = []
        for (i, c) in enumerate(reversed(code.args[1:])):
            o.extend(_compile_to_assembly(c, withargs, existing_labels, break_dest, height + i))
        o.extend(['_sym_' + code.args[0].value, 'JUMP'])
        return o
    elif code.value == 'symbol':
        return ['_sym_' + code.args[0].value]
    elif code.value == 'label':
        label_name = code.args[0].value
        assert isinstance(label_name, str)
        if label_name in existing_labels:
            raise Exception(f'Label with name {label_name} already exists!')
        else:
            existing_labels.add(label_name)
        if code.args[1].value != 'var_list':
            raise CodegenPanic('2nd arg to label must be var_list')
        var_args = code.args[1].args
        body = code.args[2]
        height = 0
        withargs = {}
        for arg in reversed(var_args):
            assert isinstance(arg.value, str)
            withargs[arg.value] = height
            height += 1
        body_asm = _compile_to_assembly(body, withargs=withargs, existing_labels=existing_labels, height=height)
        pop_scoped_vars = []
        return ['_sym_' + label_name, 'JUMPDEST'] + body_asm + pop_scoped_vars
    elif code.value == 'unique_symbol':
        symbol = code.args[0].value
        assert isinstance(symbol, str)
        if symbol in existing_labels:
            raise Exception(f'symbol {symbol} already exists!')
        else:
            existing_labels.add(symbol)
        return []
    elif code.value == 'exit_to':
        raise CodegenPanic('exit_to not implemented yet!')
    elif code.value == 'debugger':
        return mkdebug(pc_debugger=False, source_pos=code.source_pos)
    elif code.value == 'pc_debugger':
        return mkdebug(pc_debugger=True, source_pos=code.source_pos)
    else:
        raise ValueError(f'Weird code element: {type(code)} {code}')

def note_line_num(line_number_map, item, pos):
    if False:
        i = 10
        return i + 15
    if isinstance(item, Instruction):
        if item.lineno is not None:
            offsets = (item.lineno, item.col_offset, item.end_lineno, item.end_col_offset)
        else:
            offsets = None
        line_number_map['pc_pos_map'][pos] = offsets
        if item.error_msg is not None:
            line_number_map['error_map'][pos] = item.error_msg
    added_line_breakpoint = note_breakpoint(line_number_map, item, pos)
    return added_line_breakpoint

def note_breakpoint(line_number_map, item, pos):
    if False:
        i = 10
        return i + 15
    if item == 'DEBUG':
        if item.pc_debugger:
            line_number_map['pc_breakpoints'].add(pos)
        else:
            line_number_map['breakpoints'].add(item.lineno + 1)
_TERMINAL_OPS = ('JUMP', 'RETURN', 'REVERT', 'STOP', 'INVALID')

def _prune_unreachable_code(assembly):
    if False:
        print('Hello World!')
    changed = False
    i = 0
    while i < len(assembly) - 2:
        instr = assembly[i]
        if isinstance(instr, list):
            instr = assembly[i][-1]
        if assembly[i] in _TERMINAL_OPS and (not (is_symbol(assembly[i + 1]) or isinstance(assembly[i + 1], list))):
            changed = True
            del assembly[i + 1]
        else:
            i += 1
    return changed

def _prune_inefficient_jumps(assembly):
    if False:
        for i in range(10):
            print('nop')
    changed = False
    i = 0
    while i < len(assembly) - 4:
        if is_symbol(assembly[i]) and assembly[i + 1] == 'JUMP' and (assembly[i] == assembly[i + 2]) and (assembly[i + 3] == 'JUMPDEST'):
            changed = True
            del assembly[i:i + 2]
        else:
            i += 1
    return changed

def _merge_jumpdests(assembly):
    if False:
        print('Hello World!')
    changed = False
    i = 0
    while i < len(assembly) - 3:
        if is_symbol(assembly[i]) and assembly[i + 1] == 'JUMPDEST':
            current_symbol = assembly[i]
            if is_symbol(assembly[i + 2]) and assembly[i + 3] == 'JUMPDEST':
                new_symbol = assembly[i + 2]
                for j in range(len(assembly)):
                    if assembly[j] == current_symbol and i != j:
                        assembly[j] = new_symbol
                        changed = True
            elif is_symbol(assembly[i + 2]) and assembly[i + 3] == 'JUMP':
                new_symbol = assembly[i + 2]
                for j in range(len(assembly)):
                    if assembly[j] == current_symbol and i != j:
                        assembly[j] = new_symbol
                        changed = True
        i += 1
    return changed
_RETURNS_ZERO_OR_ONE = {'LT', 'GT', 'SLT', 'SGT', 'EQ', 'ISZERO', 'CALL', 'STATICCALL', 'CALLCODE', 'DELEGATECALL'}

def _merge_iszero(assembly):
    if False:
        i = 10
        return i + 15
    changed = False
    i = 0
    while i < len(assembly) - 2:
        if isinstance(assembly[i], str) and assembly[i] in _RETURNS_ZERO_OR_ONE and (assembly[i + 1:i + 3] == ['ISZERO', 'ISZERO']):
            changed = True
            del assembly[i + 1:i + 3]
        else:
            i += 1
    i = 0
    while i < len(assembly) - 3:
        if assembly[i:i + 2] == ['ISZERO', 'ISZERO'] and is_symbol(assembly[i + 2]) and (assembly[i + 3] == 'JUMPI'):
            changed = True
            del assembly[i:i + 2]
        else:
            i += 1
    return changed

def is_symbol_map_indicator(asm_node):
    if False:
        return 10
    return asm_node == 'JUMPDEST'

def _prune_unused_jumpdests(assembly):
    if False:
        for i in range(10):
            print('nop')
    changed = False
    used_jumpdests = set()
    for i in range(len(assembly) - 1):
        if is_symbol(assembly[i]) and (not is_symbol_map_indicator(assembly[i + 1])):
            used_jumpdests.add(assembly[i])
    for item in assembly:
        if isinstance(item, list) and isinstance(item[0], _DataHeader):
            for t in item:
                if is_symbol(t):
                    used_jumpdests.add(t)
    i = 0
    while i < len(assembly) - 2:
        if is_symbol(assembly[i]) and assembly[i] not in used_jumpdests:
            changed = True
            del assembly[i:i + 2]
        else:
            i += 1
    return changed

def _stack_peephole_opts(assembly):
    if False:
        return 10
    changed = False
    i = 0
    while i < len(assembly) - 2:
        if assembly[i:i + 3] == ['DUP1', 'SWAP1', 'POP']:
            changed = True
            del assembly[i:i + 3]
            continue
        if assembly[i:i + 3] == ['SWAP1', 'POP', 'POP']:
            changed = True
            del assembly[i]
            continue
        i += 1
    return changed

def _optimize_assembly(assembly):
    if False:
        print('Hello World!')
    for x in assembly:
        if isinstance(x, list) and isinstance(x[0], _RuntimeHeader):
            _optimize_assembly(x)
    for _ in range(1024):
        changed = False
        changed |= _prune_unreachable_code(assembly)
        changed |= _merge_iszero(assembly)
        changed |= _merge_jumpdests(assembly)
        changed |= _prune_inefficient_jumps(assembly)
        changed |= _prune_unused_jumpdests(assembly)
        changed |= _stack_peephole_opts(assembly)
        if not changed:
            return
    raise CompilerPanic('infinite loop detected during assembly reduction')

def adjust_pc_maps(pc_maps, ofst):
    if False:
        print('Hello World!')
    assert ofst >= 0
    ret = {}
    ret['breakpoints'] = pc_maps['breakpoints'].copy()
    ret['pc_breakpoints'] = {pc + ofst for pc in pc_maps['pc_breakpoints']}
    ret['pc_jump_map'] = {k + ofst: v for (k, v) in pc_maps['pc_jump_map'].items()}
    ret['pc_pos_map'] = {k + ofst: v for (k, v) in pc_maps['pc_pos_map'].items()}
    ret['error_map'] = {k + ofst: v for (k, v) in pc_maps['error_map'].items()}
    return ret
SYMBOL_SIZE = 2

def _data_to_evm(assembly, symbol_map):
    if False:
        return 10
    ret = bytearray()
    assert isinstance(assembly[0], _DataHeader)
    for item in assembly[1:]:
        if is_symbol(item):
            symbol = symbol_map[item].to_bytes(SYMBOL_SIZE, 'big')
            ret.extend(symbol)
        elif isinstance(item, int):
            ret.append(item)
        elif isinstance(item, bytes):
            ret.extend(item)
        else:
            raise ValueError(f'invalid data {type(item)} {item}')
    return ret

def _length_of_data(assembly):
    if False:
        while True:
            i = 10
    ret = 0
    assert isinstance(assembly[0], _DataHeader)
    for item in assembly[1:]:
        if is_symbol(item):
            ret += SYMBOL_SIZE
        elif isinstance(item, int):
            assert 0 <= item < 256, f'invalid data byte {item}'
            ret += 1
        elif isinstance(item, bytes):
            ret += len(item)
        else:
            raise ValueError(f'invalid data {type(item)} {item}')
    return ret

@dataclass
class _RuntimeHeader:
    label: str
    ctor_mem_size: int
    immutables_len: int

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'<RUNTIME {self.label} mem @{self.ctor_mem_size} imms @{self.immutables_len}>'

@dataclass
class _DataHeader:
    label: str

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'DATA {self.label}'

def _relocate_segments(assembly):
    if False:
        i = 10
        return i + 15
    data_segments = []
    non_data_segments = []
    code_segments = []
    for t in assembly:
        if isinstance(t, list):
            if isinstance(t[0], _DataHeader):
                data_segments.append(t)
            else:
                _relocate_segments(t)
                assert isinstance(t[0], _RuntimeHeader)
                code_segments.append(t)
        else:
            non_data_segments.append(t)
    assembly.clear()
    assembly.extend(non_data_segments)
    assembly.extend(code_segments)
    assembly.extend(data_segments)

def assembly_to_evm(assembly, pc_ofst=0, insert_compiler_metadata=False):
    if False:
        i = 10
        return i + 15
    (bytecode, source_maps, _) = assembly_to_evm_with_symbol_map(assembly, pc_ofst=pc_ofst, insert_compiler_metadata=insert_compiler_metadata)
    return (bytecode, source_maps)

def assembly_to_evm_with_symbol_map(assembly, pc_ofst=0, insert_compiler_metadata=False):
    if False:
        i = 10
        return i + 15
    '\n    Assembles assembly into EVM\n\n    assembly: list of asm instructions\n    pc_ofst: when constructing the source map, the amount to offset all\n             pcs by (no effect until we add deploy code source map)\n    insert_compiler_metadata: whether to append vyper metadata to output\n                            (should be true for runtime code)\n    '
    line_number_map = {'breakpoints': set(), 'pc_breakpoints': set(), 'pc_jump_map': {0: '-'}, 'pc_pos_map': {}, 'error_map': {}}
    pc = 0
    symbol_map = {}
    (runtime_code, runtime_code_start, runtime_code_end) = (None, None, None)
    (mem_ofst_size, ctor_mem_size) = (None, None)
    max_mem_ofst = 0
    for (i, item) in enumerate(assembly):
        if isinstance(item, list) and isinstance(item[0], _RuntimeHeader):
            assert runtime_code is None, 'Multiple subcodes'
            assert ctor_mem_size is None
            ctor_mem_size = item[0].ctor_mem_size
            (runtime_code, runtime_map) = assembly_to_evm(item[1:])
            (runtime_code_start, runtime_code_end) = _runtime_code_offsets(ctor_mem_size, len(runtime_code))
            assert runtime_code_end - runtime_code_start == len(runtime_code)
        if is_ofst(item) and is_mem_sym(assembly[i + 1]):
            max_mem_ofst = max(assembly[i + 2], max_mem_ofst)
    if runtime_code_end is not None:
        mem_ofst_size = calc_mem_ofst_size(runtime_code_end + max_mem_ofst)
    data_section_lengths = []
    immutables_len = None
    for (i, item) in enumerate(assembly):
        note_line_num(line_number_map, item, pc)
        if item == 'DEBUG':
            continue
        if item == 'JUMP':
            last = assembly[i - 1]
            if is_symbol(last) and last.startswith('_sym_internal'):
                if last.endswith('cleanup'):
                    line_number_map['pc_jump_map'][pc] = 'o'
                else:
                    line_number_map['pc_jump_map'][pc] = 'i'
            else:
                line_number_map['pc_jump_map'][pc] = '-'
        elif item in ('JUMPI', 'JUMPDEST'):
            line_number_map['pc_jump_map'][pc] = '-'
        if is_symbol(item):
            if is_symbol_map_indicator(assembly[i + 1]):
                if item in symbol_map:
                    raise CompilerPanic(f'duplicate jumpdest {item}')
                symbol_map[item] = pc
            else:
                pc += SYMBOL_SIZE + 1
        elif is_mem_sym(item):
            pc += mem_ofst_size + 1
        elif is_ofst(item):
            assert is_symbol(assembly[i + 1]) or is_mem_sym(assembly[i + 1])
            assert isinstance(assembly[i + 2], int)
            pc -= 1
        elif isinstance(item, list) and isinstance(item[0], _RuntimeHeader):
            symbol_map[item[0].label] = pc
            t = adjust_pc_maps(runtime_map, pc)
            for key in line_number_map:
                line_number_map[key].update(t[key])
            immutables_len = item[0].immutables_len
            pc += len(runtime_code)
            for t in item:
                if isinstance(t, list) and isinstance(t[0], _DataHeader):
                    data_section_lengths.append(_length_of_data(t))
        elif isinstance(item, list) and isinstance(item[0], _DataHeader):
            symbol_map[item[0].label] = pc
            pc += _length_of_data(item)
        else:
            pc += 1
    bytecode_suffix = b''
    if insert_compiler_metadata:
        assert immutables_len is not None
        metadata = (len(runtime_code), data_section_lengths, immutables_len, {'vyper': version_tuple})
        bytecode_suffix += cbor2.dumps(metadata)
        suffix_len = len(bytecode_suffix) + 2
        bytecode_suffix += suffix_len.to_bytes(2, 'big')
    pc += len(bytecode_suffix)
    symbol_map['_sym_code_end'] = pc
    symbol_map['_mem_deploy_start'] = runtime_code_start
    symbol_map['_mem_deploy_end'] = runtime_code_end
    if runtime_code is not None:
        symbol_map['_sym_subcode_size'] = len(runtime_code)
    ret = bytearray()
    to_skip = 0
    for (i, item) in enumerate(assembly):
        if to_skip > 0:
            to_skip -= 1
            continue
        if item in ('DEBUG',):
            continue
        elif is_symbol(item):
            if not is_symbol_map_indicator(assembly[i + 1]):
                (bytecode, _) = assembly_to_evm(PUSH_N(symbol_map[item], n=SYMBOL_SIZE))
                ret.extend(bytecode)
        elif is_mem_sym(item):
            (bytecode, _) = assembly_to_evm(PUSH_N(symbol_map[item], n=mem_ofst_size))
            ret.extend(bytecode)
        elif is_ofst(item):
            ofst = symbol_map[assembly[i + 1]] + assembly[i + 2]
            n = mem_ofst_size if is_mem_sym(assembly[i + 1]) else SYMBOL_SIZE
            (bytecode, _) = assembly_to_evm(PUSH_N(ofst, n))
            ret.extend(bytecode)
            to_skip = 2
        elif isinstance(item, int):
            ret.append(item)
        elif isinstance(item, str) and item.upper() in get_opcodes():
            ret.append(get_opcodes()[item.upper()][0])
        elif item[:4] == 'PUSH':
            ret.append(PUSH_OFFSET + int(item[4:]))
        elif item[:3] == 'DUP':
            ret.append(DUP_OFFSET + int(item[3:]))
        elif item[:4] == 'SWAP':
            ret.append(SWAP_OFFSET + int(item[4:]))
        elif isinstance(item, list) and isinstance(item[0], _RuntimeHeader):
            ret.extend(runtime_code)
        elif isinstance(item, list) and isinstance(item[0], _DataHeader):
            ret.extend(_data_to_evm(item, symbol_map))
        else:
            raise ValueError(f'Weird symbol in assembly: {type(item)} {item}')
    ret.extend(bytecode_suffix)
    line_number_map['breakpoints'] = list(line_number_map['breakpoints'])
    line_number_map['pc_breakpoints'] = list(line_number_map['pc_breakpoints'])
    return (bytes(ret), line_number_map, symbol_map)