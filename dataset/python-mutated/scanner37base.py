"""
Python 37 bytecode scanner/deparser base.

Also we *modify* the instruction sequence to assist deparsing code.
For example:
 -  we add "COME_FROM" instructions to help in figuring out
    conditional branching and looping.
 -  LOAD_CONSTs are classified further into the type of thing
    they load:
      lambda's, genexpr's, {dict,set,list} comprehension's,
 -  PARAMETER counts appended  {CALL,MAKE}_FUNCTION, BUILD_{TUPLE,SET,SLICE}

Finally we save token information.
"""
from typing import Any, Dict, List, Set, Tuple
from xdis import iscode, instruction_size, Instruction
from xdis.bytecode import _get_const_info
from uncompyle6.scanner import Token
import xdis
import xdis.opcodes.opcode_37 as op3
from uncompyle6.scanner import Scanner
import sys
globals().update(op3.opmap)
CONST_COLLECTIONS = ('CONST_LIST', 'CONST_SET', 'CONST_DICT')

class Scanner37Base(Scanner):

    def __init__(self, version: Tuple[int, int], show_asm=None, debug='', is_pypy=False):
        if False:
            for i in range(10):
                print('nop')
        super(Scanner37Base, self).__init__(version, show_asm, is_pypy)
        self.offset2tok_index = None
        self.debug = debug
        self.is_pypy = is_pypy
        if self.version < (3, 8):
            setup_ops = [self.opc.SETUP_LOOP, self.opc.SETUP_EXCEPT, self.opc.SETUP_FINALLY]
            self.setup_ops_no_loop = frozenset(setup_ops) - frozenset([self.opc.SETUP_LOOP])
        else:
            setup_ops = [self.opc.SETUP_FINALLY]
            self.setup_ops_no_loop = frozenset(setup_ops)
            self.opc.BREAK_LOOP = 80
            self.opc.CONTINUE_LOOP = 119
            pass
        setup_ops.append(self.opc.SETUP_WITH)
        self.setup_ops = frozenset(setup_ops)
        self.pop_jump_tf = frozenset([self.opc.PJIF, self.opc.PJIT])
        self.not_continue_follow = ('END_FINALLY', 'POP_BLOCK')
        statement_opcodes = [self.opc.POP_BLOCK, self.opc.STORE_FAST, self.opc.DELETE_FAST, self.opc.STORE_DEREF, self.opc.STORE_GLOBAL, self.opc.DELETE_GLOBAL, self.opc.STORE_NAME, self.opc.DELETE_NAME, self.opc.STORE_ATTR, self.opc.DELETE_ATTR, self.opc.STORE_SUBSCR, self.opc.POP_TOP, self.opc.DELETE_SUBSCR, self.opc.END_FINALLY, self.opc.RETURN_VALUE, self.opc.RAISE_VARARGS, self.opc.PRINT_EXPR, self.opc.JUMP_ABSOLUTE, self.opc.BREAK_LOOP, self.opc.CONTINUE_LOOP]
        self.statement_opcodes = frozenset(statement_opcodes) | self.setup_ops_no_loop
        self.designator_ops = frozenset([self.opc.STORE_FAST, self.opc.STORE_NAME, self.opc.STORE_GLOBAL, self.opc.STORE_DEREF, self.opc.STORE_ATTR, self.opc.STORE_SUBSCR, self.opc.UNPACK_SEQUENCE, self.opc.JUMP_ABSOLUTE, self.opc.UNPACK_EX])
        self.jump_if_pop = frozenset([self.opc.JUMP_IF_FALSE_OR_POP, self.opc.JUMP_IF_TRUE_OR_POP])
        self.pop_jump_if_pop = frozenset([self.opc.JUMP_IF_FALSE_OR_POP, self.opc.JUMP_IF_TRUE_OR_POP, self.opc.POP_JUMP_IF_TRUE, self.opc.POP_JUMP_IF_FALSE])
        self.statement_opcode_sequences = [(self.opc.POP_JUMP_IF_FALSE, self.opc.JUMP_FORWARD), (self.opc.POP_JUMP_IF_FALSE, self.opc.JUMP_ABSOLUTE), (self.opc.POP_JUMP_IF_TRUE, self.opc.JUMP_FORWARD), (self.opc.POP_JUMP_IF_TRUE, self.opc.JUMP_ABSOLUTE)]
        varargs_ops = set([self.opc.BUILD_LIST, self.opc.BUILD_TUPLE, self.opc.BUILD_SET, self.opc.BUILD_SLICE, self.opc.BUILD_MAP, self.opc.UNPACK_SEQUENCE, self.opc.RAISE_VARARGS])
        varargs_ops.add(self.opc.CALL_METHOD)
        varargs_ops |= set([self.opc.BUILD_SET_UNPACK, self.opc.BUILD_MAP_UNPACK, self.opc.BUILD_LIST_UNPACK, self.opc.BUILD_TUPLE_UNPACK])
        varargs_ops.add(self.opc.BUILD_CONST_KEY_MAP)
        self.MAKE_FUNCTION_FLAGS = tuple('\n            default keyword-only annotation closure'.split())
        self.varargs_ops = frozenset(varargs_ops)
        return

    def ingest(self, co, classname=None, code_objects={}, show_asm=None):
        if False:
            while True:
                i = 10
        'Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the\n              opcode name, e.g.:\n                *  BUILD_LIST, BUILD_SET\n                *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional\n                   arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set\n        which will cause custom grammar rules. Specifically, variable\n        arg tokens like MAKE_FUNCTION or BUILD_LIST cause specific\n        rules for the specific number of arguments they take.\n\n        '

        def tokens_append(j, token):
            if False:
                print('Hello World!')
            tokens.append(token)
            self.offset2tok_index[token.offset] = j
            j += 1
            assert j == len(tokens)
            return j
        if not show_asm:
            show_asm = self.show_asm
        bytecode = self.build_instructions(co)
        if show_asm in ('both', 'before'):
            print('\n# ---- before tokenization:')
            self.insts = bytecode.disassemble_bytes(co.co_code, varnames=co.co_varnames, names=co.co_names, constants=co.co_consts, cells=bytecode._cell_names, linestarts=bytecode._linestarts, asm_format='extended', filename=co.co_filename, show_source=True, first_line_number=co.co_firstlineno)
        customize = {}
        if self.is_pypy:
            customize['PyPy'] = 0
        self.load_asserts = set()
        tokens = []
        self.offset2tok_index = {}
        n = len(self.insts)
        for (i, inst) in enumerate(self.insts):
            assert_can_follow = inst.opname == 'POP_JUMP_IF_TRUE' and i + 1 < n
            if assert_can_follow:
                next_inst = self.insts[i + 1]
                if next_inst.opname == 'LOAD_GLOBAL' and next_inst.argval == 'AssertionError' and inst.argval:
                    raise_idx = self.offset2inst_index[self.prev_op[inst.argval]]
                    raise_inst = self.insts[raise_idx]
                    if raise_inst.opname.startswith('RAISE_VARARGS'):
                        self.load_asserts.add(next_inst.offset)
                    pass
                pass
        for (i, inst) in enumerate(self.insts):
            if inst.opname == 'JUMP_FORWARD':
                jump_inst = self.insts[self.offset2inst_index[inst.argval]]
                if jump_inst.has_extended_arg and jump_inst.opname.startswith('JUMP'):
                    self.insts[i] = Instruction(jump_inst.opname, jump_inst.opcode, jump_inst.optype, jump_inst.inst_size, jump_inst.arg, jump_inst.argval, jump_inst.argrepr, jump_inst.has_arg, inst.offset, inst.starts_line, inst.is_jump_target, inst.has_extended_arg, None, None)
        jump_targets = self.find_jump_targets(show_asm)
        last_op_was_break = False
        j = 0
        for (i, inst) in enumerate(self.insts):
            argval = inst.argval
            op = inst.opcode
            if inst.opname == 'EXTENDED_ARG':
                if i + 1 < n and self.insts[i + 1].opcode != self.opc.MAKE_FUNCTION:
                    continue
            if inst.offset in jump_targets:
                jump_idx = 0
                for jump_offset in sorted(jump_targets[inst.offset], reverse=True):
                    come_from_name = 'COME_FROM'
                    opname = self.opname_for_offset(jump_offset)
                    if opname == 'EXTENDED_ARG':
                        k = xdis.next_offset(op, self.opc, jump_offset)
                        opname = self.opname_for_offset(k)
                    if opname.startswith('SETUP_'):
                        come_from_type = opname[len('SETUP_'):]
                        come_from_name = 'COME_FROM_%s' % come_from_type
                        pass
                    elif inst.offset in self.except_targets:
                        come_from_name = 'COME_FROM_EXCEPT_CLAUSE'
                    j = tokens_append(j, Token(opname=come_from_name, attr=jump_offset, pattr=repr(jump_offset), offset='%s_%s' % (inst.offset, jump_idx), has_arg=True, opc=self.opc, has_extended_arg=False))
                    jump_idx += 1
                    pass
                pass
            pattr = inst.argrepr
            opname = inst.opname
            if op in self.opc.CONST_OPS:
                const = argval
                if iscode(const):
                    if const.co_name == '<lambda>':
                        assert opname == 'LOAD_CONST'
                        opname = 'LOAD_LAMBDA'
                    elif const.co_name == '<genexpr>':
                        opname = 'LOAD_GENEXPR'
                    elif const.co_name == '<dictcomp>':
                        opname = 'LOAD_DICTCOMP'
                    elif const.co_name == '<setcomp>':
                        opname = 'LOAD_SETCOMP'
                    elif const.co_name == '<listcomp>':
                        opname = 'LOAD_LISTCOMP'
                    else:
                        opname = 'LOAD_CODE'
                    pattr = '<code_object ' + const.co_name + '>'
                elif isinstance(const, str):
                    opname = 'LOAD_STR'
                else:
                    if isinstance(inst.arg, int) and inst.arg < len(co.co_consts):
                        (argval, _) = _get_const_info(inst.arg, co.co_consts)
                    pattr = const
                    pass
            elif opname == 'IMPORT_NAME':
                if '.' in inst.argval:
                    opname = 'IMPORT_NAME_ATTR'
                    pass
            elif opname == 'LOAD_FAST' and argval == '.0':
                opname = 'LOAD_ARG'
            elif opname in ('MAKE_FUNCTION', 'MAKE_CLOSURE'):
                flags = argval
                opname = 'MAKE_FUNCTION_%d' % flags
                attr = []
                for flag in self.MAKE_FUNCTION_FLAGS:
                    bit = flags & 1
                    attr.append(bit)
                    flags >>= 1
                attr = attr[:4]
                j = tokens_append(j, Token(opname=opname, attr=attr, pattr=pattr, offset=inst.offset, linestart=inst.starts_line, op=op, has_arg=inst.has_arg, opc=self.opc, has_extended_arg=inst.has_extended_arg))
                continue
            elif op in self.varargs_ops:
                pos_args = argval
                if self.is_pypy and (not pos_args) and (opname == 'BUILD_MAP'):
                    opname = 'BUILD_MAP_n'
                else:
                    opname = '%s_%d' % (opname, pos_args)
            elif self.is_pypy and opname == 'JUMP_IF_NOT_DEBUG':
                customize[opname] = 0
            elif opname == 'UNPACK_EX':
                before_args = argval & 255
                after_args = argval >> 8 & 255
                pattr = '%d before vararg, %d after' % (before_args, after_args)
                argval = (before_args, after_args)
                opname = '%s_%d+%d' % (opname, before_args, after_args)
            elif op == self.opc.JUMP_ABSOLUTE:
                pattr = argval
                target = inst.argval
                if target <= inst.offset:
                    next_opname = self.insts[i + 1].opname
                    is_continue = self.insts[self.offset2inst_index[target]].opname == 'FOR_ITER' and self.insts[i + 1].opname == 'JUMP_FORWARD'
                    if self.version < (3, 8) and (is_continue or (inst.offset in self.stmts and (inst.starts_line and next_opname not in self.not_continue_follow))):
                        opname = 'CONTINUE'
                    else:
                        opname = 'JUMP_BACK'
                        if tokens[-1].kind == 'JUMP_BACK' and tokens[-1].attr <= argval:
                            if tokens[-2].kind == 'BREAK_LOOP':
                                del tokens[-1]
                            else:
                                tokens[-1].kind = sys.intern('CONTINUE')
                    if last_op_was_break and opname == 'CONTINUE':
                        last_op_was_break = False
                        continue
            elif inst.offset in self.load_asserts:
                opname = 'LOAD_ASSERT'
            last_op_was_break = opname == 'BREAK_LOOP'
            j = tokens_append(j, Token(opname=opname, attr=argval, pattr=pattr, offset=inst.offset, linestart=inst.starts_line, op=op, has_arg=inst.has_arg, opc=self.opc, has_extended_arg=inst.has_extended_arg))
            pass
        if show_asm in ('both', 'after'):
            print('\n# ---- after tokenization:')
            for t in tokens:
                print(t.format(line_prefix=''))
            print()
        return (tokens, customize)

    def find_jump_targets(self, debug: str) -> dict:
        if False:
            return 10
        '\n        Detect all offsets in a byte code which are jump targets\n        where we might insert a COME_FROM instruction.\n\n        Return the list of offsets.\n\n        Return the list of offsets. An instruction can be jumped\n        to in from multiple instructions.\n        '
        code = self.code
        n = len(code)
        self.structs = [{'type': 'root', 'start': 0, 'end': n - 1}]
        self.loops: List[int] = []
        self.fixed_jumps: Dict[int, int] = {}
        self.except_targets = {}
        self.ignore_if: Set[int] = set()
        self.build_statement_indices()
        self.not_continue: Set[int] = set()
        self.return_end_ifs: Set[int] = set()
        self.setup_loop_targets = {}
        self.setup_loops = {}
        targets = {}
        for (i, inst) in enumerate(self.insts):
            offset = inst.offset
            op = inst.opcode
            self.detect_control_flow(offset, targets, i)
            if inst.has_arg:
                label = self.fixed_jumps.get(offset)
                oparg = inst.arg
                if self.code[offset] == self.opc.EXTENDED_ARG:
                    j = xdis.next_offset(op, self.opc, offset)
                    next_offset = xdis.next_offset(op, self.opc, j)
                else:
                    next_offset = xdis.next_offset(op, self.opc, offset)
                if label is None:
                    if op in self.opc.hasjrel and op != self.opc.FOR_ITER:
                        label = next_offset + oparg
                    elif op in self.opc.hasjabs:
                        if op in self.jump_if_pop:
                            if oparg > offset:
                                label = oparg
                if label is not None and label != -1:
                    targets[label] = targets.get(label, []) + [offset]
            elif op == self.opc.END_FINALLY and offset in self.fixed_jumps:
                label = self.fixed_jumps[offset]
                targets[label] = targets.get(label, []) + [offset]
                pass
            pass
        if debug in ('both', 'after'):
            import pprint as pp
            pp.pprint(self.structs)
        return targets

    def build_statement_indices(self):
        if False:
            i = 10
            return i + 15
        code = self.code
        start = 0
        end = codelen = len(code)
        prelim = self.inst_matches(start, end, self.statement_opcodes)
        stmts = self.stmts = set(prelim)
        pass_stmts = set()
        for sequence in self.statement_opcode_sequences:
            for i in self.op_range(start, end - (len(sequence) + 1)):
                match = True
                for elem in sequence:
                    if elem != code[i]:
                        match = False
                        break
                    i += instruction_size(code[i], self.opc)
                if match is True:
                    i = self.prev_op[i]
                    stmts.add(i)
                    pass_stmts.add(i)
        if pass_stmts:
            stmt_offset_list = list(stmts)
            stmt_offset_list.sort()
        else:
            stmt_offset_list = prelim
        self.next_stmt = slist = []
        last_stmt_offset = -1
        i = 0
        for stmt_offset in stmt_offset_list:
            if code[stmt_offset] == self.opc.JUMP_ABSOLUTE and stmt_offset not in pass_stmts:
                target = self.get_target(stmt_offset)
                if target > stmt_offset or self.lines[last_stmt_offset].l_no == self.lines[stmt_offset].l_no:
                    stmts.remove(stmt_offset)
                    continue
                j = self.prev_op[stmt_offset]
                while code[j] == self.opc.JUMP_ABSOLUTE and j > 0:
                    j = self.prev_op[j]
                if code[j] == self.opc.LIST_APPEND:
                    stmts.remove(stmt_offset)
                    continue
            elif code[stmt_offset] == self.opc.POP_TOP and code[self.prev_op[stmt_offset]] == self.opc.ROT_TWO:
                stmts.remove(stmt_offset)
                continue
            elif code[stmt_offset] in self.designator_ops:
                j = self.prev_op[stmt_offset]
                while code[j] in self.designator_ops:
                    j = self.prev_op[j]
                if code[j] == self.opc.FOR_ITER:
                    stmts.remove(stmt_offset)
                    continue
            slist += [stmt_offset] * (stmt_offset - i)
            last_stmt_offset = stmt_offset
            i = stmt_offset
        slist += [codelen] * (codelen - len(slist))

    def detect_control_flow(self, offset: int, targets: Dict[Any, Any], inst_index: int):
        if False:
            while True:
                i = 10
        '\n        Detect type of block structures and their boundaries to fix optimized jumps\n        in python2.3+\n        '
        code = self.code
        inst = self.insts[inst_index]
        op = inst.opcode
        parent: Dict[str, Any] = self.structs[0]
        start: int = parent['start']
        end: int = parent['end']
        for struct in self.structs:
            current_start = struct['start']
            current_end = struct['end']
            if current_start <= offset < current_end and (current_start >= start and current_end <= end):
                start = current_start
                end = current_end
                parent = struct
        if self.version < (3, 8) and op == self.opc.SETUP_LOOP:
            start += inst.inst_size
            target = self.get_target(offset)
            end = self.restrict_to_parent(target, parent)
            self.setup_loops[target] = offset
            if target != end:
                self.fixed_jumps[offset] = end
            (line_no, next_line_byte) = self.lines[offset]
            jump_back = self.last_instr(start, end, self.opc.JUMP_ABSOLUTE, next_line_byte, False)
            if jump_back:
                jump_forward_offset = xdis.next_offset(code[jump_back], self.opc, jump_back)
            else:
                jump_forward_offset = None
            return_val_offset1 = self.prev[self.prev[end]]
            if jump_back and jump_back != self.prev_op[end] and self.is_jump_forward(jump_forward_offset):
                if code[self.prev_op[end]] == self.opc.RETURN_VALUE or (code[self.prev_op[end]] == self.opc.POP_BLOCK and code[return_val_offset1] == self.opc.RETURN_VALUE):
                    jump_back = None
            if not jump_back:
                jump_back = self.last_instr(start, end, self.opc.RETURN_VALUE)
                if not jump_back:
                    return
                jb_inst = self.get_inst(jump_back)
                jump_back = self.next_offset(jb_inst.opcode, jump_back)
                if_offset = None
                if code[self.prev_op[next_line_byte]] not in self.pop_jump_tf:
                    if_offset = self.prev[next_line_byte]
                if if_offset:
                    loop_type = 'while'
                    self.ignore_if.add(if_offset)
                else:
                    loop_type = 'for'
                target = next_line_byte
                end = xdis.next_offset(code[jump_back], self.opc, jump_back)
            else:
                if self.get_target(jump_back) >= next_line_byte:
                    jump_back = self.last_instr(start, end, self.opc.JUMP_ABSOLUTE, start, False)
                jb_inst = self.get_inst(jump_back)
                jb_next_offset = self.next_offset(jb_inst.opcode, jump_back)
                if end > jb_next_offset and self.is_jump_forward(end):
                    if self.is_jump_forward(jb_next_offset):
                        if self.get_target(jb_next_offset) == self.get_target(end):
                            self.fixed_jumps[offset] = jb_next_offset
                            end = jb_next_offset
                elif target < offset:
                    self.fixed_jumps[offset] = jb_next_offset
                    end = jb_next_offset
                target = self.get_target(jump_back)
                if code[target] in (self.opc.FOR_ITER, self.opc.GET_ITER):
                    loop_type = 'for'
                else:
                    loop_type = 'while'
                    test = self.prev_op[next_line_byte]
                    if test == offset:
                        loop_type = 'while 1'
                    elif self.code[test] in self.opc.JUMP_OPs:
                        self.ignore_if.add(test)
                        test_target = self.get_target(test)
                        if test_target > jump_back + 3:
                            jump_back = test_target
                self.not_continue.add(jump_back)
            self.loops.append(target)
            self.structs.append({'type': loop_type + '-loop', 'start': target, 'end': jump_back})
            after_jump_offset = xdis.next_offset(code[jump_back], self.opc, jump_back)
            if after_jump_offset != end:
                self.structs.append({'type': loop_type + '-else', 'start': after_jump_offset, 'end': end})
        elif op in self.pop_jump_tf:
            target = inst.argval
            self.fixed_jumps[offset] = target
        elif self.version < (3, 8) and op == self.opc.SETUP_EXCEPT:
            target = self.get_target(offset)
            end = self.restrict_to_parent(target, parent)
            self.fixed_jumps[offset] = end
        elif op == self.opc.POP_EXCEPT:
            next_offset = xdis.next_offset(op, self.opc, offset)
            target = self.get_target(next_offset)
            if target > next_offset:
                next_op = code[next_offset]
                if self.opc.JUMP_ABSOLUTE == next_op and self.opc.END_FINALLY != code[xdis.next_offset(next_op, self.opc, next_offset)]:
                    self.fixed_jumps[next_offset] = target
                    self.except_targets[target] = next_offset
        elif op == self.opc.SETUP_FINALLY:
            target = self.get_target(offset)
            end = self.restrict_to_parent(target, parent)
            self.fixed_jumps[offset] = end
        elif op in self.jump_if_pop:
            target = self.get_target(offset)
            if target > offset:
                unop_target = self.last_instr(offset, target, self.opc.JUMP_FORWARD, target)
                if unop_target and code[unop_target + 3] != self.opc.ROT_TWO:
                    self.fixed_jumps[offset] = unop_target
                else:
                    self.fixed_jumps[offset] = self.restrict_to_parent(target, parent)
                    pass
                pass
        elif op == self.opc.RETURN_VALUE:
            next_offset = xdis.next_offset(op, self.opc, offset)
            if next_offset < len(code) and (code[next_offset] == self.opc.JUMP_ABSOLUTE and offset in self.return_end_ifs):
                self.return_end_ifs.remove(offset)
                pass
            pass
        elif op == self.opc.JUMP_FORWARD:
            rtarget = self.get_target(offset)
            rtarget_prev = self.prev[rtarget]
            if code[rtarget_prev] == self.opc.RETURN_VALUE and rtarget_prev in self.return_end_ifs:
                i = rtarget_prev
                while i != offset:
                    if code[i] in [op3.JUMP_FORWARD, op3.JUMP_ABSOLUTE]:
                        return
                    i = self.prev[i]
                self.return_end_ifs.remove(rtarget_prev)
            pass
        return

    def next_except_jump(self, start):
        if False:
            print('Hello World!')
        '\n        Return the next jump that was generated by an except SomeException:\n        construct in a try...except...else clause or None if not found.\n        '
        if self.code[start] == self.opc.DUP_TOP:
            except_match = self.first_instr(start, len(self.code), self.opc.POP_JUMP_IF_FALSE)
            if except_match:
                jmp = self.prev_op[self.get_target(except_match)]
                self.ignore_if.add(except_match)
                self.not_continue.add(jmp)
                return jmp
        count_END_FINALLY = 0
        count_SETUP_ = 0
        for i in self.op_range(start, len(self.code)):
            op = self.code[i]
            if op == self.opc.END_FINALLY:
                if count_END_FINALLY == count_SETUP_:
                    assert self.code[self.prev_op[i]] in frozenset([self.opc.JUMP_ABSOLUTE, self.opc.JUMP_FORWARD, self.opc.RETURN_VALUE])
                    self.not_continue.add(self.prev_op[i])
                    return self.prev_op[i]
                count_END_FINALLY += 1
            elif op in self.setup_opts_no_loop:
                count_SETUP_ += 1
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (3, 7):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner37Base(PYTHON_VERSION_TRIPLE).ingest(co)
        for t in tokens:
            print(t)
    else:
        print(f'Need to be Python 3.7 to demo; I am version {version_tuple_to_str()}.')
    pass