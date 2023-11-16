"""
Python 2 Generic bytecode scanner/deparser

This overlaps various Python3's dis module, but it can be run from
Python versions other than the version running this code. Notably,
run from Python version 2.

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
from __future__ import print_function
from copy import copy
from xdis import code2num, iscode, op_has_argument, instruction_size
from xdis.bytecode import _get_const_info
from uncompyle6.scanner import Scanner, Token
from sys import intern

class Scanner2(Scanner):

    def __init__(self, version, show_asm=None, is_pypy=False):
        if False:
            return 10
        Scanner.__init__(self, version, show_asm, is_pypy)
        self.pop_jump_if = frozenset([self.opc.PJIF, self.opc.PJIT])
        self.jump_forward = frozenset([self.opc.JUMP_ABSOLUTE, self.opc.JUMP_FORWARD])
        self.genexpr_name = '<genexpr>'
        self.load_asserts = set([])
        self.statement_opcodes = frozenset([self.opc.SETUP_LOOP, self.opc.BREAK_LOOP, self.opc.SETUP_FINALLY, self.opc.END_FINALLY, self.opc.SETUP_EXCEPT, self.opc.POP_BLOCK, self.opc.STORE_FAST, self.opc.DELETE_FAST, self.opc.STORE_DEREF, self.opc.STORE_GLOBAL, self.opc.DELETE_GLOBAL, self.opc.STORE_NAME, self.opc.DELETE_NAME, self.opc.STORE_ATTR, self.opc.DELETE_ATTR, self.opc.STORE_SUBSCR, self.opc.DELETE_SUBSCR, self.opc.RETURN_VALUE, self.opc.RAISE_VARARGS, self.opc.POP_TOP, self.opc.PRINT_EXPR, self.opc.PRINT_ITEM, self.opc.PRINT_NEWLINE, self.opc.PRINT_ITEM_TO, self.opc.PRINT_NEWLINE_TO, self.opc.CONTINUE_LOOP, self.opc.JUMP_ABSOLUTE, self.opc.EXEC_STMT])
        self.designator_ops = frozenset([self.opc.STORE_FAST, self.opc.STORE_NAME, self.opc.STORE_GLOBAL, self.opc.STORE_DEREF, self.opc.STORE_ATTR, self.opc.STORE_SLICE_0, self.opc.STORE_SLICE_1, self.opc.STORE_SLICE_2, self.opc.STORE_SLICE_3, self.opc.STORE_SUBSCR, self.opc.UNPACK_SEQUENCE, self.opc.JUMP_ABSOLUTE])
        self.pop_jump_if_or_pop = frozenset([])
        self.varargs_ops = frozenset([self.opc.BUILD_LIST, self.opc.BUILD_TUPLE, self.opc.BUILD_SLICE, self.opc.UNPACK_SEQUENCE, self.opc.MAKE_FUNCTION, self.opc.CALL_FUNCTION, self.opc.MAKE_CLOSURE, self.opc.CALL_FUNCTION_VAR, self.opc.CALL_FUNCTION_KW, self.opc.CALL_FUNCTION_VAR_KW, self.opc.DUP_TOPX, self.opc.RAISE_VARARGS])

    @staticmethod
    def extended_arg_val(arg):
        if False:
            print('Hello World!')
        'Return integer value of an EXTENDED_ARG operand.\n        In Python2 this always the operand value shifted 16 bits since\n        the operand is always 2 bytes. In Python 3.6+ this changes to one byte.\n        '
        return arg << 16

    @staticmethod
    def unmangle_name(name, classname):
        if False:
            i = 10
            return i + 15
        'Remove __ from the end of _name_ if it starts with __classname__\n        return the "unmangled" name.\n        '
        if name.startswith(classname) and name[-2:] != '__':
            return name[len(classname) - 2:]
        return name

    @classmethod
    def unmangle_code_names(self, co, classname):
        if False:
            return 10
        'Remove __ from the end of _name_ if it starts with __classname__\n        return the "unmangled" name.\n        '
        if classname:
            classname = '_' + classname.lstrip('_') + '__'
            if hasattr(co, 'co_cellvars'):
                free = [self.unmangle_name(name, classname) for name in co.co_cellvars + co.co_freevars]
            else:
                free = ()
            names = [self.unmangle_name(name, classname) for name in co.co_names]
            varnames = [self.unmangle_name(name, classname) for name in co.co_varnames]
        else:
            if hasattr(co, 'co_cellvars'):
                free = co.co_cellvars + co.co_freevars
            else:
                free = ()
            names = co.co_names
            varnames = co.co_varnames
        return (free, names, varnames)

    def ingest(self, co, classname=None, code_objects={}, show_asm=None):
        if False:
            i = 10
            return i + 15
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause custom\n        grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or BUILD_LIST\n        cause specific rules for the specific number of arguments they take.\n        '
        if not show_asm:
            show_asm = self.show_asm
        bytecode = self.build_instructions(co)
        if show_asm in ('both', 'before'):
            print('\n# ---- before tokenization:')
            bytecode.disassemble_bytes(co.co_code, varnames=co.co_varnames, names=co.co_names, constants=co.co_consts, cells=bytecode._cell_names, linestarts=bytecode._linestarts, asm_format='extended')
        new_tokens = []
        customize = {}
        if self.is_pypy:
            customize['PyPy'] = 0
        codelen = len(self.code)
        (free, names, varnames) = self.unmangle_code_names(co, classname)
        self.names = names
        self.load_asserts = set()
        for i in self.op_range(0, codelen):
            if self.is_pypy:
                have_pop_jump = self.code[i] in (self.opc.PJIF, self.opc.PJIT)
            else:
                have_pop_jump = self.code[i] == self.opc.PJIT
            if have_pop_jump and self.code[i + 3] == self.opc.LOAD_GLOBAL:
                if names[self.get_argument(i + 3)] == 'AssertionError':
                    self.load_asserts.add(i + 3)
        load_asserts_save = copy(self.load_asserts)
        jump_targets = self.find_jump_targets(show_asm)
        self.load_asserts = load_asserts_save
        last_stmt = self.next_stmt[0]
        i = self.next_stmt[last_stmt]
        replace = {}
        while i < codelen - 1:
            if self.lines[last_stmt].next > i:
                if self.code[last_stmt] == self.opc.PRINT_ITEM:
                    if self.code[i] == self.opc.PRINT_ITEM:
                        replace[i] = 'PRINT_ITEM_CONT'
                    elif self.code[i] == self.opc.PRINT_NEWLINE:
                        replace[i] = 'PRINT_NEWLINE_CONT'
            last_stmt = i
            i = self.next_stmt[i]
        extended_arg = 0
        for offset in self.op_range(0, codelen):
            if offset in jump_targets:
                jump_idx = 0
                for jump_offset in sorted(jump_targets[offset], reverse=True):
                    come_from_name = 'COME_FROM'
                    op_name = self.opname_for_offset(jump_offset)
                    if op_name.startswith('SETUP_') and self.version[:2] == (2, 7):
                        come_from_type = op_name[len('SETUP_'):]
                        if come_from_type not in ('LOOP', 'EXCEPT'):
                            come_from_name = 'COME_FROM_%s' % come_from_type
                        pass
                    new_tokens.append(Token(come_from_name, jump_offset, repr(jump_offset), offset='%s_%d' % (offset, jump_idx), has_arg=True))
                    jump_idx += 1
                    pass
            op = self.code[offset]
            op_name = self.op_name(op)
            oparg = None
            pattr = None
            has_arg = op_has_argument(op, self.opc)
            if has_arg:
                oparg = self.get_argument(offset) + extended_arg
                extended_arg = 0
                if op == self.opc.EXTENDED_ARG:
                    extended_arg += self.extended_arg_val(oparg)
                    continue
                if op_name in ('BUILD_LIST', 'BUILD_SET'):
                    t = Token(op_name, oparg, pattr, offset, self.linestarts.get(offset, None), op, has_arg, self.opc)
                    collection_type = op_name.split('_')[1]
                    next_tokens = self.bound_collection_from_tokens(new_tokens, t, len(new_tokens), 'CONST_%s' % collection_type)
                    if next_tokens is not None:
                        new_tokens = next_tokens
                        continue
                if op in self.opc.CONST_OPS:
                    const = co.co_consts[oparg]
                    if iscode(const):
                        oparg = const
                        if const.co_name == '<lambda>':
                            assert op_name == 'LOAD_CONST'
                            op_name = 'LOAD_LAMBDA'
                        elif const.co_name == '<genexpr>':
                            op_name = 'LOAD_GENEXPR'
                        elif const.co_name == '<dictcomp>':
                            op_name = 'LOAD_DICTCOMP'
                        elif const.co_name == '<setcomp>':
                            op_name = 'LOAD_SETCOMP'
                        else:
                            op_name = 'LOAD_CODE'
                        pattr = '<code_object ' + const.co_name + '>'
                    else:
                        if oparg < len(co.co_consts):
                            (argval, _) = _get_const_info(oparg, co.co_consts)
                        pattr = const
                        pass
                elif op in self.opc.NAME_OPS:
                    pattr = names[oparg]
                elif op in self.opc.JREL_OPS:
                    if self.version[:2] == (2, 7):
                        self.patch_continue(new_tokens, offset, op)
                    pattr = repr(offset + 3 + oparg)
                elif op in self.opc.JABS_OPS:
                    if self.version[:2] == (2, 7):
                        self.patch_continue(new_tokens, offset, op)
                    pattr = repr(oparg)
                elif op in self.opc.LOCAL_OPS:
                    pattr = varnames[oparg]
                elif op in self.opc.COMPARE_OPS:
                    pattr = self.opc.cmp_op[oparg]
                elif op in self.opc.FREE_OPS:
                    pattr = free[oparg]
            if op in self.varargs_ops:
                if op == self.opc.BUILD_TUPLE and self.code[self.prev[offset]] == self.opc.LOAD_CLOSURE:
                    continue
                else:
                    if self.is_pypy and (not oparg) and (op_name == 'BUILD_MAP'):
                        op_name = 'BUILD_MAP_n'
                    else:
                        op_name = '%s_%d' % (op_name, oparg)
                        pass
                    customize[op_name] = oparg
            elif op == self.opc.JUMP_ABSOLUTE:
                target = self.get_target(offset)
                if target <= offset:
                    op_name = 'JUMP_BACK'
                    j = self.offset2inst_index.get(offset)
                    if j is None and offset > self.opc.ARG_MAX_VALUE:
                        j = self.offset2inst_index[offset - 3]
                    target_index = self.offset2inst_index[target]
                    is_continue = self.insts[target_index - 1].opname == 'SETUP_LOOP' and self.insts[j + 1].opname == 'JUMP_FORWARD'
                    if is_continue:
                        op_name = 'CONTINUE'
                    if offset in self.stmts and self.code[offset + 3] not in (self.opc.END_FINALLY, self.opc.POP_BLOCK):
                        if offset in self.linestarts and self.code[self.prev[offset]] == self.opc.JUMP_ABSOLUTE or self.code[target] == self.opc.FOR_ITER or offset not in self.not_continue:
                            op_name = 'CONTINUE'
            elif op == self.opc.LOAD_GLOBAL:
                if offset in self.load_asserts:
                    op_name = 'LOAD_ASSERT'
            elif op == self.opc.RETURN_VALUE:
                if offset in self.return_end_ifs:
                    op_name = 'RETURN_END_IF'
            linestart = self.linestarts.get(offset, None)
            if offset not in replace:
                new_tokens.append(Token(op_name, oparg, pattr, offset, linestart, op, has_arg, self.opc))
            else:
                new_tokens.append(Token(replace[offset], oparg, pattr, offset, linestart, op, has_arg, self.opc))
                pass
            pass
        if show_asm in ('both', 'after'):
            print('\n# ---- after tokenization:')
            for t in new_tokens:
                print(t.format(line_prefix=''))
            print()
        return (new_tokens, customize)

    def build_statement_indices(self):
        if False:
            return 10
        code = self.code
        start = 0
        end = len(code)
        stmt_opcode_seqs = frozenset([(self.opc.PJIF, self.opc.JUMP_FORWARD), (self.opc.PJIF, self.opc.JUMP_ABSOLUTE), (self.opc.PJIT, self.opc.JUMP_FORWARD), (self.opc.PJIT, self.opc.JUMP_ABSOLUTE)])
        prelim = self.all_instr(start, end, self.statement_opcodes)
        stmts = self.stmts = set(prelim)
        pass_stmts = set()
        for seq in stmt_opcode_seqs:
            for i in self.op_range(start, end - (len(seq) + 1)):
                match = True
                for elem in seq:
                    if elem != code[i]:
                        match = False
                        break
                    i += instruction_size(code[i], self.opc)
                if match:
                    i = self.prev[i]
                    stmts.add(i)
                    pass_stmts.add(i)
        if pass_stmts:
            stmt_list = list(stmts)
            stmt_list.sort()
        else:
            stmt_list = prelim
        last_stmt = -1
        self.next_stmt = []
        slist = self.next_stmt = []
        i = 0
        for s in stmt_list:
            if code[s] == self.opc.JUMP_ABSOLUTE and s not in pass_stmts:
                target = self.get_target(s)
                if target > s or (self.lines and self.lines[last_stmt].l_no == self.lines[s].l_no):
                    stmts.remove(s)
                    continue
                j = self.prev[s]
                while code[j] == self.opc.JUMP_ABSOLUTE:
                    j = self.prev[j]
                if self.version >= (2, 3) and self.opname_for_offset(j) == 'LIST_APPEND':
                    stmts.remove(s)
                    continue
            elif code[s] == self.opc.POP_TOP:
                prev = code[self.prev[s]]
                if prev == self.opc.ROT_TWO or (self.version < (2, 7) and prev in (self.opc.JUMP_IF_FALSE, self.opc.JUMP_IF_TRUE, self.opc.RETURN_VALUE)):
                    stmts.remove(s)
                    continue
            elif code[s] in self.designator_ops:
                j = self.prev[s]
                while code[j] in self.designator_ops:
                    j = self.prev[j]
                if self.version > (2, 1) and code[j] == self.opc.FOR_ITER:
                    stmts.remove(s)
                    continue
            last_stmt = s
            slist += [s] * (s - i)
            i = s
        slist += [end] * (end - len(slist))

    def next_except_jump(self, start):
        if False:
            print('Hello World!')
        '\n        Return the next jump that was generated by an except SomeException:\n        construct in a try...except...else clause or None if not found.\n        '
        if self.code[start] == self.opc.DUP_TOP:
            except_match = self.first_instr(start, len(self.code), self.opc.PJIF)
            if except_match:
                jmp = self.prev[self.get_target(except_match)]
                if self.version < (2, 7) and self.code[jmp] in self.jump_forward:
                    self.not_continue.add(jmp)
                    jmp = self.get_target(jmp)
                    prev_offset = self.prev[except_match]
                    if self.code[prev_offset] == self.opc.COMPARE_OP and self.code[prev_offset + 1] != 10:
                        return None
                    if jmp not in self.pop_jump_if | self.jump_forward:
                        self.ignore_if.add(except_match)
                        return None
                self.ignore_if.add(except_match)
                self.not_continue.add(jmp)
                return jmp
        count_END_FINALLY = 0
        count_SETUP_ = 0
        for i in self.op_range(start, len(self.code)):
            op = self.code[i]
            if op == self.opc.END_FINALLY:
                if count_END_FINALLY == count_SETUP_:
                    if self.version[:2] == (2, 7):
                        assert self.code[self.prev[i]] in self.jump_forward | frozenset([self.opc.RETURN_VALUE])
                    self.not_continue.add(self.prev[i])
                    return self.prev[i]
                count_END_FINALLY += 1
            elif op in self.setup_ops:
                count_SETUP_ += 1

    def detect_control_flow(self, offset, op, extended_arg):
        if False:
            print('Hello World!')
        '\n        Detect type of block structures and their boundaries to fix optimized jumps\n        in python2.3+\n        '
        code = self.code
        parent = self.structs[0]
        start = parent['start']
        end = parent['end']
        next_line_byte = end
        for struct in self.structs:
            current_start = struct['start']
            current_end = struct['end']
            if current_start <= offset < current_end and (current_start >= start and current_end <= end):
                start = current_start
                end = current_end
                parent = struct
        if op == self.opc.SETUP_LOOP:
            inst = self.insts[self.offset2inst_index[offset]]
            start += instruction_size(op, self.opc)
            setup_target = inst.argval
            loop_end_offset = self.restrict_to_parent(setup_target, parent)
            self.setup_loop_targets[offset] = setup_target
            self.setup_loops[setup_target] = offset
            if setup_target != loop_end_offset:
                self.fixed_jumps[offset] = loop_end_offset
            if self.lines:
                (line_no, next_line_byte) = self.lines[offset]
            jump_back_offset = self.last_instr(start, loop_end_offset, self.opc.JUMP_ABSOLUTE, next_line_byte, False)
            if jump_back_offset:
                if self.version < (2, 7):
                    jump_forward_offset = jump_back_offset + 4
                    return_val_offset1 = self.prev[self.prev[self.prev[loop_end_offset]]]
                    jump_target = self.get_target(jump_back_offset, code[jump_back_offset])
                    if jump_target > jump_back_offset or code[jump_back_offset + 3] in [self.opc.JUMP_FORWARD, self.opc.JUMP_ABSOLUTE]:
                        jump_back_offset = None
                        pass
                else:
                    jump_forward_offset = jump_back_offset + 3
                    return_val_offset1 = self.prev[self.prev[loop_end_offset]]
            if jump_back_offset and jump_back_offset != self.prev[loop_end_offset] and (code[jump_forward_offset] in self.jump_forward):
                if code[self.prev[loop_end_offset]] == self.opc.RETURN_VALUE or (code[self.prev[loop_end_offset]] == self.opc.POP_BLOCK and code[return_val_offset1] == self.opc.RETURN_VALUE):
                    jump_back_offset = None
            if not jump_back_offset:
                jump_back_offset = self.last_instr(start, loop_end_offset, self.opc.RETURN_VALUE)
                if not jump_back_offset:
                    return
                jump_back_offset += 1
                if_offset = None
                if self.version < (2, 7):
                    if code[self.prev[next_line_byte]] == self.opc.POP_TOP and code[self.prev[self.prev[next_line_byte]]] in self.pop_jump_if:
                        if_offset = self.prev[self.prev[next_line_byte]]
                elif code[self.prev[next_line_byte]] in self.pop_jump_if:
                    if_offset = self.prev[next_line_byte]
                if if_offset:
                    loop_type = 'while'
                    self.ignore_if.add(if_offset)
                    if self.version < (2, 7) and code[self.prev[jump_back_offset]] == self.opc.RETURN_VALUE:
                        self.ignore_if.add(self.prev[jump_back_offset])
                        pass
                    pass
                else:
                    loop_type = 'for'
                setup_target = next_line_byte
                loop_end_offset = jump_back_offset + 3
            else:
                if self.get_target(jump_back_offset) >= next_line_byte:
                    jump_back_offset = self.last_instr(start, loop_end_offset, self.opc.JUMP_ABSOLUTE, start, False)
                if loop_end_offset > jump_back_offset + 4 and code[loop_end_offset] in self.jump_forward:
                    if code[jump_back_offset + 4] in self.jump_forward:
                        if self.get_target(jump_back_offset + 4) == self.get_target(loop_end_offset):
                            self.fixed_jumps[offset] = jump_back_offset + 4
                            loop_end_offset = jump_back_offset + 4
                elif setup_target < offset:
                    self.fixed_jumps[offset] = jump_back_offset + 4
                    loop_end_offset = jump_back_offset + 4
                setup_target = self.get_target(jump_back_offset, self.opc.JUMP_ABSOLUTE)
                if self.version > (2, 1) and code[setup_target] in (self.opc.FOR_ITER, self.opc.GET_ITER):
                    loop_type = 'for'
                else:
                    loop_type = 'while'
                    if self.version < (2, 7) and self.code[self.prev[next_line_byte]] == self.opc.POP_TOP:
                        test_op_offset = self.prev[self.prev[next_line_byte]]
                    else:
                        test_op_offset = self.prev[next_line_byte]
                    if test_op_offset == offset:
                        loop_type = 'while 1'
                    elif self.code[test_op_offset] in self.opc.JUMP_OPs:
                        test_target = self.get_target(test_op_offset)
                        self.ignore_if.add(test_op_offset)
                        if test_target > jump_back_offset + 3:
                            jump_back_offset = test_target
                self.not_continue.add(jump_back_offset)
            self.loops.append(setup_target)
            self.structs.append({'type': loop_type + '-loop', 'start': setup_target, 'end': jump_back_offset})
            if jump_back_offset + 3 != loop_end_offset:
                self.structs.append({'type': loop_type + '-else', 'start': jump_back_offset + 3, 'end': loop_end_offset})
        elif op == self.opc.SETUP_EXCEPT:
            start = offset + instruction_size(op, self.opc)
            target = self.get_target(offset, op)
            end_offset = self.restrict_to_parent(target, parent)
            if target != end_offset:
                self.fixed_jumps[offset] = end_offset
            self.structs.append({'type': 'try', 'start': start - 3, 'end': end_offset - 4})
            end_else = start_else = self.get_target(self.prev[end_offset])
            end_finally_offset = end_offset
            setup_except_nest = 0
            while end_finally_offset < len(self.code):
                if self.code[end_finally_offset] == self.opc.END_FINALLY:
                    if setup_except_nest == 0:
                        break
                    else:
                        setup_except_nest -= 1
                elif self.code[end_finally_offset] == self.opc.SETUP_EXCEPT:
                    setup_except_nest += 1
                end_finally_offset += instruction_size(code[end_finally_offset], self.opc)
                pass
            i = end_offset
            while i < len(self.code) and i < end_finally_offset:
                jmp = self.next_except_jump(i)
                if jmp is None:
                    i = self.next_stmt[i]
                    continue
                if self.code[jmp] == self.opc.RETURN_VALUE:
                    self.structs.append({'type': 'except', 'start': i, 'end': jmp + 1})
                    i = jmp + 1
                else:
                    target = self.get_target(jmp)
                    if target != start_else:
                        end_else = self.get_target(jmp)
                    if self.code[jmp] == self.opc.JUMP_FORWARD:
                        if self.version <= (2, 6):
                            self.fixed_jumps[jmp] = target
                        else:
                            self.fixed_jumps[jmp] = -1
                    self.structs.append({'type': 'except', 'start': i, 'end': jmp})
                    i = jmp + 3
            if end_else != start_else:
                r_end_else = self.restrict_to_parent(end_else, parent)
                if self.version[:2] == (2, 7):
                    self.structs.append({'type': 'try-else', 'start': i + 1, 'end': r_end_else})
                    self.fixed_jumps[i] = r_end_else
            else:
                self.fixed_jumps[i] = i + 1
        elif op in self.pop_jump_if:
            target = self.get_target(offset, op)
            rtarget = self.restrict_to_parent(target, parent)
            if target != rtarget and parent['type'] == 'and/or':
                self.fixed_jumps[offset] = rtarget
                return
            jump_if_offset = offset
            start = offset + 3
            pre = self.prev
            test_target = target
            if self.version < (2, 7):
                if code[pre[test_target]] == self.opc.POP_TOP:
                    test_target = pre[test_target]
                test_set = self.pop_jump_if
            else:
                test_set = self.pop_jump_if_or_pop | self.pop_jump_if
            if code[pre[test_target]] in test_set and target > offset:
                self.fixed_jumps[offset] = pre[target]
                self.structs.append({'type': 'and/or', 'start': start, 'end': pre[target]})
                return
            pre_rtarget = pre[rtarget]
            if op == self.opc.PJIF:
                match = self.rem_or(start, self.next_stmt[offset], self.opc.PJIF, target)
                if match:
                    if code[pre_rtarget] in self.jump_forward and pre_rtarget not in self.stmts and (self.restrict_to_parent(self.get_target(pre_rtarget), parent) == rtarget):
                        if code[pre[pre_rtarget]] == self.opc.JUMP_ABSOLUTE and self.remove_mid_line_ifs([offset]) and (target == self.get_target(pre[pre_rtarget])) and (pre[pre_rtarget] not in self.stmts or self.get_target(pre[pre_rtarget]) > pre[pre_rtarget]) and (1 == len(self.remove_mid_line_ifs(self.rem_or(start, pre[pre_rtarget], self.pop_jump_if, target)))):
                            pass
                        elif code[pre[pre_rtarget]] == self.opc.RETURN_VALUE and self.remove_mid_line_ifs([offset]) and (1 == len(set(self.remove_mid_line_ifs(self.rem_or(start, pre[pre_rtarget], self.pop_jump_if, target))) | set(self.remove_mid_line_ifs(self.rem_or(start, pre[pre_rtarget], (self.opc.PJIF, self.opc.PJIT, self.opc.JUMP_ABSOLUTE), pre_rtarget, True))))):
                            pass
                        else:
                            fix = None
                            jump_ifs = self.all_instr(start, self.next_stmt[offset], self.opc.PJIF)
                            last_jump_good = True
                            for j in jump_ifs:
                                if target == self.get_target(j):
                                    if self.lines[j].next == j + 3 and last_jump_good:
                                        fix = j
                                        break
                                else:
                                    last_jump_good = False
                            self.fixed_jumps[offset] = fix or match[-1]
                            return
                    else:
                        if self.version < (2, 7) and parent['type'] in ('root', 'for-loop', 'if-then', 'else', 'try'):
                            self.fixed_jumps[offset] = rtarget
                        else:
                            self.fixed_jumps[offset] = match[-1]
                        return
            else:
                if self.version < (2, 7) and code[offset + 3] == self.opc.POP_TOP:
                    assert_offset = offset + 4
                else:
                    assert_offset = offset + 3
                if assert_offset in self.load_asserts:
                    if code[pre_rtarget] == self.opc.RAISE_VARARGS:
                        return
                    self.load_asserts.remove(assert_offset)
                next = self.next_stmt[offset]
                if pre[next] == offset:
                    pass
                elif code[next] in self.jump_forward and target == self.get_target(next):
                    if code[pre[next]] == self.opc.PJIF:
                        if code[next] == self.opc.JUMP_FORWARD or target != rtarget or code[pre[pre_rtarget]] not in (self.opc.JUMP_ABSOLUTE, self.opc.RETURN_VALUE):
                            self.fixed_jumps[offset] = pre[next]
                            return
                elif code[next] == self.opc.JUMP_ABSOLUTE and code[target] in self.jump_forward:
                    next_target = self.get_target(next)
                    if self.get_target(target) == next_target:
                        self.fixed_jumps[offset] = pre[next]
                        return
                    elif code[next_target] in self.jump_forward and self.get_target(next_target) == self.get_target(target):
                        self.fixed_jumps[offset] = pre[next]
                        return
            if offset in self.ignore_if:
                return
            if self.version == (2, 7):
                if code[pre_rtarget] == self.opc.JUMP_ABSOLUTE and pre_rtarget in self.stmts and (pre_rtarget != offset) and (pre[pre_rtarget] != offset):
                    if code[rtarget] == self.opc.JUMP_ABSOLUTE and code[rtarget + 3] == self.opc.POP_BLOCK:
                        if code[pre[pre_rtarget]] != self.opc.JUMP_ABSOLUTE:
                            pass
                        elif self.get_target(pre[pre_rtarget]) != target:
                            pass
                        else:
                            rtarget = pre_rtarget
                    else:
                        rtarget = pre_rtarget
                    pre_rtarget = pre[rtarget]
            code_pre_rtarget = code[pre_rtarget]
            if code_pre_rtarget in self.jump_forward:
                if_end = self.get_target(pre_rtarget)
                if if_end < pre_rtarget and pre[if_end] in self.setup_loop_targets:
                    if if_end > start:
                        return
                    else:
                        next_offset = target + instruction_size(self.code[target], self.opc)
                        next_op = self.code[next_offset]
                        if self.op_name(next_op) == 'JUMP_FORWARD':
                            jump_target = self.get_target(next_offset, next_op)
                            if jump_target in self.setup_loops:
                                self.structs.append({'type': 'while-loop', 'start': jump_if_offset, 'end': jump_target})
                                self.fixed_jumps[jump_if_offset] = jump_target
                                return
                end_offset = self.restrict_to_parent(if_end, parent)
                if_then_maybe = None
                if (2, 2) <= self.version <= (2, 6):
                    if self.opname_for_offset(jump_if_offset).startswith('JUMP_IF'):
                        jump_if_target = code[jump_if_offset + 1]
                        if self.opname_for_offset(jump_if_target + jump_if_offset + 3) == 'POP_TOP':
                            jump_inst = jump_if_target + jump_if_offset
                            jump_offset = code[jump_inst + 1]
                            jump_op = self.opname_for_offset(jump_inst)
                            if jump_op == 'JUMP_FORWARD' and jump_offset == 1:
                                self.structs.append({'type': 'if-then', 'start': start - 3, 'end': pre_rtarget})
                                self.thens[start] = end_offset
                            elif jump_op == 'JUMP_ABSOLUTE':
                                if_then_maybe = {'type': 'if-then', 'start': start - 3, 'end': pre_rtarget}
                elif self.version[:2] == (2, 7):
                    self.structs.append({'type': 'if-then', 'start': start - 3, 'end': pre_rtarget})
                if pre_rtarget not in self.linestarts or self.version < (2, 7):
                    self.not_continue.add(pre_rtarget)
                if rtarget < end_offset:
                    if if_then_maybe and jump_op == 'JUMP_ABSOLUTE':
                        jump_target = self.get_target(jump_inst, code[jump_inst])
                        if self.opname_for_offset(end_offset) == 'JUMP_FORWARD':
                            end_target = self.get_target(end_offset, code[end_offset])
                            if jump_target == end_target:
                                self.structs.append(if_then_maybe)
                                self.thens[start] = end_offset
                    self.structs.append({'type': 'else', 'start': rtarget, 'end': end_offset})
            elif code_pre_rtarget == self.opc.RETURN_VALUE:
                if self.version[:2] == (2, 7) or pre_rtarget not in self.ignore_if:
                    if self.code[self.prev[offset]] != self.opc.COMPARE_OP or self.code[self.prev[offset] + 1] != 10:
                        self.structs.append({'type': 'if-then', 'start': start, 'end': rtarget})
                        self.thens[start] = rtarget
                        if self.version[:2] == (2, 7) or code[pre_rtarget + 1] != self.opc.JUMP_FORWARD:
                            self.fixed_jumps[offset] = rtarget
                            if self.version[:2] == (2, 7) and self.insts[self.offset2inst_index[pre[pre_rtarget]]].is_jump_target:
                                self.return_end_ifs.add(pre[pre_rtarget])
                                pass
                            else:
                                self.return_end_ifs.add(pre_rtarget)
                            pass
                        pass
                    pass
        elif op in self.pop_jump_if_or_pop:
            target = self.get_target(offset, op)
            self.fixed_jumps[offset] = self.restrict_to_parent(target, parent)

    def find_jump_targets(self, debug):
        if False:
            while True:
                i = 10
        '\n        Detect all offsets in a byte code which are jump targets\n        where we might insert a pseudo "COME_FROM" instruction.\n        "COME_FROM" instructions are used in detecting overall\n        control flow. The more detailed information about the\n        control flow is captured in self.structs.\n        Since this stuff is tricky, consult self.structs when\n        something goes amiss.\n\n        Return the list of offsets. An instruction can be jumped\n        to in from multiple instructions.\n        '
        code = self.code
        n = len(code)
        self.structs = [{'type': 'root', 'start': 0, 'end': n - 1}]
        self.loops = []
        self.fixed_jumps = {}
        self.ignore_if = set()
        self.build_statement_indices()
        self.not_continue = set()
        self.return_end_ifs = set()
        self.setup_loop_targets = {}
        self.setup_loops = {}
        self.thens = {}
        targets = {}
        extended_arg = 0
        for offset in self.op_range(0, n):
            op = code[offset]
            if op == self.opc.EXTENDED_ARG:
                arg = code2num(code, offset + 1) | extended_arg
                extended_arg += self.extended_arg_val(arg)
                continue
            self.detect_control_flow(offset, op, extended_arg)
            if op_has_argument(op, self.opc):
                label = self.fixed_jumps.get(offset)
                oparg = self.get_argument(offset)
                if label is None:
                    if op in self.opc.JREL_OPS and self.op_name(op) != 'FOR_ITER':
                        label = offset + 3 + oparg
                    elif self.version[:2] == (2, 7) and op in self.opc.JABS_OPS:
                        if op in (self.opc.JUMP_IF_FALSE_OR_POP, self.opc.JUMP_IF_TRUE_OR_POP):
                            if oparg > offset:
                                label = oparg
                                pass
                            pass
                if label is not None and label != -1:
                    if self.version[:2] == (2, 7):
                        if label in self.setup_loops:
                            source = self.setup_loops[label]
                        else:
                            source = offset
                        targets[label] = targets.get(label, []) + [source]
                    elif not (code[label] == self.opc.POP_TOP and code[self.prev[label]] == self.opc.RETURN_VALUE):
                        skip_come_from = code[offset + 3] == self.opc.END_FINALLY or (code[offset + 3] == self.opc.POP_TOP and code[offset + 4] == self.opc.END_FINALLY)
                        if skip_come_from and op == self.opc.JUMP_FORWARD:
                            skip_come_from = False
                        if not skip_come_from:
                            if offset not in set(self.ignore_if):
                                if label in self.setup_loops:
                                    source = self.setup_loops[label]
                                else:
                                    source = offset
                                if self.version > (2, 6) or self.code[source] != self.opc.SETUP_LOOP or self.code[label] != self.opc.JUMP_FORWARD:
                                    targets[label] = targets.get(label, []) + [source]
                                pass
                            pass
                        pass
                    pass
            elif op == self.opc.END_FINALLY and offset in self.fixed_jumps and (self.version[:2] == (2, 7)):
                label = self.fixed_jumps[offset]
                targets[label] = targets.get(label, []) + [offset]
                pass
            extended_arg = 0
            pass
        if debug in ('both', 'after'):
            print(targets)
            import pprint as pp
            pp.pprint(self.structs)
        return targets

    def patch_continue(self, tokens, offset, op):
        if False:
            for i in range(10):
                print('nop')
        if op in (self.opc.JUMP_FORWARD, self.opc.JUMP_ABSOLUTE):
            n = len(tokens)
            if n > 2 and tokens[-1].kind == 'JUMP_BACK' and (self.code[offset + 3] == self.opc.END_FINALLY):
                tokens[-1].kind = intern('CONTINUE')

    def rem_or(self, start, end, instr, target=None, include_beyond_target=False):
        if False:
            return 10
        '\n        Find all <instr> in the block from start to end.\n        <instr> is any python bytecode instruction or a list of opcodes\n        If <instr> is an opcode with a target (like a jump), a target\n        destination can be specified which must match precisely.\n\n        Return a list with indexes to them or [] if none found.\n        '
        assert start >= 0 and end <= len(self.code) and (start <= end)
        try:
            None in instr
        except:
            instr = [instr]
        instr_offsets = []
        for i in self.op_range(start, end):
            op = self.code[i]
            if op in instr:
                if target is None:
                    instr_offsets.append(i)
                else:
                    t = self.get_target(i, op)
                    if include_beyond_target and t >= target:
                        instr_offsets.append(i)
                    elif t == target:
                        instr_offsets.append(i)
        pjits = self.all_instr(start, end, self.opc.PJIT)
        filtered = []
        for pjit in pjits:
            tgt = self.get_target(pjit) - 3
            for i in instr_offsets:
                if i <= pjit or i >= tgt:
                    filtered.append(i)
            instr_offsets = filtered
            filtered = []
        return instr_offsets