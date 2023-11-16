"""
If/else statement reduction check for Python 2.6 (and older?)
"""
IFELSE_STMT_RULES = frozenset([('ifelsestmt', ('testexpr_then', 'pass', 'filler', 'else_suitel', 'COME_FROM', 'POP_TOP')), ('ifelsestmt', ('testexpr_then', 'c_stmts_opt', '\\e_filler', 'else_suitel', 'come_froms', 'POP_TOP')), ('ifelsestmt', ('testexpr_then', '\\e_c_stmts_opt', '\\e_filler', 'else_suitel', 'come_froms', 'POP_TOP'))])

def ifelsestmt2(self, lhs, n, rule, tree, tokens, first, last):
    if False:
        i = 10
        return i + 15
    if last + 1 < n and tokens[last + 1] == 'COME_FROM_LOOP' and (lhs != 'ifelsestmtc'):
        return True
    if rule not in IFELSE_STMT_RULES:
        return False
    stmts = tree[1]
    if stmts in ('c_stmts',) and len(stmts) == 1:
        raise_stmt1 = stmts[0]
        if raise_stmt1 == 'raise_stmt1' and raise_stmt1[0] in ('LOAD_ASSERT',):
            return True
    if len(tree) == 6 and tree[-1] == 'POP_TOP':
        last_token = tree[-2]
        if last_token == 'COME_FROM' and tokens[first].offset > last_token.attr:
            if self.insts[self.offset2inst_index[last_token.attr]].opname != 'SETUP_LOOP':
                return True
    testexpr = tree[0]
    if testexpr[0] in ('testtrue', 'testfalse', 'testfalse_then'):
        if_condition = testexpr[0]
        else_suite = tree[3]
        assert else_suite.kind.startswith('else_suite')
        if len(if_condition) > 1 and if_condition[1].kind.startswith('jmp_'):
            if last == n:
                last -= 1
            jmp = if_condition[1]
            jmp_target = int(jmp[0].pattr)
            if tree[2] == 'filler':
                jump_else_end = tree[3]
            else:
                jump_else_end = tree[2]
            if jump_else_end == 'jf_cfs':
                jump_else_end = jump_else_end[0]
            if jump_else_end == 'JUMP_FORWARD':
                endif_target = int(jump_else_end.pattr)
                last_offset = tokens[last].off2int()
                if endif_target != last_offset:
                    return True
            last_offset = tokens[last].off2int(prefer_last=False)
            if jmp_target <= last_offset:
                return True
            if jump_else_end in ('jf_cfs', 'jump_forward_else') and jump_else_end[0] == 'JUMP_FORWARD':
                jump_else_forward = jump_else_end[0]
                jump_else_forward_target = jump_else_forward.attr
                if jump_else_forward_target < last_offset:
                    return True
                pass
            if jump_else_end in ('jb_elsec', 'jb_elsel', 'jf_cfs', 'jb_cfs') and jump_else_end[-1] == 'COME_FROM':
                if jump_else_end[-1].off2int() != jmp_target:
                    return True
            if tokens[first].off2int() > jmp_target:
                return True
            return jmp_target > last_offset and tokens[last] != 'JUMP_FORWARD'
    return False