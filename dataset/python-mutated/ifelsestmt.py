from uncompyle6.scanners.tok import Token
IFELSE_STMT_RULES = frozenset([('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jump_forward_else', 'else_suite', '_come_froms')), ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jump_forward_else', 'else_suite', '\\e__come_froms')), ('ifelsestmtl', ('testexpr', 'c_stmts_opt', 'jump_forward_else', 'else_suitec')), ('ifelsestmtc', ('testexpr', 'c_stmts_opt', 'jump_forward_else', 'else_suitec', '\\e__come_froms')), ('ifelsestmtc', ('testexpr', 'c_stmts_opt', 'jump_absolute_else', 'else_suitec')), ('ifelsestmtc', ('testexpr', 'c_stmts_opt', 'JUMP_ABSOLUTE', 'else_suitec')), ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jf_cfs', 'else_suite', '\\e_opt_come_from_except')), ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'JUMP_FORWARD', 'else_suite', 'come_froms')), ('ifelsestmtc', ('testexpr', 'c_stmts_opt', 'JUMP_FORWARD', 'else_suite', 'come_froms')), ('ifelsestmt', ('testexpr', 'c_stmts', 'come_froms', 'else_suite', 'come_froms')), ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jf_cfs', 'else_suite', 'opt_come_from_except')), ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jf_cf_pop', 'else_suite')), ('ifelsestmt', ('testexpr', 'stmts', 'jf_cfs', 'else_suite_opt', 'opt_come_from_except')), ('ifelsestmt', ('testexpr', 'stmts', 'jf_cfs', '\\e_else_suite_opt', '\\e_opt_come_from_except')), ('ifelsestmt', ('testexpr', 'stmts', 'jf_cfs', '\\e_else_suite_opt', 'opt_come_from_except'))])

def ifelsestmt(self, lhs, n, rule, tree, tokens, first, last):
    if False:
        print('Hello World!')
    if last + 1 < n and tokens[last + 1] == 'COME_FROM_LOOP' and (lhs != 'ifelsestmtc'):
        return True
    first_offset = tokens[first].off2int()
    if rule not in IFELSE_STMT_RULES:
        return False
    stmts = tree[1]
    if stmts in ('c_stmts',) and len(stmts) == 1:
        raise_stmt1 = stmts[0]
        if raise_stmt1 == 'raise_stmt1' and raise_stmt1[0] in ('LOAD_ASSERT',):
            return True
    if len(tree) == 5:
        end_come_froms = tree[-1]
        if end_come_froms.kind != 'else_suite' and self.version >= (3, 0):
            if end_come_froms == 'opt_come_from_except' and len(end_come_froms) > 0:
                end_come_froms = end_come_froms[0]
            if not isinstance(end_come_froms, Token):
                if len(end_come_froms):
                    return first_offset > end_come_froms[-1].attr
            elif first_offset > end_come_froms.attr:
                return True
        if self.version < (3, 0):
            last_token = tree[-1]
        else:
            last_token = tokens[last]
        if last_token == 'COME_FROM' and first_offset > last_token.attr:
            if self.version < (3, 0) and self.insts[self.offset2inst_index[last_token.attr]].opname != 'SETUP_LOOP':
                return True
    testexpr = tree[0]
    if testexpr[0] in ('testtrue', 'testfalse'):
        if_condition = testexpr[0]
        else_suite = tree[3]
        assert else_suite.kind.startswith('else_suite')
        if len(if_condition) > 1 and if_condition[1].kind.startswith('jmp_'):
            if last == n:
                last -= 1
            jmp = if_condition[1]
            if self.version >= (2, 7):
                jmp_target = jmp[0].attr
            else:
                jmp_target = int(jmp[0].pattr)
            jump_else_end = tree[2]
            if jump_else_end == 'jf_cf_pop':
                jump_else_end = jump_else_end[0]
            if jump_else_end == 'JUMP_FORWARD':
                endif_target = int(jump_else_end.pattr)
                last_offset = tokens[last].off2int()
                if endif_target != last_offset:
                    return True
            last_offset = tokens[last].off2int(prefer_last=False)
            if jmp_target == last_offset:
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
            if first_offset > jmp_target:
                if lhs in ('ifelsestmtl', 'ifelsestmtc'):
                    jump_false = jmp
                    if tree[2].kind == 'JUMP_FORWARD' and jump_false == 'jmp_false' and (len(else_suite) == 1):
                        suite_stmts = else_suite[0]
                        continue_stmt = suite_stmts[0]
                        if suite_stmts == 'suite_stmts' and len(suite_stmts) == 1 and (continue_stmt == 'continue') and (jump_false[0].attr == continue_stmt[0].attr):
                            return False
                return True
            return jmp_target > last_offset and tokens[last] != 'JUMP_FORWARD'
    return False