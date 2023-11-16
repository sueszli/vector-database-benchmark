ASSERT_OPS = frozenset(['LOAD_ASSERT', 'RAISE_VARARGS_1'])

def or_check(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        print('Hello World!')
    rhs = rule[1]
    if rhs[0:2] in (('expr_jt', 'expr'), ('expr_jitop', 'expr'), ('expr_jit', 'expr')):
        if tokens[last] in ASSERT_OPS or tokens[last - 1] in ASSERT_OPS:
            return True
        load_global = tokens[last - 1]
        if load_global == 'LOAD_GLOBAL' and load_global.attr == 'AssertionError':
            return True
        first_offset = tokens[first].off2int()
        expr_jt = ast[0]
        if expr_jt == 'expr_jitop':
            jump_true = expr_jt[1]
        else:
            jump_true = expr_jt[1][0]
        jmp_true_target = jump_true.attr
        last_token = tokens[last]
        last_token_offset = last_token.off2int()
        if jmp_true_target < first_offset:
            return False
        elif jmp_true_target < last_token_offset:
            return True
        if last_token == 'POP_JUMP_IF_FALSE' and (not self.version[:2] in ((2, 7), (3, 5), (3, 6))):
            if last_token.attr < last_token_offset:
                last_token = tokens[last + 1]
            return not (last_token_offset <= jmp_true_target <= last_token_offset + 3 or jmp_true_target < tokens[first].off2int())
        elif last_token == 'JUMP_FORWARD' and expr_jt.kind != 'expr_jitop':
            return True
    return False