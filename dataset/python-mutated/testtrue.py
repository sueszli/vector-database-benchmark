def testtrue(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        for i in range(10):
            print('nop')
    if self.version[:2] != (3, 7):
        return False
    if rule == ('testtrue', ('expr', 'jmp_true')):
        pjit = tokens[min(last - 1, n - 2)]
        if pjit == 'POP_JUMP_IF_TRUE' and tokens[first].off2int() > pjit.attr:
            assert_next = tokens[min(last + 1, n - 1)]
            return assert_next != 'RAISE_VARARGS_1'
    elif rule == ('testfalsel', ('expr', 'jmp_true')):
        pjit = tokens[min(last - 1, n - 2)]
        if pjit == 'POP_JUMP_IF_TRUE' and tokens[first].off2int() > pjit.attr:
            assert_next = tokens[min(last + 1, n - 1)]
            return assert_next == 'RAISE_VARARGS_1'
    return False