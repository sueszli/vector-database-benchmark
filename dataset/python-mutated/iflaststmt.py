def iflaststmt(self, lhs: str, n: int, rule, tree, tokens: list, first: int, last: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    testexpr = tree[0]
    if testexpr[0] in ('testtrue', 'testfalse'):
        test = testexpr[0]
        if len(test) > 1 and test[1].kind.startswith('jmp_'):
            if last == n:
                last -= 1
            jmp_target = test[1][0].attr
            if tokens[first].off2int() <= jmp_target < tokens[last].off2int():
                return True
            if last + 1 < n and tokens[last - 1] != 'JUMP_BACK' and (tokens[last + 1] == 'COME_FROM_LOOP'):
                return True
            if first > 0 and tokens[first - 1] == 'POP_JUMP_IF_FALSE':
                return tokens[first - 1].attr == jmp_target
            if jmp_target > tokens[last].off2int():
                if jmp_target == tokens[last - 1].attr:
                    return False
                if last < n and tokens[last].kind.startswith('JUMP'):
                    return False
                return True
        pass
    return False