def whilestmt38_check(self, lhs: str, n: int, rule, ast, tokens: list, first: int, last: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if tokens[last] != 'COME_FROM' and tokens[last - 1] == 'COME_FROM':
        last -= 1
    if tokens[last - 1].kind.startswith('RAISE_VARARGS'):
        return True
    while tokens[last] == 'COME_FROM':
        last -= 1
    first_offset = tokens[first].off2int()
    if tokens[last] == 'JUMP_LOOP' and (tokens[last].attr == first_offset or tokens[last - 1].attr == first_offset):
        return False
    return True