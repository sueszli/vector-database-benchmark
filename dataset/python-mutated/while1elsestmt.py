def while1elsestmt(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        i = 10
        return i + 15
    if last == n:
        last -= 1
    if tokens[last] == 'COME_FROM_LOOP':
        last -= 1
    elif tokens[last - 1] == 'COME_FROM_LOOP':
        last -= 2
    if tokens[last] in ('JUMP_BACK', 'CONTINUE'):
        return True
    last += 1
    return self.version < (3, 8) and tokens[first].attr > tokens[last].off2int()