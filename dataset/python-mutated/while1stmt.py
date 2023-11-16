def while1stmt(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        for i in range(10):
            print('nop')
    if last == n or tokens[last - 1] == 'COME_FROM_LOOP':
        cfl = last - 1
    else:
        cfl = last
    assert tokens[cfl] == 'COME_FROM_LOOP'
    for loop_end in range(cfl - 1, first, -1):
        if tokens[loop_end] != 'POP_BLOCK':
            break
    if tokens[loop_end].kind not in ('JUMP_BACK', 'RETURN_VALUE', 'RAISE_VARARGS_1'):
        if not tokens[loop_end].kind.startswith('COME_FROM'):
            return True
    if 0 <= last and tokens[last] in ('COME_FROM_LOOP', 'JUMP_BACK'):
        last += 1
    if last == n:
        last -= 1
    offset = tokens[last].off2int()
    assert tokens[first] == 'SETUP_LOOP'
    if tokens[loop_end] == 'JUMP_BACK':
        loop_end += 1
    loop_end_offset = tokens[loop_end].off2int(prefer_last=False)
    for t in range(first + 1, loop_end):
        token = tokens[t]
        if token.opc.opmap.get(token.kind, 'LOAD_CONST') in token.opc.JUMP_OPS:
            if token.attr >= loop_end_offset:
                return True
    return tokens[first].attr not in (offset, offset + 2)