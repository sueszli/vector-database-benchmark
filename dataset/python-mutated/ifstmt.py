def ifstmt(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        for i in range(10):
            print('nop')
    first_offset = tokens[first].off2int(prefer_last=False)
    if lhs == 'ifstmtl':
        if last == n:
            last -= 1
            pass
        if tokens[last].attr and isinstance(tokens[last].attr, int):
            if first_offset >= tokens[last].attr:
                return True
            pass
        pass
    l = last
    if l == n:
        l -= 1
    if isinstance(tokens[l].offset, str):
        last_offset = int(tokens[l].offset.split('_')[0], 10)
    else:
        last_offset = tokens[l].offset
    for i in range(first, l):
        t = tokens[i]
        if t.kind in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'):
            pjif_target = t.attr
            target_instr = self.insts[self.offset2inst_index[pjif_target]]
            if lhs == 'iflaststmtl' and target_instr.opname == 'JUMP_ABSOLUTE':
                pjif_target = target_instr.arg
            if pjif_target > last_offset:
                if tokens[l] == 'JUMP_FORWARD':
                    return tokens[l].attr != pjif_target
                return True
            elif lhs == 'ifstmtl' and first_offset > pjif_target:
                return False
            pass
        pass
    pass
    if ast:
        testexpr = ast[0]
        if last + 1 < n and tokens[last + 1] == 'COME_FROM_LOOP':
            return True
        if testexpr[0] in ('testtrue', 'testfalse'):
            test = testexpr[0]
            if len(test) > 1 and test[1].kind.startswith('jmp_'):
                jmp_target = test[1][0].attr
                if first_offset <= jmp_target < tokens[last].off2int(prefer_last=False):
                    return True
                if jmp_target > tokens[last].off2int():
                    if jmp_target == tokens[last - 1].attr:
                        return False
                    if last < n and tokens[last].kind.startswith('JUMP'):
                        return False
                    return True
            pass
        pass
    return False