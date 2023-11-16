"""
If statement reduction check for Python 2.6 (and older?)
"""

def ifstmt2(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        return 10
    if lhs == 'ifstmtl':
        if last == n:
            last -= 1
            pass
        if tokens[last].attr and isinstance(tokens[last].attr, int):
            if tokens[first].offset >= tokens[last].attr:
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
        if t.kind in ('JUMP_IF_FALSE', 'JUMP_IF_TRUE'):
            jif_target = int(t.pattr)
            target_instr = self.insts[self.offset2inst_index[jif_target]]
            if lhs == 'iflaststmtl' and target_instr.opname == 'JUMP_ABSOLUTE':
                jif_target = target_instr.arg
            if jif_target > last_offset:
                if tokens[l] == 'JUMP_FORWARD':
                    return tokens[l].attr != jif_target
                return True
            elif lhs == 'ifstmtl' and tokens[first].off2int() > jif_target:
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
            jmp = test[1]
            if len(test) > 1 and jmp.kind.startswith('jmp_'):
                jmp_target = int(jmp[0].pattr)
                if last == len(tokens):
                    last -= 1
                if_end_offset = tokens[last].off2int(prefer_last=False)
                if tokens[first].off2int(prefer_last=True) <= jmp_target < if_end_offset:
                    previous_inst_index = self.offset2inst_index[jmp_target] - 1
                    previous_inst = self.insts[previous_inst_index]
                    if previous_inst.opname != 'JUMP_ABSOLUTE' and previous_inst.argval != if_end_offset:
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