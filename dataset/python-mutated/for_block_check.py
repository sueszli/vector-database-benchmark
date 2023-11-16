from uncompyle6.scanners.tok import Token

def for_block_invalid(self, lhs, n, rule, tree, tokens, first: int, last: int) -> bool:
    if False:
        i = 10
        return i + 15
    if rule == ('for_block', ('l_stmts_opt', 'JUMP_ABSOLUTE', 'JUMP_BACK', 'JUMP_BACK')):
        jump_back1 = tokens[last - 2]
        jump_back2 = tokens[last - 1]
        if jump_back1.attr != jump_back2.attr:
            return True
        jump_absolute = tokens[last - 3]
        if jump_absolute.attr != jump_back2.offset:
            return True
        if self.version[:2] == (2, 7):
            return False
    if len(rule[1]) <= 1 or not tree:
        return False
    come_froms = tree[-1]
    pop_jump_index = first - 1
    while pop_jump_index > 0 and tokens[pop_jump_index] in ('ELSE', 'POP_TOP', 'JUMP_FORWARD', 'COME_FROM'):
        pop_jump_index -= 1
    if tokens[pop_jump_index].attr > 256:
        return False
    pop_jump_offset = tokens[pop_jump_index].off2int(prefer_last=False)
    if isinstance(come_froms, Token):
        if tokens[pop_jump_index].attr < pop_jump_offset and tree[0] != 'pass':
            return False
        return come_froms.attr is not None and pop_jump_offset > come_froms.attr
    elif len(come_froms) == 0:
        return False
    else:
        return pop_jump_offset > come_froms[-1].attr