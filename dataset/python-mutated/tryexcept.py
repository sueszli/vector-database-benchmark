def tryexcept(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        while True:
            i = 10
    come_from_except = ast[-1]
    if rule == ('try_except', ('SETUP_EXCEPT', 'suite_stmts_opt', 'POP_BLOCK', 'except_handler', 'opt_come_from_except')):
        if come_from_except[0] == 'COME_FROM':
            return True
        pass
    elif rule == ('try_except', ('SETUP_EXCEPT', 'suite_stmts_opt', 'POP_BLOCK', 'except_handler', 'COME_FROM')):
        return come_from_except.attr < tokens[first].offset
    elif rule == ('try_except', ('SETUP_EXCEPT', 'suite_stmts_opt', 'POP_BLOCK', 'except_handler', '\\e_opt_come_from_except')):
        for i in range(last, first, -1):
            if tokens[i] == 'END_FINALLY':
                jump_before_finally = tokens[i - 1]
                if jump_before_finally.kind.startswith('JUMP'):
                    if jump_before_finally == 'JUMP_FORWARD':
                        return tokens[i - 1].attr > tokens[last].off2int(prefer_last=True)
                    elif jump_before_finally == 'JUMP_BACK':
                        except_handler = ast[3]
                        if except_handler == 'except_handler' and except_handler[0] == 'JUMP_FORWARD':
                            return True
                        return False
                    pass
                pass
            pass
        return False