def p_statement_assign(t):
    if False:
        for i in range(10):
            print('nop')
    'statement : NAME EQUALS expression'
    names[t[1]] = t[3]

def p_statement_expr(t):
    if False:
        return 10
    'statement : expression'
    t[0] = t[1]