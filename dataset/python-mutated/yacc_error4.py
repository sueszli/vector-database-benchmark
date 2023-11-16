import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.yacc as yacc
from calclex import tokens
precedence = (('left', 'PLUS', 'MINUS'), ('left', 'TIMES', 'DIVIDE'), ('right', 'UMINUS'))
names = {}

def p_statement_assign(t):
    if False:
        print('Hello World!')
    'statement : NAME EQUALS expression'
    names[t[1]] = t[3]

def p_statement_expr(t):
    if False:
        print('Hello World!')
    'statement : expression'
    print(t[1])

def p_expression_binop(t):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression PLUS expression\n                  | expression MINUS expression\n                  | expression TIMES expression\n                  | expression DIVIDE expression'
    if t[2] == '+':
        t[0] = t[1] + t[3]
    elif t[2] == '-':
        t[0] = t[1] - t[3]
    elif t[2] == '*':
        t[0] = t[1] * t[3]
    elif t[2] == '/':
        t[0] = t[1] / t[3]

def p_expression_uminus(t):
    if False:
        while True:
            i = 10
    'expression : MINUS expression %prec UMINUS'
    t[0] = -t[2]

def p_expression_group(t):
    if False:
        return 10
    'expression : LPAREN expression RPAREN'
    t[0] = t[2]

def p_expression_number(t):
    if False:
        print('Hello World!')
    'expression : NUMBER'
    t[0] = t[1]

def p_expression_name(t):
    if False:
        while True:
            i = 10
    'expression : NAME'
    try:
        t[0] = names[t[1]]
    except LookupError:
        print("Undefined name '%s'" % t[1])
        t[0] = 0

def p_error_handler(t):
    if False:
        while True:
            i = 10
    'error : NAME'
    pass

def p_error(t):
    if False:
        for i in range(10):
            print('nop')
    pass
yacc.yacc()