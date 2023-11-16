import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.yacc as yacc
from calclex import tokens
precedence = (('left', 'PLUS', 'MINUS'), ('left', 'TIMES', 'DIVIDE'), ('right', 'UMINUS'))

def p_statements(t):
    if False:
        for i in range(10):
            print('nop')
    'statements : statements statement'
    pass

def p_statements_1(t):
    if False:
        print('Hello World!')
    'statements : statement'
    pass

def p_statement_assign(p):
    if False:
        i = 10
        return i + 15
    'statement : LPAREN NAME EQUALS expression RPAREN'
    print('%s=%s' % (p[2], p[4]))

def p_statement_expr(t):
    if False:
        print('Hello World!')
    'statement : LPAREN expression RPAREN'
    print(t[1])

def p_expression_binop(t):
    if False:
        print('Hello World!')
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
        return 10
    'expression : MINUS expression %prec UMINUS'
    t[0] = -t[2]

def p_expression_number(t):
    if False:
        i = 10
        return i + 15
    'expression : NUMBER'
    t[0] = t[1]

def p_error(p):
    if False:
        print('Hello World!')
    if p:
        print("Line %d: Syntax error at '%s'" % (p.lineno, p.value))
    while True:
        tok = parser.token()
        if not tok or tok.type == 'RPAREN':
            break
    if tok:
        parser.restart()
    return None
parser = yacc.yacc()
import calclex
calclex.lexer.lineno = 1
parser.parse('\n(a = 3 + 4)\n(b = 4 + * 5 - 6 + *)\n(c = 10 + 11)\n')