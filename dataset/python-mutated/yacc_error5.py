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

def p_statement_assign_error(t):
    if False:
        return 10
    'statement : NAME EQUALS error'
    (line_start, line_end) = t.linespan(3)
    (pos_start, pos_end) = t.lexspan(3)
    print('Assignment Error at %d:%d to %d:%d' % (line_start, pos_start, line_end, pos_end))

def p_statement_expr(t):
    if False:
        return 10
    'statement : expression'
    print(t[1])

def p_expression_binop(t):
    if False:
        return 10
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
        for i in range(10):
            print('nop')
    'expression : MINUS expression %prec UMINUS'
    t[0] = -t[2]

def p_expression_group(t):
    if False:
        for i in range(10):
            print('nop')
    'expression : LPAREN expression RPAREN'
    (line_start, line_end) = t.linespan(2)
    (pos_start, pos_end) = t.lexspan(2)
    print('Group at %d:%d to %d:%d' % (line_start, pos_start, line_end, pos_end))
    t[0] = t[2]

def p_expression_group_error(t):
    if False:
        return 10
    'expression : LPAREN error RPAREN'
    (line_start, line_end) = t.linespan(2)
    (pos_start, pos_end) = t.lexspan(2)
    print('Syntax error at %d:%d to %d:%d' % (line_start, pos_start, line_end, pos_end))
    t[0] = 0

def p_expression_number(t):
    if False:
        while True:
            i = 10
    'expression : NUMBER'
    t[0] = t[1]

def p_expression_name(t):
    if False:
        i = 10
        return i + 15
    'expression : NAME'
    try:
        t[0] = names[t[1]]
    except LookupError:
        print("Undefined name '%s'" % t[1])
        t[0] = 0

def p_error(t):
    if False:
        return 10
    print("Syntax error at '%s'" % t.value)
parser = yacc.yacc()
import calclex
calclex.lexer.lineno = 1
parser.parse('\na = 3 +\n(4*5) +\n(a b c) +\n+ 6 + 7\n', tracking=True)