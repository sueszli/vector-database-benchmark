import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
from ply import lex, yacc
t_A = 'A'
t_B = 'B'
t_C = 'C'
tokens = ('A', 'B', 'C')
the_lexer = lex.lex()

def t_error(t):
    if False:
        print('Hello World!')
    pass

def p_error(p):
    if False:
        return 10
    pass

def p_start(t):
    if False:
        return 10
    'start : A nest C'
    pass

def p_nest(t):
    if False:
        return 10
    'nest : B'
    print(t[-1])
the_parser = yacc.yacc(debug=False, write_tables=False)
the_parser.parse('ABC', the_lexer)
the_parser.parse('ABC', the_lexer, tracking=True)
the_parser.parse('ABC', the_lexer, tracking=True, debug=1)