import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER', 'MINUS']
t_PLUS = '\\+'
t_MINUS = '-'

def t_NUMBER(t):
    if False:
        for i in range(10):
            print('nop')
    '\\d+'
    return t

def t_error(t):
    if False:
        return 10
    pass
lex.lex()