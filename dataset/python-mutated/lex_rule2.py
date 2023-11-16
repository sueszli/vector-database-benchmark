import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
t_PLUS = '\\+'
t_MINUS = '-'

def t_NUMBER():
    if False:
        i = 10
        return i + 15
    '\\d+'
    return t

def t_error(t):
    if False:
        for i in range(10):
            print('nop')
    pass
lex.lex()