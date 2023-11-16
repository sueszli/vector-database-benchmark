import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
t_PLUS = '\\+'
t_MINUS = '-'
t_NUMBER = '\\d+'

def t_ignore(t):
    if False:
        while True:
            i = 10
    ' \t'
    pass

def t_error(t):
    if False:
        i = 10
        return i + 15
    pass
import sys
lex.lex()