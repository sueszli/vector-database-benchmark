import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'NUMBER']
t_PLUS = '\\+'
t_MINUS = '-'
t_NUMBER = '\\d+'

def t_error(t):
    if False:
        return 10
    pass
lex.lex()