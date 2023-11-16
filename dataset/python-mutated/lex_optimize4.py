import re
import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
t_PLUS = '\\+?'
t_MINUS = '-'
t_NUMBER = '(\\d+)'

def t_error(t):
    if False:
        i = 10
        return i + 15
    pass
lex.lex(optimize=True, lextab='opt4tab', reflags=re.UNICODE)
lex.runmain(data='3+4')