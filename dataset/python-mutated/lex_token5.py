import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
t_PLUS = '\\+'
t_MINUS = '-'

def t_NUMBER(t):
    if False:
        print('Hello World!')
    '\\d+'
    t.type = 'NUM'
    return t

def t_error(t):
    if False:
        i = 10
        return i + 15
    pass
lex.lex()
lex.input('1234')
t = lex.token()