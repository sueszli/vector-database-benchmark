import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
t_PLUS = '\\+'
t_MINUS = '-'

def t_NUMBER(t):
    if False:
        while True:
            i = 10
    '\\d+'
    pass

def t_NUMBER(t):
    if False:
        i = 10
        return i + 15
    '\\d+'
    pass

def t_error(t):
    if False:
        while True:
            i = 10
    pass
lex.lex()