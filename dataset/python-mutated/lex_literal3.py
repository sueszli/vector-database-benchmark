import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['NUMBER']
literals = []

def t_NUMBER(t):
    if False:
        i = 10
        return i + 15
    '\\d+'
    return t

def t_error(t):
    if False:
        while True:
            i = 10
    pass
lex.lex()