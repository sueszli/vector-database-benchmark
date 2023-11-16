import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['NUMBER']
literals = 23

def t_NUMBER(t):
    if False:
        print('Hello World!')
    '\\d+'
    return t

def t_error(t):
    if False:
        return 10
    pass
lex.lex()