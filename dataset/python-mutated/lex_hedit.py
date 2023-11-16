import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ('H_EDIT_DESCRIPTOR',)
t_ignore = ' \t\n'

def t_H_EDIT_DESCRIPTOR(t):
    if False:
        for i in range(10):
            print('nop')
    '\\d+H.*'
    i = t.value.index('H')
    n = eval(t.value[:i])
    t.lexer.lexpos -= len(t.value) - (i + 1 + n)
    t.value = t.value[i + 1:i + 1 + n]
    return t

def t_error(t):
    if False:
        while True:
            i = 10
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
lex.lex()
lex.runmain(data='3Habc 10Habcdefghij 2Hxy')