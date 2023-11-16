import sys
sys.path.insert(0, '../..')
tokens = ('H_EDIT_DESCRIPTOR',)
t_ignore = ' \t\n'

def t_H_EDIT_DESCRIPTOR(t):
    if False:
        return 10
    '\\d+H.*'
    i = t.value.index('H')
    n = eval(t.value[:i])
    t.lexer.lexpos -= len(t.value) - (i + 1 + n)
    t.value = t.value[i + 1:i + 1 + n]
    return t

def t_error(t):
    if False:
        for i in range(10):
            print('nop')
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
import ply.lex as lex
lex.lex()
lex.runmain()