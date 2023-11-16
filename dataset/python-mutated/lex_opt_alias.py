import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
tokens = ('NAME', 'NUMBER')
states = (('instdef', 'inclusive'), ('spam', 'exclusive'))
literals = ['=', '+', '-', '*', '/', '(', ')']

def t_instdef_spam_BITS(t):
    if False:
        for i in range(10):
            print('nop')
    '[01-]+'
    return t
t_NAME = '[a-zA-Z_][a-zA-Z0-9_]*'

def NUMBER(t):
    if False:
        for i in range(10):
            print('nop')
    '\\d+'
    try:
        t.value = int(t.value)
    except ValueError:
        print('Integer value too large %s' % t.value)
        t.value = 0
    return t
t_ANY_NUMBER = NUMBER
t_ignore = ' \t'
t_spam_ignore = t_ignore

def t_newline(t):
    if False:
        i = 10
        return i + 15
    '\\n+'
    t.lexer.lineno += t.value.count('\n')

def t_error(t):
    if False:
        i = 10
        return i + 15
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
t_spam_error = t_error
import ply.lex as lex
lex.lex(optimize=1, lextab='aliastab')
lex.runmain(data='3+4')