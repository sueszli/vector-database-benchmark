import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ('NAME', 'NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'EQUALS', 'LPAREN', 'RPAREN')

def make_calc():
    if False:
        while True:
            i = 10
    t_PLUS = '\\+'
    t_MINUS = '-'
    t_TIMES = '\\*'
    t_DIVIDE = '/'
    t_EQUALS = '='
    t_LPAREN = '\\('
    t_RPAREN = '\\)'
    t_NAME = '[a-zA-Z_][a-zA-Z0-9_]*'

    def t_NUMBER(t):
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
    t_ignore = ' \t'

    def t_newline(t):
        if False:
            i = 10
            return i + 15
        '\\n+'
        t.lineno += t.value.count('\n')

    def t_error(t):
        if False:
            for i in range(10):
                print('nop')
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
    return lex.lex()
make_calc()
lex.runmain(data='3+4')