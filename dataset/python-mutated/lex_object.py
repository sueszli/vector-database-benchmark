import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex

class CalcLexer:
    tokens = ('NAME', 'NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'EQUALS', 'LPAREN', 'RPAREN')
    t_PLUS = '\\+'
    t_MINUS = '-'
    t_TIMES = '\\*'
    t_DIVIDE = '/'
    t_EQUALS = '='
    t_LPAREN = '\\('
    t_RPAREN = '\\)'
    t_NAME = '[a-zA-Z_][a-zA-Z0-9_]*'

    def t_NUMBER(self, t):
        if False:
            print('Hello World!')
        '\\d+'
        try:
            t.value = int(t.value)
        except ValueError:
            print('Integer value too large %s' % t.value)
            t.value = 0
        return t
    t_ignore = ' \t'

    def t_newline(self, t):
        if False:
            while True:
                i = 10
        '\\n+'
        t.lineno += t.value.count('\n')

    def t_error(self, t):
        if False:
            for i in range(10):
                print('nop')
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
calc = CalcLexer()
lex.lex(object=calc)
lex.runmain(data='3+4')