import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
states = 'comment'
t_PLUS = '\\+'
t_MINUS = '-'
t_NUMBER = '\\d+'

def t_comment(t):
    if False:
        for i in range(10):
            print('nop')
    '/\\*'
    t.lexer.begin('comment')
    print('Entering comment state')

def t_comment_body_part(t):
    if False:
        i = 10
        return i + 15
    '(.|\\n)*\\*/'
    print('comment body %s' % t)
    t.lexer.begin('INITIAL')

def t_error(t):
    if False:
        for i in range(10):
            print('nop')
    pass
lex.lex()