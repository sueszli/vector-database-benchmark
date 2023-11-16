import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
states = (('comment', 'exclusive'),)
t_PLUS = '\\+'
t_MINUS = '-'
t_NUMBER = '\\d+'
t_ignore = ' \t'

def t_comment(t):
    if False:
        return 10
    '/\\*'
    t.lexer.begin('comment')
    print('Entering comment state')

def t_comment_body_part(t):
    if False:
        while True:
            i = 10
    '(.|\\n)*\\*/'
    print('comment body %s' % t)
    t.lexer.begin('INITIAL')

def t_error(t):
    if False:
        return 10
    pass
t_comment_error = t_error
t_comment_ignore = t_ignore
lex.lex()
data = '3 + 4 /* This is a comment */ + 10'
lex.runmain(data=data)