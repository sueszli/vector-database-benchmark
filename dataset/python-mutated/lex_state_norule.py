import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.lex as lex
tokens = ['PLUS', 'MINUS', 'NUMBER']
states = (('comment', 'exclusive'), ('example', 'exclusive'))
t_PLUS = '\\+'
t_MINUS = '-'
t_NUMBER = '\\d+'

def t_comment(t):
    if False:
        return 10
    '/\\*'
    t.lexer.begin('comment')
    print('Entering comment state')

def t_comment_body_part(t):
    if False:
        print('Hello World!')
    '(.|\\n)*\\*/'
    print('comment body %s' % t)
    t.lexer.begin('INITIAL')

def t_error(t):
    if False:
        i = 10
        return i + 15
    pass
lex.lex()