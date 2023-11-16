import sys
sys.path.append('../..')
from ply import *
tokens = ('LITERAL', 'SECTION', 'TOKEN', 'LEFT', 'RIGHT', 'PREC', 'START', 'TYPE', 'NONASSOC', 'UNION', 'CODE', 'ID', 'QLITERAL', 'NUMBER')
states = (('code', 'exclusive'),)
literals = [';', ',', '<', '>', '|', ':']
t_ignore = ' \t'
t_TOKEN = '%token'
t_LEFT = '%left'
t_RIGHT = '%right'
t_NONASSOC = '%nonassoc'
t_PREC = '%prec'
t_START = '%start'
t_TYPE = '%type'
t_UNION = '%union'
t_ID = '[a-zA-Z_][a-zA-Z_0-9]*'
t_QLITERAL = '(?P<quote>[\'"]).*?(?P=quote)'
t_NUMBER = '\\d+'

def t_SECTION(t):
    if False:
        print('Hello World!')
    '%%'
    if getattr(t.lexer, 'lastsection', 0):
        t.value = t.lexer.lexdata[t.lexpos + 2:]
        t.lexer.lexpos = len(t.lexer.lexdata)
    else:
        t.lexer.lastsection = 0
    return t

def t_ccomment(t):
    if False:
        while True:
            i = 10
    '/\\*(.|\\n)*?\\*/'
    t.lexer.lineno += t.value.count('\n')
t_ignore_cppcomment = '//.*'

def t_LITERAL(t):
    if False:
        while True:
            i = 10
    '%\\{(.|\\n)*?%\\}'
    t.lexer.lineno += t.value.count('\n')
    return t

def t_NEWLINE(t):
    if False:
        print('Hello World!')
    '\\n'
    t.lexer.lineno += 1

def t_code(t):
    if False:
        i = 10
        return i + 15
    '\\{'
    t.lexer.codestart = t.lexpos
    t.lexer.level = 1
    t.lexer.begin('code')

def t_code_ignore_string(t):
    if False:
        return 10
    '\\"([^\\\\\\n]|(\\\\.))*?\\"'

def t_code_ignore_char(t):
    if False:
        for i in range(10):
            print('nop')
    "\\'([^\\\\\\n]|(\\\\.))*?\\'"

def t_code_ignore_comment(t):
    if False:
        i = 10
        return i + 15
    '/\\*(.|\\n)*?\\*/'

def t_code_ignore_cppcom(t):
    if False:
        return 10
    '//.*'

def t_code_lbrace(t):
    if False:
        while True:
            i = 10
    '\\{'
    t.lexer.level += 1

def t_code_rbrace(t):
    if False:
        i = 10
        return i + 15
    '\\}'
    t.lexer.level -= 1
    if t.lexer.level == 0:
        t.type = 'CODE'
        t.value = t.lexer.lexdata[t.lexer.codestart:t.lexpos + 1]
        t.lexer.begin('INITIAL')
        t.lexer.lineno += t.value.count('\n')
        return t
t_code_ignore_nonspace = '[^\\s\\}\\\'\\"\\{]+'
t_code_ignore_whitespace = '\\s+'
t_code_ignore = ''

def t_code_error(t):
    if False:
        return 10
    raise RuntimeError

def t_error(t):
    if False:
        while True:
            i = 10
    print("%d: Illegal character '%s'" % (t.lexer.lineno, t.value[0]))
    print(t.value)
    t.lexer.skip(1)
lex.lex()
if __name__ == '__main__':
    lex.runmain()