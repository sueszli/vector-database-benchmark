from ply import *
keywords = ('LET', 'READ', 'DATA', 'PRINT', 'GOTO', 'IF', 'THEN', 'FOR', 'NEXT', 'TO', 'STEP', 'END', 'STOP', 'DEF', 'GOSUB', 'DIM', 'REM', 'RETURN', 'RUN', 'LIST', 'NEW')
tokens = keywords + ('EQUALS', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'POWER', 'LPAREN', 'RPAREN', 'LT', 'LE', 'GT', 'GE', 'NE', 'COMMA', 'SEMI', 'INTEGER', 'FLOAT', 'STRING', 'ID', 'NEWLINE')
t_ignore = ' \t'

def t_REM(t):
    if False:
        for i in range(10):
            print('nop')
    'REM .*'
    return t

def t_ID(t):
    if False:
        for i in range(10):
            print('nop')
    '[A-Z][A-Z0-9]*'
    if t.value in keywords:
        t.type = t.value
    return t
t_EQUALS = '='
t_PLUS = '\\+'
t_MINUS = '-'
t_TIMES = '\\*'
t_POWER = '\\^'
t_DIVIDE = '/'
t_LPAREN = '\\('
t_RPAREN = '\\)'
t_LT = '<'
t_LE = '<='
t_GT = '>'
t_GE = '>='
t_NE = '<>'
t_COMMA = '\\,'
t_SEMI = ';'
t_INTEGER = '\\d+'
t_FLOAT = '((\\d*\\.\\d+)(E[\\+-]?\\d+)?|([1-9]\\d*E[\\+-]?\\d+))'
t_STRING = '\\".*?\\"'

def t_NEWLINE(t):
    if False:
        return 10
    '\\n'
    t.lexer.lineno += 1
    return t

def t_error(t):
    if False:
        print('Hello World!')
    print('Illegal character %s' % t.value[0])
    t.lexer.skip(1)
lex.lex(debug=0)