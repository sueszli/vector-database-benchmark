import sys
sys.path.insert(0, '../..')
import ply.lex as lex
reserved = ('AUTO', 'BREAK', 'CASE', 'CHAR', 'CONST', 'CONTINUE', 'DEFAULT', 'DO', 'DOUBLE', 'ELSE', 'ENUM', 'EXTERN', 'FLOAT', 'FOR', 'GOTO', 'IF', 'INT', 'LONG', 'REGISTER', 'RETURN', 'SHORT', 'SIGNED', 'SIZEOF', 'STATIC', 'STRUCT', 'SWITCH', 'TYPEDEF', 'UNION', 'UNSIGNED', 'VOID', 'VOLATILE', 'WHILE')
tokens = reserved + ('ID', 'TYPEID', 'ICONST', 'FCONST', 'SCONST', 'CCONST', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD', 'OR', 'AND', 'NOT', 'XOR', 'LSHIFT', 'RSHIFT', 'LOR', 'LAND', 'LNOT', 'LT', 'LE', 'GT', 'GE', 'EQ', 'NE', 'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL', 'PLUSEQUAL', 'MINUSEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL', 'ANDEQUAL', 'XOREQUAL', 'OREQUAL', 'PLUSPLUS', 'MINUSMINUS', 'ARROW', 'CONDOP', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE', 'COMMA', 'PERIOD', 'SEMI', 'COLON', 'ELLIPSIS')
t_ignore = ' \t\x0c'

def t_NEWLINE(t):
    if False:
        print('Hello World!')
    '\\n+'
    t.lexer.lineno += t.value.count('\n')
t_PLUS = '\\+'
t_MINUS = '-'
t_TIMES = '\\*'
t_DIVIDE = '/'
t_MOD = '%'
t_OR = '\\|'
t_AND = '&'
t_NOT = '~'
t_XOR = '\\^'
t_LSHIFT = '<<'
t_RSHIFT = '>>'
t_LOR = '\\|\\|'
t_LAND = '&&'
t_LNOT = '!'
t_LT = '<'
t_GT = '>'
t_LE = '<='
t_GE = '>='
t_EQ = '=='
t_NE = '!='
t_EQUALS = '='
t_TIMESEQUAL = '\\*='
t_DIVEQUAL = '/='
t_MODEQUAL = '%='
t_PLUSEQUAL = '\\+='
t_MINUSEQUAL = '-='
t_LSHIFTEQUAL = '<<='
t_RSHIFTEQUAL = '>>='
t_ANDEQUAL = '&='
t_OREQUAL = '\\|='
t_XOREQUAL = '\\^='
t_PLUSPLUS = '\\+\\+'
t_MINUSMINUS = '--'
t_ARROW = '->'
t_CONDOP = '\\?'
t_LPAREN = '\\('
t_RPAREN = '\\)'
t_LBRACKET = '\\['
t_RBRACKET = '\\]'
t_LBRACE = '\\{'
t_RBRACE = '\\}'
t_COMMA = ','
t_PERIOD = '\\.'
t_SEMI = ';'
t_COLON = ':'
t_ELLIPSIS = '\\.\\.\\.'
reserved_map = {}
for r in reserved:
    reserved_map[r.lower()] = r

def t_ID(t):
    if False:
        return 10
    '[A-Za-z_][\\w_]*'
    t.type = reserved_map.get(t.value, 'ID')
    return t
t_ICONST = '\\d+([uU]|[lL]|[uU][lL]|[lL][uU])?'
t_FCONST = '((\\d+)(\\.\\d+)(e(\\+|-)?(\\d+))? | (\\d+)e(\\+|-)?(\\d+))([lL]|[fF])?'
t_SCONST = '\\"([^\\\\\\n]|(\\\\.))*?\\"'
t_CCONST = "(L)?\\'([^\\\\\\n]|(\\\\.))*?\\'"

def t_comment(t):
    if False:
        while True:
            i = 10
    '/\\*(.|\\n)*?\\*/'
    t.lexer.lineno += t.value.count('\n')

def t_preprocessor(t):
    if False:
        print('Hello World!')
    '\\#(.)*?\\n'
    t.lexer.lineno += 1

def t_error(t):
    if False:
        while True:
            i = 10
    print('Illegal character %s' % repr(t.value[0]))
    t.lexer.skip(1)
lexer = lex.lex()
if __name__ == '__main__':
    lex.runmain(lexer)