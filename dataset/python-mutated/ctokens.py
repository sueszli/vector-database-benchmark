tokens = ['ID', 'TYPEID', 'INTEGER', 'FLOAT', 'STRING', 'CHARACTER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'OR', 'AND', 'NOT', 'XOR', 'LSHIFT', 'RSHIFT', 'LOR', 'LAND', 'LNOT', 'LT', 'LE', 'GT', 'GE', 'EQ', 'NE', 'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL', 'PLUSEQUAL', 'MINUSEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL', 'ANDEQUAL', 'XOREQUAL', 'OREQUAL', 'INCREMENT', 'DECREMENT', 'ARROW', 'TERNARY', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE', 'COMMA', 'PERIOD', 'SEMI', 'COLON', 'ELLIPSIS']
t_PLUS = '\\+'
t_MINUS = '-'
t_TIMES = '\\*'
t_DIVIDE = '/'
t_MODULO = '%'
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
t_INCREMENT = '\\+\\+'
t_DECREMENT = '--'
t_ARROW = '->'
t_TERNARY = '\\?'
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
t_ID = '[A-Za-z_][A-Za-z0-9_]*'
t_INTEGER = '\\d+([uU]|[lL]|[uU][lL]|[lL][uU])?'
t_FLOAT = '((\\d+)(\\.\\d+)(e(\\+|-)?(\\d+))? | (\\d+)e(\\+|-)?(\\d+))([lL]|[fF])?'
t_STRING = '\\"([^\\\\\\n]|(\\\\.))*?\\"'
t_CHARACTER = "(L)?\\'([^\\\\\\n]|(\\\\.))*?\\'"

def t_COMMENT(t):
    if False:
        print('Hello World!')
    '/\\*(.|\\n)*?\\*/'
    t.lexer.lineno += t.value.count('\n')
    return t

def t_CPPCOMMENT(t):
    if False:
        return 10
    '//.*\\n'
    t.lexer.lineno += 1
    return t