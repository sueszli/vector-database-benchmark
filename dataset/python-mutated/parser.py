import ply.yacc as yacc
import copy
from ..smtlib import Operators
import ply.lex as lex
import re

class ParserException(Exception):
    """
    Parser exception
    """
    pass
tokens = ('NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'AND', 'OR', 'NEG', 'LPAREN', 'RPAREN', 'LBRAKET', 'RBRAKET', 'REGISTER', 'SEGMENT', 'COLOM', 'PTR', 'TYPE', 'RSHIFT', 'LSHIFT', 'LOR', 'LAND', 'LNOT', 'EQ', 'LT', 'LE', 'GT', 'GE')
t_PLUS = '\\+'
t_MINUS = '-'
t_TIMES = '\\*'
t_DIVIDE = '/'
t_LPAREN = '\\('
t_RPAREN = '\\)'
t_LBRAKET = '\\['
t_RBRAKET = '\\]'
t_COLOM = ':'
t_AND = '&'
t_OR = '\\|'
t_NEG = '~'
t_LSHIFT = '<<'
t_RSHIFT = '>>'
t_LAND = '&&'
t_LOR = '\\|\\|'
t_LNOT = '!'
t_EQ = '=='
t_LT = '<'
t_LE = '<='
t_GT = '>'
t_GE = '>='
re_NUMBER = re.compile('^(0x[a-fA-F0-9]+|[a-fA-F0-9]+)$')
re_REGISTER = re.compile('^(EAX|EBX|ECX|EDX|ESI|EDI|ESP|EBP|RAX|RBX|RCX|RDX|RSI|RDI|RSP|RBP|ZF|CF|SF|EFLAGS)$')
re_SEGMENT = re.compile('^(SS|DS|ES|SS|CS)$')
re_TYPE = re.compile('^(QWORD|DWORD|WORD|BYTE)$')
re_PTR = re.compile('^PTR$')

def t_TOKEN(t):
    if False:
        i = 10
        return i + 15
    '[a-zA-Z0-9]+'
    if re_TYPE.match(t.value):
        t.type = 'TYPE'
    elif re_PTR.match(t.value):
        t.type = 'PTR'
    elif re_NUMBER.match(t.value):
        if t.value.startswith('0x'):
            t.value = t.value[2:]
        t.value = int(t.value, 16)
        t.type = 'NUMBER'
    elif re_REGISTER.match(t.value):
        t.type = 'REGISTER'
    elif re_SEGMENT.match(t.value):
        t.type = 'SEGMENT'
    else:
        raise ParserException(f'Unknown:<{t.value}>')
    return t

def t_newline(t):
    if False:
        print('Hello World!')
    '\\n+'
    t.lexer.lineno += len(t.value)
t_ignore = ' \t'

def t_error(t):
    if False:
        return 10
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
lexer = lex.lex()
precedence = (('left', 'PLUS', 'MINUS'), ('left', 'DIVIDE'), ('left', 'TIMES'), ('left', 'AND', 'OR'), ('right', 'NEG'))

def default_read_memory(address, size):
    if False:
        i = 10
        return i + 15
    return f'READM({address:08x},{size})'

def default_read_register(reg):
    if False:
        return 10
    return f'REG({reg})'

def default_get_descriptor(selector):
    if False:
        return 10
    return (0, 4294963200, 'rwx')
default_sizes_32 = {'QWORD': 8, 'DWORD': 4, 'WORD': 2, 'BYTE': 1}
default_sizes_64 = {'QWORD': 8, 'DWORD': 4, 'WORD': 2, 'BYTE': 1}
functions = {'read_memory': default_read_memory, 'read_register': default_read_register, 'get_descriptor': default_get_descriptor}
sizes = copy.copy(default_sizes_32)

def p_expression_div(p):
    if False:
        return 10
    'expression : expression DIVIDE expression'
    p[0] = p[1] // p[3]

def p_expression_mul(p):
    if False:
        i = 10
        return i + 15
    'expression : expression TIMES expression'
    p[0] = p[1] * p[3]

def p_expression_plus(p):
    if False:
        i = 10
        return i + 15
    'expression : expression PLUS expression'
    p[0] = p[1] + p[3]

def p_expression_minus(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression MINUS expression'
    p[0] = p[1] - p[3]

def p_expression_and(p):
    if False:
        i = 10
        return i + 15
    'expression : expression AND expression'
    p[0] = p[1] & p[3]

def p_expression_or(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression OR expression'
    p[0] = p[1] | p[3]

def p_expression_neg(p):
    if False:
        i = 10
        return i + 15
    'expression : NEG expression'
    p[0] = ~p[1]

def p_expression_lshift(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression LSHIFT expression'
    p[0] = p[1] << p[3]

def p_expression_rshift(p):
    if False:
        while True:
            i = 10
    'expression : expression RSHIFT expression'
    p[0] = p[1] >> p[3]

def p_expression_deref(p):
    if False:
        i = 10
        return i + 15
    'expression : TYPE PTR LBRAKET expression RBRAKET'
    size = sizes[p[1]]
    address = p[4]
    char_list = functions['read_memory'](address, size)
    value = Operators.CONCAT(8 * len(char_list), *reversed(map(Operators.ORD, char_list)))
    p[0] = value

def p_expression_derefseg(p):
    if False:
        return 10
    'expression : TYPE PTR SEGMENT COLOM LBRAKET expression RBRAKET'
    size = sizes[p[1]]
    address = p[6]
    seg = functions['read_register'](p[3])
    (base, limit, _) = functions['get_descriptor'](seg)
    address = base + address
    char_list = functions['read_memory'](address, size)
    value = Operators.CONCAT(8 * len(char_list), *reversed(map(Operators.ORD, char_list)))
    p[0] = value

def p_expression_term(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : term'
    p[0] = p[1]

def p_factor_expr(p):
    if False:
        print('Hello World!')
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]

def p_term_num(p):
    if False:
        print('Hello World!')
    'term : NUMBER'
    p[0] = p[1]

def p_term_reg(p):
    if False:
        i = 10
        return i + 15
    'term : REGISTER'
    p[0] = functions['read_register'](p[1])

def p_expression_eq(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression EQ expression'
    p[0] = p[1] == p[3]

def p_expression_land(p):
    if False:
        return 10
    'expression : expression LAND expression'
    p[0] = p[1] and p[3]

def p_expression_lor(p):
    if False:
        i = 10
        return i + 15
    'expression : expression LOR expression'
    p[0] = p[1] or p[3]

def p_expression_lnot(p):
    if False:
        i = 10
        return i + 15
    'expression : LNOT expression'
    p[0] = not p[1]

def p_expression_lt(p):
    if False:
        for i in range(10):
            print('nop')
    'expression : expression LT expression'
    p[0] = Operators.ULT(p[1], p[3])

def p_expression_le(p):
    if False:
        while True:
            i = 10
    'expression : expression LE expression'
    p[0] = Operators.ULE(p[1], p[3])

def p_expression_gt(p):
    if False:
        return 10
    'expression : expression GT expression'
    p[0] = Operators.UGT(p[1], p[3])

def p_expression_ge(p):
    if False:
        print('Hello World!')
    'expression : expression GE expression'
    p[0] = Operators.UGE(p[1], p[3])

def p_error(p):
    if False:
        for i in range(10):
            print('nop')
    print('Syntax error in input:', p)
parser = yacc.yacc(debug=0, write_tables=0)

def parse(expression, read_memory=None, read_register=None, get_descriptor=None, word_size=32):
    if False:
        while True:
            i = 10
    global functions, sizes
    if read_memory is not None:
        functions['read_memory'] = read_memory
    else:
        functions['read_memory'] = default_read_memory
    if read_register is not None:
        functions['read_register'] = read_register
    else:
        functions['read_register'] = default_read_register
    if get_descriptor is not None:
        functions['get_descriptor'] = get_descriptor
    else:
        functions['get_descriptor'] = default_get_descriptor
    if word_size == 32:
        sizes = copy.copy(default_sizes_32)
    elif word_size == 64:
        sizes = copy.copy(default_sizes_64)
    else:
        raise ParserException('Got unsupported word size')
    result = parser.parse(expression, tracking=True)
    del functions['read_memory']
    del functions['read_register']
    del functions['get_descriptor']
    return result
if __name__ == '__main__':
    while True:
        try:
            s = input('calc > ')
        except EOFError:
            break
        if not s:
            continue
        result = parse(s)
        print(result)