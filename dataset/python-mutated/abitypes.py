import warnings
from ..exceptions import EthereumError
import ply.yacc as yacc
import ply.lex as lex
import re
tokens = ('COMMA', 'LPAREN', 'RPAREN', 'LBRAKET', 'RBRAKET', 'NUMBER', 'UINTN', 'INTN', 'UINT', 'INT', 'BOOL', 'FIXEDMN', 'UFIXEDMN', 'ADDRESS', 'FIXED', 'UFIXED', 'FUNCTION', 'BYTESM', 'BYTES', 'STRING')
t_LPAREN = '\\('
t_RPAREN = '\\)'
t_LBRAKET = '\\['
t_RBRAKET = '\\]'
t_COMMA = '\\,'

def t_NUMBER(t):
    if False:
        for i in range(10):
            print('nop')
    '\\d+'
    t.value = int(t.value)
    return t

def t_UINTN(t):
    if False:
        print('Hello World!')
    'uint(?P<size>256|248|240|232|224|216|208|200|192|184|176|168|160|152|144|136|128|120|112|104|96|88|80|72|64|56|48|40|32|24|16|8)'
    size = int(t.lexer.lexmatch.group('size'))
    t.value = ('uint', size)
    return t

def t_ADDRESS(t):
    if False:
        i = 10
        return i + 15
    'address'
    t.value = ('uint', 160)
    return t

def t_BOOL(t):
    if False:
        while True:
            i = 10
    'bool'
    t.value = ('uint', 8)
    return t

def t_UINT(t):
    if False:
        print('Hello World!')
    'uint'
    t.value = ('uint', 256)
    return t

def t_INTN(t):
    if False:
        while True:
            i = 10
    'int(?P<size>256|248|240|232|224|216|208|200|192|184|176|168|160|152|144|136|128|120|112|104|96|88|80|72|64|56|48|40|32|24|16|8)'
    size = int(t.lexer.lexmatch.group('size'))
    t.value = ('int', size)
    return t

def t_INT(t):
    if False:
        return 10
    'int'
    t.value = ('int', 256)
    return t

def t_FIXEDMN(t):
    if False:
        for i in range(10):
            print('nop')
    '^fixed(?P<M>256|248|240|232|224|216|208|200|192|184|176|168|160|152|144|136|128|120|112|104|96|88|80|72|64|56|48|40|32|24|16|8)x(?P<N>80|79|78|77|76|75|74|73|72|71|70|69|68|67|66|65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1)'
    M = int(t.lexer.lexmatch.group('M'))
    N = int(t.lexer.lexmatch.group('N'))
    t.value = ('fixed', M, N)
    return t

def t_FIXED(t):
    if False:
        while True:
            i = 10
    'fixed'
    t.value = ('fixed', 128, 18)
    return t

def t_UFIXEDMN(t):
    if False:
        print('Hello World!')
    'ufixed(?P<M>256|248|240|232|224|216|208|200|192|184|176|168|160|152|144|136|128|120|112|104|96|88|80|72|64|56|48|40|32|24|16|8)x(?P<N>80|79|78|77|76|75|74|73|72|71|70|69|68|67|66|65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1)'
    M = int(t.lexer.lexmatch.group('M'))
    N = int(t.lexer.lexmatch.group('N'))
    t.value = ('ufixed', M, N)
    return t

def t_UFIXED(t):
    if False:
        i = 10
        return i + 15
    'ufixed'
    t.value = ('ufixed', 128, 18)
    return t

def t_BYTESM(t):
    if False:
        while True:
            i = 10
    'bytes(?P<nbytes>32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1)'
    size = int(t.lexer.lexmatch.group('nbytes'))
    t.value = ('bytesM', size)
    return t

def t_BYTES(t):
    if False:
        while True:
            i = 10
    'bytes'
    t.value = ('bytes',)
    return t

def t_STRING(t):
    if False:
        for i in range(10):
            print('nop')
    'string'
    t.value = ('string',)
    return t

def t_FUNCTION(t):
    if False:
        while True:
            i = 10
    'function'
    t.value = ('function',)
    return t

def t_error(t):
    if False:
        while True:
            i = 10
    raise EthereumError("Illegal character '%s'" % t.value[0])
lexer = lex.lex()

def p_basic_type(p):
    if False:
        while True:
            i = 10
    '\n    T : UINTN\n    T : UINT\n    T : INTN\n    T : INT\n    T : ADDRESS\n    T : BOOL\n    T : FIXEDMN\n    T : UFIXEDMN\n    T : FIXED\n    T : UFIXED\n    T : BYTESM\n    T : FUNCTION\n    T : BYTES\n    T : STRING\n\n    '
    p[0] = p[1]

def p_type_list_one(p):
    if False:
        i = 10
        return i + 15
    '\n    TL : T\n    '
    p[0] = (p[1],)

def p_type_list(p):
    if False:
        print('Hello World!')
    '\n    TL : T COMMA TL\n    '
    p[0] = (p[1],) + p[3]

def p_tuple(p):
    if False:
        for i in range(10):
            print('nop')
    '\n    T : LPAREN TL RPAREN\n    '
    p[0] = ('tuple', p[2])

def p_tuple_empty(p):
    if False:
        for i in range(10):
            print('nop')
    '\n    T : LPAREN RPAREN\n    '
    p[0] = ('tuple', ())

def p_dynamic_type(p):
    if False:
        return 10
    '\n    T : T LBRAKET RBRAKET\n    '
    reps = None
    base_type = p[1]
    p[0] = ('array', reps, base_type)

def p_dynamic_fixed_type(p):
    if False:
        for i in range(10):
            print('nop')
    '\n    T : T LBRAKET NUMBER RBRAKET\n    '
    reps = int(p[3])
    base_type = p[1]
    p[0] = ('array', reps, base_type)

def p_error(p):
    if False:
        while True:
            i = 10
    raise EthereumError('Syntax Error at abitypes')
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=ResourceWarning)
    parser = yacc.yacc(debug=False)
parse = parser.parse
if __name__ == '__main__':
    while True:
        try:
            s = input('abitype > ')
        except EOFError:
            break
        print('R:', parser.parse(s, debug=True, tracking=True))