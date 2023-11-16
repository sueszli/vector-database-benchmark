from ply import *
import basiclex
tokens = basiclex.tokens
precedence = (('left', 'PLUS', 'MINUS'), ('left', 'TIMES', 'DIVIDE'), ('left', 'POWER'), ('right', 'UMINUS'))

def p_program(p):
    if False:
        i = 10
        return i + 15
    'program : program statement\n               | statement'
    if len(p) == 2 and p[1]:
        p[0] = {}
        (line, stat) = p[1]
        p[0][line] = stat
    elif len(p) == 3:
        p[0] = p[1]
        if not p[0]:
            p[0] = {}
        if p[2]:
            (line, stat) = p[2]
            p[0][line] = stat

def p_program_error(p):
    if False:
        while True:
            i = 10
    'program : error'
    p[0] = None
    p.parser.error = 1

def p_statement(p):
    if False:
        print('Hello World!')
    'statement : INTEGER command NEWLINE'
    if isinstance(p[2], str):
        print('%s %s %s' % (p[2], 'AT LINE', p[1]))
        p[0] = None
        p.parser.error = 1
    else:
        lineno = int(p[1])
        p[0] = (lineno, p[2])

def p_statement_interactive(p):
    if False:
        while True:
            i = 10
    'statement : RUN NEWLINE\n                 | LIST NEWLINE\n                 | NEW NEWLINE'
    p[0] = (0, (p[1], 0))

def p_statement_blank(p):
    if False:
        for i in range(10):
            print('nop')
    'statement : INTEGER NEWLINE'
    p[0] = (0, ('BLANK', int(p[1])))

def p_statement_bad(p):
    if False:
        return 10
    'statement : INTEGER error NEWLINE'
    print('MALFORMED STATEMENT AT LINE %s' % p[1])
    p[0] = None
    p.parser.error = 1

def p_statement_newline(p):
    if False:
        return 10
    'statement : NEWLINE'
    p[0] = None

def p_command_let(p):
    if False:
        for i in range(10):
            print('nop')
    'command : LET variable EQUALS expr'
    p[0] = ('LET', p[2], p[4])

def p_command_let_bad(p):
    if False:
        while True:
            i = 10
    'command : LET variable EQUALS error'
    p[0] = 'BAD EXPRESSION IN LET'

def p_command_read(p):
    if False:
        for i in range(10):
            print('nop')
    'command : READ varlist'
    p[0] = ('READ', p[2])

def p_command_read_bad(p):
    if False:
        i = 10
        return i + 15
    'command : READ error'
    p[0] = 'MALFORMED VARIABLE LIST IN READ'

def p_command_data(p):
    if False:
        return 10
    'command : DATA numlist'
    p[0] = ('DATA', p[2])

def p_command_data_bad(p):
    if False:
        i = 10
        return i + 15
    'command : DATA error'
    p[0] = 'MALFORMED NUMBER LIST IN DATA'

def p_command_print(p):
    if False:
        while True:
            i = 10
    'command : PRINT plist optend'
    p[0] = ('PRINT', p[2], p[3])

def p_command_print_bad(p):
    if False:
        for i in range(10):
            print('nop')
    'command : PRINT error'
    p[0] = 'MALFORMED PRINT STATEMENT'

def p_optend(p):
    if False:
        while True:
            i = 10
    'optend : COMMA \n              | SEMI\n              |'
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = None

def p_command_print_empty(p):
    if False:
        i = 10
        return i + 15
    'command : PRINT'
    p[0] = ('PRINT', [], None)

def p_command_goto(p):
    if False:
        while True:
            i = 10
    'command : GOTO INTEGER'
    p[0] = ('GOTO', int(p[2]))

def p_command_goto_bad(p):
    if False:
        i = 10
        return i + 15
    'command : GOTO error'
    p[0] = 'INVALID LINE NUMBER IN GOTO'

def p_command_if(p):
    if False:
        for i in range(10):
            print('nop')
    'command : IF relexpr THEN INTEGER'
    p[0] = ('IF', p[2], int(p[4]))

def p_command_if_bad(p):
    if False:
        print('Hello World!')
    'command : IF error THEN INTEGER'
    p[0] = 'BAD RELATIONAL EXPRESSION'

def p_command_if_bad2(p):
    if False:
        while True:
            i = 10
    'command : IF relexpr THEN error'
    p[0] = 'INVALID LINE NUMBER IN THEN'

def p_command_for(p):
    if False:
        return 10
    'command : FOR ID EQUALS expr TO expr optstep'
    p[0] = ('FOR', p[2], p[4], p[6], p[7])

def p_command_for_bad_initial(p):
    if False:
        i = 10
        return i + 15
    'command : FOR ID EQUALS error TO expr optstep'
    p[0] = 'BAD INITIAL VALUE IN FOR STATEMENT'

def p_command_for_bad_final(p):
    if False:
        while True:
            i = 10
    'command : FOR ID EQUALS expr TO error optstep'
    p[0] = 'BAD FINAL VALUE IN FOR STATEMENT'

def p_command_for_bad_step(p):
    if False:
        for i in range(10):
            print('nop')
    'command : FOR ID EQUALS expr TO expr STEP error'
    p[0] = 'MALFORMED STEP IN FOR STATEMENT'

def p_optstep(p):
    if False:
        for i in range(10):
            print('nop')
    'optstep : STEP expr\n               | empty'
    if len(p) == 3:
        p[0] = p[2]
    else:
        p[0] = None

def p_command_next(p):
    if False:
        for i in range(10):
            print('nop')
    'command : NEXT ID'
    p[0] = ('NEXT', p[2])

def p_command_next_bad(p):
    if False:
        print('Hello World!')
    'command : NEXT error'
    p[0] = 'MALFORMED NEXT'

def p_command_end(p):
    if False:
        for i in range(10):
            print('nop')
    'command : END'
    p[0] = ('END',)

def p_command_rem(p):
    if False:
        for i in range(10):
            print('nop')
    'command : REM'
    p[0] = ('REM', p[1])

def p_command_stop(p):
    if False:
        return 10
    'command : STOP'
    p[0] = ('STOP',)

def p_command_def(p):
    if False:
        print('Hello World!')
    'command : DEF ID LPAREN ID RPAREN EQUALS expr'
    p[0] = ('FUNC', p[2], p[4], p[7])

def p_command_def_bad_rhs(p):
    if False:
        for i in range(10):
            print('nop')
    'command : DEF ID LPAREN ID RPAREN EQUALS error'
    p[0] = 'BAD EXPRESSION IN DEF STATEMENT'

def p_command_def_bad_arg(p):
    if False:
        return 10
    'command : DEF ID LPAREN error RPAREN EQUALS expr'
    p[0] = 'BAD ARGUMENT IN DEF STATEMENT'

def p_command_gosub(p):
    if False:
        while True:
            i = 10
    'command : GOSUB INTEGER'
    p[0] = ('GOSUB', int(p[2]))

def p_command_gosub_bad(p):
    if False:
        i = 10
        return i + 15
    'command : GOSUB error'
    p[0] = 'INVALID LINE NUMBER IN GOSUB'

def p_command_return(p):
    if False:
        return 10
    'command : RETURN'
    p[0] = ('RETURN',)

def p_command_dim(p):
    if False:
        i = 10
        return i + 15
    'command : DIM dimlist'
    p[0] = ('DIM', p[2])

def p_command_dim_bad(p):
    if False:
        print('Hello World!')
    'command : DIM error'
    p[0] = 'MALFORMED VARIABLE LIST IN DIM'

def p_dimlist(p):
    if False:
        return 10
    'dimlist : dimlist COMMA dimitem\n               | dimitem'
    if len(p) == 4:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_dimitem_single(p):
    if False:
        while True:
            i = 10
    'dimitem : ID LPAREN INTEGER RPAREN'
    p[0] = (p[1], eval(p[3]), 0)

def p_dimitem_double(p):
    if False:
        print('Hello World!')
    'dimitem : ID LPAREN INTEGER COMMA INTEGER RPAREN'
    p[0] = (p[1], eval(p[3]), eval(p[5]))

def p_expr_binary(p):
    if False:
        return 10
    'expr : expr PLUS expr\n            | expr MINUS expr\n            | expr TIMES expr\n            | expr DIVIDE expr\n            | expr POWER expr'
    p[0] = ('BINOP', p[2], p[1], p[3])

def p_expr_number(p):
    if False:
        return 10
    'expr : INTEGER\n            | FLOAT'
    p[0] = ('NUM', eval(p[1]))

def p_expr_variable(p):
    if False:
        while True:
            i = 10
    'expr : variable'
    p[0] = ('VAR', p[1])

def p_expr_group(p):
    if False:
        for i in range(10):
            print('nop')
    'expr : LPAREN expr RPAREN'
    p[0] = ('GROUP', p[2])

def p_expr_unary(p):
    if False:
        return 10
    'expr : MINUS expr %prec UMINUS'
    p[0] = ('UNARY', '-', p[2])

def p_relexpr(p):
    if False:
        print('Hello World!')
    'relexpr : expr LT expr\n               | expr LE expr\n               | expr GT expr\n               | expr GE expr\n               | expr EQUALS expr\n               | expr NE expr'
    p[0] = ('RELOP', p[2], p[1], p[3])

def p_variable(p):
    if False:
        while True:
            i = 10
    'variable : ID\n              | ID LPAREN expr RPAREN\n              | ID LPAREN expr COMMA expr RPAREN'
    if len(p) == 2:
        p[0] = (p[1], None, None)
    elif len(p) == 5:
        p[0] = (p[1], p[3], None)
    else:
        p[0] = (p[1], p[3], p[5])

def p_varlist(p):
    if False:
        i = 10
        return i + 15
    'varlist : varlist COMMA variable\n               | variable'
    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_numlist(p):
    if False:
        print('Hello World!')
    'numlist : numlist COMMA number\n               | number'
    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_number(p):
    if False:
        i = 10
        return i + 15
    'number  : INTEGER\n               | FLOAT'
    p[0] = eval(p[1])

def p_number_signed(p):
    if False:
        i = 10
        return i + 15
    'number  : MINUS INTEGER\n               | MINUS FLOAT'
    p[0] = eval('-' + p[2])

def p_plist(p):
    if False:
        return 10
    'plist   : plist COMMA pitem\n               | pitem'
    if len(p) > 3:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_item_string(p):
    if False:
        return 10
    'pitem : STRING'
    p[0] = (p[1][1:-1], None)

def p_item_string_expr(p):
    if False:
        for i in range(10):
            print('nop')
    'pitem : STRING expr'
    p[0] = (p[1][1:-1], p[2])

def p_item_expr(p):
    if False:
        i = 10
        return i + 15
    'pitem : expr'
    p[0] = ('', p[1])

def p_empty(p):
    if False:
        while True:
            i = 10
    'empty : '

def p_error(p):
    if False:
        i = 10
        return i + 15
    if not p:
        print('SYNTAX ERROR AT EOF')
bparser = yacc.yacc()

def parse(data, debug=0):
    if False:
        for i in range(10):
            print('nop')
    bparser.error = 0
    p = bparser.parse(data, debug=debug)
    if bparser.error:
        return None
    return p