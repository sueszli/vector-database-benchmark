import ylex
tokens = ylex.tokens
from ply import *
tokenlist = []
preclist = []
emit_code = 1

def p_yacc(p):
    if False:
        for i in range(10):
            print('nop')
    'yacc : defsection rulesection'

def p_defsection(p):
    if False:
        print('Hello World!')
    'defsection : definitions SECTION\n                  | SECTION'
    p.lexer.lastsection = 1
    print('tokens = ', repr(tokenlist))
    print()
    print('precedence = ', repr(preclist))
    print()
    print('# -------------- RULES ----------------')
    print()

def p_rulesection(p):
    if False:
        i = 10
        return i + 15
    'rulesection : rules SECTION'
    print('# -------------- RULES END ----------------')
    print_code(p[2], 0)

def p_definitions(p):
    if False:
        i = 10
        return i + 15
    'definitions : definitions definition\n                   | definition'

def p_definition_literal(p):
    if False:
        return 10
    'definition : LITERAL'
    print_code(p[1], 0)

def p_definition_start(p):
    if False:
        print('Hello World!')
    'definition : START ID'
    print("start = '%s'" % p[2])

def p_definition_token(p):
    if False:
        return 10
    'definition : toktype opttype idlist optsemi '
    for i in p[3]:
        if i[0] not in '\'"':
            tokenlist.append(i)
    if p[1] == '%left':
        preclist.append(('left',) + tuple(p[3]))
    elif p[1] == '%right':
        preclist.append(('right',) + tuple(p[3]))
    elif p[1] == '%nonassoc':
        preclist.append(('nonassoc',) + tuple(p[3]))

def p_toktype(p):
    if False:
        for i in range(10):
            print('nop')
    'toktype : TOKEN\n               | LEFT\n               | RIGHT\n               | NONASSOC'
    p[0] = p[1]

def p_opttype(p):
    if False:
        while True:
            i = 10
    "opttype : '<' ID '>'\n               | empty"

def p_idlist(p):
    if False:
        return 10
    'idlist  : idlist optcomma tokenid\n               | tokenid'
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]
        p[1].append(p[3])

def p_tokenid(p):
    if False:
        return 10
    'tokenid : ID \n               | ID NUMBER\n               | QLITERAL\n               | QLITERAL NUMBER'
    p[0] = p[1]

def p_optsemi(p):
    if False:
        i = 10
        return i + 15
    "optsemi : ';'\n               | empty"

def p_optcomma(p):
    if False:
        i = 10
        return i + 15
    "optcomma : ','\n                | empty"

def p_definition_type(p):
    if False:
        return 10
    "definition : TYPE '<' ID '>' namelist optsemi"

def p_namelist(p):
    if False:
        return 10
    'namelist : namelist optcomma ID\n                | ID'

def p_definition_union(p):
    if False:
        i = 10
        return i + 15
    'definition : UNION CODE optsemi'

def p_rules(p):
    if False:
        for i in range(10):
            print('nop')
    'rules   : rules rule\n               | rule'
    if len(p) == 2:
        rule = p[1]
    else:
        rule = p[2]
    embedded = []
    embed_count = 0
    rulename = rule[0]
    rulecount = 1
    for r in rule[1]:
        print('def p_%s_%d(p):' % (rulename, rulecount))
        prod = []
        prodcode = ''
        for i in range(len(r)):
            item = r[i]
            if item[0] == '{':
                if i == len(r) - 1:
                    prodcode = item
                    break
                else:
                    embed_name = '_embed%d_%s' % (embed_count, rulename)
                    prod.append(embed_name)
                    embedded.append((embed_name, item))
                    embed_count += 1
            else:
                prod.append(item)
        print("    '''%s : %s'''" % (rulename, ' '.join(prod)))
        print_code(prodcode, 4)
        print()
        rulecount += 1
    for (e, code) in embedded:
        print('def p_%s(p):' % e)
        print("    '''%s : '''" % e)
        print_code(code, 4)
        print()

def p_rule(p):
    if False:
        i = 10
        return i + 15
    "rule : ID ':' rulelist ';' "
    p[0] = (p[1], [p[3]])

def p_rule2(p):
    if False:
        return 10
    "rule : ID ':' rulelist morerules ';' "
    p[4].insert(0, p[3])
    p[0] = (p[1], p[4])

def p_rule_empty(p):
    if False:
        return 10
    "rule : ID ':' ';' "
    p[0] = (p[1], [[]])

def p_rule_empty2(p):
    if False:
        print('Hello World!')
    "rule : ID ':' morerules ';' "
    p[3].insert(0, [])
    p[0] = (p[1], p[3])

def p_morerules(p):
    if False:
        print('Hello World!')
    "morerules : morerules '|' rulelist\n                 | '|' rulelist\n                 | '|'  "
    if len(p) == 2:
        p[0] = [[]]
    elif len(p) == 3:
        p[0] = [p[2]]
    else:
        p[0] = p[1]
        p[0].append(p[3])

def p_rulelist(p):
    if False:
        i = 10
        return i + 15
    'rulelist : rulelist ruleitem\n                | ruleitem'
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]
        p[1].append(p[2])

def p_ruleitem(p):
    if False:
        i = 10
        return i + 15
    'ruleitem : ID\n                | QLITERAL\n                | CODE\n                | PREC'
    p[0] = p[1]

def p_empty(p):
    if False:
        return 10
    'empty : '

def p_error(p):
    if False:
        while True:
            i = 10
    pass
yacc.yacc(debug=0)

def print_code(code, indent):
    if False:
        return 10
    if not emit_code:
        return
    codelines = code.splitlines()
    for c in codelines:
        print('%s# %s' % (' ' * indent, c))