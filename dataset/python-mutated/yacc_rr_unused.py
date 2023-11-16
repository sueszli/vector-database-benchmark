import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
import ply.yacc as yacc
tokens = ('A', 'B', 'C')

def p_grammar(p):
    if False:
        for i in range(10):
            print('nop')
    '\n   rule1 : rule2 B\n         | rule2 C\n\n   rule2 : rule3 B\n         | rule4\n         | rule5\n\n   rule3 : A\n\n   rule4 : A\n\n   rule5 : A\n   '
yacc.yacc()