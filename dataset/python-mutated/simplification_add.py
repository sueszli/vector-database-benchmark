from __future__ import print_function
import miasm.expression.expression as m2_expr
from miasm.expression.simplifications import ExpressionSimplifier
simp = ExpressionSimplifier()
print('\nExpression simplification demo: Adding a simplification:\na + a + a == a * 3\n\nMore detailed examples can be found in miasm/expression/simplification*.\n')

def simp_add_mul(expr_simp, expr):
    if False:
        i = 10
        return i + 15
    'Naive Simplification: a + a + a == a * 3'
    if expr.op == '+' and len(expr.args) == 3 and (expr.args.count(expr.args[0]) == len(expr.args)):
        return m2_expr.ExprOp('*', expr.args[0], m2_expr.ExprInt(3, expr.args[0].size))
    else:
        return expr
a = m2_expr.ExprId('a', 32)
base_expr = a + a + a
print('Without adding the simplification:')
print('\t%s = %s' % (base_expr, simp(base_expr)))
simp.enable_passes({m2_expr.ExprOp: [simp_add_mul]})
print('After adding the simplification:')
print('\t%s = %s' % (base_expr, simp(base_expr)))
assert simp(base_expr) == m2_expr.ExprOp('*', a, m2_expr.ExprInt(3, a.size))