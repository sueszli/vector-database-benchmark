from __future__ import print_function
from miasm.expression.expression import *
from pdb import pm
print('\nExpression simplification demo.\n(and regression test)\n')
a = ExprId('a', 32)
b = ExprId('b', 32)
c = ExprId('c', 32)
d = ExprId('d', 32)
e = ExprId('e', 32)
m = ExprMem(a, 32)
s = a[:8]
i1 = ExprInt(1, 32)
i2 = ExprInt(2, 32)
cc = ExprCond(a, b, c)
o = ExprCompose(a[8:16], a[:8])
o2 = ExprCompose(a[8:16], a[:8])
l = [a[:8], b[:8], c[:8], m[:8], s, i1[:8], i2[:8], o[:8]]
l2 = l[::-1]
x = ExprMem(a + b + ExprInt(66, 32), 32)

def replace_expr(e):
    if False:
        print('Hello World!')
    dct = {c + ExprInt(66, 32): d, a + b: c}
    if e in dct:
        return dct[e]
    return e
print(x)
y = x.visit(replace_expr)
print(y)
print(x.copy())
print(y.copy())
print(y == y.copy())
print(repr(y), repr(y.copy()))
z = ExprCompose(a[5:5 + 8], b[:16], x[:8])
print(z)
print(z.copy())
print(z[:31].copy().visit(replace_expr))
print('replace')
print(x.replace_expr({c + ExprInt(66, 32): d, a + b: c}))
print(z.replace_expr({c + ExprInt(66, 32): d, a + b: c}))
u = z.copy()
print(u)