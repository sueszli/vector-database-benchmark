from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.mulexpression_canon import mulexpression_canon

def pf_eigenvalue_canon(expr, args):
    if False:
        for i in range(10):
            print('nop')
    X = args[0]
    lambd = Variable()
    v = Variable(X.shape[0])
    lhs = matmul(X, v)
    rhs = lambd * v
    (lhs, _) = mulexpression_canon(lhs, lhs.args)
    (rhs, _) = mul_canon(rhs, rhs.args)
    return (lambd, [lhs <= rhs])