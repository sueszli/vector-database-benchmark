from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.canonicalizers.mulexpression_canon import mulexpression_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.one_minus_pos_canon import one_minus_pos_canon

def eye_minus_inv_canon(expr, args):
    if False:
        for i in range(10):
            print('nop')
    X = args[0]
    U = Variable(X.shape)
    YX = matmul(U, X)
    (YX_canon, _) = mulexpression_canon(YX, YX.args)
    one_minus = one_minus_pos(YX_canon - U)
    (canon, _) = one_minus_pos_canon(one_minus, one_minus.args)
    lhs_canon = diag(U + canon)
    return (U, [lhs_canon >= 0])