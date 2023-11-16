import numpy as np
from cvxpy import SOC, Variable, hstack
from cvxpy.constraints.exponential import ExpCone
from cvxpy.reductions.solvers.conic_solvers.scs_conif import scs_psdvec_to_psdmat

def suppfunc_canon(expr, args):
    if False:
        for i in range(10):
            print('nop')
    y = args[0].flatten()
    parent = expr._parent
    (A, b, K_sels) = parent.conic_repr_of_set()
    eta = Variable(shape=(b.size,))
    expr._eta = eta
    n = A.shape[1]
    n0 = y.size
    if n > n0:
        y_lift = hstack([y, np.zeros(shape=(n - n0,))])
    else:
        y_lift = y
    local_cons = [A.T @ eta + y_lift == 0]
    nonnegsel = K_sels['nonneg']
    if nonnegsel.size > 0:
        temp_expr = eta[nonnegsel]
        local_cons.append(temp_expr >= 0)
    socsels = K_sels['soc']
    for socsel in socsels:
        tempsca = eta[socsel[0]]
        tempvec = eta[socsel[1:]]
        soccon = SOC(tempsca, tempvec)
        local_cons.append(soccon)
    psdsels = K_sels['psd']
    for psdsel in psdsels:
        curmat = scs_psdvec_to_psdmat(eta, psdsel)
        local_cons.append(curmat >> 0)
    expsel = K_sels['exp']
    if expsel.size > 0:
        matexpsel = np.reshape(expsel, (-1, 3))
        curr_u = eta[matexpsel[:, 0]]
        curr_v = eta[matexpsel[:, 1]]
        curr_w = eta[matexpsel[:, 2]]
        ec = ExpCone(-curr_v, -curr_u, np.exp(1) * curr_w)
        local_cons.append(ec)
    epigraph = b @ eta
    return (epigraph, local_cons)