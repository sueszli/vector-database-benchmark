from cvxpy.constraints.zero import Zero

def zero_constr_canon(expr, args):
    if False:
        while True:
            i = 10
    assert len(args) == 2
    return (Zero(args[0] - args[1], constr_id=expr.id), [])