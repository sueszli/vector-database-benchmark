from cvxpy.constraints.nonpos import NonPos

def nonpos_constr_canon(expr, args):
    if False:
        print('Hello World!')
    assert len(args) == 2
    return (NonPos(args[0] - args[1], constr_id=expr.id), [])