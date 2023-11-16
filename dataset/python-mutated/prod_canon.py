from cvxpy.atoms.affine.sum import sum as sum_func

def prod_canon(expr, args):
    if False:
        return 10
    return (sum_func(args[0], axis=expr.axis, keepdims=expr.keepdims), [])