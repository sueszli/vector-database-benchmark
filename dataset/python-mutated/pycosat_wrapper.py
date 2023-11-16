from sympy.assumptions.cnf import EncodedCNF

def pycosat_satisfiable(expr, all_models=False):
    if False:
        print('Hello World!')
    import pycosat
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False
    if not all_models:
        r = pycosat.solve(expr.data)
        result = r != 'UNSAT'
        if not result:
            return result
        return {expr.symbols[abs(lit) - 1]: lit > 0 for lit in r}
    else:
        r = pycosat.itersolve(expr.data)
        result = r != 'UNSAT'
        if not result:
            return result

        def _gen(results):
            if False:
                while True:
                    i = 10
            satisfiable = False
            try:
                while True:
                    sol = next(results)
                    yield {expr.symbols[abs(lit) - 1]: lit > 0 for lit in sol}
                    satisfiable = True
            except StopIteration:
                if not satisfiable:
                    yield False
        return _gen(r)