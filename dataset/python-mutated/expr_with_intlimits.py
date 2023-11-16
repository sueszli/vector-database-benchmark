from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.singleton import S
from sympy.core.relational import Eq

class ReorderError(NotImplementedError):
    """
    Exception raised when trying to reorder dependent limits.
    """

    def __init__(self, expr, msg):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('%s could not be reordered: %s.' % (expr, msg))

class ExprWithIntLimits(ExprWithLimits):
    """
    Superclass for Product and Sum.

    See Also
    ========

    sympy.concrete.expr_with_limits.ExprWithLimits
    sympy.concrete.products.Product
    sympy.concrete.summations.Sum
    """
    __slots__ = ()

    def change_index(self, var, trafo, newvar=None):
        if False:
            i = 10
            return i + 15
        '\n        Change index of a Sum or Product.\n\n        Perform a linear transformation `x \\mapsto a x + b` on the index variable\n        `x`. For `a` the only values allowed are `\\pm 1`. A new variable to be used\n        after the change of index can also be specified.\n\n        Explanation\n        ===========\n\n        ``change_index(expr, var, trafo, newvar=None)`` where ``var`` specifies the\n        index variable `x` to transform. The transformation ``trafo`` must be linear\n        and given in terms of ``var``. If the optional argument ``newvar`` is\n        provided then ``var`` gets replaced by ``newvar`` in the final expression.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Product, simplify\n        >>> from sympy.abc import x, y, a, b, c, d, u, v, i, j, k, l\n\n        >>> S = Sum(x, (x, a, b))\n        >>> S.doit()\n        -a**2/2 + a/2 + b**2/2 + b/2\n\n        >>> Sn = S.change_index(x, x + 1, y)\n        >>> Sn\n        Sum(y - 1, (y, a + 1, b + 1))\n        >>> Sn.doit()\n        -a**2/2 + a/2 + b**2/2 + b/2\n\n        >>> Sn = S.change_index(x, -x, y)\n        >>> Sn\n        Sum(-y, (y, -b, -a))\n        >>> Sn.doit()\n        -a**2/2 + a/2 + b**2/2 + b/2\n\n        >>> Sn = S.change_index(x, x+u)\n        >>> Sn\n        Sum(-u + x, (x, a + u, b + u))\n        >>> Sn.doit()\n        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u\n        >>> simplify(Sn.doit())\n        -a**2/2 + a/2 + b**2/2 + b/2\n\n        >>> Sn = S.change_index(x, -x - u, y)\n        >>> Sn\n        Sum(-u - y, (y, -b - u, -a - u))\n        >>> Sn.doit()\n        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u\n        >>> simplify(Sn.doit())\n        -a**2/2 + a/2 + b**2/2 + b/2\n\n        >>> P = Product(i*j**2, (i, a, b), (j, c, d))\n        >>> P\n        Product(i*j**2, (i, a, b), (j, c, d))\n        >>> P2 = P.change_index(i, i+3, k)\n        >>> P2\n        Product(j**2*(k - 3), (k, a + 3, b + 3), (j, c, d))\n        >>> P3 = P2.change_index(j, -j, l)\n        >>> P3\n        Product(l**2*(k - 3), (k, a + 3, b + 3), (l, -d, -c))\n\n        When dealing with symbols only, we can make a\n        general linear transformation:\n\n        >>> Sn = S.change_index(x, u*x+v, y)\n        >>> Sn\n        Sum((-v + y)/u, (y, b*u + v, a*u + v))\n        >>> Sn.doit()\n        -v*(a*u - b*u + 1)/u + (a**2*u**2/2 + a*u*v + a*u/2 - b**2*u**2/2 - b*u*v + b*u/2 + v)/u\n        >>> simplify(Sn.doit())\n        a**2*u/2 + a/2 - b**2*u/2 + b/2\n\n        However, the last result can be inconsistent with usual\n        summation where the index increment is always 1. This is\n        obvious as we get back the original value only for ``u``\n        equal +1 or -1.\n\n        See Also\n        ========\n\n        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,\n        reorder_limit,\n        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder,\n        sympy.concrete.summations.Sum.reverse_order,\n        sympy.concrete.products.Product.reverse_order\n        '
        if newvar is None:
            newvar = var
        limits = []
        for limit in self.limits:
            if limit[0] == var:
                p = trafo.as_poly(var)
                if p.degree() != 1:
                    raise ValueError('Index transformation is not linear')
                alpha = p.coeff_monomial(var)
                beta = p.coeff_monomial(S.One)
                if alpha.is_number:
                    if alpha == S.One:
                        limits.append((newvar, alpha * limit[1] + beta, alpha * limit[2] + beta))
                    elif alpha == S.NegativeOne:
                        limits.append((newvar, alpha * limit[2] + beta, alpha * limit[1] + beta))
                    else:
                        raise ValueError('Linear transformation results in non-linear summation stepsize')
                else:
                    limits.append((newvar, alpha * limit[2] + beta, alpha * limit[1] + beta))
            else:
                limits.append(limit)
        function = self.function.subs(var, (var - beta) / alpha)
        function = function.subs(var, newvar)
        return self.func(function, *limits)

    def index(expr, x):
        if False:
            return 10
        '\n        Return the index of a dummy variable in the list of limits.\n\n        Explanation\n        ===========\n\n        ``index(expr, x)``  returns the index of the dummy variable ``x`` in the\n        limits of ``expr``. Note that we start counting with 0 at the inner-most\n        limits tuple.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y, a, b, c, d\n        >>> from sympy import Sum, Product\n        >>> Sum(x*y, (x, a, b), (y, c, d)).index(x)\n        0\n        >>> Sum(x*y, (x, a, b), (y, c, d)).index(y)\n        1\n        >>> Product(x*y, (x, a, b), (y, c, d)).index(x)\n        0\n        >>> Product(x*y, (x, a, b), (y, c, d)).index(y)\n        1\n\n        See Also\n        ========\n\n        reorder_limit, reorder, sympy.concrete.summations.Sum.reverse_order,\n        sympy.concrete.products.Product.reverse_order\n        '
        variables = [limit[0] for limit in expr.limits]
        if variables.count(x) != 1:
            raise ValueError(expr, 'Number of instances of variable not equal to one')
        else:
            return variables.index(x)

    def reorder(expr, *arg):
        if False:
            return 10
        '\n        Reorder limits in a expression containing a Sum or a Product.\n\n        Explanation\n        ===========\n\n        ``expr.reorder(*arg)`` reorders the limits in the expression ``expr``\n        according to the list of tuples given by ``arg``. These tuples can\n        contain numerical indices or index variable names or involve both.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Product\n        >>> from sympy.abc import x, y, z, a, b, c, d, e, f\n\n        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((x, y))\n        Sum(x*y, (y, c, d), (x, a, b))\n\n        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder((x, y), (x, z), (y, z))\n        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))\n\n        >>> P = Product(x*y*z, (x, a, b), (y, c, d), (z, e, f))\n        >>> P.reorder((x, y), (x, z), (y, z))\n        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))\n\n        We can also select the index variables by counting them, starting\n        with the inner-most one:\n\n        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder((0, 1))\n        Sum(x**2, (x, c, d), (x, a, b))\n\n        And of course we can mix both schemes:\n\n        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, x))\n        Sum(x*y, (y, c, d), (x, a, b))\n        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, 0))\n        Sum(x*y, (y, c, d), (x, a, b))\n\n        See Also\n        ========\n\n        reorder_limit, index, sympy.concrete.summations.Sum.reverse_order,\n        sympy.concrete.products.Product.reverse_order\n        '
        new_expr = expr
        for r in arg:
            if len(r) != 2:
                raise ValueError(r, 'Invalid number of arguments')
            index1 = r[0]
            index2 = r[1]
            if not isinstance(r[0], int):
                index1 = expr.index(r[0])
            if not isinstance(r[1], int):
                index2 = expr.index(r[1])
            new_expr = new_expr.reorder_limit(index1, index2)
        return new_expr

    def reorder_limit(expr, x, y):
        if False:
            while True:
                i = 10
        '\n        Interchange two limit tuples of a Sum or Product expression.\n\n        Explanation\n        ===========\n\n        ``expr.reorder_limit(x, y)`` interchanges two limit tuples. The\n        arguments ``x`` and ``y`` are integers corresponding to the index\n        variables of the two limits which are to be interchanged. The\n        expression ``expr`` has to be either a Sum or a Product.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y, z, a, b, c, d, e, f\n        >>> from sympy import Sum, Product\n\n        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)\n        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))\n        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder_limit(1, 0)\n        Sum(x**2, (x, c, d), (x, a, b))\n\n        >>> Product(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)\n        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))\n\n        See Also\n        ========\n\n        index, reorder, sympy.concrete.summations.Sum.reverse_order,\n        sympy.concrete.products.Product.reverse_order\n        '
        var = {limit[0] for limit in expr.limits}
        limit_x = expr.limits[x]
        limit_y = expr.limits[y]
        if len(set(limit_x[1].free_symbols).intersection(var)) == 0 and len(set(limit_x[2].free_symbols).intersection(var)) == 0 and (len(set(limit_y[1].free_symbols).intersection(var)) == 0) and (len(set(limit_y[2].free_symbols).intersection(var)) == 0):
            limits = []
            for (i, limit) in enumerate(expr.limits):
                if i == x:
                    limits.append(limit_y)
                elif i == y:
                    limits.append(limit_x)
                else:
                    limits.append(limit)
            return type(expr)(expr.function, *limits)
        else:
            raise ReorderError(expr, 'could not interchange the two limits specified')

    @property
    def has_empty_sequence(self):
        if False:
            print('Hello World!')
        "\n        Returns True if the Sum or Product is computed for an empty sequence.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Product, Symbol\n        >>> m = Symbol('m')\n        >>> Sum(m, (m, 1, 0)).has_empty_sequence\n        True\n\n        >>> Sum(m, (m, 1, 1)).has_empty_sequence\n        False\n\n        >>> M = Symbol('M', integer=True, positive=True)\n        >>> Product(m, (m, 1, M)).has_empty_sequence\n        False\n\n        >>> Product(m, (m, 2, M)).has_empty_sequence\n\n        >>> Product(m, (m, M + 1, M)).has_empty_sequence\n        True\n\n        >>> N = Symbol('N', integer=True, positive=True)\n        >>> Sum(m, (m, N, M)).has_empty_sequence\n\n        >>> N = Symbol('N', integer=True, negative=True)\n        >>> Sum(m, (m, N, M)).has_empty_sequence\n        False\n\n        See Also\n        ========\n\n        has_reversed_limits\n        has_finite_limits\n\n        "
        ret_None = False
        for lim in self.limits:
            dif = lim[1] - lim[2]
            eq = Eq(dif, 1)
            if eq == True:
                return True
            elif eq == False:
                continue
            else:
                ret_None = True
        if ret_None:
            return None
        return False