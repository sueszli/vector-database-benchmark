"""
This File contains helper functions for nth_linear_constant_coeff_undetermined_coefficients,
nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients,
nth_linear_constant_coeff_variation_of_parameters,
and nth_linear_euler_eq_nonhomogeneous_variation_of_parameters.

All the functions in this file are used by more than one solvers so, instead of creating
instances in other classes for using them it is better to keep it here as separate helpers.

"""
from collections import defaultdict
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, atan2, conjugate
from sympy.integrals import Integral
from sympy.polys import Poly, RootOf, rootof, roots
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants

def _test_term(coeff, func, order):
    if False:
        while True:
            i = 10
    '\n    Linear Euler ODEs have the form  K*x**order*diff(y(x), x, order) = F(x),\n    where K is independent of x and y(x), order>= 0.\n    So we need to check that for each term, coeff == K*x**order from\n    some K.  We have a few cases, since coeff may have several\n    different types.\n    '
    x = func.args[0]
    f = func.func
    if order < 0:
        raise ValueError('order should be greater than 0')
    if coeff == 0:
        return True
    if order == 0:
        if x in coeff.free_symbols:
            return False
        return True
    if coeff.is_Mul:
        if coeff.has(f(x)):
            return False
        return x ** order in coeff.args
    elif coeff.is_Pow:
        return coeff.as_base_exp() == (x, order)
    elif order == 1:
        return x == coeff
    return False

def _get_euler_characteristic_eq_sols(eq, func, match_obj):
    if False:
        while True:
            i = 10
    '\n    Returns the solution of homogeneous part of the linear euler ODE and\n    the list of roots of characteristic equation.\n\n    The parameter ``match_obj`` is a dict of order:coeff terms, where order is the order\n    of the derivative on each term, and coeff is the coefficient of that derivative.\n\n    '
    x = func.args[0]
    f = func.func
    (chareq, symbol) = (S.Zero, Dummy('x'))
    for i in match_obj:
        if i >= 0:
            chareq += (match_obj[i] * diff(x ** symbol, x, i) * x ** (-symbol)).expand()
    chareq = Poly(chareq, symbol)
    chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    collectterms = []
    constants = list(get_numbered_constants(eq, num=chareq.degree() * 2))
    constants.reverse()
    charroots = defaultdict(int)
    for root in chareqroots:
        charroots[root] += 1
    gsol = S.Zero
    ln = log
    for (root, multiplicity) in charroots.items():
        for i in range(multiplicity):
            if isinstance(root, RootOf):
                gsol += x ** root * constants.pop()
                if multiplicity != 1:
                    raise ValueError('Value should be 1')
                collectterms = [(0, root, 0)] + collectterms
            elif root.is_real:
                gsol += ln(x) ** i * x ** root * constants.pop()
                collectterms = [(i, root, 0)] + collectterms
            else:
                reroot = re(root)
                imroot = im(root)
                gsol += ln(x) ** i * x ** reroot * (constants.pop() * sin(abs(imroot) * ln(x)) + constants.pop() * cos(imroot * ln(x)))
                collectterms = [(i, reroot, imroot)] + collectterms
    gsol = Eq(f(x), gsol)
    gensols = []
    for (i, reroot, imroot) in collectterms:
        if imroot == 0:
            gensols.append(ln(x) ** i * x ** reroot)
        else:
            sin_form = ln(x) ** i * x ** reroot * sin(abs(imroot) * ln(x))
            if sin_form in gensols:
                cos_form = ln(x) ** i * x ** reroot * cos(imroot * ln(x))
                gensols.append(cos_form)
            else:
                gensols.append(sin_form)
    return (gsol, gensols)

def _solve_variation_of_parameters(eq, func, roots, homogen_sol, order, match_obj, simplify_flag=True):
    if False:
        while True:
            i = 10
    '\n    Helper function for the method of variation of parameters and nonhomogeneous euler eq.\n\n    See the\n    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffVariationOfParameters`\n    docstring for more information on this method.\n\n    The parameter are ``match_obj`` should be a dictionary that has the following\n    keys:\n\n    ``list``\n    A list of solutions to the homogeneous equation.\n\n    ``sol``\n    The general solution.\n\n    '
    f = func.func
    x = func.args[0]
    r = match_obj
    psol = 0
    wr = wronskian(roots, x)
    if simplify_flag:
        wr = simplify(wr)
        wr = trigsimp(wr, deep=True, recursive=True)
    if not wr:
        raise NotImplementedError('Cannot find ' + str(order) + ' solutions to the homogeneous equation necessary to apply ' + 'variation of parameters to ' + str(eq) + ' (Wronskian == 0)')
    if len(roots) != order:
        raise NotImplementedError('Cannot find ' + str(order) + ' solutions to the homogeneous equation necessary to apply ' + 'variation of parameters to ' + str(eq) + ' (number of terms != order)')
    negoneterm = S.NegativeOne ** order
    for i in roots:
        psol += negoneterm * Integral(wronskian([sol for sol in roots if sol != i], x) * r[-1] / wr, x) * i / r[order]
        negoneterm *= -1
    if simplify_flag:
        psol = simplify(psol)
        psol = trigsimp(psol, deep=True)
    return Eq(f(x), homogen_sol.rhs + psol)

def _get_const_characteristic_eq_sols(r, func, order):
    if False:
        return 10
    '\n    Returns the roots of characteristic equation of constant coefficient\n    linear ODE and list of collectterms which is later on used by simplification\n    to use collect on solution.\n\n    The parameter `r` is a dict of order:coeff terms, where order is the order of the\n    derivative on each term, and coeff is the coefficient of that derivative.\n\n    '
    x = func.args[0]
    (chareq, symbol) = (S.Zero, Dummy('x'))
    for i in r.keys():
        if isinstance(i, str) or i < 0:
            pass
        else:
            chareq += r[i] * symbol ** i
    chareq = Poly(chareq, symbol)
    chareqroots = roots(chareq, multiple=True)
    if len(chareqroots) != order:
        chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    chareq_is_complex = not all((i.is_real for i in chareq.all_coeffs()))
    charroots = defaultdict(int)
    for root in chareqroots:
        charroots[root] += 1
    collectterms = []
    gensols = []
    conjugate_roots = []
    for root in chareqroots:
        if root not in charroots:
            continue
        multiplicity = charroots.pop(root)
        for i in range(multiplicity):
            if chareq_is_complex:
                gensols.append(x ** i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
                continue
            reroot = re(root)
            imroot = im(root)
            if imroot.has(atan2) and reroot.has(atan2):
                gensols.append(x ** i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
            else:
                if root in conjugate_roots:
                    collectterms = [(i, reroot, imroot)] + collectterms
                    continue
                if imroot == 0:
                    gensols.append(x ** i * exp(reroot * x))
                    collectterms = [(i, reroot, 0)] + collectterms
                    continue
                conjugate_roots.append(conjugate(root))
                gensols.append(x ** i * exp(reroot * x) * sin(abs(imroot) * x))
                gensols.append(x ** i * exp(reroot * x) * cos(imroot * x))
                collectterms = [(i, reroot, imroot)] + collectterms
    return (gensols, collectterms)

def _get_simplified_sol(sol, func, collectterms):
    if False:
        i = 10
        return i + 15
    '\n    Helper function which collects the solution on\n    collectterms. Ideally this should be handled by odesimp.It is used\n    only when the simplify is set to True in dsolve.\n\n    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is\n    the multiplicity of the root, reroot is real part and imroot being the imaginary part.\n\n    '
    f = func.func
    x = func.args[0]
    collectterms.sort(key=default_sort_key)
    collectterms.reverse()
    assert len(sol) == 1 and sol[0].lhs == f(x)
    sol = sol[0].rhs
    sol = expand_mul(sol)
    for (i, reroot, imroot) in collectterms:
        sol = collect(sol, x ** i * exp(reroot * x) * sin(abs(imroot) * x))
        sol = collect(sol, x ** i * exp(reroot * x) * cos(imroot * x))
    for (i, reroot, imroot) in collectterms:
        sol = collect(sol, x ** i * exp(reroot * x))
    sol = powsimp(sol)
    return Eq(f(x), sol)

def _undetermined_coefficients_match(expr, x, func=None, eq_homogeneous=S.Zero):
    if False:
        print('Hello World!')
    "\n    Returns a trial function match if undetermined coefficients can be applied\n    to ``expr``, and ``None`` otherwise.\n\n    A trial expression can be found for an expression for use with the method\n    of undetermined coefficients if the expression is an\n    additive/multiplicative combination of constants, polynomials in `x` (the\n    independent variable of expr), `\\sin(a x + b)`, `\\cos(a x + b)`, and\n    `e^{a x}` terms (in other words, it has a finite number of linearly\n    independent derivatives).\n\n    Note that you may still need to multiply each term returned here by\n    sufficient `x` to make it linearly independent with the solutions to the\n    homogeneous equation.\n\n    This is intended for internal use by ``undetermined_coefficients`` hints.\n\n    SymPy currently has no way to convert `\\sin^n(x) \\cos^m(y)` into a sum of\n    only `\\sin(a x)` and `\\cos(b x)` terms, so these are not implemented.  So,\n    for example, you will need to manually convert `\\sin^2(x)` into `[1 +\n    \\cos(2 x)]/2` to properly apply the method of undetermined coefficients on\n    it.\n\n    Examples\n    ========\n\n    >>> from sympy import log, exp\n    >>> from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match\n    >>> from sympy.abc import x\n    >>> _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x)\n    {'test': True, 'trialset': {x*exp(x), exp(-x), exp(x)}}\n    >>> _undetermined_coefficients_match(log(x), x)\n    {'test': False}\n\n    "
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    expr = powsimp(expr, combine='exp')
    retdict = {}

    def _test_term(expr, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if ``expr`` fits the proper form for undetermined coefficients.\n        '
        if not expr.has(x):
            return True
        elif expr.is_Add:
            return all((_test_term(i, x) for i in expr.args))
        elif expr.is_Mul:
            if expr.has(sin, cos):
                foundtrig = False
                for i in expr.args:
                    if i.has(sin, cos):
                        if foundtrig:
                            return False
                        else:
                            foundtrig = True
            return all((_test_term(i, x) for i in expr.args))
        elif expr.is_Function:
            if expr.func in (sin, cos, exp, sinh, cosh):
                if expr.args[0].match(a * x + b):
                    return True
                else:
                    return False
            else:
                return False
        elif expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer and (expr.exp >= 0):
            return True
        elif expr.is_Pow and expr.base.is_number:
            if expr.exp.match(a * x + b):
                return True
            else:
                return False
        elif expr.is_Symbol or expr.is_number:
            return True
        else:
            return False

    def _get_trial_set(expr, x, exprs=set()):
        if False:
            while True:
                i = 10
        '\n        Returns a set of trial terms for undetermined coefficients.\n\n        The idea behind undetermined coefficients is that the terms expression\n        repeat themselves after a finite number of derivatives, except for the\n        coefficients (they are linearly dependent).  So if we collect these,\n        we should have the terms of our trial function.\n        '

        def _remove_coefficient(expr, x):
            if False:
                print('Hello World!')
            '\n            Returns the expression without a coefficient.\n\n            Similar to expr.as_independent(x)[1], except it only works\n            multiplicatively.\n            '
            term = S.One
            if expr.is_Mul:
                for i in expr.args:
                    if i.has(x):
                        term *= i
            elif expr.has(x):
                term = expr
            return term
        expr = expand_mul(expr)
        if expr.is_Add:
            for term in expr.args:
                if _remove_coefficient(term, x) in exprs:
                    pass
                else:
                    exprs.add(_remove_coefficient(term, x))
                    exprs = exprs.union(_get_trial_set(term, x, exprs))
        else:
            term = _remove_coefficient(expr, x)
            tmpset = exprs.union({term})
            oldset = set()
            while tmpset != oldset:
                oldset = tmpset.copy()
                expr = expr.diff(x)
                term = _remove_coefficient(expr, x)
                if term.is_Add:
                    tmpset = tmpset.union(_get_trial_set(term, x, tmpset))
                else:
                    tmpset.add(term)
            exprs = tmpset
        return exprs

    def is_homogeneous_solution(term):
        if False:
            i = 10
            return i + 15
        ' This function checks whether the given trialset contains any root\n            of homogeneous equation'
        return expand(sub_func_doit(eq_homogeneous, func, term)).is_zero
    retdict['test'] = _test_term(expr, x)
    if retdict['test']:
        temp_set = set()
        for i in Add.make_args(expr):
            act = _get_trial_set(i, x)
            if eq_homogeneous is not S.Zero:
                while any((is_homogeneous_solution(ts) for ts in act)):
                    act = {x * ts for ts in act}
            temp_set = temp_set.union(act)
        retdict['trialset'] = temp_set
    return retdict

def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    if False:
        print('Hello World!')
    "\n    Helper function for the method of undetermined coefficients.\n\n    See the\n    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`\n    docstring for more information on this method.\n\n    The parameter ``trialset`` is the set of trial functions as returned by\n    ``_undetermined_coefficients_match()['trialset']``.\n\n    The parameter ``match`` should be a dictionary that has the following\n    keys:\n\n    ``list``\n    A list of solutions to the homogeneous equation.\n\n    ``sol``\n    The general solution.\n\n    "
    r = match
    coeffs = numbered_symbols('a', cls=Dummy)
    coefflist = []
    gensols = r['list']
    gsol = r['sol']
    f = func.func
    x = func.args[0]
    if len(gensols) != order:
        raise NotImplementedError('Cannot find ' + str(order) + ' solutions to the homogeneous equation necessary to apply' + ' undetermined coefficients to ' + str(eq) + ' (number of terms != order)')
    trialfunc = 0
    for i in trialset:
        c = next(coeffs)
        coefflist.append(c)
        trialfunc += c * i
    eqs = sub_func_doit(eq, f(x), trialfunc)
    coeffsdict = dict(list(zip(trialset, [0] * (len(trialset) + 1))))
    eqs = _mexpand(eqs)
    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[x])
        if coeffsdict.get(s[x]):
            coeffsdict[s[x]] += s['coeff']
        else:
            coeffsdict[s[x]] = s['coeff']
    coeffvals = solve(list(coeffsdict.values()), coefflist)
    if not coeffvals:
        raise NotImplementedError('Could not solve `%s` using the method of undetermined coefficients (unable to solve for coefficients).' % eq)
    psol = trialfunc.subs(coeffvals)
    return Eq(f(x), gsol.rhs + psol)