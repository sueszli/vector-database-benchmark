from collections import defaultdict
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.constructor import construct_domain
from sympy.polys.solvers import PolyNonlinearError
from .sdm import SDM, sdm_irref, sdm_particular_from_rref, sdm_nullspace_from_rref
from sympy.utilities.misc import filldedent

def _linsolve(eqs, syms):
    if False:
        for i in range(10):
            print('nop')
    "Solve a linear system of equations.\n\n    Examples\n    ========\n\n    Solve a linear system with a unique solution:\n\n    >>> from sympy import symbols, Eq\n    >>> from sympy.polys.matrices.linsolve import _linsolve\n    >>> x, y = symbols('x, y')\n    >>> eqs = [Eq(x + y, 1), Eq(x - y, 2)]\n    >>> _linsolve(eqs, [x, y])\n    {x: 3/2, y: -1/2}\n\n    In the case of underdetermined systems the solution will be expressed in\n    terms of the unknown symbols that are unconstrained:\n\n    >>> _linsolve([Eq(x + y, 0)], [x, y])\n    {x: -y, y: y}\n\n    "
    nsyms = len(syms)
    (eqsdict, const) = _linear_eq_to_dict(eqs, syms)
    Aaug = sympy_dict_to_dm(eqsdict, const, syms)
    K = Aaug.domain
    if K.is_RealField or K.is_ComplexField:
        Aaug = Aaug.to_ddm().rref()[0].to_sdm()
    (Arref, pivots, nzcols) = sdm_irref(Aaug)
    if pivots and pivots[-1] == nsyms:
        return None
    P = sdm_particular_from_rref(Arref, nsyms + 1, pivots)
    (V, nonpivots) = sdm_nullspace_from_rref(Arref, K.one, nsyms, pivots, nzcols)
    sol = defaultdict(list)
    for (i, v) in P.items():
        sol[syms[i]].append(K.to_sympy(v))
    for (npi, Vi) in zip(nonpivots, V):
        sym = syms[npi]
        for (i, v) in Vi.items():
            sol[syms[i]].append(sym * K.to_sympy(v))
    sol = {s: Add(*terms) for (s, terms) in sol.items()}
    zero = S.Zero
    for s in set(syms) - set(sol):
        sol[s] = zero
    return sol

def sympy_dict_to_dm(eqs_coeffs, eqs_rhs, syms):
    if False:
        i = 10
        return i + 15
    'Convert a system of dict equations to a sparse augmented matrix'
    elems = set(eqs_rhs).union(*(e.values() for e in eqs_coeffs))
    (K, elems_K) = construct_domain(elems, field=True, extension=True)
    elem_map = dict(zip(elems, elems_K))
    neqs = len(eqs_coeffs)
    nsyms = len(syms)
    sym2index = dict(zip(syms, range(nsyms)))
    eqsdict = []
    for (eq, rhs) in zip(eqs_coeffs, eqs_rhs):
        eqdict = {sym2index[s]: elem_map[c] for (s, c) in eq.items()}
        if rhs:
            eqdict[nsyms] = -elem_map[rhs]
        if eqdict:
            eqsdict.append(eqdict)
    sdm_aug = SDM(enumerate(eqsdict), (neqs, nsyms + 1), K)
    return sdm_aug

def _linear_eq_to_dict(eqs, syms):
    if False:
        print('Hello World!')
    'Convert a system Expr/Eq equations into dict form, returning\n    the coefficient dictionaries and a list of syms-independent terms\n    from each expression in ``eqs```.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.matrices.linsolve import _linear_eq_to_dict\n    >>> from sympy.abc import x\n    >>> _linear_eq_to_dict([2*x + 3], {x})\n    ([{x: 2}], [3])\n    '
    coeffs = []
    ind = []
    symset = set(syms)
    for (i, e) in enumerate(eqs):
        if e.is_Equality:
            (coeff, terms) = _lin_eq2dict(e.lhs, symset)
            (cR, tR) = _lin_eq2dict(e.rhs, symset)
            coeff -= cR
            for (k, v) in tR.items():
                if k in terms:
                    terms[k] -= v
                else:
                    terms[k] = -v
            terms = {k: v for (k, v) in terms.items() if v}
            (c, d) = (coeff, terms)
        else:
            (c, d) = _lin_eq2dict(e, symset)
        coeffs.append(d)
        ind.append(c)
    return (coeffs, ind)

def _lin_eq2dict(a, symset):
    if False:
        return 10
    'return (c, d) where c is the sym-independent part of ``a`` and\n    ``d`` is an efficiently calculated dictionary mapping symbols to\n    their coefficients. A PolyNonlinearError is raised if non-linearity\n    is detected.\n\n    The values in the dictionary will be non-zero.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.matrices.linsolve import _lin_eq2dict\n    >>> from sympy.abc import x, y\n    >>> _lin_eq2dict(x + 2*y + 3, {x, y})\n    (3, {x: 1, y: 2})\n    '
    if a in symset:
        return (S.Zero, {a: S.One})
    elif a.is_Add:
        terms_list = defaultdict(list)
        coeff_list = []
        for ai in a.args:
            (ci, ti) = _lin_eq2dict(ai, symset)
            coeff_list.append(ci)
            for (mij, cij) in ti.items():
                terms_list[mij].append(cij)
        coeff = Add(*coeff_list)
        terms = {sym: Add(*coeffs) for (sym, coeffs) in terms_list.items()}
        return (coeff, terms)
    elif a.is_Mul:
        terms = terms_coeff = None
        coeff_list = []
        for ai in a.args:
            (ci, ti) = _lin_eq2dict(ai, symset)
            if not ti:
                coeff_list.append(ci)
            elif terms is None:
                terms = ti
                terms_coeff = ci
            else:
                raise PolyNonlinearError(filldedent('\n                    nonlinear cross-term: %s' % a))
        coeff = Mul._from_args(coeff_list)
        if terms is None:
            return (coeff, {})
        else:
            terms = {sym: coeff * c for (sym, c) in terms.items()}
            return (coeff * terms_coeff, terms)
    elif not a.has_xfree(symset):
        return (a, {})
    else:
        raise PolyNonlinearError('nonlinear term: %s' % a)