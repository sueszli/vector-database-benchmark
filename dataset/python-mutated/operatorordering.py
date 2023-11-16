"""Functions for reordering operator expressions."""
import warnings
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
__all__ = ['normal_order', 'normal_ordered_form']

def _expand_powers(factors):
    if False:
        while True:
            i = 10
    '\n    Helper function for normal_ordered_form and normal_order: Expand a\n    power expression to a multiplication expression so that that the\n    expression can be handled by the normal ordering functions.\n    '
    new_factors = []
    for factor in factors.args:
        if isinstance(factor, Pow) and isinstance(factor.args[1], Integer) and (factor.args[1] > 0):
            for n in range(factor.args[1]):
                new_factors.append(factor.args[0])
        else:
            new_factors.append(factor)
    return new_factors

def _normal_ordered_form_factor(product, independent=False, recursive_limit=10, _recursive_depth=0):
    if False:
        print('Hello World!')
    '\n    Helper function for normal_ordered_form_factor: Write multiplication\n    expression with bosonic or fermionic operators on normally ordered form,\n    using the bosonic and fermionic commutation relations. The resulting\n    operator expression is equivalent to the argument, but will in general be\n    a sum of operator products instead of a simple product.\n    '
    factors = _expand_powers(product)
    new_factors = []
    n = 0
    while n < len(factors) - 1:
        (current, next) = (factors[n], factors[n + 1])
        if any((not isinstance(f, (FermionOp, BosonOp)) for f in (current, next))):
            new_factors.append(current)
            n += 1
            continue
        key_1 = (current.is_annihilation, str(current.name))
        key_2 = (next.is_annihilation, str(next.name))
        if key_1 <= key_2:
            new_factors.append(current)
            n += 1
            continue
        n += 2
        if current.is_annihilation and (not next.is_annihilation):
            if isinstance(current, BosonOp) and isinstance(next, BosonOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = Commutator(current, next)
                    new_factors.append(next * current + c)
                else:
                    new_factors.append(next * current + 1)
            elif isinstance(current, FermionOp) and isinstance(next, FermionOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = AntiCommutator(current, next)
                    new_factors.append(-next * current + c)
                else:
                    new_factors.append(-next * current + 1)
        elif current.is_annihilation == next.is_annihilation and isinstance(current, FermionOp) and isinstance(next, FermionOp):
            new_factors.append(-next * current)
        else:
            new_factors.append(next * current)
    if n == len(factors) - 1:
        new_factors.append(factors[-1])
    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_ordered_form(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth + 1, independent=independent)

def _normal_ordered_form_terms(expr, independent=False, recursive_limit=10, _recursive_depth=0):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for normal_ordered_form: loop through each term in an\n    addition expression and call _normal_ordered_form_factor to perform the\n    factor to an normally ordered expression.\n    '
    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_ordered_form_factor(term, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth, independent=independent)
            new_terms.append(new_term)
        else:
            new_terms.append(term)
    return Add(*new_terms)

def normal_ordered_form(expr, independent=False, recursive_limit=10, _recursive_depth=0):
    if False:
        i = 10
        return i + 15
    'Write an expression with bosonic or fermionic operators on normal\n    ordered form, where each term is normally ordered. Note that this\n    normal ordered form is equivalent to the original expression.\n\n    Parameters\n    ==========\n\n    expr : expression\n        The expression write on normal ordered form.\n    independent : bool (default False)\n        Whether to consider operator with different names as operating in\n        different Hilbert spaces. If False, the (anti-)commutation is left\n        explicit.\n    recursive_limit : int (default 10)\n        The number of allowed recursive applications of the function.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum import Dagger\n    >>> from sympy.physics.quantum.boson import BosonOp\n    >>> from sympy.physics.quantum.operatorordering import normal_ordered_form\n    >>> a = BosonOp("a")\n    >>> normal_ordered_form(a * Dagger(a))\n    1 + Dagger(a)*a\n    '
    if _recursive_depth > recursive_limit:
        warnings.warn('Too many recursions, aborting')
        return expr
    if isinstance(expr, Add):
        return _normal_ordered_form_terms(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth, independent=independent)
    elif isinstance(expr, Mul):
        return _normal_ordered_form_factor(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth, independent=independent)
    else:
        return expr

def _normal_order_factor(product, recursive_limit=10, _recursive_depth=0):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for normal_order: Normal order a multiplication expression\n    with bosonic or fermionic operators. In general the resulting operator\n    expression will not be equivalent to original product.\n    '
    factors = _expand_powers(product)
    n = 0
    new_factors = []
    while n < len(factors) - 1:
        if isinstance(factors[n], BosonOp) and factors[n].is_annihilation:
            if not isinstance(factors[n + 1], BosonOp):
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    new_factors.append(factors[n + 1] * factors[n])
                else:
                    new_factors.append(factors[n + 1] * factors[n])
                n += 1
        elif isinstance(factors[n], FermionOp) and factors[n].is_annihilation:
            if not isinstance(factors[n + 1], FermionOp):
                new_factors.append(factors[n])
            elif factors[n + 1].is_annihilation:
                new_factors.append(factors[n])
            else:
                if factors[n].args[0] != factors[n + 1].args[0]:
                    new_factors.append(-factors[n + 1] * factors[n])
                else:
                    new_factors.append(-factors[n + 1] * factors[n])
                n += 1
        else:
            new_factors.append(factors[n])
        n += 1
    if n == len(factors) - 1:
        new_factors.append(factors[-1])
    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_order(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth + 1)

def _normal_order_terms(expr, recursive_limit=10, _recursive_depth=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for normal_order: look through each term in an addition\n    expression and call _normal_order_factor to perform the normal ordering\n    on the factors.\n    '
    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_order_factor(term, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth)
            new_terms.append(new_term)
        else:
            new_terms.append(term)
    return Add(*new_terms)

def normal_order(expr, recursive_limit=10, _recursive_depth=0):
    if False:
        while True:
            i = 10
    'Normal order an expression with bosonic or fermionic operators. Note\n    that this normal order is not equivalent to the original expression, but\n    the creation and annihilation operators in each term in expr is reordered\n    so that the expression becomes normal ordered.\n\n    Parameters\n    ==========\n\n    expr : expression\n        The expression to normal order.\n\n    recursive_limit : int (default 10)\n        The number of allowed recursive applications of the function.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum import Dagger\n    >>> from sympy.physics.quantum.boson import BosonOp\n    >>> from sympy.physics.quantum.operatorordering import normal_order\n    >>> a = BosonOp("a")\n    >>> normal_order(a * Dagger(a))\n    Dagger(a)*a\n    '
    if _recursive_depth > recursive_limit:
        warnings.warn('Too many recursions, aborting')
        return expr
    if isinstance(expr, Add):
        return _normal_order_terms(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth)
    elif isinstance(expr, Mul):
        return _normal_order_factor(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth)
    else:
        return expr