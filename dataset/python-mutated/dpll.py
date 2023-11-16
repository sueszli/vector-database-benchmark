"""Implementation of DPLL algorithm

Further improvements: eliminate calls to pl_true, implement branching rules,
efficient unit propagation.

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
  - https://www.researchgate.net/publication/242384772_Implementations_of_the_DPLL_Algorithm
"""
from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, to_int_repr, _find_predicates
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol

def dpll_satisfiable(expr):
    if False:
        return 10
    '\n    Check satisfiability of a propositional sentence.\n    It returns a model rather than True when it succeeds\n\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.algorithms.dpll import dpll_satisfiable\n    >>> dpll_satisfiable(A & ~B)\n    {A: True, B: False}\n    >>> dpll_satisfiable(A & ~A)\n    False\n\n    '
    if not isinstance(expr, CNF):
        clauses = conjuncts(to_cnf(expr))
    else:
        clauses = expr.clauses
    if False in clauses:
        return False
    symbols = sorted(_find_predicates(expr), key=default_sort_key)
    symbols_int_repr = set(range(1, len(symbols) + 1))
    clauses_int_repr = to_int_repr(clauses, symbols)
    result = dpll_int_repr(clauses_int_repr, symbols_int_repr, {})
    if not result:
        return result
    output = {}
    for key in result:
        output.update({symbols[key - 1]: result[key]})
    return output

def dpll(clauses, symbols, model):
    if False:
        print('Hello World!')
    '\n    Compute satisfiability in a partial model.\n    Clauses is an array of conjuncts.\n\n    >>> from sympy.abc import A, B, D\n    >>> from sympy.logic.algorithms.dpll import dpll\n    >>> dpll([A, B, D], [A, B], {D: False})\n    False\n\n    '
    (P, value) = find_unit_clause(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        (P, value) = find_unit_clause(clauses, model)
    (P, value) = find_pure_symbol(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        (P, value) = find_pure_symbol(symbols, clauses)
    unknown_clauses = []
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    if not clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols[:]
    return dpll(unit_propagate(unknown_clauses, P), symbols, model) or dpll(unit_propagate(unknown_clauses, Not(P)), symbols_copy, model_copy)

def dpll_int_repr(clauses, symbols, model):
    if False:
        i = 10
        return i + 15
    '\n    Compute satisfiability in a partial model.\n    Arguments are expected to be in integer representation\n\n    >>> from sympy.logic.algorithms.dpll import dpll_int_repr\n    >>> dpll_int_repr([{1}, {2}, {3}], {1, 2}, {3: False})\n    False\n\n    '
    (P, value) = find_unit_clause_int_repr(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        (P, value) = find_unit_clause_int_repr(clauses, model)
    (P, value) = find_pure_symbol_int_repr(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        (P, value) = find_pure_symbol_int_repr(symbols, clauses)
    unknown_clauses = []
    for c in clauses:
        val = pl_true_int_repr(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols.copy()
    return dpll_int_repr(unit_propagate_int_repr(unknown_clauses, P), symbols, model) or dpll_int_repr(unit_propagate_int_repr(unknown_clauses, -P), symbols_copy, model_copy)

def pl_true_int_repr(clause, model={}):
    if False:
        while True:
            i = 10
    '\n    Lightweight version of pl_true.\n    Argument clause represents the set of args of an Or clause. This is used\n    inside dpll_int_repr, it is not meant to be used directly.\n\n    >>> from sympy.logic.algorithms.dpll import pl_true_int_repr\n    >>> pl_true_int_repr({1, 2}, {1: False})\n    >>> pl_true_int_repr({1, 2}, {1: False, 2: False})\n    False\n\n    '
    result = False
    for lit in clause:
        if lit < 0:
            p = model.get(-lit)
            if p is not None:
                p = not p
        else:
            p = model.get(lit)
        if p is True:
            return True
        elif p is None:
            result = None
    return result

def unit_propagate(clauses, symbol):
    if False:
        return 10
    '\n    Returns an equivalent set of clauses\n    If a set of clauses contains the unit clause l, the other clauses are\n    simplified by the application of the two following rules:\n\n      1. every clause containing l is removed\n      2. in every clause that contains ~l this literal is deleted\n\n    Arguments are expected to be in CNF.\n\n    >>> from sympy.abc import A, B, D\n    >>> from sympy.logic.algorithms.dpll import unit_propagate\n    >>> unit_propagate([A | B, D | ~B, B], B)\n    [D, B]\n\n    '
    output = []
    for c in clauses:
        if c.func != Or:
            output.append(c)
            continue
        for arg in c.args:
            if arg == ~symbol:
                output.append(Or(*[x for x in c.args if x != ~symbol]))
                break
            if arg == symbol:
                break
        else:
            output.append(c)
    return output

def unit_propagate_int_repr(clauses, s):
    if False:
        for i in range(10):
            print('nop')
    '\n    Same as unit_propagate, but arguments are expected to be in integer\n    representation\n\n    >>> from sympy.logic.algorithms.dpll import unit_propagate_int_repr\n    >>> unit_propagate_int_repr([{1, 2}, {3, -2}, {2}], 2)\n    [{3}]\n\n    '
    negated = {-s}
    return [clause - negated for clause in clauses if s not in clause]

def find_pure_symbol(symbols, unknown_clauses):
    if False:
        i = 10
        return i + 15
    '\n    Find a symbol and its value if it appears only as a positive literal\n    (or only as a negative) in clauses.\n\n    >>> from sympy.abc import A, B, D\n    >>> from sympy.logic.algorithms.dpll import find_pure_symbol\n    >>> find_pure_symbol([A, B, D], [A|~B,~B|~D,D|A])\n    (A, True)\n\n    '
    for sym in symbols:
        (found_pos, found_neg) = (False, False)
        for c in unknown_clauses:
            if not found_pos and sym in disjuncts(c):
                found_pos = True
            if not found_neg and Not(sym) in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return (sym, found_pos)
    return (None, None)

def find_pure_symbol_int_repr(symbols, unknown_clauses):
    if False:
        i = 10
        return i + 15
    '\n    Same as find_pure_symbol, but arguments are expected\n    to be in integer representation\n\n    >>> from sympy.logic.algorithms.dpll import find_pure_symbol_int_repr\n    >>> find_pure_symbol_int_repr({1,2,3},\n    ...     [{1, -2}, {-2, -3}, {3, 1}])\n    (1, True)\n\n    '
    all_symbols = set().union(*unknown_clauses)
    found_pos = all_symbols.intersection(symbols)
    found_neg = all_symbols.intersection([-s for s in symbols])
    for p in found_pos:
        if -p not in found_neg:
            return (p, True)
    for p in found_neg:
        if -p not in found_pos:
            return (-p, False)
    return (None, None)

def find_unit_clause(clauses, model):
    if False:
        while True:
            i = 10
    '\n    A unit clause has only 1 variable that is not bound in the model.\n\n    >>> from sympy.abc import A, B, D\n    >>> from sympy.logic.algorithms.dpll import find_unit_clause\n    >>> find_unit_clause([A | B | D, B | ~D, A | ~B], {A:True})\n    (B, False)\n\n    '
    for clause in clauses:
        num_not_in_model = 0
        for literal in disjuncts(clause):
            sym = literal_symbol(literal)
            if sym not in model:
                num_not_in_model += 1
                (P, value) = (sym, not isinstance(literal, Not))
        if num_not_in_model == 1:
            return (P, value)
    return (None, None)

def find_unit_clause_int_repr(clauses, model):
    if False:
        while True:
            i = 10
    '\n    Same as find_unit_clause, but arguments are expected to be in\n    integer representation.\n\n    >>> from sympy.logic.algorithms.dpll import find_unit_clause_int_repr\n    >>> find_unit_clause_int_repr([{1, 2, 3},\n    ...     {2, -3}, {1, -2}], {1: True})\n    (2, False)\n\n    '
    bound = set(model) | {-sym for sym in model}
    for clause in clauses:
        unbound = clause - bound
        if len(unbound) == 1:
            p = unbound.pop()
            if p < 0:
                return (-p, False)
            else:
                return (p, True)
    return (None, None)