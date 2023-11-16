"""Inference in propositional logic"""
from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module

def literal_symbol(literal):
    if False:
        return 10
    '\n    The symbol in this literal (without the negation).\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A\n    >>> from sympy.logic.inference import literal_symbol\n    >>> literal_symbol(A)\n    A\n    >>> literal_symbol(~A)\n    A\n\n    '
    if literal is True or literal is False:
        return literal
    elif literal.is_Symbol:
        return literal
    elif literal.is_Not:
        return literal_symbol(literal.args[0])
    else:
        raise ValueError('Argument must be a boolean literal.')

def satisfiable(expr, algorithm=None, all_models=False, minimal=False, use_lra_theory=False):
    if False:
        return 10
    '\n    Check satisfiability of a propositional sentence.\n    Returns a model when it succeeds.\n    Returns {true: true} for trivially true expressions.\n\n    On setting all_models to True, if given expr is satisfiable then\n    returns a generator of models. However, if expr is unsatisfiable\n    then returns a generator containing the single element False.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.inference import satisfiable\n    >>> satisfiable(A & ~B)\n    {A: True, B: False}\n    >>> satisfiable(A & ~A)\n    False\n    >>> satisfiable(True)\n    {True: True}\n    >>> next(satisfiable(A & ~A, all_models=True))\n    False\n    >>> models = satisfiable((A >> B) & B, all_models=True)\n    >>> next(models)\n    {A: False, B: True}\n    >>> next(models)\n    {A: True, B: True}\n    >>> def use_models(models):\n    ...     for model in models:\n    ...         if model:\n    ...             # Do something with the model.\n    ...             print(model)\n    ...         else:\n    ...             # Given expr is unsatisfiable.\n    ...             print("UNSAT")\n    >>> use_models(satisfiable(A >> ~A, all_models=True))\n    {A: False}\n    >>> use_models(satisfiable(A ^ A, all_models=True))\n    UNSAT\n\n    '
    if use_lra_theory:
        if algorithm is not None and algorithm != 'dpll2':
            raise ValueError(f'Currently only dpll2 can handle using lra theory. {algorithm} is not handled.')
        algorithm = 'dpll2'
    if algorithm is None or algorithm == 'pycosat':
        pycosat = import_module('pycosat')
        if pycosat is not None:
            algorithm = 'pycosat'
        else:
            if algorithm == 'pycosat':
                raise ImportError('pycosat module is not present')
            algorithm = 'dpll2'
    if algorithm == 'minisat22':
        pysat = import_module('pysat')
        if pysat is None:
            algorithm = 'dpll2'
    if algorithm == 'z3':
        z3 = import_module('z3')
        if z3 is None:
            algorithm = 'dpll2'
    if algorithm == 'dpll':
        from sympy.logic.algorithms.dpll import dpll_satisfiable
        return dpll_satisfiable(expr)
    elif algorithm == 'dpll2':
        from sympy.logic.algorithms.dpll2 import dpll_satisfiable
        return dpll_satisfiable(expr, all_models, use_lra_theory=use_lra_theory)
    elif algorithm == 'pycosat':
        from sympy.logic.algorithms.pycosat_wrapper import pycosat_satisfiable
        return pycosat_satisfiable(expr, all_models)
    elif algorithm == 'minisat22':
        from sympy.logic.algorithms.minisat22_wrapper import minisat22_satisfiable
        return minisat22_satisfiable(expr, all_models, minimal)
    elif algorithm == 'z3':
        from sympy.logic.algorithms.z3_wrapper import z3_satisfiable
        return z3_satisfiable(expr, all_models)
    raise NotImplementedError

def valid(expr):
    if False:
        return 10
    '\n    Check validity of a propositional sentence.\n    A valid propositional sentence is True under every assignment.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.inference import valid\n    >>> valid(A | ~A)\n    True\n    >>> valid(A | B)\n    False\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Validity\n\n    '
    return not satisfiable(Not(expr))

def pl_true(expr, model=None, deep=False):
    if False:
        while True:
            i = 10
    "\n    Returns whether the given assignment is a model or not.\n\n    If the assignment does not specify the value for every proposition,\n    this may return None to indicate 'not obvious'.\n\n    Parameters\n    ==========\n\n    model : dict, optional, default: {}\n        Mapping of symbols to boolean values to indicate assignment.\n    deep: boolean, optional, default: False\n        Gives the value of the expression under partial assignments\n        correctly. May still return None to indicate 'not obvious'.\n\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.inference import pl_true\n    >>> pl_true( A & B, {A: True, B: True})\n    True\n    >>> pl_true(A & B, {A: False})\n    False\n    >>> pl_true(A & B, {A: True})\n    >>> pl_true(A & B, {A: True}, deep=True)\n    >>> pl_true(A >> (B >> A))\n    >>> pl_true(A >> (B >> A), deep=True)\n    True\n    >>> pl_true(A & ~A)\n    >>> pl_true(A & ~A, deep=True)\n    False\n    >>> pl_true(A & B & (~A | ~B), {A: True})\n    >>> pl_true(A & B & (~A | ~B), {A: True}, deep=True)\n    False\n\n    "
    from sympy.core.symbol import Symbol
    boolean = (True, False)

    def _validate(expr):
        if False:
            i = 10
            return i + 15
        if isinstance(expr, Symbol) or expr in boolean:
            return True
        if not isinstance(expr, BooleanFunction):
            return False
        return all((_validate(arg) for arg in expr.args))
    if expr in boolean:
        return expr
    expr = sympify(expr)
    if not _validate(expr):
        raise ValueError('%s is not a valid boolean expression' % expr)
    if not model:
        model = {}
    model = {k: v for (k, v) in model.items() if v in boolean}
    result = expr.subs(model)
    if result in boolean:
        return bool(result)
    if deep:
        model = {k: True for k in result.atoms()}
        if pl_true(result, model):
            if valid(result):
                return True
        elif not satisfiable(result):
            return False
    return None

def entails(expr, formula_set=None):
    if False:
        return 10
    '\n    Check whether the given expr_set entail an expr.\n    If formula_set is empty then it returns the validity of expr.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B, C\n    >>> from sympy.logic.inference import entails\n    >>> entails(A, [A >> B, B >> C])\n    False\n    >>> entails(C, [A >> B, B >> C, A])\n    True\n    >>> entails(A >> B)\n    False\n    >>> entails(A >> (B >> A))\n    True\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Logical_consequence\n\n    '
    if formula_set:
        formula_set = list(formula_set)
    else:
        formula_set = []
    formula_set.append(Not(expr))
    return not satisfiable(And(*formula_set))

class KB:
    """Base class for all knowledge bases"""

    def __init__(self, sentence=None):
        if False:
            for i in range(10):
                print('nop')
        self.clauses_ = set()
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def ask(self, query):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def retract(self, sentence):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @property
    def clauses(self):
        if False:
            while True:
                i = 10
        return list(ordered(self.clauses_))

class PropKB(KB):
    """A KB for Propositional Logic.  Inefficient, with no indexing."""

    def tell(self, sentence):
        if False:
            return 10
        "Add the sentence's clauses to the KB\n\n        Examples\n        ========\n\n        >>> from sympy.logic.inference import PropKB\n        >>> from sympy.abc import x, y\n        >>> l = PropKB()\n        >>> l.clauses\n        []\n\n        >>> l.tell(x | y)\n        >>> l.clauses\n        [x | y]\n\n        >>> l.tell(y)\n        >>> l.clauses\n        [y, x | y]\n\n        "
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.add(c)

    def ask(self, query):
        if False:
            return 10
        'Checks if the query is true given the set of clauses.\n\n        Examples\n        ========\n\n        >>> from sympy.logic.inference import PropKB\n        >>> from sympy.abc import x, y\n        >>> l = PropKB()\n        >>> l.tell(x & ~y)\n        >>> l.ask(x)\n        True\n        >>> l.ask(y)\n        False\n\n        '
        return entails(query, self.clauses_)

    def retract(self, sentence):
        if False:
            i = 10
            return i + 15
        "Remove the sentence's clauses from the KB\n\n        Examples\n        ========\n\n        >>> from sympy.logic.inference import PropKB\n        >>> from sympy.abc import x, y\n        >>> l = PropKB()\n        >>> l.clauses\n        []\n\n        >>> l.tell(x | y)\n        >>> l.clauses\n        [x | y]\n\n        >>> l.retract(x | y)\n        >>> l.clauses\n        []\n\n        "
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.discard(c)