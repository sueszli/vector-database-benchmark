"""Module for querying SymPy objects about assumptions."""
from sympy.assumptions.assume import global_assumptions, Predicate, AppliedPredicate
from sympy.assumptions.cnf import CNF, EncodedCNF, Literal
from sympy.core import sympify
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.inference import satisfiable
from sympy.utilities.decorator import memoize_property
from sympy.utilities.exceptions import sympy_deprecation_warning, SymPyDeprecationWarning, ignore_warnings

class AssumptionKeys:
    """
    This class contains all the supported keys by ``ask``.
    It should be accessed via the instance ``sympy.Q``.

    """

    @memoize_property
    def hermitian(self):
        if False:
            while True:
                i = 10
        from .handlers.sets import HermitianPredicate
        return HermitianPredicate()

    @memoize_property
    def antihermitian(self):
        if False:
            while True:
                i = 10
        from .handlers.sets import AntihermitianPredicate
        return AntihermitianPredicate()

    @memoize_property
    def real(self):
        if False:
            return 10
        from .handlers.sets import RealPredicate
        return RealPredicate()

    @memoize_property
    def extended_real(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.sets import ExtendedRealPredicate
        return ExtendedRealPredicate()

    @memoize_property
    def imaginary(self):
        if False:
            return 10
        from .handlers.sets import ImaginaryPredicate
        return ImaginaryPredicate()

    @memoize_property
    def complex(self):
        if False:
            return 10
        from .handlers.sets import ComplexPredicate
        return ComplexPredicate()

    @memoize_property
    def algebraic(self):
        if False:
            i = 10
            return i + 15
        from .handlers.sets import AlgebraicPredicate
        return AlgebraicPredicate()

    @memoize_property
    def transcendental(self):
        if False:
            for i in range(10):
                print('nop')
        from .predicates.sets import TranscendentalPredicate
        return TranscendentalPredicate()

    @memoize_property
    def integer(self):
        if False:
            i = 10
            return i + 15
        from .handlers.sets import IntegerPredicate
        return IntegerPredicate()

    @memoize_property
    def noninteger(self):
        if False:
            return 10
        from .predicates.sets import NonIntegerPredicate
        return NonIntegerPredicate()

    @memoize_property
    def rational(self):
        if False:
            print('Hello World!')
        from .handlers.sets import RationalPredicate
        return RationalPredicate()

    @memoize_property
    def irrational(self):
        if False:
            print('Hello World!')
        from .handlers.sets import IrrationalPredicate
        return IrrationalPredicate()

    @memoize_property
    def finite(self):
        if False:
            i = 10
            return i + 15
        from .handlers.calculus import FinitePredicate
        return FinitePredicate()

    @memoize_property
    def infinite(self):
        if False:
            return 10
        from .handlers.calculus import InfinitePredicate
        return InfinitePredicate()

    @memoize_property
    def positive_infinite(self):
        if False:
            i = 10
            return i + 15
        from .handlers.calculus import PositiveInfinitePredicate
        return PositiveInfinitePredicate()

    @memoize_property
    def negative_infinite(self):
        if False:
            return 10
        from .handlers.calculus import NegativeInfinitePredicate
        return NegativeInfinitePredicate()

    @memoize_property
    def positive(self):
        if False:
            while True:
                i = 10
        from .handlers.order import PositivePredicate
        return PositivePredicate()

    @memoize_property
    def negative(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.order import NegativePredicate
        return NegativePredicate()

    @memoize_property
    def zero(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.order import ZeroPredicate
        return ZeroPredicate()

    @memoize_property
    def extended_positive(self):
        if False:
            i = 10
            return i + 15
        from .handlers.order import ExtendedPositivePredicate
        return ExtendedPositivePredicate()

    @memoize_property
    def extended_negative(self):
        if False:
            print('Hello World!')
        from .handlers.order import ExtendedNegativePredicate
        return ExtendedNegativePredicate()

    @memoize_property
    def nonzero(self):
        if False:
            i = 10
            return i + 15
        from .handlers.order import NonZeroPredicate
        return NonZeroPredicate()

    @memoize_property
    def nonpositive(self):
        if False:
            print('Hello World!')
        from .handlers.order import NonPositivePredicate
        return NonPositivePredicate()

    @memoize_property
    def nonnegative(self):
        if False:
            i = 10
            return i + 15
        from .handlers.order import NonNegativePredicate
        return NonNegativePredicate()

    @memoize_property
    def extended_nonzero(self):
        if False:
            while True:
                i = 10
        from .handlers.order import ExtendedNonZeroPredicate
        return ExtendedNonZeroPredicate()

    @memoize_property
    def extended_nonpositive(self):
        if False:
            print('Hello World!')
        from .handlers.order import ExtendedNonPositivePredicate
        return ExtendedNonPositivePredicate()

    @memoize_property
    def extended_nonnegative(self):
        if False:
            while True:
                i = 10
        from .handlers.order import ExtendedNonNegativePredicate
        return ExtendedNonNegativePredicate()

    @memoize_property
    def even(self):
        if False:
            return 10
        from .handlers.ntheory import EvenPredicate
        return EvenPredicate()

    @memoize_property
    def odd(self):
        if False:
            return 10
        from .handlers.ntheory import OddPredicate
        return OddPredicate()

    @memoize_property
    def prime(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.ntheory import PrimePredicate
        return PrimePredicate()

    @memoize_property
    def composite(self):
        if False:
            print('Hello World!')
        from .handlers.ntheory import CompositePredicate
        return CompositePredicate()

    @memoize_property
    def commutative(self):
        if False:
            return 10
        from .handlers.common import CommutativePredicate
        return CommutativePredicate()

    @memoize_property
    def is_true(self):
        if False:
            while True:
                i = 10
        from .handlers.common import IsTruePredicate
        return IsTruePredicate()

    @memoize_property
    def symmetric(self):
        if False:
            i = 10
            return i + 15
        from .handlers.matrices import SymmetricPredicate
        return SymmetricPredicate()

    @memoize_property
    def invertible(self):
        if False:
            i = 10
            return i + 15
        from .handlers.matrices import InvertiblePredicate
        return InvertiblePredicate()

    @memoize_property
    def orthogonal(self):
        if False:
            return 10
        from .handlers.matrices import OrthogonalPredicate
        return OrthogonalPredicate()

    @memoize_property
    def unitary(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.matrices import UnitaryPredicate
        return UnitaryPredicate()

    @memoize_property
    def positive_definite(self):
        if False:
            while True:
                i = 10
        from .handlers.matrices import PositiveDefinitePredicate
        return PositiveDefinitePredicate()

    @memoize_property
    def upper_triangular(self):
        if False:
            print('Hello World!')
        from .handlers.matrices import UpperTriangularPredicate
        return UpperTriangularPredicate()

    @memoize_property
    def lower_triangular(self):
        if False:
            print('Hello World!')
        from .handlers.matrices import LowerTriangularPredicate
        return LowerTriangularPredicate()

    @memoize_property
    def diagonal(self):
        if False:
            while True:
                i = 10
        from .handlers.matrices import DiagonalPredicate
        return DiagonalPredicate()

    @memoize_property
    def fullrank(self):
        if False:
            while True:
                i = 10
        from .handlers.matrices import FullRankPredicate
        return FullRankPredicate()

    @memoize_property
    def square(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.matrices import SquarePredicate
        return SquarePredicate()

    @memoize_property
    def integer_elements(self):
        if False:
            while True:
                i = 10
        from .handlers.matrices import IntegerElementsPredicate
        return IntegerElementsPredicate()

    @memoize_property
    def real_elements(self):
        if False:
            print('Hello World!')
        from .handlers.matrices import RealElementsPredicate
        return RealElementsPredicate()

    @memoize_property
    def complex_elements(self):
        if False:
            for i in range(10):
                print('nop')
        from .handlers.matrices import ComplexElementsPredicate
        return ComplexElementsPredicate()

    @memoize_property
    def singular(self):
        if False:
            return 10
        from .predicates.matrices import SingularPredicate
        return SingularPredicate()

    @memoize_property
    def normal(self):
        if False:
            return 10
        from .predicates.matrices import NormalPredicate
        return NormalPredicate()

    @memoize_property
    def triangular(self):
        if False:
            return 10
        from .predicates.matrices import TriangularPredicate
        return TriangularPredicate()

    @memoize_property
    def unit_triangular(self):
        if False:
            i = 10
            return i + 15
        from .predicates.matrices import UnitTriangularPredicate
        return UnitTriangularPredicate()

    @memoize_property
    def eq(self):
        if False:
            print('Hello World!')
        from .relation.equality import EqualityPredicate
        return EqualityPredicate()

    @memoize_property
    def ne(self):
        if False:
            print('Hello World!')
        from .relation.equality import UnequalityPredicate
        return UnequalityPredicate()

    @memoize_property
    def gt(self):
        if False:
            for i in range(10):
                print('nop')
        from .relation.equality import StrictGreaterThanPredicate
        return StrictGreaterThanPredicate()

    @memoize_property
    def ge(self):
        if False:
            while True:
                i = 10
        from .relation.equality import GreaterThanPredicate
        return GreaterThanPredicate()

    @memoize_property
    def lt(self):
        if False:
            while True:
                i = 10
        from .relation.equality import StrictLessThanPredicate
        return StrictLessThanPredicate()

    @memoize_property
    def le(self):
        if False:
            while True:
                i = 10
        from .relation.equality import LessThanPredicate
        return LessThanPredicate()
Q = AssumptionKeys()

def _extract_all_facts(assump, exprs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract all relevant assumptions from *assump* with respect to given *exprs*.\n\n    Parameters\n    ==========\n\n    assump : sympy.assumptions.cnf.CNF\n\n    exprs : tuple of expressions\n\n    Returns\n    =======\n\n    sympy.assumptions.cnf.CNF\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.cnf import CNF\n    >>> from sympy.assumptions.ask import _extract_all_facts\n    >>> from sympy.abc import x, y\n    >>> assump = CNF.from_prop(Q.positive(x) & Q.integer(y))\n    >>> exprs = (x,)\n    >>> cnf = _extract_all_facts(assump, exprs)\n    >>> cnf.clauses\n    {frozenset({Literal(Q.positive, False)})}\n\n    '
    facts = set()
    for clause in assump.clauses:
        args = []
        for literal in clause:
            if isinstance(literal.lit, AppliedPredicate) and len(literal.lit.arguments) == 1:
                if literal.lit.arg in exprs:
                    args.append(Literal(literal.lit.function, literal.is_Not))
                else:
                    break
            else:
                break
        else:
            if args:
                facts.add(frozenset(args))
    return CNF(facts)

def ask(proposition, assumptions=True, context=global_assumptions):
    if False:
        while True:
            i = 10
    '\n    Function to evaluate the proposition with assumptions.\n\n    Explanation\n    ===========\n\n    This function evaluates the proposition to ``True`` or ``False`` if\n    the truth value can be determined. If not, it returns ``None``.\n\n    It should be discerned from :func:`~.refine()` which, when applied to a\n    proposition, simplifies the argument to symbolic ``Boolean`` instead of\n    Python built-in ``True``, ``False`` or ``None``.\n\n    **Syntax**\n\n        * ask(proposition)\n            Evaluate the *proposition* in global assumption context.\n\n        * ask(proposition, assumptions)\n            Evaluate the *proposition* with respect to *assumptions* in\n            global assumption context.\n\n    Parameters\n    ==========\n\n    proposition : Boolean\n        Proposition which will be evaluated to boolean value. If this is\n        not ``AppliedPredicate``, it will be wrapped by ``Q.is_true``.\n\n    assumptions : Boolean, optional\n        Local assumptions to evaluate the *proposition*.\n\n    context : AssumptionsContext, optional\n        Default assumptions to evaluate the *proposition*. By default,\n        this is ``sympy.assumptions.global_assumptions`` variable.\n\n    Returns\n    =======\n\n    ``True``, ``False``, or ``None``\n\n    Raises\n    ======\n\n    TypeError : *proposition* or *assumptions* is not valid logical expression.\n\n    ValueError : assumptions are inconsistent.\n\n    Examples\n    ========\n\n    >>> from sympy import ask, Q, pi\n    >>> from sympy.abc import x, y\n    >>> ask(Q.rational(pi))\n    False\n    >>> ask(Q.even(x*y), Q.even(x) & Q.integer(y))\n    True\n    >>> ask(Q.prime(4*x), Q.integer(x))\n    False\n\n    If the truth value cannot be determined, ``None`` will be returned.\n\n    >>> print(ask(Q.odd(3*x))) # cannot determine unless we know x\n    None\n\n    ``ValueError`` is raised if assumptions are inconsistent.\n\n    >>> ask(Q.integer(x), Q.even(x) & Q.odd(x))\n    Traceback (most recent call last):\n      ...\n    ValueError: inconsistent assumptions Q.even(x) & Q.odd(x)\n\n    Notes\n    =====\n\n    Relations in assumptions are not implemented (yet), so the following\n    will not give a meaningful result.\n\n    >>> ask(Q.positive(x), x > 0)\n\n    It is however a work in progress.\n\n    See Also\n    ========\n\n    sympy.assumptions.refine.refine : Simplification using assumptions.\n        Proposition is not reduced to ``None`` if the truth value cannot\n        be determined.\n    '
    from sympy.assumptions.satask import satask
    from sympy.assumptions.lra_satask import lra_satask
    from sympy.logic.algorithms.lra_theory import UnhandledInput
    proposition = sympify(proposition)
    assumptions = sympify(assumptions)
    if isinstance(proposition, Predicate) or proposition.kind is not BooleanKind:
        raise TypeError('proposition must be a valid logical expression')
    if isinstance(assumptions, Predicate) or assumptions.kind is not BooleanKind:
        raise TypeError('assumptions must be a valid logical expression')
    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if isinstance(proposition, AppliedPredicate):
        (key, args) = (proposition.function, proposition.arguments)
    elif proposition.func in binrelpreds:
        (key, args) = (binrelpreds[type(proposition)], proposition.args)
    else:
        (key, args) = (Q.is_true, (proposition,))
    assump_cnf = CNF.from_prop(assumptions)
    assump_cnf.extend(context)
    local_facts = _extract_all_facts(assump_cnf, args)
    known_facts_cnf = get_all_known_facts()
    enc_cnf = EncodedCNF()
    enc_cnf.from_cnf(CNF(known_facts_cnf))
    enc_cnf.add_from_cnf(local_facts)
    if local_facts.clauses and satisfiable(enc_cnf) is False:
        raise ValueError('inconsistent assumptions %s' % assumptions)
    res = _ask_single_fact(key, local_facts)
    if res is not None:
        return res
    res = key(*args)._eval_ask(assumptions)
    if res is not None:
        return bool(res)
    res = satask(proposition, assumptions=assumptions, context=context)
    if res is not None:
        return res
    try:
        res = lra_satask(proposition, assumptions=assumptions, context=context)
    except UnhandledInput:
        return None
    return res

def _ask_single_fact(key, local_facts):
    if False:
        return 10
    '\n    Compute the truth value of single predicate using assumptions.\n\n    Parameters\n    ==========\n\n    key : sympy.assumptions.assume.Predicate\n        Proposition predicate.\n\n    local_facts : sympy.assumptions.cnf.CNF\n        Local assumption in CNF form.\n\n    Returns\n    =======\n\n    ``True``, ``False`` or ``None``\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.cnf import CNF\n    >>> from sympy.assumptions.ask import _ask_single_fact\n\n    If prerequisite of proposition is rejected by the assumption,\n    return ``False``.\n\n    >>> key, assump = Q.zero, ~Q.zero\n    >>> local_facts = CNF.from_prop(assump)\n    >>> _ask_single_fact(key, local_facts)\n    False\n    >>> key, assump = Q.zero, ~Q.even\n    >>> local_facts = CNF.from_prop(assump)\n    >>> _ask_single_fact(key, local_facts)\n    False\n\n    If assumption implies the proposition, return ``True``.\n\n    >>> key, assump = Q.even, Q.zero\n    >>> local_facts = CNF.from_prop(assump)\n    >>> _ask_single_fact(key, local_facts)\n    True\n\n    If proposition rejects the assumption, return ``False``.\n\n    >>> key, assump = Q.even, Q.odd\n    >>> local_facts = CNF.from_prop(assump)\n    >>> _ask_single_fact(key, local_facts)\n    False\n    '
    if local_facts.clauses:
        known_facts_dict = get_known_facts_dict()
        if len(local_facts.clauses) == 1:
            (cl,) = local_facts.clauses
            if len(cl) == 1:
                (f,) = cl
                prop_facts = known_facts_dict.get(key, None)
                prop_req = prop_facts[0] if prop_facts is not None else set()
                if f.is_Not and f.arg in prop_req:
                    return False
        for clause in local_facts.clauses:
            if len(clause) == 1:
                (f,) = clause
                prop_facts = known_facts_dict.get(f.arg, None) if not f.is_Not else None
                if prop_facts is None:
                    continue
                (prop_req, prop_rej) = prop_facts
                if key in prop_req:
                    return True
                elif key in prop_rej:
                    return False
    return None

def register_handler(key, handler):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register a handler in the ask system. key must be a string and handler a\n    class inheriting from AskHandler.\n\n    .. deprecated:: 1.8.\n        Use multipledispatch handler instead. See :obj:`~.Predicate`.\n\n    '
    sympy_deprecation_warning('\n        The AskHandler system is deprecated. The register_handler() function\n        should be replaced with the multipledispatch handler of Predicate.\n        ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler')
    if isinstance(key, Predicate):
        key = key.name.name
    Qkey = getattr(Q, key, None)
    if Qkey is not None:
        Qkey.add_handler(handler)
    else:
        setattr(Q, key, Predicate(key, handlers=[handler]))

def remove_handler(key, handler):
    if False:
        for i in range(10):
            print('nop')
    '\n    Removes a handler from the ask system.\n\n    .. deprecated:: 1.8.\n        Use multipledispatch handler instead. See :obj:`~.Predicate`.\n\n    '
    sympy_deprecation_warning('\n        The AskHandler system is deprecated. The remove_handler() function\n        should be replaced with the multipledispatch handler of Predicate.\n        ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler')
    if isinstance(key, Predicate):
        key = key.name.name
    with ignore_warnings(SymPyDeprecationWarning):
        getattr(Q, key).remove_handler(handler)
from sympy.assumptions.ask_generated import get_all_known_facts, get_known_facts_dict