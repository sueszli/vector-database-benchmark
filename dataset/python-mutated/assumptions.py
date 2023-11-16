"""
This module contains the machinery handling assumptions.
Do also consider the guide :ref:`assumptions-guide`.

All symbolic objects have assumption attributes that can be accessed via
``.is_<assumption name>`` attribute.

Assumptions determine certain properties of symbolic objects and can
have 3 possible values: ``True``, ``False``, ``None``.  ``True`` is returned if the
object has the property and ``False`` is returned if it does not or cannot
(i.e. does not make sense):

    >>> from sympy import I
    >>> I.is_algebraic
    True
    >>> I.is_real
    False
    >>> I.is_prime
    False

When the property cannot be determined (or when a method is not
implemented) ``None`` will be returned. For example,  a generic symbol, ``x``,
may or may not be positive so a value of ``None`` is returned for ``x.is_positive``.

By default, all symbolic values are in the largest set in the given context
without specifying the property. For example, a symbol that has a property
being integer, is also real, complex, etc.

Here follows a list of possible assumption names:

.. glossary::

    commutative
        object commutes with any other object with
        respect to multiplication operation. See [12]_.

    complex
        object can have only values from the set
        of complex numbers. See [13]_.

    imaginary
        object value is a number that can be written as a real
        number multiplied by the imaginary unit ``I``.  See
        [3]_.  Please note that ``0`` is not considered to be an
        imaginary number, see
        `issue #7649 <https://github.com/sympy/sympy/issues/7649>`_.

    real
        object can have only values from the set
        of real numbers.

    extended_real
        object can have only values from the set
        of real numbers, ``oo`` and ``-oo``.

    integer
        object can have only values from the set
        of integers.

    odd
    even
        object can have only values from the set of
        odd (even) integers [2]_.

    prime
        object is a natural number greater than 1 that has
        no positive divisors other than 1 and itself.  See [6]_.

    composite
        object is a positive integer that has at least one positive
        divisor other than 1 or the number itself.  See [4]_.

    zero
        object has the value of 0.

    nonzero
        object is a real number that is not zero.

    rational
        object can have only values from the set
        of rationals.

    algebraic
        object can have only values from the set
        of algebraic numbers [11]_.

    transcendental
        object can have only values from the set
        of transcendental numbers [10]_.

    irrational
        object value cannot be represented exactly by :class:`~.Rational`, see [5]_.

    finite
    infinite
        object absolute value is bounded (arbitrarily large).
        See [7]_, [8]_, [9]_.

    negative
    nonnegative
        object can have only negative (nonnegative)
        values [1]_.

    positive
    nonpositive
        object can have only positive (nonpositive) values.

    extended_negative
    extended_nonnegative
    extended_positive
    extended_nonpositive
    extended_nonzero
        as without the extended part, but also including infinity with
        corresponding sign, e.g., extended_positive includes ``oo``

    hermitian
    antihermitian
        object belongs to the field of Hermitian
        (antihermitian) operators.

Examples
========

    >>> from sympy import Symbol
    >>> x = Symbol('x', real=True); x
    x
    >>> x.is_real
    True
    >>> x.is_complex
    True

See Also
========

.. seealso::

    :py:class:`sympy.core.numbers.ImaginaryUnit`
    :py:class:`sympy.core.numbers.Zero`
    :py:class:`sympy.core.numbers.One`
    :py:class:`sympy.core.numbers.Infinity`
    :py:class:`sympy.core.numbers.NegativeInfinity`
    :py:class:`sympy.core.numbers.ComplexInfinity`

Notes
=====

The fully-resolved assumptions for any SymPy expression
can be obtained as follows:

    >>> from sympy.core.assumptions import assumptions
    >>> x = Symbol('x',positive=True)
    >>> assumptions(x + I)
    {'commutative': True, 'complex': True, 'composite': False, 'even':
    False, 'extended_negative': False, 'extended_nonnegative': False,
    'extended_nonpositive': False, 'extended_nonzero': False,
    'extended_positive': False, 'extended_real': False, 'finite': True,
    'imaginary': False, 'infinite': False, 'integer': False, 'irrational':
    False, 'negative': False, 'noninteger': False, 'nonnegative': False,
    'nonpositive': False, 'nonzero': False, 'odd': False, 'positive':
    False, 'prime': False, 'rational': False, 'real': False, 'zero':
    False}

Developers Notes
================

The current (and possibly incomplete) values are stored
in the ``obj._assumptions dictionary``; queries to getter methods
(with property decorators) or attributes of objects/classes
will return values and update the dictionary.

    >>> eq = x**2 + I
    >>> eq._assumptions
    {}
    >>> eq.is_finite
    True
    >>> eq._assumptions
    {'finite': True, 'infinite': False}

For a :class:`~.Symbol`, there are two locations for assumptions that may
be of interest. The ``assumptions0`` attribute gives the full set of
assumptions derived from a given set of initial assumptions. The
latter assumptions are stored as ``Symbol._assumptions_orig``

    >>> Symbol('x', prime=True, even=True)._assumptions_orig
    {'even': True, 'prime': True}

The ``_assumptions_orig`` are not necessarily canonical nor are they filtered
in any way: they records the assumptions used to instantiate a Symbol and (for
storage purposes) represent a more compact representation of the assumptions
needed to recreate the full set in ``Symbol.assumptions0``.


References
==========

.. [1] https://en.wikipedia.org/wiki/Negative_number
.. [2] https://en.wikipedia.org/wiki/Parity_%28mathematics%29
.. [3] https://en.wikipedia.org/wiki/Imaginary_number
.. [4] https://en.wikipedia.org/wiki/Composite_number
.. [5] https://en.wikipedia.org/wiki/Irrational_number
.. [6] https://en.wikipedia.org/wiki/Prime_number
.. [7] https://en.wikipedia.org/wiki/Finite
.. [8] https://docs.python.org/3/library/math.html#math.isfinite
.. [9] https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
.. [10] https://en.wikipedia.org/wiki/Transcendental_number
.. [11] https://en.wikipedia.org/wiki/Algebraic_number
.. [12] https://en.wikipedia.org/wiki/Commutative_property
.. [13] https://en.wikipedia.org/wiki/Complex_number

"""
from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions

def _load_pre_generated_assumption_rules():
    if False:
        for i in range(10):
            print('nop')
    ' Load the assumption rules from pre-generated data\n\n    To update the pre-generated data, see :method::`_generate_assumption_rules`\n    '
    _assume_rules = FactRules._from_python(_assumptions)
    return _assume_rules

def _generate_assumption_rules():
    if False:
        for i in range(10):
            print('nop')
    ' Generate the default assumption rules\n\n    This method should only be called to update the pre-generated\n    assumption rules.\n\n    To update the pre-generated assumptions run: bin/ask_update.py\n\n    '
    _assume_rules = FactRules(['integer        ->  rational', 'rational       ->  real', 'rational       ->  algebraic', 'algebraic      ->  complex', 'transcendental ==  complex & !algebraic', 'real           ->  hermitian', 'imaginary      ->  complex', 'imaginary      ->  antihermitian', 'extended_real  ->  commutative', 'complex        ->  commutative', 'complex        ->  finite', 'odd            ==  integer & !even', 'even           ==  integer & !odd', 'real           ->  complex', 'extended_real  ->  real | infinite', 'real           ==  extended_real & finite', 'extended_real        ==  extended_negative | zero | extended_positive', 'extended_negative    ==  extended_nonpositive & extended_nonzero', 'extended_positive    ==  extended_nonnegative & extended_nonzero', 'extended_nonpositive ==  extended_real & !extended_positive', 'extended_nonnegative ==  extended_real & !extended_negative', 'real           ==  negative | zero | positive', 'negative       ==  nonpositive & nonzero', 'positive       ==  nonnegative & nonzero', 'nonpositive    ==  real & !positive', 'nonnegative    ==  real & !negative', 'positive       ==  extended_positive & finite', 'negative       ==  extended_negative & finite', 'nonpositive    ==  extended_nonpositive & finite', 'nonnegative    ==  extended_nonnegative & finite', 'nonzero        ==  extended_nonzero & finite', 'zero           ->  even & finite', 'zero           ==  extended_nonnegative & extended_nonpositive', 'zero           ==  nonnegative & nonpositive', 'nonzero        ->  real', 'prime          ->  integer & positive', 'composite      ->  integer & positive & !prime', '!composite     ->  !positive | !even | prime', 'irrational     ==  real & !rational', 'imaginary      ->  !extended_real', 'infinite       ==  !finite', 'noninteger     ==  extended_real & !integer', 'extended_nonzero == extended_real & !zero'])
    return _assume_rules
_assume_rules = _load_pre_generated_assumption_rules()
_assume_defined = _assume_rules.defined_facts.copy()
_assume_defined.add('polar')
_assume_defined = frozenset(_assume_defined)

def assumptions(expr, _check=None):
    if False:
        print('Hello World!')
    'return the T/F assumptions of ``expr``'
    n = sympify(expr)
    if n.is_Symbol:
        rv = n.assumptions0
        if _check is not None:
            rv = {k: rv[k] for k in set(rv) & set(_check)}
        return rv
    rv = {}
    for k in _assume_defined if _check is None else _check:
        v = getattr(n, 'is_{}'.format(k))
        if v is not None:
            rv[k] = v
    return rv

def common_assumptions(exprs, check=None):
    if False:
        while True:
            i = 10
    "return those assumptions which have the same True or False\n    value for all the given expressions.\n\n    Examples\n    ========\n\n    >>> from sympy.core import common_assumptions\n    >>> from sympy import oo, pi, sqrt\n    >>> common_assumptions([-4, 0, sqrt(2), 2, pi, oo])\n    {'commutative': True, 'composite': False,\n    'extended_real': True, 'imaginary': False, 'odd': False}\n\n    By default, all assumptions are tested; pass an iterable of the\n    assumptions to limit those that are reported:\n\n    >>> common_assumptions([0, 1, 2], ['positive', 'integer'])\n    {'integer': True}\n    "
    check = _assume_defined if check is None else set(check)
    if not check or not exprs:
        return {}
    assume = [assumptions(i, _check=check) for i in sympify(exprs)]
    for (i, e) in enumerate(assume):
        assume[i] = {k: e[k] for k in set(e) & check}
    common = set.intersection(*[set(i) for i in assume])
    a = assume[0]
    return {k: a[k] for k in common if all((a[k] == b[k] for b in assume))}

def failing_assumptions(expr, **assumptions):
    if False:
        print('Hello World!')
    "\n    Return a dictionary containing assumptions with values not\n    matching those of the passed assumptions.\n\n    Examples\n    ========\n\n    >>> from sympy import failing_assumptions, Symbol\n\n    >>> x = Symbol('x', positive=True)\n    >>> y = Symbol('y')\n    >>> failing_assumptions(6*x + y, positive=True)\n    {'positive': None}\n\n    >>> failing_assumptions(x**2 - 1, positive=True)\n    {'positive': None}\n\n    If *expr* satisfies all of the assumptions, an empty dictionary is returned.\n\n    >>> failing_assumptions(x**2, positive=True)\n    {}\n\n    "
    expr = sympify(expr)
    failed = {}
    for k in assumptions:
        test = getattr(expr, 'is_%s' % k, None)
        if test is not assumptions[k]:
            failed[k] = test
    return failed

def check_assumptions(expr, against=None, **assume):
    if False:
        return 10
    "\n    Checks whether assumptions of ``expr`` match the T/F assumptions\n    given (or possessed by ``against``). True is returned if all\n    assumptions match; False is returned if there is a mismatch and\n    the assumption in ``expr`` is not None; else None is returned.\n\n    Explanation\n    ===========\n\n    *assume* is a dict of assumptions with True or False values\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol, pi, I, exp, check_assumptions\n    >>> check_assumptions(-5, integer=True)\n    True\n    >>> check_assumptions(pi, real=True, integer=False)\n    True\n    >>> check_assumptions(pi, negative=True)\n    False\n    >>> check_assumptions(exp(I*pi/7), real=False)\n    True\n    >>> x = Symbol('x', positive=True)\n    >>> check_assumptions(2*x + 1, positive=True)\n    True\n    >>> check_assumptions(-2*x - 5, positive=True)\n    False\n\n    To check assumptions of *expr* against another variable or expression,\n    pass the expression or variable as ``against``.\n\n    >>> check_assumptions(2*x + 1, x)\n    True\n\n    To see if a number matches the assumptions of an expression, pass\n    the number as the first argument, else its specific assumptions\n    may not have a non-None value in the expression:\n\n    >>> check_assumptions(x, 3)\n    >>> check_assumptions(3, x)\n    True\n\n    ``None`` is returned if ``check_assumptions()`` could not conclude.\n\n    >>> check_assumptions(2*x - 1, x)\n\n    >>> z = Symbol('z')\n    >>> check_assumptions(z, real=True)\n\n    See Also\n    ========\n\n    failing_assumptions\n\n    "
    expr = sympify(expr)
    if against is not None:
        if assume:
            raise ValueError('Expecting `against` or `assume`, not both.')
        assume = assumptions(against)
    known = True
    for (k, v) in assume.items():
        if v is None:
            continue
        e = getattr(expr, 'is_' + k, None)
        if e is None:
            known = None
        elif v != e:
            return False
    return known

class StdFactKB(FactKB):
    """A FactKB specialized for the built-in rules

    This is the only kind of FactKB that Basic objects should use.
    """

    def __init__(self, facts=None):
        if False:
            return 10
        super().__init__(_assume_rules)
        if not facts:
            self._generator = {}
        elif not isinstance(facts, FactKB):
            self._generator = facts.copy()
        else:
            self._generator = facts.generator
        if facts:
            self.deduce_all_facts(facts)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(self)

    @property
    def generator(self):
        if False:
            while True:
                i = 10
        return self._generator.copy()

def as_property(fact):
    if False:
        return 10
    'Convert a fact name to the name of the corresponding property'
    return 'is_%s' % fact

def make_property(fact):
    if False:
        print('Hello World!')
    'Create the automagic property corresponding to a fact.'

    def getit(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._assumptions[fact]
        except KeyError:
            if self._assumptions is self.default_assumptions:
                self._assumptions = self.default_assumptions.copy()
            return _ask(fact, self)
    getit.func_name = as_property(fact)
    return property(getit)

def _ask(fact, obj):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the truth value for a property of an object.\n\n    This function is called when a request is made to see what a fact\n    value is.\n\n    For this we use several techniques:\n\n    First, the fact-evaluation function is tried, if it exists (for\n    example _eval_is_integer). Then we try related facts. For example\n\n        rational   -->   integer\n\n    another example is joined rule:\n\n        integer & !odd  --> even\n\n    so in the latter case if we are looking at what 'even' value is,\n    'integer' and 'odd' facts will be asked.\n\n    In all cases, when we settle on some fact value, its implications are\n    deduced, and the result is cached in ._assumptions.\n    "
    assumptions = obj._assumptions
    handler_map = obj._prop_handler
    facts_to_check = [fact]
    facts_queued = {fact}
    for fact_i in facts_to_check:
        if fact_i in assumptions:
            continue
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)
    if fact in assumptions:
        return assumptions[fact]
    assumptions._tell(fact, None)
    return None

def _prepare_class_assumptions(cls):
    if False:
        print('Hello World!')
    'Precompute class level assumptions and generate handlers.\n\n    This is called by Basic.__init_subclass__ each time a Basic subclass is\n    defined.\n    '
    local_defs = {}
    for k in _assume_defined:
        attrname = as_property(k)
        v = cls.__dict__.get(attrname, '')
        if isinstance(v, (bool, int, type(None))):
            if v is not None:
                v = bool(v)
            local_defs[k] = v
    defs = {}
    for base in reversed(cls.__bases__):
        assumptions = getattr(base, '_explicit_class_assumptions', None)
        if assumptions is not None:
            defs.update(assumptions)
    defs.update(local_defs)
    cls._explicit_class_assumptions = defs
    cls.default_assumptions = StdFactKB(defs)
    cls._prop_handler = {}
    for k in _assume_defined:
        eval_is_meth = getattr(cls, '_eval_is_%s' % k, None)
        if eval_is_meth is not None:
            cls._prop_handler[k] = eval_is_meth
    for (k, v) in cls.default_assumptions.items():
        setattr(cls, as_property(k), v)
    derived_from_bases = set()
    for base in cls.__bases__:
        default_assumptions = getattr(base, 'default_assumptions', None)
        if default_assumptions is not None:
            derived_from_bases.update(default_assumptions)
    for fact in derived_from_bases - set(cls.default_assumptions):
        pname = as_property(fact)
        if pname not in cls.__dict__:
            setattr(cls, pname, make_property(fact))
    for fact in _assume_defined:
        pname = as_property(fact)
        if not hasattr(cls, pname):
            setattr(cls, pname, make_property(fact))

class ManagedProperties(type):

    def __init__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        msg = 'The ManagedProperties metaclass. Basic does not use metaclasses any more'
        sympy_deprecation_warning(msg, deprecated_since_version='1.12', active_deprecations_target='managedproperties')
        _prepare_class_assumptions(cls)