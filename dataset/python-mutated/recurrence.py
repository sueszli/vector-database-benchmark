"""Recurrence Operators"""
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.printing import sstr
from sympy.core.sympify import sympify

def RecurrenceOperators(base, generator):
    if False:
        print('Hello World!')
    "\n    Returns an Algebra of Recurrence Operators and the operator for\n    shifting i.e. the `Sn` operator.\n    The first argument needs to be the base polynomial ring for the algebra\n    and the second argument must be a generator which can be either a\n    noncommutative Symbol or a string.\n\n    Examples\n    ========\n\n    >>> from sympy import ZZ\n    >>> from sympy import symbols\n    >>> from sympy.holonomic.recurrence import RecurrenceOperators\n    >>> n = symbols('n', integer=True)\n    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')\n    "
    ring = RecurrenceOperatorAlgebra(base, generator)
    return (ring, ring.shift_operator)

class RecurrenceOperatorAlgebra:
    """
    A Recurrence Operator Algebra is a set of noncommutative polynomials
    in intermediate `Sn` and coefficients in a base ring A. It follows the
    commutation rule:
    Sn * a(n) = a(n + 1) * Sn

    This class represents a Recurrence Operator Algebra and serves as the parent ring
    for Recurrence Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    >>> R
    Univariate Recurrence Operator Algebra in intermediate Sn over the base ring
    ZZ[n]

    See Also
    ========

    RecurrenceOperator
    """

    def __init__(self, base, generator):
        if False:
            return 10
        self.base = base
        self.shift_operator = RecurrenceOperator([base.zero, base.one], self)
        if generator is None:
            self.gen_symbol = symbols('Sn', commutative=False)
        elif isinstance(generator, str):
            self.gen_symbol = symbols(generator, commutative=False)
        elif isinstance(generator, Symbol):
            self.gen_symbol = generator

    def __str__(self):
        if False:
            i = 10
            return i + 15
        string = 'Univariate Recurrence Operator Algebra in intermediate ' + sstr(self.gen_symbol) + ' over the base ring ' + self.base.__str__()
        return string
    __repr__ = __str__

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self.base == other.base and self.gen_symbol == other.gen_symbol:
            return True
        else:
            return False

def _add_lists(list1, list2):
    if False:
        return 10
    if len(list1) <= len(list2):
        sol = [a + b for (a, b) in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for (a, b) in zip(list1, list2)] + list1[len(list2):]
    return sol

class RecurrenceOperator:
    """
    The Recurrence Operators are defined by a list of polynomials
    in the base ring and the parent ring of the Operator.

    Explanation
    ===========

    Takes a list of polynomials for each power of Sn and the
    parent ring which must be an instance of RecurrenceOperatorAlgebra.

    A Recurrence Operator can be created easily using
    the operator `Sn`. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.recurrence import RecurrenceOperator, RecurrenceOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n),'Sn')

    >>> RecurrenceOperator([0, 1, n**2], R)
    (1)Sn + (n**2)Sn**2

    >>> Sn*n
    (n + 1)Sn

    >>> n*Sn*n + 1 - Sn**2*n
    (1) + (n**2 + n)Sn + (-n - 2)Sn**2

    See Also
    ========

    DifferentialOperatorAlgebra
    """
    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        if False:
            print('Hello World!')
        self.parent = parent
        if isinstance(list_of_poly, list):
            for (i, j) in enumerate(list_of_poly):
                if isinstance(j, int):
                    list_of_poly[i] = self.parent.base.from_sympy(S(j))
                elif not isinstance(j, self.parent.base.dtype):
                    list_of_poly[i] = self.parent.base.from_sympy(j)
            self.listofpoly = list_of_poly
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        '\n        Multiplies two Operators and returns another\n        RecurrenceOperator instance using the commutation rule\n        Sn * a(n) = a(n + 1) * Sn\n        '
        listofself = self.listofpoly
        base = self.parent.base
        if not isinstance(other, RecurrenceOperator):
            if not isinstance(other, self.parent.base.dtype):
                listofother = [self.parent.base.from_sympy(sympify(other))]
            else:
                listofother = [other]
        else:
            listofother = other.listofpoly

        def _mul_dmp_diffop(b, listofother):
            if False:
                print('Hello World!')
            if isinstance(listofother, list):
                sol = []
                for i in listofother:
                    sol.append(i * b)
                return sol
            else:
                return [b * listofother]
        sol = _mul_dmp_diffop(listofself[0], listofother)

        def _mul_Sni_b(b):
            if False:
                while True:
                    i = 10
            sol = [base.zero]
            if isinstance(b, list):
                for i in b:
                    j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
                    sol.append(base.from_sympy(j))
            else:
                j = b.subs(base.gens[0], base.gens[0] + S.One)
                sol.append(base.from_sympy(j))
            return sol
        for i in range(1, len(listofself)):
            listofother = _mul_Sni_b(listofother)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))
        return RecurrenceOperator(sol, self.parent)

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, RecurrenceOperator):
            if isinstance(other, int):
                other = S(other)
            if not isinstance(other, self.parent.base.dtype):
                other = self.parent.base.from_sympy(other)
            sol = []
            for j in self.listofpoly:
                sol.append(other * j)
            return RecurrenceOperator(sol, self.parent)

    def __add__(self, other):
        if False:
            return 10
        if isinstance(other, RecurrenceOperator):
            sol = _add_lists(self.listofpoly, other.listofpoly)
            return RecurrenceOperator(sol, self.parent)
        else:
            if isinstance(other, int):
                other = S(other)
            list_self = self.listofpoly
            if not isinstance(other, self.parent.base.dtype):
                list_other = [self.parent.base.from_sympy(other)]
            else:
                list_other = [other]
            sol = []
            sol.append(list_self[0] + list_other[0])
            sol += list_self[1:]
            return RecurrenceOperator(sol, self.parent)
    __radd__ = __add__

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        return self + -1 * other

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return -1 * self + other

    def __pow__(self, n):
        if False:
            return 10
        if n == 1:
            return self
        if n == 0:
            return RecurrenceOperator([self.parent.base.one], self.parent)
        if self.listofpoly == self.parent.shift_operator.listofpoly:
            sol = []
            for i in range(0, n):
                sol.append(self.parent.base.zero)
            sol.append(self.parent.base.one)
            return RecurrenceOperator(sol, self.parent)
        elif n % 2 == 1:
            powreduce = self ** (n - 1)
            return powreduce * self
        elif n % 2 == 0:
            powreduce = self ** (n / 2)
            return powreduce * powreduce

    def __str__(self):
        if False:
            return 10
        listofpoly = self.listofpoly
        print_str = ''
        for (i, j) in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue
            j = self.parent.base.to_sympy(j)
            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue
            if print_str:
                print_str += ' + '
            if i == 1:
                print_str += '(' + sstr(j) + ')Sn'
                continue
            print_str += '(' + sstr(j) + ')' + 'Sn**' + sstr(i)
        return print_str
    __repr__ = __str__

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, RecurrenceOperator):
            if self.listofpoly == other.listofpoly and self.parent == other.parent:
                return True
            else:
                return False
        elif self.listofpoly[0] == other:
            for i in self.listofpoly[1:]:
                if i is not self.parent.base.zero:
                    return False
            return True
        else:
            return False

class HolonomicSequence:
    """
    A Holonomic Sequence is a type of sequence satisfying a linear homogeneous
    recurrence relation with Polynomial coefficients. Alternatively, A sequence
    is Holonomic if and only if its generating function is a Holonomic Function.
    """

    def __init__(self, recurrence, u0=[]):
        if False:
            i = 10
            return i + 15
        self.recurrence = recurrence
        if not isinstance(u0, list):
            self.u0 = [u0]
        else:
            self.u0 = u0
        if len(self.u0) == 0:
            self._have_init_cond = False
        else:
            self._have_init_cond = True
        self.n = recurrence.parent.base.gens[0]

    def __repr__(self):
        if False:
            return 10
        str_sol = 'HolonomicSequence(%s, %s)' % (self.recurrence.__repr__(), sstr(self.n))
        if not self._have_init_cond:
            return str_sol
        else:
            cond_str = ''
            seq_str = 0
            for i in self.u0:
                cond_str += ', u(%s) = %s' % (sstr(seq_str), sstr(i))
                seq_str += 1
            sol = str_sol + cond_str
            return sol
    __str__ = __repr__

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self.recurrence == other.recurrence:
            if self.n == other.n:
                if self._have_init_cond and other._have_init_cond:
                    if self.u0 == other.u0:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
        else:
            return False