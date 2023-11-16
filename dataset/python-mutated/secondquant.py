"""
Second quantization operators and states for bosons.

This follow the formulation of Fetter and Welecka, "Quantum Theory
of Many-Particle Systems."
"""
from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
__all__ = ['Dagger', 'KroneckerDelta', 'BosonicOperator', 'AnnihilateBoson', 'CreateBoson', 'AnnihilateFermion', 'CreateFermion', 'FockState', 'FockStateBra', 'FockStateKet', 'FockStateBosonKet', 'FockStateBosonBra', 'FockStateFermionKet', 'FockStateFermionBra', 'BBra', 'BKet', 'FBra', 'FKet', 'F', 'Fd', 'B', 'Bd', 'apply_operators', 'InnerProduct', 'BosonicBasis', 'VarBosonicBasis', 'FixedBosonicBasis', 'Commutator', 'matrix_rep', 'contraction', 'wicks', 'NO', 'evaluate_deltas', 'AntiSymmetricTensor', 'substitute_dummies', 'PermutationOperator', 'simplify_index_permutations']

class SecondQuantizationError(Exception):
    pass

class AppliesOnlyToSymbolicIndex(SecondQuantizationError):
    pass

class ContractionAppliesOnlyToFermions(SecondQuantizationError):
    pass

class ViolationOfPauliPrinciple(SecondQuantizationError):
    pass

class SubstitutionOfAmbigousOperatorFailed(SecondQuantizationError):
    pass

class WicksTheoremDoesNotApply(SecondQuantizationError):
    pass

class Dagger(Expr):
    """
    Hermitian conjugate of creation/annihilation operators.

    Examples
    ========

    >>> from sympy import I
    >>> from sympy.physics.secondquant import Dagger, B, Bd
    >>> Dagger(2*I)
    -2*I
    >>> Dagger(B(0))
    CreateBoson(0)
    >>> Dagger(Bd(0))
    AnnihilateBoson(0)

    """

    def __new__(cls, arg):
        if False:
            i = 10
            return i + 15
        arg = sympify(arg)
        r = cls.eval(arg)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, arg)
        return obj

    @classmethod
    def eval(cls, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluates the Dagger instance.\n\n        Examples\n        ========\n\n        >>> from sympy import I\n        >>> from sympy.physics.secondquant import Dagger, B, Bd\n        >>> Dagger(2*I)\n        -2*I\n        >>> Dagger(B(0))\n        CreateBoson(0)\n        >>> Dagger(Bd(0))\n        AnnihilateBoson(0)\n\n        The eval() method is called automatically.\n\n        '
        dagger = getattr(arg, '_dagger_', None)
        if dagger is not None:
            return dagger()
        if isinstance(arg, Basic):
            if arg.is_Add:
                return Add(*tuple(map(Dagger, arg.args)))
            if arg.is_Mul:
                return Mul(*tuple(map(Dagger, reversed(arg.args))))
            if arg.is_Number:
                return arg
            if arg.is_Pow:
                return Pow(Dagger(arg.args[0]), arg.args[1])
            if arg == I:
                return -arg
        else:
            return None

    def _dagger_(self):
        if False:
            while True:
                i = 10
        return self.args[0]

class TensorSymbol(Expr):
    is_commutative = True

class AntiSymmetricTensor(TensorSymbol):
    """Stores upper and lower indices in separate Tuple's.

    Each group of indices is assumed to be antisymmetric.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import AntiSymmetricTensor
    >>> i, j = symbols('i j', below_fermi=True)
    >>> a, b = symbols('a b', above_fermi=True)
    >>> AntiSymmetricTensor('v', (a, i), (b, j))
    AntiSymmetricTensor(v, (a, i), (b, j))
    >>> AntiSymmetricTensor('v', (i, a), (b, j))
    -AntiSymmetricTensor(v, (a, i), (b, j))

    As you can see, the indices are automatically sorted to a canonical form.

    """

    def __new__(cls, symbol, upper, lower):
        if False:
            while True:
                i = 10
        try:
            (upper, signu) = _sort_anticommuting_fermions(upper, key=cls._sortkey)
            (lower, signl) = _sort_anticommuting_fermions(lower, key=cls._sortkey)
        except ViolationOfPauliPrinciple:
            return S.Zero
        symbol = sympify(symbol)
        upper = Tuple(*upper)
        lower = Tuple(*lower)
        if (signu + signl) % 2:
            return -TensorSymbol.__new__(cls, symbol, upper, lower)
        else:
            return TensorSymbol.__new__(cls, symbol, upper, lower)

    @classmethod
    def _sortkey(cls, index):
        if False:
            for i in range(10):
                print('nop')
        'Key for sorting of indices.\n\n        particle < hole < general\n\n        FIXME: This is a bottle-neck, can we do it faster?\n        '
        h = hash(index)
        label = str(index)
        if isinstance(index, Dummy):
            if index.assumptions0.get('above_fermi'):
                return (20, label, h)
            elif index.assumptions0.get('below_fermi'):
                return (21, label, h)
            else:
                return (22, label, h)
        if index.assumptions0.get('above_fermi'):
            return (10, label, h)
        elif index.assumptions0.get('below_fermi'):
            return (11, label, h)
        else:
            return (12, label, h)

    def _latex(self, printer):
        if False:
            return 10
        return '{%s^{%s}_{%s}}' % (self.symbol, ''.join([i.name for i in self.args[1]]), ''.join([i.name for i in self.args[2]]))

    @property
    def symbol(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the symbol of the tensor.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import AntiSymmetricTensor\n        >>> i, j = symbols('i,j', below_fermi=True)\n        >>> a, b = symbols('a,b', above_fermi=True)\n        >>> AntiSymmetricTensor('v', (a, i), (b, j))\n        AntiSymmetricTensor(v, (a, i), (b, j))\n        >>> AntiSymmetricTensor('v', (a, i), (b, j)).symbol\n        v\n\n        "
        return self.args[0]

    @property
    def upper(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the upper indices.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import AntiSymmetricTensor\n        >>> i, j = symbols('i,j', below_fermi=True)\n        >>> a, b = symbols('a,b', above_fermi=True)\n        >>> AntiSymmetricTensor('v', (a, i), (b, j))\n        AntiSymmetricTensor(v, (a, i), (b, j))\n        >>> AntiSymmetricTensor('v', (a, i), (b, j)).upper\n        (a, i)\n\n\n        "
        return self.args[1]

    @property
    def lower(self):
        if False:
            return 10
        "\n        Returns the lower indices.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import AntiSymmetricTensor\n        >>> i, j = symbols('i,j', below_fermi=True)\n        >>> a, b = symbols('a,b', above_fermi=True)\n        >>> AntiSymmetricTensor('v', (a, i), (b, j))\n        AntiSymmetricTensor(v, (a, i), (b, j))\n        >>> AntiSymmetricTensor('v', (a, i), (b, j)).lower\n        (b, j)\n\n        "
        return self.args[2]

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s(%s,%s)' % self.args

class SqOperator(Expr):
    """
    Base class for Second Quantization operators.
    """
    op_symbol = 'sq'
    is_commutative = False

    def __new__(cls, k):
        if False:
            i = 10
            return i + 15
        obj = Basic.__new__(cls, sympify(k))
        return obj

    @property
    def state(self):
        if False:
            return 10
        "\n        Returns the state index related to this operator.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F, Fd, B, Bd\n        >>> p = Symbol('p')\n        >>> F(p).state\n        p\n        >>> Fd(p).state\n        p\n        >>> B(p).state\n        p\n        >>> Bd(p).state\n        p\n\n        "
        return self.args[0]

    @property
    def is_symbolic(self):
        if False:
            print('Hello World!')
        "\n        Returns True if the state is a symbol (as opposed to a number).\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> p = Symbol('p')\n        >>> F(p).is_symbolic\n        True\n        >>> F(1).is_symbolic\n        False\n\n        "
        if self.state.is_Integer:
            return False
        else:
            return True

    def __repr__(self):
        if False:
            return 10
        return NotImplemented

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s(%r)' % (self.op_symbol, self.state)

    def apply_operator(self, state):
        if False:
            while True:
                i = 10
        '\n        Applies an operator to itself.\n        '
        raise NotImplementedError('implement apply_operator in a subclass')

class BosonicOperator(SqOperator):
    pass

class Annihilator(SqOperator):
    pass

class Creator(SqOperator):
    pass

class AnnihilateBoson(BosonicOperator, Annihilator):
    """
    Bosonic annihilation operator.

    Examples
    ========

    >>> from sympy.physics.secondquant import B
    >>> from sympy.abc import x
    >>> B(x)
    AnnihilateBoson(x)
    """
    op_symbol = 'b'

    def _dagger_(self):
        if False:
            for i in range(10):
                print('nop')
        return CreateBoson(self.state)

    def apply_operator(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Apply state to self if self is not symbolic and state is a FockStateKet, else\n        multiply self by state.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import B, BKet\n        >>> from sympy.abc import x, y, n\n        >>> B(x).apply_operator(y)\n        y*AnnihilateBoson(x)\n        >>> B(0).apply_operator(BKet((n,)))\n        sqrt(n)*FockStateBosonKet((n - 1,))\n\n        '
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element])
            return amp * state.down(element)
        else:
            return Mul(self, state)

    def __repr__(self):
        if False:
            return 10
        return 'AnnihilateBoson(%s)' % self.state

    def _latex(self, printer):
        if False:
            return 10
        if self.state is S.Zero:
            return 'b_{0}'
        else:
            return 'b_{%s}' % self.state.name

class CreateBoson(BosonicOperator, Creator):
    """
    Bosonic creation operator.
    """
    op_symbol = 'b+'

    def _dagger_(self):
        if False:
            while True:
                i = 10
        return AnnihilateBoson(self.state)

    def apply_operator(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Apply state to self if self is not symbolic and state is a FockStateKet, else\n        multiply self by state.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import B, Dagger, BKet\n        >>> from sympy.abc import x, y, n\n        >>> Dagger(B(x)).apply_operator(y)\n        y*CreateBoson(x)\n        >>> B(0).apply_operator(BKet((n,)))\n        sqrt(n)*FockStateBosonKet((n - 1,))\n        '
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element] + 1)
            return amp * state.up(element)
        else:
            return Mul(self, state)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'CreateBoson(%s)' % self.state

    def _latex(self, printer):
        if False:
            while True:
                i = 10
        if self.state is S.Zero:
            return '{b^\\dagger_{0}}'
        else:
            return '{b^\\dagger_{%s}}' % self.state.name
B = AnnihilateBoson
Bd = CreateBoson

class FermionicOperator(SqOperator):

    @property
    def is_restricted(self):
        if False:
            print('Hello World!')
        "\n        Is this FermionicOperator restricted with respect to fermi level?\n\n        Returns\n        =======\n\n        1  : restricted to orbits above fermi\n        0  : no restriction\n        -1 : restricted to orbits below fermi\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F, Fd\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_restricted\n        1\n        >>> Fd(a).is_restricted\n        1\n        >>> F(i).is_restricted\n        -1\n        >>> Fd(i).is_restricted\n        -1\n        >>> F(p).is_restricted\n        0\n        >>> Fd(p).is_restricted\n        0\n\n        "
        ass = self.args[0].assumptions0
        if ass.get('below_fermi'):
            return -1
        if ass.get('above_fermi'):
            return 1
        return 0

    @property
    def is_above_fermi(self):
        if False:
            return 10
        "\n        Does the index of this FermionicOperator allow values above fermi?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_above_fermi\n        True\n        >>> F(i).is_above_fermi\n        False\n        >>> F(p).is_above_fermi\n        True\n\n        Note\n        ====\n\n        The same applies to creation operators Fd\n\n        "
        return not self.args[0].assumptions0.get('below_fermi')

    @property
    def is_below_fermi(self):
        if False:
            i = 10
            return i + 15
        "\n        Does the index of this FermionicOperator allow values below fermi?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_below_fermi\n        False\n        >>> F(i).is_below_fermi\n        True\n        >>> F(p).is_below_fermi\n        True\n\n        The same applies to creation operators Fd\n\n        "
        return not self.args[0].assumptions0.get('above_fermi')

    @property
    def is_only_below_fermi(self):
        if False:
            i = 10
            return i + 15
        "\n        Is the index of this FermionicOperator restricted to values below fermi?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_only_below_fermi\n        False\n        >>> F(i).is_only_below_fermi\n        True\n        >>> F(p).is_only_below_fermi\n        False\n\n        The same applies to creation operators Fd\n        "
        return self.is_below_fermi and (not self.is_above_fermi)

    @property
    def is_only_above_fermi(self):
        if False:
            return 10
        "\n        Is the index of this FermionicOperator restricted to values above fermi?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_only_above_fermi\n        True\n        >>> F(i).is_only_above_fermi\n        False\n        >>> F(p).is_only_above_fermi\n        False\n\n        The same applies to creation operators Fd\n        "
        return self.is_above_fermi and (not self.is_below_fermi)

    def _sortkey(self):
        if False:
            for i in range(10):
                print('nop')
        h = hash(self)
        label = str(self.args[0])
        if self.is_only_q_creator:
            return (1, label, h)
        if self.is_only_q_annihilator:
            return (4, label, h)
        if isinstance(self, Annihilator):
            return (3, label, h)
        if isinstance(self, Creator):
            return (2, label, h)

class AnnihilateFermion(FermionicOperator, Annihilator):
    """
    Fermionic annihilation operator.
    """
    op_symbol = 'f'

    def _dagger_(self):
        if False:
            print('Hello World!')
        return CreateFermion(self.state)

    def apply_operator(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply state to self if self is not symbolic and state is a FockStateKet, else\n        multiply self by state.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import B, Dagger, BKet\n        >>> from sympy.abc import x, y, n\n        >>> Dagger(B(x)).apply_operator(y)\n        y*CreateBoson(x)\n        >>> B(0).apply_operator(BKet((n,)))\n        sqrt(n)*FockStateBosonKet((n - 1,))\n        '
        if isinstance(state, FockStateFermionKet):
            element = self.state
            return state.down(element)
        elif isinstance(state, Mul):
            (c_part, nc_part) = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                element = self.state
                return Mul(*c_part + [nc_part[0].down(element)] + nc_part[1:])
            else:
                return Mul(self, state)
        else:
            return Mul(self, state)

    @property
    def is_q_creator(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Can we create a quasi-particle?  (create hole or create particle)\n        If so, would that be above or below the fermi surface?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_q_creator\n        0\n        >>> F(i).is_q_creator\n        -1\n        >>> F(p).is_q_creator\n        -1\n\n        "
        if self.is_below_fermi:
            return -1
        return 0

    @property
    def is_q_annihilator(self):
        if False:
            print('Hello World!')
        "\n        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)\n        If so, would that be above or below the fermi surface?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=1)\n        >>> i = Symbol('i', below_fermi=1)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_q_annihilator\n        1\n        >>> F(i).is_q_annihilator\n        0\n        >>> F(p).is_q_annihilator\n        1\n\n        "
        if self.is_above_fermi:
            return 1
        return 0

    @property
    def is_only_q_creator(self):
        if False:
            return 10
        "\n        Always create a quasi-particle?  (create hole or create particle)\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_only_q_creator\n        False\n        >>> F(i).is_only_q_creator\n        True\n        >>> F(p).is_only_q_creator\n        False\n\n        "
        return self.is_only_below_fermi

    @property
    def is_only_q_annihilator(self):
        if False:
            print('Hello World!')
        "\n        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import F\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> F(a).is_only_q_annihilator\n        True\n        >>> F(i).is_only_q_annihilator\n        False\n        >>> F(p).is_only_q_annihilator\n        False\n\n        "
        return self.is_only_above_fermi

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'AnnihilateFermion(%s)' % self.state

    def _latex(self, printer):
        if False:
            print('Hello World!')
        if self.state is S.Zero:
            return 'a_{0}'
        else:
            return 'a_{%s}' % self.state.name

class CreateFermion(FermionicOperator, Creator):
    """
    Fermionic creation operator.
    """
    op_symbol = 'f+'

    def _dagger_(self):
        if False:
            i = 10
            return i + 15
        return AnnihilateFermion(self.state)

    def apply_operator(self, state):
        if False:
            while True:
                i = 10
        '\n        Apply state to self if self is not symbolic and state is a FockStateKet, else\n        multiply self by state.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import B, Dagger, BKet\n        >>> from sympy.abc import x, y, n\n        >>> Dagger(B(x)).apply_operator(y)\n        y*CreateBoson(x)\n        >>> B(0).apply_operator(BKet((n,)))\n        sqrt(n)*FockStateBosonKet((n - 1,))\n        '
        if isinstance(state, FockStateFermionKet):
            element = self.state
            return state.up(element)
        elif isinstance(state, Mul):
            (c_part, nc_part) = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                element = self.state
                return Mul(*c_part + [nc_part[0].up(element)] + nc_part[1:])
        return Mul(self, state)

    @property
    def is_q_creator(self):
        if False:
            i = 10
            return i + 15
        "\n        Can we create a quasi-particle?  (create hole or create particle)\n        If so, would that be above or below the fermi surface?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import Fd\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> Fd(a).is_q_creator\n        1\n        >>> Fd(i).is_q_creator\n        0\n        >>> Fd(p).is_q_creator\n        1\n\n        "
        if self.is_above_fermi:
            return 1
        return 0

    @property
    def is_q_annihilator(self):
        if False:
            return 10
        "\n        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)\n        If so, would that be above or below the fermi surface?\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import Fd\n        >>> a = Symbol('a', above_fermi=1)\n        >>> i = Symbol('i', below_fermi=1)\n        >>> p = Symbol('p')\n\n        >>> Fd(a).is_q_annihilator\n        0\n        >>> Fd(i).is_q_annihilator\n        -1\n        >>> Fd(p).is_q_annihilator\n        -1\n\n        "
        if self.is_below_fermi:
            return -1
        return 0

    @property
    def is_only_q_creator(self):
        if False:
            i = 10
            return i + 15
        "\n        Always create a quasi-particle?  (create hole or create particle)\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import Fd\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> Fd(a).is_only_q_creator\n        True\n        >>> Fd(i).is_only_q_creator\n        False\n        >>> Fd(p).is_only_q_creator\n        False\n\n        "
        return self.is_only_above_fermi

    @property
    def is_only_q_annihilator(self):
        if False:
            i = 10
            return i + 15
        "\n        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import Fd\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> Fd(a).is_only_q_annihilator\n        False\n        >>> Fd(i).is_only_q_annihilator\n        True\n        >>> Fd(p).is_only_q_annihilator\n        False\n\n        "
        return self.is_only_below_fermi

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'CreateFermion(%s)' % self.state

    def _latex(self, printer):
        if False:
            print('Hello World!')
        if self.state is S.Zero:
            return '{a^\\dagger_{0}}'
        else:
            return '{a^\\dagger_{%s}}' % self.state.name
Fd = CreateFermion
F = AnnihilateFermion

class FockState(Expr):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Anywhere you can have a FockState, you can also have S.Zero.
    All code must check for this!

    Base class to represent FockStates.
    """
    is_commutative = False

    def __new__(cls, occupations):
        if False:
            return 10
        "\n        occupations is a list with two possible meanings:\n\n        - For bosons it is a list of occupation numbers.\n          Element i is the number of particles in state i.\n\n        - For fermions it is a list of occupied orbits.\n          Element 0 is the state that was occupied first, element i\n          is the i'th occupied state.\n        "
        occupations = list(map(sympify, occupations))
        obj = Basic.__new__(cls, Tuple(*occupations))
        return obj

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        i = int(i)
        return self.args[0][i]

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'FockState(%r)' % self.args

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s%r%s' % (getattr(self, 'lbracket', ''), self._labels(), getattr(self, 'rbracket', ''))

    def _labels(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.args[0])

    def _latex(self, printer):
        if False:
            i = 10
            return i + 15
        return '%s%s%s' % (getattr(self, 'lbracket_latex', ''), printer._print(self._labels()), getattr(self, 'rbracket_latex', ''))

class BosonState(FockState):
    """
    Base class for FockStateBoson(Ket/Bra).
    """

    def up(self, i):
        if False:
            return 10
        '\n        Performs the action of a creation operator.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import BBra\n        >>> b = BBra([1, 2])\n        >>> b\n        FockStateBosonBra((1, 2))\n        >>> b.up(1)\n        FockStateBosonBra((1, 3))\n        '
        i = int(i)
        new_occs = list(self.args[0])
        new_occs[i] = new_occs[i] + S.One
        return self.__class__(new_occs)

    def down(self, i):
        if False:
            return 10
        '\n        Performs the action of an annihilation operator.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import BBra\n        >>> b = BBra([1, 2])\n        >>> b\n        FockStateBosonBra((1, 2))\n        >>> b.down(1)\n        FockStateBosonBra((1, 1))\n        '
        i = int(i)
        new_occs = list(self.args[0])
        if new_occs[i] == S.Zero:
            return S.Zero
        else:
            new_occs[i] = new_occs[i] - S.One
            return self.__class__(new_occs)

class FermionState(FockState):
    """
    Base class for FockStateFermion(Ket/Bra).
    """
    fermi_level = 0

    def __new__(cls, occupations, fermi_level=0):
        if False:
            for i in range(10):
                print('nop')
        occupations = list(map(sympify, occupations))
        if len(occupations) > 1:
            try:
                (occupations, sign) = _sort_anticommuting_fermions(occupations, key=hash)
            except ViolationOfPauliPrinciple:
                return S.Zero
        else:
            sign = 0
        cls.fermi_level = fermi_level
        if cls._count_holes(occupations) > fermi_level:
            return S.Zero
        if sign % 2:
            return S.NegativeOne * FockState.__new__(cls, occupations)
        else:
            return FockState.__new__(cls, occupations)

    def up(self, i):
        if False:
            i = 10
            return i + 15
        "\n        Performs the action of a creation operator.\n\n        Explanation\n        ===========\n\n        If below fermi we try to remove a hole,\n        if above fermi we try to create a particle.\n\n        If general index p we return ``Kronecker(p,i)*self``\n        where ``i`` is a new symbol with restriction above or below.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import FKet\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        >>> FKet([]).up(a)\n        FockStateFermionKet((a,))\n\n        A creator acting on vacuum below fermi vanishes\n\n        >>> FKet([]).up(i)\n        0\n\n\n        "
        present = i in self.args[0]
        if self._only_above_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif self._only_below_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        elif present:
            hole = Dummy('i', below_fermi=True)
            return KroneckerDelta(i, hole) * self._remove_orbit(i)
        else:
            particle = Dummy('a', above_fermi=True)
            return KroneckerDelta(i, particle) * self._add_orbit(i)

    def down(self, i):
        if False:
            while True:
                i = 10
        "\n        Performs the action of an annihilation operator.\n\n        Explanation\n        ===========\n\n        If below fermi we try to create a hole,\n        If above fermi we try to remove a particle.\n\n        If general index p we return ``Kronecker(p,i)*self``\n        where ``i`` is a new symbol with restriction above or below.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.physics.secondquant import FKet\n        >>> a = Symbol('a', above_fermi=True)\n        >>> i = Symbol('i', below_fermi=True)\n        >>> p = Symbol('p')\n\n        An annihilator acting on vacuum above fermi vanishes\n\n        >>> FKet([]).down(a)\n        0\n\n        Also below fermi, it vanishes, unless we specify a fermi level > 0\n\n        >>> FKet([]).down(i)\n        0\n        >>> FKet([],4).down(i)\n        FockStateFermionKet((i,))\n\n        "
        present = i in self.args[0]
        if self._only_above_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        elif self._only_below_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif present:
            hole = Dummy('i', below_fermi=True)
            return KroneckerDelta(i, hole) * self._add_orbit(i)
        else:
            particle = Dummy('a', above_fermi=True)
            return KroneckerDelta(i, particle) * self._remove_orbit(i)

    @classmethod
    def _only_below_fermi(cls, i):
        if False:
            while True:
                i = 10
        '\n        Tests if given orbit is only below fermi surface.\n\n        If nothing can be concluded we return a conservative False.\n        '
        if i.is_number:
            return i <= cls.fermi_level
        if i.assumptions0.get('below_fermi'):
            return True
        return False

    @classmethod
    def _only_above_fermi(cls, i):
        if False:
            return 10
        '\n        Tests if given orbit is only above fermi surface.\n\n        If fermi level has not been set we return True.\n        If nothing can be concluded we return a conservative False.\n        '
        if i.is_number:
            return i > cls.fermi_level
        if i.assumptions0.get('above_fermi'):
            return True
        return not cls.fermi_level

    def _remove_orbit(self, i):
        if False:
            return 10
        '\n        Removes particle/fills hole in orbit i. No input tests performed here.\n        '
        new_occs = list(self.args[0])
        pos = new_occs.index(i)
        del new_occs[pos]
        if pos % 2:
            return S.NegativeOne * self.__class__(new_occs, self.fermi_level)
        else:
            return self.__class__(new_occs, self.fermi_level)

    def _add_orbit(self, i):
        if False:
            return 10
        '\n        Adds particle/creates hole in orbit i. No input tests performed here.\n        '
        return self.__class__((i,) + self.args[0], self.fermi_level)

    @classmethod
    def _count_holes(cls, list):
        if False:
            return 10
        '\n        Returns the number of identified hole states in list.\n        '
        return len([i for i in list if cls._only_below_fermi(i)])

    def _negate_holes(self, list):
        if False:
            for i in range(10):
                print('nop')
        return tuple([-i if i <= self.fermi_level else i for i in list])

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.fermi_level:
            return 'FockStateKet(%r, fermi_level=%s)' % (self.args[0], self.fermi_level)
        else:
            return 'FockStateKet(%r)' % (self.args[0],)

    def _labels(self):
        if False:
            i = 10
            return i + 15
        return self._negate_holes(self.args[0])

class FockStateKet(FockState):
    """
    Representation of a ket.
    """
    lbracket = '|'
    rbracket = '>'
    lbracket_latex = '\\left|'
    rbracket_latex = '\\right\\rangle'

class FockStateBra(FockState):
    """
    Representation of a bra.
    """
    lbracket = '<'
    rbracket = '|'
    lbracket_latex = '\\left\\langle'
    rbracket_latex = '\\right|'

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, FockStateKet):
            return InnerProduct(self, other)
        else:
            return Expr.__mul__(self, other)

class FockStateBosonKet(BosonState, FockStateKet):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Occupation numbers can be any integer >= 0.

    Examples
    ========

    >>> from sympy.physics.secondquant import BKet
    >>> BKet([1, 2])
    FockStateBosonKet((1, 2))
    """

    def _dagger_(self):
        if False:
            while True:
                i = 10
        return FockStateBosonBra(*self.args)

class FockStateBosonBra(BosonState, FockStateBra):
    """
    Describes a collection of BosonBra particles.

    Examples
    ========

    >>> from sympy.physics.secondquant import BBra
    >>> BBra([1, 2])
    FockStateBosonBra((1, 2))
    """

    def _dagger_(self):
        if False:
            while True:
                i = 10
        return FockStateBosonKet(*self.args)

class FockStateFermionKet(FermionState, FockStateKet):
    """
    Many-particle Fock state with a sequence of occupied orbits.

    Explanation
    ===========

    Each state can only have one particle, so we choose to store a list of
    occupied orbits rather than a tuple with occupation numbers (zeros and ones).

    states below fermi level are holes, and are represented by negative labels
    in the occupation list.

    For symbolic state labels, the fermi_level caps the number of allowed hole-
    states.

    Examples
    ========

    >>> from sympy.physics.secondquant import FKet
    >>> FKet([1, 2])
    FockStateFermionKet((1, 2))
    """

    def _dagger_(self):
        if False:
            i = 10
            return i + 15
        return FockStateFermionBra(*self.args)

class FockStateFermionBra(FermionState, FockStateBra):
    """
    See Also
    ========

    FockStateFermionKet

    Examples
    ========

    >>> from sympy.physics.secondquant import FBra
    >>> FBra([1, 2])
    FockStateFermionBra((1, 2))
    """

    def _dagger_(self):
        if False:
            print('Hello World!')
        return FockStateFermionKet(*self.args)
BBra = FockStateBosonBra
BKet = FockStateBosonKet
FBra = FockStateFermionBra
FKet = FockStateFermionKet

def _apply_Mul(m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take a Mul instance with operators and apply them to states.\n\n    Explanation\n    ===========\n\n    This method applies all operators with integer state labels\n    to the actual states.  For symbolic state labels, nothing is done.\n    When inner products of FockStates are encountered (like <a|b>),\n    they are converted to instances of InnerProduct.\n\n    This does not currently work on double inner products like,\n    <a|b><c|d>.\n\n    If the argument is not a Mul, it is simply returned as is.\n    '
    if not isinstance(m, Mul):
        return m
    (c_part, nc_part) = m.args_cnc()
    n_nc = len(nc_part)
    if n_nc in (0, 1):
        return m
    else:
        last = nc_part[-1]
        next_to_last = nc_part[-2]
        if isinstance(last, FockStateKet):
            if isinstance(next_to_last, SqOperator):
                if next_to_last.is_symbolic:
                    return m
                else:
                    result = next_to_last.apply_operator(last)
                    if result == 0:
                        return S.Zero
                    else:
                        return _apply_Mul(Mul(*c_part + nc_part[:-2] + [result]))
            elif isinstance(next_to_last, Pow):
                if isinstance(next_to_last.base, SqOperator) and next_to_last.exp.is_Integer:
                    if next_to_last.base.is_symbolic:
                        return m
                    else:
                        result = last
                        for i in range(next_to_last.exp):
                            result = next_to_last.base.apply_operator(result)
                            if result == 0:
                                break
                        if result == 0:
                            return S.Zero
                        else:
                            return _apply_Mul(Mul(*c_part + nc_part[:-2] + [result]))
                else:
                    return m
            elif isinstance(next_to_last, FockStateBra):
                result = InnerProduct(next_to_last, last)
                if result == 0:
                    return S.Zero
                else:
                    return _apply_Mul(Mul(*c_part + nc_part[:-2] + [result]))
            else:
                return m
        else:
            return m

def apply_operators(e):
    if False:
        while True:
            i = 10
    '\n    Take a SymPy expression with operators and states and apply the operators.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.secondquant import apply_operators\n    >>> from sympy import sympify\n    >>> apply_operators(sympify(3)+4)\n    7\n    '
    e = e.expand()
    muls = e.atoms(Mul)
    subs_list = [(m, _apply_Mul(m)) for m in iter(muls)]
    return e.subs(subs_list)

class InnerProduct(Basic):
    """
    An unevaluated inner product between a bra and ket.

    Explanation
    ===========

    Currently this class just reduces things to a product of
    Kronecker Deltas.  In the future, we could introduce abstract
    states like ``|a>`` and ``|b>``, and leave the inner product unevaluated as
    ``<a|b>``.

    """
    is_commutative = True

    def __new__(cls, bra, ket):
        if False:
            return 10
        if not isinstance(bra, FockStateBra):
            raise TypeError('must be a bra')
        if not isinstance(ket, FockStateKet):
            raise TypeError('must be a ket')
        return cls.eval(bra, ket)

    @classmethod
    def eval(cls, bra, ket):
        if False:
            return 10
        result = S.One
        for (i, j) in zip(bra.args[0], ket.args[0]):
            result *= KroneckerDelta(i, j)
            if result == 0:
                break
        return result

    @property
    def bra(self):
        if False:
            while True:
                i = 10
        'Returns the bra part of the state'
        return self.args[0]

    @property
    def ket(self):
        if False:
            while True:
                i = 10
        'Returns the ket part of the state'
        return self.args[1]

    def __repr__(self):
        if False:
            while True:
                i = 10
        sbra = repr(self.bra)
        sket = repr(self.ket)
        return '%s|%s' % (sbra[:-1], sket[1:])

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.__repr__()

def matrix_rep(op, basis):
    if False:
        while True:
            i = 10
    '\n    Find the representation of an operator in a basis.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.secondquant import VarBosonicBasis, B, matrix_rep\n    >>> b = VarBosonicBasis(5)\n    >>> o = B(0)\n    >>> matrix_rep(o, b)\n    Matrix([\n    [0, 1,       0,       0, 0],\n    [0, 0, sqrt(2),       0, 0],\n    [0, 0,       0, sqrt(3), 0],\n    [0, 0,       0,       0, 2],\n    [0, 0,       0,       0, 0]])\n    '
    a = zeros(len(basis))
    for i in range(len(basis)):
        for j in range(len(basis)):
            a[i, j] = apply_operators(Dagger(basis[i]) * op * basis[j])
    return a

class BosonicBasis:
    """
    Base class for a basis set of bosonic Fock states.
    """
    pass

class VarBosonicBasis:
    """
    A single state, variable particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis
    >>> b = VarBosonicBasis(5)
    >>> b
    [FockState((0,)), FockState((1,)), FockState((2,)),
     FockState((3,)), FockState((4,))]
    """

    def __init__(self, n_max):
        if False:
            return 10
        self.n_max = n_max
        self._build_states()

    def _build_states(self):
        if False:
            for i in range(10):
                print('nop')
        self.basis = []
        for i in range(self.n_max):
            self.basis.append(FockStateBosonKet([i]))
        self.n_basis = len(self.basis)

    def index(self, state):
        if False:
            while True:
                i = 10
        '\n        Returns the index of state in basis.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import VarBosonicBasis\n        >>> b = VarBosonicBasis(3)\n        >>> state = b.state(1)\n        >>> b\n        [FockState((0,)), FockState((1,)), FockState((2,))]\n        >>> state\n        FockStateBosonKet((1,))\n        >>> b.index(state)\n        1\n        '
        return self.basis.index(state)

    def state(self, i):
        if False:
            print('Hello World!')
        '\n        The state of a single basis.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import VarBosonicBasis\n        >>> b = VarBosonicBasis(5)\n        >>> b.state(3)\n        FockStateBosonKet((3,))\n        '
        return self.basis[i]

    def __getitem__(self, i):
        if False:
            return 10
        return self.state(i)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.basis)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return repr(self.basis)

class FixedBosonicBasis(BosonicBasis):
    """
    Fixed particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import FixedBosonicBasis
    >>> b = FixedBosonicBasis(2, 2)
    >>> state = b.state(1)
    >>> b
    [FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]
    >>> state
    FockStateBosonKet((1, 1))
    >>> b.index(state)
    1
    """

    def __init__(self, n_particles, n_levels):
        if False:
            for i in range(10):
                print('nop')
        self.n_particles = n_particles
        self.n_levels = n_levels
        self._build_particle_locations()
        self._build_states()

    def _build_particle_locations(self):
        if False:
            return 10
        tup = ['i%i' % i for i in range(self.n_particles)]
        first_loop = 'for i0 in range(%i)' % self.n_levels
        other_loops = ''
        for (cur, prev) in zip(tup[1:], tup):
            temp = 'for %s in range(%s + 1) ' % (cur, prev)
            other_loops = other_loops + temp
        tup_string = '(%s)' % ', '.join(tup)
        list_comp = '[%s %s %s]' % (tup_string, first_loop, other_loops)
        result = eval(list_comp)
        if self.n_particles == 1:
            result = [(item,) for item in result]
        self.particle_locations = result

    def _build_states(self):
        if False:
            return 10
        self.basis = []
        for tuple_of_indices in self.particle_locations:
            occ_numbers = self.n_levels * [0]
            for level in tuple_of_indices:
                occ_numbers[level] += 1
            self.basis.append(FockStateBosonKet(occ_numbers))
        self.n_basis = len(self.basis)

    def index(self, state):
        if False:
            return 10
        'Returns the index of state in basis.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import FixedBosonicBasis\n        >>> b = FixedBosonicBasis(2, 3)\n        >>> b.index(b.state(3))\n        3\n        '
        return self.basis.index(state)

    def state(self, i):
        if False:
            print('Hello World!')
        'Returns the state that lies at index i of the basis\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import FixedBosonicBasis\n        >>> b = FixedBosonicBasis(2, 3)\n        >>> b.state(3)\n        FockStateBosonKet((1, 0, 1))\n        '
        return self.basis[i]

    def __getitem__(self, i):
        if False:
            while True:
                i = 10
        return self.state(i)

    def __len__(self):
        if False:
            return 10
        return len(self.basis)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.basis)

class Commutator(Function):
    """
    The Commutator:  [A, B] = A*B - B*A

    The arguments are ordered according to .__cmp__()

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import Commutator
    >>> A, B = symbols('A,B', commutative=False)
    >>> Commutator(B, A)
    -Commutator(A, B)

    Evaluate the commutator with .doit()

    >>> comm = Commutator(A,B); comm
    Commutator(A, B)
    >>> comm.doit()
    A*B - B*A


    For two second quantization operators the commutator is evaluated
    immediately:

    >>> from sympy.physics.secondquant import Fd, F
    >>> a = symbols('a', above_fermi=True)
    >>> i = symbols('i', below_fermi=True)
    >>> p,q = symbols('p,q')

    >>> Commutator(Fd(a),Fd(i))
    2*NO(CreateFermion(a)*CreateFermion(i))

    But for more complicated expressions, the evaluation is triggered by
    a call to .doit()

    >>> comm = Commutator(Fd(p)*Fd(q),F(i)); comm
    Commutator(CreateFermion(p)*CreateFermion(q), AnnihilateFermion(i))
    >>> comm.doit(wicks=True)
    -KroneckerDelta(i, p)*CreateFermion(q) +
     KroneckerDelta(i, q)*CreateFermion(p)

    """
    is_commutative = False

    @classmethod
    def eval(cls, a, b):
        if False:
            print('Hello World!')
        '\n        The Commutator [A,B] is on canonical form if A < B.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import Commutator, F, Fd\n        >>> from sympy.abc import x\n        >>> c1 = Commutator(F(x), Fd(x))\n        >>> c2 = Commutator(Fd(x), F(x))\n        >>> Commutator.eval(c1, c2)\n        0\n        '
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero
        a = a.expand()
        if isinstance(a, Add):
            return Add(*[cls(term, b) for term in a.args])
        b = b.expand()
        if isinstance(b, Add):
            return Add(*[cls(a, term) for term in b.args])
        (ca, nca) = a.args_cnc()
        (cb, ncb) = b.args_cnc()
        c_part = list(ca) + list(cb)
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))
        if isinstance(a, BosonicOperator) and isinstance(b, BosonicOperator):
            if isinstance(b, CreateBoson) and isinstance(a, AnnihilateBoson):
                return KroneckerDelta(a.state, b.state)
            if isinstance(a, CreateBoson) and isinstance(b, AnnihilateBoson):
                return S.NegativeOne * KroneckerDelta(a.state, b.state)
            else:
                return S.Zero
        if isinstance(a, FermionicOperator) and isinstance(b, FermionicOperator):
            return wicks(a * b) - wicks(b * a)
        if a.sort_key() > b.sort_key():
            return S.NegativeOne * cls(b, a)

    def doit(self, **hints):
        if False:
            while True:
                i = 10
        "\n        Enables the computation of complex expressions.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import Commutator, F, Fd\n        >>> from sympy import symbols\n        >>> i, j = symbols('i,j', below_fermi=True)\n        >>> a, b = symbols('a,b', above_fermi=True)\n        >>> c = Commutator(Fd(a)*F(i),Fd(b)*F(j))\n        >>> c.doit(wicks=True)\n        0\n        "
        a = self.args[0]
        b = self.args[1]
        if hints.get('wicks'):
            a = a.doit(**hints)
            b = b.doit(**hints)
            try:
                return wicks(a * b) - wicks(b * a)
            except ContractionAppliesOnlyToFermions:
                pass
            except WicksTheoremDoesNotApply:
                pass
        return (a * b - b * a).doit(**hints)

    def __repr__(self):
        if False:
            return 10
        return 'Commutator(%s,%s)' % (self.args[0], self.args[1])

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '[%s,%s]' % (self.args[0], self.args[1])

    def _latex(self, printer):
        if False:
            return 10
        return '\\left[%s,%s\\right]' % tuple([printer._print(arg) for arg in self.args])

class NO(Expr):
    """
    This Object is used to represent normal ordering brackets.

    i.e.  {abcd}  sometimes written  :abcd:

    Explanation
    ===========

    Applying the function NO(arg) to an argument means that all operators in
    the argument will be assumed to anticommute, and have vanishing
    contractions.  This allows an immediate reordering to canonical form
    upon object creation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import NO, F, Fd
    >>> p,q = symbols('p,q')
    >>> NO(Fd(p)*F(q))
    NO(CreateFermion(p)*AnnihilateFermion(q))
    >>> NO(F(q)*Fd(p))
    -NO(CreateFermion(p)*AnnihilateFermion(q))


    Note
    ====

    If you want to generate a normal ordered equivalent of an expression, you
    should use the function wicks().  This class only indicates that all
    operators inside the brackets anticommute, and have vanishing contractions.
    Nothing more, nothing less.

    """
    is_commutative = False

    def __new__(cls, arg):
        if False:
            print('Hello World!')
        '\n        Use anticommutation to get canonical form of operators.\n\n        Explanation\n        ===========\n\n        Employ associativity of normal ordered product: {ab{cd}} = {abcd}\n        but note that {ab}{cd} /= {abcd}.\n\n        We also employ distributivity: {ab + cd} = {ab} + {cd}.\n\n        Canonical form also implies expand() {ab(c+d)} = {abc} + {abd}.\n\n        '
        arg = sympify(arg)
        arg = arg.expand()
        if arg.is_Add:
            return Add(*[cls(term) for term in arg.args])
        if arg.is_Mul:
            (c_part, seq) = arg.args_cnc()
            if c_part:
                coeff = Mul(*c_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One
            newseq = []
            foundit = False
            for fac in seq:
                if isinstance(fac, NO):
                    newseq.extend(fac.args)
                    foundit = True
                else:
                    newseq.append(fac)
            if foundit:
                return coeff * cls(Mul(*newseq))
            if isinstance(seq[0], BosonicOperator):
                raise NotImplementedError
            try:
                (newseq, sign) = _sort_anticommuting_fermions(seq)
            except ViolationOfPauliPrinciple:
                return S.Zero
            if sign % 2:
                return S.NegativeOne * coeff * cls(Mul(*newseq))
            elif sign:
                return coeff * cls(Mul(*newseq))
            else:
                pass
            if coeff != S.One:
                return coeff * cls(Mul(*newseq))
            return Expr.__new__(cls, Mul(*newseq))
        if isinstance(arg, NO):
            return arg
        return arg

    @property
    def has_q_creators(self):
        if False:
            return 10
        "\n        Return 0 if the leftmost argument of the first argument is a not a\n        q_creator, else 1 if it is above fermi or -1 if it is below fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import NO, F, Fd\n\n        >>> a = symbols('a', above_fermi=True)\n        >>> i = symbols('i', below_fermi=True)\n        >>> NO(Fd(a)*Fd(i)).has_q_creators\n        1\n        >>> NO(F(i)*F(a)).has_q_creators\n        -1\n        >>> NO(Fd(i)*F(a)).has_q_creators           #doctest: +SKIP\n        0\n\n        "
        return self.args[0].args[0].is_q_creator

    @property
    def has_q_annihilators(self):
        if False:
            i = 10
            return i + 15
        "\n        Return 0 if the rightmost argument of the first argument is a not a\n        q_annihilator, else 1 if it is above fermi or -1 if it is below fermi.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import NO, F, Fd\n\n        >>> a = symbols('a', above_fermi=True)\n        >>> i = symbols('i', below_fermi=True)\n        >>> NO(Fd(a)*Fd(i)).has_q_annihilators\n        -1\n        >>> NO(F(i)*F(a)).has_q_annihilators\n        1\n        >>> NO(Fd(a)*F(i)).has_q_annihilators\n        0\n\n        "
        return self.args[0].args[-1].is_q_annihilator

    def doit(self, **hints):
        if False:
            i = 10
            return i + 15
        "\n        Either removes the brackets or enables complex computations\n        in its arguments.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.secondquant import NO, Fd, F\n        >>> from textwrap import fill\n        >>> from sympy import symbols, Dummy\n        >>> p,q = symbols('p,q', cls=Dummy)\n        >>> print(fill(str(NO(Fd(p)*F(q)).doit())))\n        KroneckerDelta(_a, _p)*KroneckerDelta(_a,\n        _q)*CreateFermion(_a)*AnnihilateFermion(_a) + KroneckerDelta(_a,\n        _p)*KroneckerDelta(_i, _q)*CreateFermion(_a)*AnnihilateFermion(_i) -\n        KroneckerDelta(_a, _q)*KroneckerDelta(_i,\n        _p)*AnnihilateFermion(_a)*CreateFermion(_i) - KroneckerDelta(_i,\n        _p)*KroneckerDelta(_i, _q)*AnnihilateFermion(_i)*CreateFermion(_i)\n        "
        if hints.get('remove_brackets', True):
            return self._remove_brackets()
        else:
            return self.__new__(type(self), self.args[0].doit(**hints))

    def _remove_brackets(self):
        if False:
            print('Hello World!')
        '\n        Returns the sorted string without normal order brackets.\n\n        The returned string have the property that no nonzero\n        contractions exist.\n        '
        subslist = []
        for i in self.iter_q_creators():
            if self[i].is_q_annihilator:
                assume = self[i].state.assumptions0
                if isinstance(self[i].state, Dummy):
                    assume.pop('above_fermi', None)
                    assume['below_fermi'] = True
                    below = Dummy('i', **assume)
                    assume.pop('below_fermi', None)
                    assume['above_fermi'] = True
                    above = Dummy('a', **assume)
                    cls = type(self[i])
                    split = self[i].__new__(cls, below) * KroneckerDelta(below, self[i].state) + self[i].__new__(cls, above) * KroneckerDelta(above, self[i].state)
                    subslist.append((self[i], split))
                else:
                    raise SubstitutionOfAmbigousOperatorFailed(self[i])
        if subslist:
            result = NO(self.subs(subslist))
            if isinstance(result, Add):
                return Add(*[term.doit() for term in result.args])
        else:
            return self.args[0]

    def _expand_operators(self):
        if False:
            while True:
                i = 10
        '\n        Returns a sum of NO objects that contain no ambiguous q-operators.\n\n        Explanation\n        ===========\n\n        If an index q has range both above and below fermi, the operator F(q)\n        is ambiguous in the sense that it can be both a q-creator and a q-annihilator.\n        If q is dummy, it is assumed to be a summation variable and this method\n        rewrites it into a sum of NO terms with unambiguous operators:\n\n        {Fd(p)*F(q)} = {Fd(a)*F(b)} + {Fd(a)*F(i)} + {Fd(j)*F(b)} -{F(i)*Fd(j)}\n\n        where a,b are above and i,j are below fermi level.\n        '
        return NO(self._remove_brackets)

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return [self.args[0].args[i] for i in range(*indices)]
        else:
            return self.args[0].args[i]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.args[0].args)

    def iter_q_annihilators(self):
        if False:
            while True:
                i = 10
        "\n        Iterates over the annihilation operators.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j', below_fermi=True)\n        >>> a, b = symbols('a b', above_fermi=True)\n        >>> from sympy.physics.secondquant import NO, F, Fd\n        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))\n\n        >>> no.iter_q_creators()\n        <generator object... at 0x...>\n        >>> list(no.iter_q_creators())\n        [0, 1]\n        >>> list(no.iter_q_annihilators())\n        [3, 2]\n\n        "
        ops = self.args[0].args
        iter = range(len(ops) - 1, -1, -1)
        for i in iter:
            if ops[i].is_q_annihilator:
                yield i
            else:
                break

    def iter_q_creators(self):
        if False:
            i = 10
            return i + 15
        "\n        Iterates over the creation operators.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> i, j = symbols('i j', below_fermi=True)\n        >>> a, b = symbols('a b', above_fermi=True)\n        >>> from sympy.physics.secondquant import NO, F, Fd\n        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))\n\n        >>> no.iter_q_creators()\n        <generator object... at 0x...>\n        >>> list(no.iter_q_creators())\n        [0, 1]\n        >>> list(no.iter_q_annihilators())\n        [3, 2]\n\n        "
        ops = self.args[0].args
        iter = range(0, len(ops))
        for i in iter:
            if ops[i].is_q_creator:
                yield i
            else:
                break

    def get_subNO(self, i):
        if False:
            while True:
                i = 10
        "\n        Returns a NO() without FermionicOperator at index i.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols\n        >>> from sympy.physics.secondquant import F, NO\n        >>> p, q, r = symbols('p,q,r')\n\n        >>> NO(F(p)*F(q)*F(r)).get_subNO(1)\n        NO(AnnihilateFermion(p)*AnnihilateFermion(r))\n\n        "
        arg0 = self.args[0]
        mul = arg0._new_rawargs(*arg0.args[:i] + arg0.args[i + 1:])
        return NO(mul)

    def _latex(self, printer):
        if False:
            print('Hello World!')
        return '\\left\\{%s\\right\\}' % printer._print(self.args[0])

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'NO(%s)' % self.args[0]

    def __str__(self):
        if False:
            return 10
        return ':%s:' % self.args[0]

def contraction(a, b):
    if False:
        while True:
            i = 10
    "\n    Calculates contraction of Fermionic operators a and b.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.physics.secondquant import F, Fd, contraction\n    >>> p, q = symbols('p,q')\n    >>> a, b = symbols('a,b', above_fermi=True)\n    >>> i, j = symbols('i,j', below_fermi=True)\n\n    A contraction is non-zero only if a quasi-creator is to the right of a\n    quasi-annihilator:\n\n    >>> contraction(F(a),Fd(b))\n    KroneckerDelta(a, b)\n    >>> contraction(Fd(i),F(j))\n    KroneckerDelta(i, j)\n\n    For general indices a non-zero result restricts the indices to below/above\n    the fermi surface:\n\n    >>> contraction(Fd(p),F(q))\n    KroneckerDelta(_i, q)*KroneckerDelta(p, q)\n    >>> contraction(F(p),Fd(q))\n    KroneckerDelta(_a, q)*KroneckerDelta(p, q)\n\n    Two creators or two annihilators always vanishes:\n\n    >>> contraction(F(p),F(q))\n    0\n    >>> contraction(Fd(p),Fd(q))\n    0\n\n    "
    if isinstance(b, FermionicOperator) and isinstance(a, FermionicOperator):
        if isinstance(a, AnnihilateFermion) and isinstance(b, CreateFermion):
            if b.state.assumptions0.get('below_fermi'):
                return S.Zero
            if a.state.assumptions0.get('below_fermi'):
                return S.Zero
            if b.state.assumptions0.get('above_fermi'):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get('above_fermi'):
                return KroneckerDelta(a.state, b.state)
            return KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('a', above_fermi=True))
        if isinstance(b, AnnihilateFermion) and isinstance(a, CreateFermion):
            if b.state.assumptions0.get('above_fermi'):
                return S.Zero
            if a.state.assumptions0.get('above_fermi'):
                return S.Zero
            if b.state.assumptions0.get('below_fermi'):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get('below_fermi'):
                return KroneckerDelta(a.state, b.state)
            return KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('i', below_fermi=True))
        return S.Zero
    else:
        t = (isinstance(i, FermionicOperator) for i in (a, b))
        raise ContractionAppliesOnlyToFermions(*t)

def _sqkey(sq_operator):
    if False:
        for i in range(10):
            print('nop')
    'Generates key for canonical sorting of SQ operators.'
    return sq_operator._sortkey()

def _sort_anticommuting_fermions(string1, key=_sqkey):
    if False:
        return 10
    "Sort fermionic operators to canonical order, assuming all pairs anticommute.\n\n    Explanation\n    ===========\n\n    Uses a bidirectional bubble sort.  Items in string1 are not referenced\n    so in principle they may be any comparable objects.   The sorting depends on the\n    operators '>' and '=='.\n\n    If the Pauli principle is violated, an exception is raised.\n\n    Returns\n    =======\n\n    tuple (sorted_str, sign)\n\n    sorted_str: list containing the sorted operators\n    sign: int telling how many times the sign should be changed\n          (if sign==0 the string was already sorted)\n    "
    verified = False
    sign = 0
    rng = list(range(len(string1) - 1))
    rev = list(range(len(string1) - 3, -1, -1))
    keys = list(map(key, string1))
    key_val = dict(list(zip(keys, string1)))
    while not verified:
        verified = True
        for i in rng:
            left = keys[i]
            right = keys[i + 1]
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])
            if left > right:
                verified = False
                keys[i:i + 2] = [right, left]
                sign = sign + 1
        if verified:
            break
        for i in rev:
            left = keys[i]
            right = keys[i + 1]
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])
            if left > right:
                verified = False
                keys[i:i + 2] = [right, left]
                sign = sign + 1
    string1 = [key_val[k] for k in keys]
    return (string1, sign)

def evaluate_deltas(e):
    if False:
        for i in range(10):
            print('nop')
    "\n    We evaluate KroneckerDelta symbols in the expression assuming Einstein summation.\n\n    Explanation\n    ===========\n\n    If one index is repeated it is summed over and in effect substituted with\n    the other one. If both indices are repeated we substitute according to what\n    is the preferred index.  this is determined by\n    KroneckerDelta.preferred_index and KroneckerDelta.killable_index.\n\n    In case there are no possible substitutions or if a substitution would\n    imply a loss of information, nothing is done.\n\n    In case an index appears in more than one KroneckerDelta, the resulting\n    substitution depends on the order of the factors.  Since the ordering is platform\n    dependent, the literal expression resulting from this function may be hard to\n    predict.\n\n    Examples\n    ========\n\n    We assume the following:\n\n    >>> from sympy import symbols, Function, Dummy, KroneckerDelta\n    >>> from sympy.physics.secondquant import evaluate_deltas\n    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)\n    >>> a,b = symbols('a b', above_fermi=True, cls=Dummy)\n    >>> p,q = symbols('p q', cls=Dummy)\n    >>> f = Function('f')\n    >>> t = Function('t')\n\n    The order of preference for these indices according to KroneckerDelta is\n    (a, b, i, j, p, q).\n\n    Trivial cases:\n\n    >>> evaluate_deltas(KroneckerDelta(i,j)*f(i))       # d_ij f(i) -> f(j)\n    f(_j)\n    >>> evaluate_deltas(KroneckerDelta(i,j)*f(j))       # d_ij f(j) -> f(i)\n    f(_i)\n    >>> evaluate_deltas(KroneckerDelta(i,p)*f(p))       # d_ip f(p) -> f(i)\n    f(_i)\n    >>> evaluate_deltas(KroneckerDelta(q,p)*f(p))       # d_qp f(p) -> f(q)\n    f(_q)\n    >>> evaluate_deltas(KroneckerDelta(q,p)*f(q))       # d_qp f(q) -> f(p)\n    f(_p)\n\n    More interesting cases:\n\n    >>> evaluate_deltas(KroneckerDelta(i,p)*t(a,i)*f(p,q))\n    f(_i, _q)*t(_a, _i)\n    >>> evaluate_deltas(KroneckerDelta(a,p)*t(a,i)*f(p,q))\n    f(_a, _q)*t(_a, _i)\n    >>> evaluate_deltas(KroneckerDelta(p,q)*f(p,q))\n    f(_p, _p)\n\n    Finally, here are some cases where nothing is done, because that would\n    imply a loss of information:\n\n    >>> evaluate_deltas(KroneckerDelta(i,p)*f(q))\n    f(_q)*KroneckerDelta(_i, _p)\n    >>> evaluate_deltas(KroneckerDelta(i,p)*f(i))\n    f(_i)*KroneckerDelta(_i, _p)\n    "
    accepted_functions = (Add,)
    if isinstance(e, accepted_functions):
        return e.func(*[evaluate_deltas(arg) for arg in e.args])
    elif isinstance(e, Mul):
        deltas = []
        indices = {}
        for i in e.args:
            for s in i.free_symbols:
                if s in indices:
                    indices[s] += 1
                else:
                    indices[s] = 0
            if isinstance(i, KroneckerDelta):
                deltas.append(i)
        for d in deltas:
            if d.killable_index.is_Symbol and indices[d.killable_index]:
                e = e.subs(d.killable_index, d.preferred_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            elif d.preferred_index.is_Symbol and indices[d.preferred_index] and d.indices_contain_equal_information:
                e = e.subs(d.preferred_index, d.killable_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            else:
                pass
        return e
    else:
        return e

def substitute_dummies(expr, new_indices=False, pretty_indices={}):
    if False:
        while True:
            i = 10
    "\n    Collect terms by substitution of dummy variables.\n\n    Explanation\n    ===========\n\n    This routine allows simplification of Add expressions containing terms\n    which differ only due to dummy variables.\n\n    The idea is to substitute all dummy variables consistently depending on\n    the structure of the term.  For each term, we obtain a sequence of all\n    dummy variables, where the order is determined by the index range, what\n    factors the index belongs to and its position in each factor.  See\n    _get_ordered_dummies() for more information about the sorting of dummies.\n    The index sequence is then substituted consistently in each term.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Function, Dummy\n    >>> from sympy.physics.secondquant import substitute_dummies\n    >>> a,b,c,d = symbols('a b c d', above_fermi=True, cls=Dummy)\n    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)\n    >>> f = Function('f')\n\n    >>> expr = f(a,b) + f(c,d); expr\n    f(_a, _b) + f(_c, _d)\n\n    Since a, b, c and d are equivalent summation indices, the expression can be\n    simplified to a single term (for which the dummy indices are still summed over)\n\n    >>> substitute_dummies(expr)\n    2*f(_a, _b)\n\n\n    Controlling output:\n\n    By default the dummy symbols that are already present in the expression\n    will be reused in a different permutation.  However, if new_indices=True,\n    new dummies will be generated and inserted.  The keyword 'pretty_indices'\n    can be used to control this generation of new symbols.\n\n    By default the new dummies will be generated on the form i_1, i_2, a_1,\n    etc.  If you supply a dictionary with key:value pairs in the form:\n\n        { index_group: string_of_letters }\n\n    The letters will be used as labels for the new dummy symbols.  The\n    index_groups must be one of 'above', 'below' or 'general'.\n\n    >>> expr = f(a,b,i,j)\n    >>> my_dummies = { 'above':'st', 'below':'uv' }\n    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)\n    f(_s, _t, _u, _v)\n\n    If we run out of letters, or if there is no keyword for some index_group\n    the default dummy generator will be used as a fallback:\n\n    >>> p,q = symbols('p q', cls=Dummy)  # general indices\n    >>> expr = f(p,q)\n    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)\n    f(_p_0, _p_1)\n\n    "
    if new_indices:
        letters_above = pretty_indices.get('above', '')
        letters_below = pretty_indices.get('below', '')
        letters_general = pretty_indices.get('general', '')
        len_above = len(letters_above)
        len_below = len(letters_below)
        len_general = len(letters_general)

        def _i(number):
            if False:
                while True:
                    i = 10
            try:
                return letters_below[number]
            except IndexError:
                return 'i_' + str(number - len_below)

        def _a(number):
            if False:
                print('Hello World!')
            try:
                return letters_above[number]
            except IndexError:
                return 'a_' + str(number - len_above)

        def _p(number):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return letters_general[number]
            except IndexError:
                return 'p_' + str(number - len_general)
    aboves = []
    belows = []
    generals = []
    dummies = expr.atoms(Dummy)
    if not new_indices:
        dummies = sorted(dummies, key=default_sort_key)
    a = i = p = 0
    for d in dummies:
        assum = d.assumptions0
        if assum.get('above_fermi'):
            if new_indices:
                sym = _a(a)
                a += 1
            l1 = aboves
        elif assum.get('below_fermi'):
            if new_indices:
                sym = _i(i)
                i += 1
            l1 = belows
        else:
            if new_indices:
                sym = _p(p)
                p += 1
            l1 = generals
        if new_indices:
            l1.append(Dummy(sym, **assum))
        else:
            l1.append(d)
    expr = expr.expand()
    terms = Add.make_args(expr)
    new_terms = []
    for term in terms:
        i = iter(belows)
        a = iter(aboves)
        p = iter(generals)
        ordered = _get_ordered_dummies(term)
        subsdict = {}
        for d in ordered:
            if d.assumptions0.get('below_fermi'):
                subsdict[d] = next(i)
            elif d.assumptions0.get('above_fermi'):
                subsdict[d] = next(a)
            else:
                subsdict[d] = next(p)
        subslist = []
        final_subs = []
        for (k, v) in subsdict.items():
            if k == v:
                continue
            if v in subsdict:
                if subsdict[v] in subsdict:
                    x = Dummy('x')
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    final_subs.insert(0, (k, v))
            else:
                subslist.append((k, v))
        subslist.extend(final_subs)
        new_terms.append(term.subs(subslist))
    return Add(*new_terms)

class KeyPrinter(StrPrinter):
    """Printer for which only equal objects are equal in print"""

    def _print_Dummy(self, expr):
        if False:
            i = 10
            return i + 15
        return '(%s_%i)' % (expr.name, expr.dummy_index)

def __kprint(expr):
    if False:
        return 10
    p = KeyPrinter()
    return p.doprint(expr)

def _get_ordered_dummies(mul, verbose=False):
    if False:
        i = 10
        return i + 15
    "Returns all dummies in the mul sorted in canonical order.\n\n    Explanation\n    ===========\n\n    The purpose of the canonical ordering is that dummies can be substituted\n    consistently across terms with the result that equivalent terms can be\n    simplified.\n\n    It is not possible to determine if two terms are equivalent based solely on\n    the dummy order.  However, a consistent substitution guided by the ordered\n    dummies should lead to trivially (non-)equivalent terms, thereby revealing\n    the equivalence.  This also means that if two terms have identical sequences of\n    dummies, the (non-)equivalence should already be apparent.\n\n    Strategy\n    --------\n\n    The canonical order is given by an arbitrary sorting rule.  A sort key\n    is determined for each dummy as a tuple that depends on all factors where\n    the index is present.  The dummies are thereby sorted according to the\n    contraction structure of the term, instead of sorting based solely on the\n    dummy symbol itself.\n\n    After all dummies in the term has been assigned a key, we check for identical\n    keys, i.e. unorderable dummies.  If any are found, we call a specialized\n    method, _determine_ambiguous(), that will determine a unique order based\n    on recursive calls to _get_ordered_dummies().\n\n    Key description\n    ---------------\n\n    A high level description of the sort key:\n\n        1. Range of the dummy index\n        2. Relation to external (non-dummy) indices\n        3. Position of the index in the first factor\n        4. Position of the index in the second factor\n\n    The sort key is a tuple with the following components:\n\n        1. A single character indicating the range of the dummy (above, below\n           or general.)\n        2. A list of strings with fully masked string representations of all\n           factors where the dummy is present.  By masked, we mean that dummies\n           are represented by a symbol to indicate either below fermi, above or\n           general.  No other information is displayed about the dummies at\n           this point.  The list is sorted stringwise.\n        3. An integer number indicating the position of the index, in the first\n           factor as sorted in 2.\n        4. An integer number indicating the position of the index, in the second\n           factor as sorted in 2.\n\n    If a factor is either of type AntiSymmetricTensor or SqOperator, the index\n    position in items 3 and 4 is indicated as 'upper' or 'lower' only.\n    (Creation operators are considered upper and annihilation operators lower.)\n\n    If the masked factors are identical, the two factors cannot be ordered\n    unambiguously in item 2.  In this case, items 3, 4 are left out.  If several\n    indices are contracted between the unorderable factors, it will be handled by\n    _determine_ambiguous()\n\n\n    "
    args = Mul.make_args(mul)
    fac_dum = {fac: fac.atoms(Dummy) for fac in args}
    fac_repr = {fac: __kprint(fac) for fac in args}
    all_dums = set().union(*fac_dum.values())
    mask = {}
    for d in all_dums:
        if d.assumptions0.get('below_fermi'):
            mask[d] = '0'
        elif d.assumptions0.get('above_fermi'):
            mask[d] = '1'
        else:
            mask[d] = '2'
    dum_repr = {d: __kprint(d) for d in all_dums}

    def _key(d):
        if False:
            for i in range(10):
                print('nop')
        dumstruct = [fac for fac in fac_dum if d in fac_dum[fac]]
        other_dums = set().union(*[fac_dum[fac] for fac in dumstruct])
        fac = dumstruct[-1]
        if other_dums is fac_dum[fac]:
            other_dums = fac_dum[fac].copy()
        other_dums.remove(d)
        masked_facs = [fac_repr[fac] for fac in dumstruct]
        for d2 in other_dums:
            masked_facs = [fac.replace(dum_repr[d2], mask[d2]) for fac in masked_facs]
        all_masked = [fac.replace(dum_repr[d], mask[d]) for fac in masked_facs]
        masked_facs = dict(list(zip(dumstruct, masked_facs)))
        if has_dups(all_masked):
            all_masked.sort()
            return (mask[d], tuple(all_masked))
        keydict = dict(list(zip(dumstruct, all_masked)))
        dumstruct.sort(key=lambda x: keydict[x])
        all_masked.sort()
        pos_val = []
        for fac in dumstruct:
            if isinstance(fac, AntiSymmetricTensor):
                if d in fac.upper:
                    pos_val.append('u')
                if d in fac.lower:
                    pos_val.append('l')
            elif isinstance(fac, Creator):
                pos_val.append('u')
            elif isinstance(fac, Annihilator):
                pos_val.append('l')
            elif isinstance(fac, NO):
                ops = [op for op in fac if op.has(d)]
                for op in ops:
                    if isinstance(op, Creator):
                        pos_val.append('u')
                    else:
                        pos_val.append('l')
            else:
                facpos = -1
                while 1:
                    facpos = masked_facs[fac].find(dum_repr[d], facpos + 1)
                    if facpos == -1:
                        break
                    pos_val.append(facpos)
        return (mask[d], tuple(all_masked), pos_val[0], pos_val[-1])
    dumkey = dict(list(zip(all_dums, list(map(_key, all_dums)))))
    result = sorted(all_dums, key=lambda x: dumkey[x])
    if has_dups(iter(dumkey.values())):
        unordered = defaultdict(set)
        for (d, k) in dumkey.items():
            unordered[k].add(d)
        for k in [k for k in unordered if len(unordered[k]) < 2]:
            del unordered[k]
        unordered = [unordered[k] for k in sorted(unordered)]
        result = _determine_ambiguous(mul, result, unordered)
    return result

def _determine_ambiguous(term, ordered, ambiguous_groups):
    if False:
        while True:
            i = 10
    all_ambiguous = set()
    for dummies in ambiguous_groups:
        all_ambiguous |= dummies
    all_ordered = set(ordered) - all_ambiguous
    if not all_ordered:
        group = [d for d in ordered if d in ambiguous_groups[0]]
        d = group[0]
        all_ordered.add(d)
        ambiguous_groups[0].remove(d)
    stored_counter = _symbol_factory._counter
    subslist = []
    for d in [d for d in ordered if d in all_ordered]:
        nondum = _symbol_factory._next()
        subslist.append((d, nondum))
    newterm = term.subs(subslist)
    neworder = _get_ordered_dummies(newterm)
    _symbol_factory._set_counter(stored_counter)
    for group in ambiguous_groups:
        ordered_group = [d for d in neworder if d in group]
        ordered_group.reverse()
        result = []
        for d in ordered:
            if d in group:
                result.append(ordered_group.pop())
            else:
                result.append(d)
        ordered = result
    return ordered

class _SymbolFactory:

    def __init__(self, label):
        if False:
            i = 10
            return i + 15
        self._counterVar = 0
        self._label = label

    def _set_counter(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Sets counter to value.\n        '
        self._counterVar = value

    @property
    def _counter(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        What counter is currently at.\n        '
        return self._counterVar

    def _next(self):
        if False:
            return 10
        '\n        Generates the next symbols and increments counter by 1.\n        '
        s = Symbol('%s%i' % (self._label, self._counterVar))
        self._counterVar += 1
        return s
_symbol_factory = _SymbolFactory('_]"]_')

@cacheit
def _get_contractions(string1, keep_only_fully_contracted=False):
    if False:
        i = 10
        return i + 15
    '\n    Returns Add-object with contracted terms.\n\n    Uses recursion to find all contractions. -- Internal helper function --\n\n    Will find nonzero contractions in string1 between indices given in\n    leftrange and rightrange.\n\n    '
    if keep_only_fully_contracted and string1:
        result = []
    else:
        result = [NO(Mul(*string1))]
    for i in range(len(string1) - 1):
        for j in range(i + 1, len(string1)):
            c = contraction(string1[i], string1[j])
            if c:
                sign = (j - i + 1) % 2
                if sign:
                    coeff = S.NegativeOne * c
                else:
                    coeff = c
                oplist = string1[i + 1:j] + string1[j + 1:]
                if oplist:
                    result.append(coeff * NO(Mul(*string1[:i]) * _get_contractions(oplist, keep_only_fully_contracted=keep_only_fully_contracted)))
                else:
                    result.append(coeff * NO(Mul(*string1[:i])))
        if keep_only_fully_contracted:
            break
    return Add(*result)

def wicks(e, **kw_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the normal ordered equivalent of an expression using Wicks Theorem.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Dummy\n    >>> from sympy.physics.secondquant import wicks, F, Fd\n    >>> p, q, r = symbols('p,q,r')\n    >>> wicks(Fd(p)*F(q))\n    KroneckerDelta(_i, q)*KroneckerDelta(p, q) + NO(CreateFermion(p)*AnnihilateFermion(q))\n\n    By default, the expression is expanded:\n\n    >>> wicks(F(p)*(F(q)+F(r)))\n    NO(AnnihilateFermion(p)*AnnihilateFermion(q)) + NO(AnnihilateFermion(p)*AnnihilateFermion(r))\n\n    With the keyword 'keep_only_fully_contracted=True', only fully contracted\n    terms are returned.\n\n    By request, the result can be simplified in the following order:\n     -- KroneckerDelta functions are evaluated\n     -- Dummy variables are substituted consistently across terms\n\n    >>> p, q, r = symbols('p q r', cls=Dummy)\n    >>> wicks(Fd(p)*(F(q)+F(r)), keep_only_fully_contracted=True)\n    KroneckerDelta(_i, _q)*KroneckerDelta(_p, _q) + KroneckerDelta(_i, _r)*KroneckerDelta(_p, _r)\n\n    "
    if not e:
        return S.Zero
    opts = {'simplify_kronecker_deltas': False, 'expand': True, 'simplify_dummies': False, 'keep_only_fully_contracted': False}
    opts.update(kw_args)
    if isinstance(e, NO):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    elif isinstance(e, FermionicOperator):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    e = e.doit(wicks=True)
    e = e.expand()
    if isinstance(e, Add):
        if opts['simplify_dummies']:
            return substitute_dummies(Add(*[wicks(term, **kw_args) for term in e.args]))
        else:
            return Add(*[wicks(term, **kw_args) for term in e.args])
    if isinstance(e, Mul):
        c_part = []
        string1 = []
        for factor in e.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                string1.append(factor)
        n = len(string1)
        if n == 0:
            result = e
        elif n == 1:
            if opts['keep_only_fully_contracted']:
                return S.Zero
            else:
                result = e
        else:
            if isinstance(string1[0], BosonicOperator):
                raise NotImplementedError
            string1 = tuple(string1)
            result = _get_contractions(string1, keep_only_fully_contracted=opts['keep_only_fully_contracted'])
            result = Mul(*c_part) * result
        if opts['expand']:
            result = result.expand()
        if opts['simplify_kronecker_deltas']:
            result = evaluate_deltas(result)
        return result
    return e

class PermutationOperator(Expr):
    """
    Represents the index permutation operator P(ij).

    P(ij)*f(i)*g(j) = f(i)*g(j) - f(j)*g(i)
    """
    is_commutative = True

    def __new__(cls, i, j):
        if False:
            i = 10
            return i + 15
        (i, j) = sorted(map(sympify, (i, j)), key=default_sort_key)
        obj = Basic.__new__(cls, i, j)
        return obj

    def get_permuted(self, expr):
        if False:
            print('Hello World!')
        "\n        Returns -expr with permuted indices.\n\n        Explanation\n        ===========\n\n        >>> from sympy import symbols, Function\n        >>> from sympy.physics.secondquant import PermutationOperator\n        >>> p,q = symbols('p,q')\n        >>> f = Function('f')\n        >>> PermutationOperator(p,q).get_permuted(f(p,q))\n        -f(q, p)\n\n        "
        i = self.args[0]
        j = self.args[1]
        if expr.has(i) and expr.has(j):
            tmp = Dummy()
            expr = expr.subs(i, tmp)
            expr = expr.subs(j, i)
            expr = expr.subs(tmp, j)
            return S.NegativeOne * expr
        else:
            return expr

    def _latex(self, printer):
        if False:
            for i in range(10):
                print('nop')
        return 'P(%s%s)' % self.args

def simplify_index_permutations(expr, permutation_operators):
    if False:
        i = 10
        return i + 15
    "\n    Performs simplification by introducing PermutationOperators where appropriate.\n\n    Explanation\n    ===========\n\n    Schematically:\n        [abij] - [abji] - [baij] + [baji] ->  P(ab)*P(ij)*[abij]\n\n    permutation_operators is a list of PermutationOperators to consider.\n\n    If permutation_operators=[P(ab),P(ij)] we will try to introduce the\n    permutation operators P(ij) and P(ab) in the expression.  If there are other\n    possible simplifications, we ignore them.\n\n    >>> from sympy import symbols, Function\n    >>> from sympy.physics.secondquant import simplify_index_permutations\n    >>> from sympy.physics.secondquant import PermutationOperator\n    >>> p,q,r,s = symbols('p,q,r,s')\n    >>> f = Function('f')\n    >>> g = Function('g')\n\n    >>> expr = f(p)*g(q) - f(q)*g(p); expr\n    f(p)*g(q) - f(q)*g(p)\n    >>> simplify_index_permutations(expr,[PermutationOperator(p,q)])\n    f(p)*g(q)*PermutationOperator(p, q)\n\n    >>> PermutList = [PermutationOperator(p,q),PermutationOperator(r,s)]\n    >>> expr = f(p,r)*g(q,s) - f(q,r)*g(p,s) + f(q,s)*g(p,r) - f(p,s)*g(q,r)\n    >>> simplify_index_permutations(expr,PermutList)\n    f(p, r)*g(q, s)*PermutationOperator(p, q)*PermutationOperator(r, s)\n\n    "

    def _get_indices(expr, ind):
        if False:
            while True:
                i = 10
        '\n        Collects indices recursively in predictable order.\n        '
        result = []
        for arg in expr.args:
            if arg in ind:
                result.append(arg)
            elif arg.args:
                result.extend(_get_indices(arg, ind))
        return result

    def _choose_one_to_keep(a, b, ind):
        if False:
            i = 10
            return i + 15
        return min(a, b, key=lambda x: default_sort_key(_get_indices(x, ind)))
    expr = expr.expand()
    if isinstance(expr, Add):
        terms = set(expr.args)
        for P in permutation_operators:
            new_terms = set()
            on_hold = set()
            while terms:
                term = terms.pop()
                permuted = P.get_permuted(term)
                if permuted in terms | on_hold:
                    try:
                        terms.remove(permuted)
                    except KeyError:
                        on_hold.remove(permuted)
                    keep = _choose_one_to_keep(term, permuted, P.args)
                    new_terms.add(P * keep)
                else:
                    permuted1 = permuted
                    permuted = substitute_dummies(permuted)
                    if permuted1 == permuted:
                        on_hold.add(term)
                    elif permuted in terms | on_hold:
                        try:
                            terms.remove(permuted)
                        except KeyError:
                            on_hold.remove(permuted)
                        keep = _choose_one_to_keep(term, permuted, P.args)
                        new_terms.add(P * keep)
                    else:
                        new_terms.add(term)
            terms = new_terms | on_hold
        return Add(*terms)
    return expr