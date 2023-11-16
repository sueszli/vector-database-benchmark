"""Fermionic quantum operators."""
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
__all__ = ['FermionOp', 'FermionFockKet', 'FermionFockBra']

class FermionOp(Operator):
    """A fermionic operator that satisfies {c, Dagger(c)} == 1.

    Parameters
    ==========

    name : str
        A string that labels the fermionic mode.

    annihilation : bool
        A bool that indicates if the fermionic operator is an annihilation
        (True, default value) or creation operator (False)

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, AntiCommutator
    >>> from sympy.physics.quantum.fermion import FermionOp
    >>> c = FermionOp("c")
    >>> AntiCommutator(c, Dagger(c)).doit()
    1
    """

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.args[0]

    @property
    def is_annihilation(self):
        if False:
            while True:
                i = 10
        return bool(self.args[1])

    @classmethod
    def default_args(self):
        if False:
            return 10
        return ('c', True)

    def __new__(cls, *args, **hints):
        if False:
            return 10
        if not len(args) in [1, 2]:
            raise ValueError('1 or 2 parameters expected, got %s' % args)
        if len(args) == 1:
            args = (args[0], S.One)
        if len(args) == 2:
            args = (args[0], Integer(args[1]))
        return Operator.__new__(cls, *args)

    def _eval_commutator_FermionOp(self, other, **hints):
        if False:
            return 10
        if 'independent' in hints and hints['independent']:
            return S.Zero
        return None

    def _eval_anticommutator_FermionOp(self, other, **hints):
        if False:
            for i in range(10):
                print('nop')
        if self.name == other.name:
            if not self.is_annihilation and other.is_annihilation:
                return S.One
        elif 'independent' in hints and hints['independent']:
            return 2 * self * other
        return None

    def _eval_anticommutator_BosonOp(self, other, **hints):
        if False:
            i = 10
            return i + 15
        return 2 * self * other

    def _eval_commutator_BosonOp(self, other, **hints):
        if False:
            while True:
                i = 10
        return S.Zero

    def _eval_adjoint(self):
        if False:
            print('Hello World!')
        return FermionOp(str(self.name), not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):
        if False:
            while True:
                i = 10
        if self.is_annihilation:
            return '{%s}' % str(self.name)
        else:
            return '{{%s}^\\dagger}' % str(self.name)

    def _print_contents(self, printer, *args):
        if False:
            return 10
        if self.is_annihilation:
            return '%s' % str(self.name)
        else:
            return 'Dagger(%s)' % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        if False:
            for i in range(10):
                print('nop')
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        else:
            return pform ** prettyForm('†')

    def _eval_power(self, exp):
        if False:
            return 10
        from sympy.core.singleton import S
        if exp == 0:
            return S.One
        elif exp == 1:
            return self
        elif (exp > 1) == True and exp.is_integer == True:
            return S.Zero
        elif (exp < 0) == True or exp.is_integer == False:
            raise ValueError('Fermionic operators can only be raised to a positive integer power')
        return Operator._eval_power(self, exp)

class FermionFockKet(Ket):
    """Fock state ket for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if False:
            for i in range(10):
                print('nop')
        if n not in (0, 1):
            raise ValueError('n must be 0 or 1')
        return Ket.__new__(cls, n)

    @property
    def n(self):
        if False:
            while True:
                i = 10
        return self.label[0]

    @classmethod
    def dual_class(self):
        if False:
            for i in range(10):
                print('nop')
        return FermionFockBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        if False:
            print('Hello World!')
        return HilbertSpace()

    def _eval_innerproduct_FermionFockBra(self, bra, **hints):
        if False:
            for i in range(10):
                print('nop')
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_FermionOp(self, op, **options):
        if False:
            i = 10
            return i + 15
        if op.is_annihilation:
            if self.n == 1:
                return FermionFockKet(0)
            else:
                return S.Zero
        elif self.n == 0:
            return FermionFockKet(1)
        else:
            return S.Zero

class FermionFockBra(Bra):
    """Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    """

    def __new__(cls, n):
        if False:
            i = 10
            return i + 15
        if n not in (0, 1):
            raise ValueError('n must be 0 or 1')
        return Bra.__new__(cls, n)

    @property
    def n(self):
        if False:
            print('Hello World!')
        return self.label[0]

    @classmethod
    def dual_class(self):
        if False:
            while True:
                i = 10
        return FermionFockKet