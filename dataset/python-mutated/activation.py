"""Activation dynamics for musclotendon models.

Musculotendon models are able to produce active force when they are activated,
which is when a chemical process has taken place within the muscle fibers
causing them to voluntarily contract. Biologically this chemical process (the
diffusion of :math:`\\textrm{Ca}^{2+}` ions) is not the input in the system,
electrical signals from the nervous system are. These are termed excitations.
Activation dynamics, which relates the normalized excitation level to the
normalized activation level, can be modeled by the models present in this
module.

"""
from abc import ABC, abstractmethod
from functools import cached_property
from sympy.core.symbol import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices.dense import MutableDenseMatrix as Matrix, zeros
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics import dynamicsymbols
__all__ = ['ActivationBase', 'FirstOrderActivationDeGroote2016', 'ZerothOrderActivation']

class ActivationBase(ABC, _NamedMixin):
    """Abstract base class for all activation dynamics classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom activation dynamics types through
    subclassing.

    """

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        'Initializer for ``ActivationBase``.'
        self.name = str(name)
        self._e = dynamicsymbols(f'e_{name}')
        self._a = dynamicsymbols(f'a_{name}')

    @classmethod
    @abstractmethod
    def with_defaults(cls, name):
        if False:
            while True:
                i = 10
        'Alternate constructor that provides recommended defaults for\n        constants.'
        pass

    @property
    def excitation(self):
        if False:
            print('Hello World!')
        'Dynamic symbol representing excitation.\n\n        Explanation\n        ===========\n\n        The alias ``e`` can also be used to access the same attribute.\n\n        '
        return self._e

    @property
    def e(self):
        if False:
            while True:
                i = 10
        'Dynamic symbol representing excitation.\n\n        Explanation\n        ===========\n\n        The alias ``excitation`` can also be used to access the same attribute.\n\n        '
        return self._e

    @property
    def activation(self):
        if False:
            for i in range(10):
                print('nop')
        'Dynamic symbol representing activation.\n\n        Explanation\n        ===========\n\n        The alias ``a`` can also be used to access the same attribute.\n\n        '
        return self._a

    @property
    def a(self):
        if False:
            i = 10
            return i + 15
        'Dynamic symbol representing activation.\n\n        Explanation\n        ===========\n\n        The alias ``activation`` can also be used to access the same attribute.\n\n        '
        return self._a

    @property
    @abstractmethod
    def order(self):
        if False:
            return 10
        'Order of the (differential) equation governing activation.'
        pass

    @property
    @abstractmethod
    def state_vars(self):
        if False:
            for i in range(10):
                print('nop')
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``x`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def x(self):
        if False:
            return 10
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``state_vars`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def input_vars(self):
        if False:
            for i in range(10):
                print('nop')
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``r`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def r(self):
        if False:
            i = 10
            return i + 15
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``input_vars`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def constants(self):
        if False:
            print('Hello World!')
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        Explanation\n        ===========\n\n        The alias ``p`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def p(self):
        if False:
            while True:
                i = 10
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        Explanation\n        ===========\n\n        The alias ``constants`` can also be used to access the same attribute.\n\n        '
        pass

    @property
    @abstractmethod
    def M(self):
        if False:
            while True:
                i = 10
        "Ordered square matrix of coefficients on the LHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The square matrix that forms part of the LHS of the linear system of\n        ordinary differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        pass

    @property
    @abstractmethod
    def F(self):
        if False:
            return 10
        "Ordered column matrix of equations on the RHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The column matrix that forms the RHS of the linear system of ordinary\n        differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        pass

    @abstractmethod
    def rhs(self):
        if False:
            while True:
                i = 10
        "\n\n        Explanation\n        ===========\n\n        The solution to the linear system of ordinary differential equations\n        governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        pass

    def __eq__(self, other):
        if False:
            return 10
        'Equality check for activation dynamics.'
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        return True

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Default representation of activation dynamics.'
        return f'{self.__class__.__name__}({self.name!r})'

class ZerothOrderActivation(ActivationBase):
    """Simple zeroth-order activation dynamics mapping excitation to
    activation.

    Explanation
    ===========

    Zeroth-order activation dynamics are useful in instances where you want to
    reduce the complexity of your musculotendon dynamics as they simple map
    exictation to activation. As a result, no additional state equations are
    introduced to your system. They also remove a potential source of delay
    between the input and dynamics of your system as no (ordinary) differential
    equations are involed.

    """

    def __init__(self, name):
        if False:
            return 10
        'Initializer for ``ZerothOrderActivation``.\n\n        Parameters\n        ==========\n\n        name : str\n            The name identifier associated with the instance. Must be a string\n            of length at least 1.\n\n        '
        super().__init__(name)
        self._a = self._e

    @classmethod
    def with_defaults(cls, name):
        if False:
            while True:
                i = 10
        "Alternate constructor that provides recommended defaults for\n        constants.\n\n        Explanation\n        ===========\n\n        As this concrete class doesn't implement any constants associated with\n        its dynamics, this ``classmethod`` simply creates a standard instance\n        of ``ZerothOrderActivation``. An implementation is provided to ensure\n        a consistent interface between all ``ActivationBase`` concrete classes.\n\n        "
        return cls(name)

    @property
    def order(self):
        if False:
            i = 10
            return i + 15
        'Order of the (differential) equation governing activation.'
        return 0

    @property
    def state_vars(self):
        if False:
            i = 10
            return i + 15
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        As zeroth-order activation dynamics simply maps excitation to\n        activation, this class has no associated state variables and so this\n        property return an empty column ``Matrix`` with shape (0, 1).\n\n        The alias ``x`` can also be used to access the same attribute.\n\n        '
        return zeros(0, 1)

    @property
    def x(self):
        if False:
            return 10
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        As zeroth-order activation dynamics simply maps excitation to\n        activation, this class has no associated state variables and so this\n        property return an empty column ``Matrix`` with shape (0, 1).\n\n        The alias ``state_vars`` can also be used to access the same attribute.\n\n        '
        return zeros(0, 1)

    @property
    def input_vars(self):
        if False:
            while True:
                i = 10
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        Excitation is the only input in zeroth-order activation dynamics and so\n        this property returns a column ``Matrix`` with one entry, ``e``, and\n        shape (1, 1).\n\n        The alias ``r`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._e])

    @property
    def r(self):
        if False:
            i = 10
            return i + 15
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        Excitation is the only input in zeroth-order activation dynamics and so\n        this property returns a column ``Matrix`` with one entry, ``e``, and\n        shape (1, 1).\n\n        The alias ``input_vars`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._e])

    @property
    def constants(self):
        if False:
            while True:
                i = 10
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        Explanation\n        ===========\n\n        As zeroth-order activation dynamics simply maps excitation to\n        activation, this class has no associated constants and so this property\n        return an empty column ``Matrix`` with shape (0, 1).\n\n        The alias ``p`` can also be used to access the same attribute.\n\n        '
        return zeros(0, 1)

    @property
    def p(self):
        if False:
            print('Hello World!')
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        Explanation\n        ===========\n\n        As zeroth-order activation dynamics simply maps excitation to\n        activation, this class has no associated constants and so this property\n        return an empty column ``Matrix`` with shape (0, 1).\n\n        The alias ``constants`` can also be used to access the same attribute.\n\n        '
        return zeros(0, 1)

    @property
    def M(self):
        if False:
            return 10
        "Ordered square matrix of coefficients on the LHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The square matrix that forms part of the LHS of the linear system of\n        ordinary differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        As zeroth-order activation dynamics have no state variables, this\n        linear system has dimension 0 and therefore ``M`` is an empty square\n        ``Matrix`` with shape (0, 0).\n\n        "
        return Matrix([])

    @property
    def F(self):
        if False:
            for i in range(10):
                print('nop')
        "Ordered column matrix of equations on the RHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The column matrix that forms the RHS of the linear system of ordinary\n        differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        As zeroth-order activation dynamics have no state variables, this\n        linear system has dimension 0 and therefore ``F`` is an empty column\n        ``Matrix`` with shape (0, 1).\n\n        "
        return zeros(0, 1)

    def rhs(self):
        if False:
            return 10
        "Ordered column matrix of equations for the solution of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The solution to the linear system of ordinary differential equations\n        governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        As zeroth-order activation dynamics have no state variables, this\n        linear has dimension 0 and therefore this method returns an empty\n        column ``Matrix`` with shape (0, 1).\n\n        "
        return zeros(0, 1)

class FirstOrderActivationDeGroote2016(ActivationBase):
    """First-order activation dynamics based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the first-order activation dynamics equation for the rate of change
    of activation with respect to time as a function of excitation and
    activation.

    The function is defined by the equation:

    .. math::

        \\frac{da}{dt} = \\left(\\frac{\\frac{1}{2} + a0}{\\tau_a \\left(\\frac{1}{2}
            + \\frac{3a}{2}\\right)} + \\frac{\\left(\\frac{1}{2}
            + \\frac{3a}{2}\\right) \\left(\\frac{1}{2} - a0\\right)}{\\tau_d}\\right)
            \\left(e - a\\right)

    where

    .. math::

        a0 = \\frac{\\tanh{\\left(b \\left(e - a\\right) \\right)}}{2}

    with constant values of :math:`tau_a = 0.015`, :math:`tau_d = 0.060`, and
    :math:`b = 10`.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    def __init__(self, name, activation_time_constant=None, deactivation_time_constant=None, smoothing_rate=None):
        if False:
            print('Hello World!')
        'Initializer for ``FirstOrderActivationDeGroote2016``.\n\n        Parameters\n        ==========\n        activation time constant : Symbol | Number | None\n            The value of the activation time constant governing the delay\n            between excitation and activation when excitation exceeds\n            activation.\n        deactivation time constant : Symbol | Number | None\n            The value of the deactivation time constant governing the delay\n            between excitation and activation when activation exceeds\n            excitation.\n        smoothing_rate : Symbol | Number | None\n            The slope of the hyperbolic tangent function used to smooth between\n            the switching of the equations where excitation exceed activation\n            and where activation exceeds excitation. The recommended value to\n            use is ``10``, but values between ``0.1`` and ``100`` can be used.\n\n        '
        super().__init__(name)
        self.activation_time_constant = activation_time_constant
        self.deactivation_time_constant = deactivation_time_constant
        self.smoothing_rate = smoothing_rate

    @classmethod
    def with_defaults(cls, name):
        if False:
            while True:
                i = 10
        'Alternate constructor that will use the published constants.\n\n        Explanation\n        ===========\n\n        Returns an instance of ``FirstOrderActivationDeGroote2016`` using the\n        three constant values specified in the original publication.\n\n        These have the values:\n\n        :math:`tau_a = 0.015`\n        :math:`tau_d = 0.060`\n        :math:`b = 10`\n\n        '
        tau_a = Float('0.015')
        tau_d = Float('0.060')
        b = Float('10.0')
        return cls(name, tau_a, tau_d, b)

    @property
    def activation_time_constant(self):
        if False:
            for i in range(10):
                print('nop')
        'Delay constant for activation.\n\n        Explanation\n        ===========\n\n        The alias ```tau_a`` can also be used to access the same attribute.\n\n        '
        return self._tau_a

    @activation_time_constant.setter
    def activation_time_constant(self, tau_a):
        if False:
            while True:
                i = 10
        if hasattr(self, '_tau_a'):
            msg = f"Can't set attribute `activation_time_constant` to {repr(tau_a)} as it is immutable and already has value {self._tau_a}."
            raise AttributeError(msg)
        self._tau_a = Symbol(f'tau_a_{self.name}') if tau_a is None else tau_a

    @property
    def tau_a(self):
        if False:
            return 10
        'Delay constant for activation.\n\n        Explanation\n        ===========\n\n        The alias ``activation_time_constant`` can also be used to access the\n        same attribute.\n\n        '
        return self._tau_a

    @property
    def deactivation_time_constant(self):
        if False:
            return 10
        'Delay constant for deactivation.\n\n        Explanation\n        ===========\n\n        The alias ``tau_d`` can also be used to access the same attribute.\n\n        '
        return self._tau_d

    @deactivation_time_constant.setter
    def deactivation_time_constant(self, tau_d):
        if False:
            print('Hello World!')
        if hasattr(self, '_tau_d'):
            msg = f"Can't set attribute `deactivation_time_constant` to {repr(tau_d)} as it is immutable and already has value {self._tau_d}."
            raise AttributeError(msg)
        self._tau_d = Symbol(f'tau_d_{self.name}') if tau_d is None else tau_d

    @property
    def tau_d(self):
        if False:
            print('Hello World!')
        'Delay constant for deactivation.\n\n        Explanation\n        ===========\n\n        The alias ``deactivation_time_constant`` can also be used to access the\n        same attribute.\n\n        '
        return self._tau_d

    @property
    def smoothing_rate(self):
        if False:
            return 10
        'Smoothing constant for the hyperbolic tangent term.\n\n        Explanation\n        ===========\n\n        The alias ``b`` can also be used to access the same attribute.\n\n        '
        return self._b

    @smoothing_rate.setter
    def smoothing_rate(self, b):
        if False:
            print('Hello World!')
        if hasattr(self, '_b'):
            msg = f"Can't set attribute `smoothing_rate` to {b!r} as it is immutable and already has value {self._b!r}."
            raise AttributeError(msg)
        self._b = Symbol(f'b_{self.name}') if b is None else b

    @property
    def b(self):
        if False:
            return 10
        'Smoothing constant for the hyperbolic tangent term.\n\n        Explanation\n        ===========\n\n        The alias ``smoothing_rate`` can also be used to access the same\n        attribute.\n\n        '
        return self._b

    @property
    def order(self):
        if False:
            while True:
                i = 10
        'Order of the (differential) equation governing activation.'
        return 1

    @property
    def state_vars(self):
        if False:
            print('Hello World!')
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``x`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._a])

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        'Ordered column matrix of functions of time that represent the state\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``state_vars`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._a])

    @property
    def input_vars(self):
        if False:
            while True:
                i = 10
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``r`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._e])

    @property
    def r(self):
        if False:
            return 10
        'Ordered column matrix of functions of time that represent the input\n        variables.\n\n        Explanation\n        ===========\n\n        The alias ``input_vars`` can also be used to access the same attribute.\n\n        '
        return Matrix([self._e])

    @property
    def constants(self):
        if False:
            while True:
                i = 10
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        Explanation\n        ===========\n\n        The alias ``p`` can also be used to access the same attribute.\n\n        '
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def p(self):
        if False:
            for i in range(10):
                print('nop')
        'Ordered column matrix of non-time varying symbols present in ``M``\n        and ``F``.\n\n        Explanation\n        ===========\n\n        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)\n        has been used instead of ``Symbol`` for a constant then that attribute\n        will not be included in the matrix returned by this property. This is\n        because the primary use of this property attribute is to provide an\n        ordered sequence of the still-free symbols that require numeric values\n        during code generation.\n\n        The alias ``constants`` can also be used to access the same attribute.\n\n        '
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def M(self):
        if False:
            for i in range(10):
                print('nop')
        "Ordered square matrix of coefficients on the LHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The square matrix that forms part of the LHS of the linear system of\n        ordinary differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        return Matrix([Integer(1)])

    @property
    def F(self):
        if False:
            while True:
                i = 10
        "Ordered column matrix of equations on the RHS of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The column matrix that forms the RHS of the linear system of ordinary\n        differential equations governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        return Matrix([self._da_eqn])

    def rhs(self):
        if False:
            i = 10
            return i + 15
        "Ordered column matrix of equations for the solution of ``M x' = F``.\n\n        Explanation\n        ===========\n\n        The solution to the linear system of ordinary differential equations\n        governing the activation dynamics:\n\n        ``M(x, r, t, p) x' = F(x, r, t, p)``.\n\n        "
        return Matrix([self._da_eqn])

    @cached_property
    def _da_eqn(self):
        if False:
            return 10
        HALF = Rational(1, 2)
        a0 = HALF * tanh(self._b * (self._e - self._a))
        a1 = HALF + Rational(3, 2) * self._a
        a2 = (HALF + a0) / (self._tau_a * a1)
        a3 = a1 * (HALF - a0) / self._tau_d
        activation_dynamics_equation = (a2 + a3) * (self._e - self._a)
        return activation_dynamics_equation

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Equality check for ``FirstOrderActivationDeGroote2016``.'
        if type(self) != type(other):
            return False
        self_attrs = (self.name, self.tau_a, self.tau_d, self.b)
        other_attrs = (other.name, other.tau_a, other.tau_d, other.b)
        if self_attrs == other_attrs:
            return True
        return False

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Representation of ``FirstOrderActivationDeGroote2016``.'
        return f'{self.__class__.__name__}({self.name!r}, activation_time_constant={self.tau_a!r}, deactivation_time_constant={self.tau_d!r}, smoothing_rate={self.b!r})'