from abc import ABC, abstractmethod
from sympy import Symbol, sympify
from sympy.physics.vector import Point
__all__ = ['BodyBase']

class BodyBase(ABC):
    """Abstract class for body type objects."""

    def __init__(self, name, masscenter=None, mass=None):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        if mass is None:
            mass = Symbol(f'{name}_mass')
        if masscenter is None:
            masscenter = Point(f'{name}_masscenter')
        self.mass = mass
        self.masscenter = masscenter
        self.potential_energy = 0
        self.points = []

    def __str__(self):
        if False:
            return 10
        return self.name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}({repr(self.name)}, masscenter={repr(self.masscenter)}, mass={repr(self.mass)})'

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of the body.'
        return self._name

    @property
    def masscenter(self):
        if False:
            while True:
                i = 10
        "The body's center of mass."
        return self._masscenter

    @masscenter.setter
    def masscenter(self, point):
        if False:
            i = 10
            return i + 15
        if not isinstance(point, Point):
            raise TypeError("The body's center of mass must be a Point object.")
        self._masscenter = point

    @property
    def mass(self):
        if False:
            print('Hello World!')
        "The body's mass."
        return self._mass

    @mass.setter
    def mass(self, mass):
        if False:
            return 10
        self._mass = sympify(mass)

    @property
    def potential_energy(self):
        if False:
            i = 10
            return i + 15
        "The potential energy of the body.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.mechanics import Particle, Point\n        >>> from sympy import symbols\n        >>> m, g, h = symbols('m g h')\n        >>> O = Point('O')\n        >>> P = Particle('P', O, m)\n        >>> P.potential_energy = m * g * h\n        >>> P.potential_energy\n        g*h*m\n\n        "
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, scalar):
        if False:
            return 10
        self._potential_energy = sympify(scalar)

    @abstractmethod
    def kinetic_energy(self, frame):
        if False:
            return 10
        pass

    @abstractmethod
    def linear_momentum(self, frame):
        if False:
            return 10
        pass

    @abstractmethod
    def angular_momentum(self, point, frame):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def parallel_axis(self, point, frame):
        if False:
            print('Hello World!')
        pass