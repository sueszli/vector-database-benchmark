"""
**Contains**

* Medium
"""
from sympy.physics.units import second, meter, kilogram, ampere
__all__ = ['Medium']
from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import speed_of_light, u0, e0
c = speed_of_light.convert_to(meter / second)
_e0mksa = e0.convert_to(ampere ** 2 * second ** 4 / (kilogram * meter ** 3))
_u0mksa = u0.convert_to(meter * kilogram / (ampere ** 2 * second ** 2))

class Medium(Basic):
    """
    This class represents an optical medium. The prime reason to implement this is
    to facilitate refraction, Fermat's principle, etc.

    Explanation
    ===========

    An optical medium is a material through which electromagnetic waves propagate.
    The permittivity and permeability of the medium define how electromagnetic
    waves propagate in it.


    Parameters
    ==========

    name: string
        The display name of the Medium.

    permittivity: Sympifyable
        Electric permittivity of the space.

    permeability: Sympifyable
        Magnetic permeability of the space.

    n: Sympifyable
        Index of refraction of the medium.


    Examples
    ========

    >>> from sympy.abc import epsilon, mu
    >>> from sympy.physics.optics import Medium
    >>> m1 = Medium('m1')
    >>> m2 = Medium('m2', epsilon, mu)
    >>> m1.intrinsic_impedance
    149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)
    >>> m2.refractive_index
    299792458*meter*sqrt(epsilon*mu)/second


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Optical_medium

    """

    def __new__(cls, name, permittivity=None, permeability=None, n=None):
        if False:
            print('Hello World!')
        if not isinstance(name, Str):
            name = Str(name)
        permittivity = _sympify(permittivity) if permittivity is not None else permittivity
        permeability = _sympify(permeability) if permeability is not None else permeability
        n = _sympify(n) if n is not None else n
        if n is not None:
            if permittivity is not None and permeability is None:
                permeability = n ** 2 / (c ** 2 * permittivity)
                return MediumPP(name, permittivity, permeability)
            elif permeability is not None and permittivity is None:
                permittivity = n ** 2 / (c ** 2 * permeability)
                return MediumPP(name, permittivity, permeability)
            elif permittivity is not None and permittivity is not None:
                raise ValueError('Specifying all of permittivity, permeability, and n is not allowed')
            else:
                return MediumN(name, n)
        elif permittivity is not None and permeability is not None:
            return MediumPP(name, permittivity, permeability)
        elif permittivity is None and permeability is None:
            return MediumPP(name, _e0mksa, _u0mksa)
        else:
            raise ValueError('Arguments are underspecified. Either specify n or any two of permittivity, permeability, and n')

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    @property
    def speed(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns speed of the electromagnetic wave travelling in the medium.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import Medium\n        >>> m = Medium('m')\n        >>> m.speed\n        299792458*meter/second\n        >>> m2 = Medium('m2', n=1)\n        >>> m.speed == m2.speed\n        True\n\n        "
        return c / self.n

    @property
    def refractive_index(self):
        if False:
            print('Hello World!')
        "\n        Returns refractive index of the medium.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import Medium\n        >>> m = Medium('m')\n        >>> m.refractive_index\n        1\n\n        "
        return c / self.speed

class MediumN(Medium):
    """
    Represents an optical medium for which only the refractive index is known.
    Useful for simple ray optics.

    This class should never be instantiated directly.
    Instead it should be instantiated indirectly by instantiating Medium with
    only n specified.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> m = Medium('m', n=2)
    >>> m
    MediumN(Str('m'), 2)
    """

    def __new__(cls, name, n):
        if False:
            print('Hello World!')
        obj = super(Medium, cls).__new__(cls, name, n)
        return obj

    @property
    def n(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[1]

class MediumPP(Medium):
    """
    Represents an optical medium for which the permittivity and permeability are known.

    This class should never be instantiated directly. Instead it should be
    instantiated indirectly by instantiating Medium with any two of
    permittivity, permeability, and n specified, or by not specifying any
    of permittivity, permeability, or n, in which case default values for
    permittivity and permeability will be used.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> from sympy.abc import epsilon, mu
    >>> m1 = Medium('m1', permittivity=epsilon, permeability=mu)
    >>> m1
    MediumPP(Str('m1'), epsilon, mu)
    >>> m2 = Medium('m2')
    >>> m2
    MediumPP(Str('m2'), 625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3), pi*kilogram*meter/(2500000*ampere**2*second**2))
    """

    def __new__(cls, name, permittivity, permeability):
        if False:
            while True:
                i = 10
        obj = super(Medium, cls).__new__(cls, name, permittivity, permeability)
        return obj

    @property
    def intrinsic_impedance(self):
        if False:
            return 10
        "\n        Returns intrinsic impedance of the medium.\n\n        Explanation\n        ===========\n\n        The intrinsic impedance of a medium is the ratio of the\n        transverse components of the electric and magnetic fields\n        of the electromagnetic wave travelling in the medium.\n        In a region with no electrical conductivity it simplifies\n        to the square root of ratio of magnetic permeability to\n        electric permittivity.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import Medium\n        >>> m = Medium('m')\n        >>> m.intrinsic_impedance\n        149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)\n\n        "
        return sqrt(self.permeability / self.permittivity)

    @property
    def permittivity(self):
        if False:
            return 10
        "\n        Returns electric permittivity of the medium.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import Medium\n        >>> m = Medium('m')\n        >>> m.permittivity\n        625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3)\n\n        "
        return self.args[1]

    @property
    def permeability(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns magnetic permeability of the medium.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import Medium\n        >>> m = Medium('m')\n        >>> m.permeability\n        pi*kilogram*meter/(2500000*ampere**2*second**2)\n\n        "
        return self.args[2]

    @property
    def n(self):
        if False:
            print('Hello World!')
        return c * sqrt(self.permittivity * self.permeability)