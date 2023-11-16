"""
Gaussian optics.

The module implements:

- Ray transfer matrices for geometrical and gaussian optics.

  See RayTransferMatrix, GeometricRay and BeamParameter

- Conjugation relations for geometrical and gaussian optics.

  See geometric_conj*, gauss_conj and conjugate_gauss_beams

The conventions for the distances are as follows:

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""
__all__ = ['RayTransferMatrix', 'FreeSpace', 'FlatRefraction', 'CurvedRefraction', 'FlatMirror', 'CurvedMirror', 'ThinLens', 'GeometricRay', 'BeamParameter', 'waist2rayleigh', 'rayleigh2waist', 'geometric_conj_ab', 'geometric_conj_af', 'geometric_conj_bf', 'gaussian_conj', 'conjugate_gauss_beams']
from sympy.core.expr import Expr
from sympy.core.numbers import I, pi
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent

class RayTransferMatrix(MutableDenseMatrix):
    """
    Base class for a Ray Transfer Matrix.

    It should be used if there is not already a more specific subclass mentioned
    in See Also.

    Parameters
    ==========

    parameters :
        A, B, C and D or 2x2 matrix (Matrix(2, 2, [A, B, C, D]))

    Examples
    ========

    >>> from sympy.physics.optics import RayTransferMatrix, ThinLens
    >>> from sympy import Symbol, Matrix

    >>> mat = RayTransferMatrix(1, 2, 3, 4)
    >>> mat
    Matrix([
    [1, 2],
    [3, 4]])

    >>> RayTransferMatrix(Matrix([[1, 2], [3, 4]]))
    Matrix([
    [1, 2],
    [3, 4]])

    >>> mat.A
    1

    >>> f = Symbol('f')
    >>> lens = ThinLens(f)
    >>> lens
    Matrix([
    [   1, 0],
    [-1/f, 1]])

    >>> lens.C
    -1/f

    See Also
    ========

    GeometricRay, BeamParameter,
    FreeSpace, FlatRefraction, CurvedRefraction,
    FlatMirror, CurvedMirror, ThinLens

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    def __new__(cls, *args):
        if False:
            return 10
        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 and isinstance(args[0], Matrix) and (args[0].shape == (2, 2)):
            temp = args[0]
        else:
            raise ValueError(filldedent('\n                Expecting 2x2 Matrix or the 4 elements of\n                the Matrix but got %s' % str(args)))
        return Matrix.__new__(cls, temp)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(Matrix.__mul__(self, other))
        elif isinstance(other, GeometricRay):
            return GeometricRay(Matrix.__mul__(self, other))
        elif isinstance(other, BeamParameter):
            temp = self * Matrix(((other.q,), (1,)))
            q = (temp[0] / temp[1]).expand(complex=True)
            return BeamParameter(other.wavelen, together(re(q)), z_r=together(im(q)))
        else:
            return Matrix.__mul__(self, other)

    @property
    def A(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The A parameter of the Matrix.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import RayTransferMatrix\n        >>> mat = RayTransferMatrix(1, 2, 3, 4)\n        >>> mat.A\n        1\n        '
        return self[0, 0]

    @property
    def B(self):
        if False:
            while True:
                i = 10
        '\n        The B parameter of the Matrix.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import RayTransferMatrix\n        >>> mat = RayTransferMatrix(1, 2, 3, 4)\n        >>> mat.B\n        2\n        '
        return self[0, 1]

    @property
    def C(self):
        if False:
            return 10
        '\n        The C parameter of the Matrix.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import RayTransferMatrix\n        >>> mat = RayTransferMatrix(1, 2, 3, 4)\n        >>> mat.C\n        3\n        '
        return self[1, 0]

    @property
    def D(self):
        if False:
            i = 10
            return i + 15
        '\n        The D parameter of the Matrix.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import RayTransferMatrix\n        >>> mat = RayTransferMatrix(1, 2, 3, 4)\n        >>> mat.D\n        4\n        '
        return self[1, 1]

class FreeSpace(RayTransferMatrix):
    """
    Ray Transfer Matrix for free space.

    Parameters
    ==========

    distance

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FreeSpace
    >>> from sympy import symbols
    >>> d = symbols('d')
    >>> FreeSpace(d)
    Matrix([
    [1, d],
    [0, 1]])
    """

    def __new__(cls, d):
        if False:
            print('Hello World!')
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)

class FlatRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction.

    Parameters
    ==========

    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatRefraction
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1 n2')
    >>> FlatRefraction(n1, n2)
    Matrix([
    [1,     0],
    [0, n1/n2]])
    """

    def __new__(cls, n1, n2):
        if False:
            i = 10
            return i + 15
        (n1, n2) = map(sympify, (n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, 0, n1 / n2)

class CurvedRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction on curved interface.

    Parameters
    ==========

    R :
        Radius of curvature (positive for concave).
    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedRefraction
    >>> from sympy import symbols
    >>> R, n1, n2 = symbols('R n1 n2')
    >>> CurvedRefraction(R, n1, n2)
    Matrix([
    [               1,     0],
    [(n1 - n2)/(R*n2), n1/n2]])
    """

    def __new__(cls, R, n1, n2):
        if False:
            print('Hello World!')
        (R, n1, n2) = map(sympify, (R, n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2) / R / n2, n1 / n2)

class FlatMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatMirror
    >>> FlatMirror()
    Matrix([
    [1, 0],
    [0, 1]])
    """

    def __new__(cls):
        if False:
            print('Hello World!')
        return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)

class CurvedMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection from curved surface.

    Parameters
    ==========

    R : radius of curvature (positive for concave)

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedMirror
    >>> from sympy import symbols
    >>> R = symbols('R')
    >>> CurvedMirror(R)
    Matrix([
    [   1, 0],
    [-2/R, 1]])
    """

    def __new__(cls, R):
        if False:
            i = 10
            return i + 15
        R = sympify(R)
        return RayTransferMatrix.__new__(cls, 1, 0, -2 / R, 1)

class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f :
        The focal distance.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    >>> f = symbols('f')
    >>> ThinLens(f)
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """

    def __new__(cls, f):
        if False:
            return 10
        f = sympify(f)
        return RayTransferMatrix.__new__(cls, 1, 0, -1 / f, 1)

class GeometricRay(MutableDenseMatrix):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        if len(args) == 1 and isinstance(args[0], Matrix) and (args[0].shape == (2, 1)):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            raise ValueError(filldedent('\n                Expecting 2x1 Matrix or the 2 elements of\n                the Matrix but got %s' % str(args)))
        return Matrix.__new__(cls, temp)

    @property
    def height(self):
        if False:
            print('Hello World!')
        "\n        The distance from the optical axis.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import GeometricRay\n        >>> from sympy import symbols\n        >>> h, angle = symbols('h, angle')\n        >>> gRay = GeometricRay(h, angle)\n        >>> gRay.height\n        h\n        "
        return self[0]

    @property
    def angle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The angle with the optical axis.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import GeometricRay\n        >>> from sympy import symbols\n        >>> h, angle = symbols('h, angle')\n        >>> gRay = GeometricRay(h, angle)\n        >>> gRay.angle\n        angle\n        "
        return self[1]

class BeamParameter(Expr):
    """
    Representation for a gaussian ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range.
    n : the refractive index of medium.

    Examples
    ========

    >>> from sympy.physics.optics import BeamParameter
    >>> p = BeamParameter(530e-9, 1, w=1e-3)
    >>> p.q
    1 + 1.88679245283019*I*pi

    >>> p.q.n()
    1.0 + 5.92753330865999*I
    >>> p.w_0.n()
    0.00100000000000000
    >>> p.z_r.n()
    5.92753330865999

    >>> from sympy.physics.optics import FreeSpace
    >>> fs = FreeSpace(10)
    >>> p1 = fs*p
    >>> p.w.n()
    0.00101413072159615
    >>> p1.w.n()
    0.00210803120913829

    See Also
    ========

    RayTransferMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_beam_parameter
    .. [2] https://en.wikipedia.org/wiki/Gaussian_beam
    """

    def __new__(cls, wavelen, z, z_r=None, w=None, n=1):
        if False:
            print('Hello World!')
        wavelen = sympify(wavelen)
        z = sympify(z)
        n = sympify(n)
        if z_r is not None and w is None:
            z_r = sympify(z_r)
        elif w is not None and z_r is None:
            z_r = waist2rayleigh(sympify(w), wavelen, n)
        elif z_r is None and w is None:
            raise ValueError('Must specify one of w and z_r.')
        return Expr.__new__(cls, wavelen, z, z_r, n)

    @property
    def wavelen(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    @property
    def z(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[1]

    @property
    def z_r(self):
        if False:
            i = 10
            return i + 15
        return self.args[2]

    @property
    def n(self):
        if False:
            i = 10
            return i + 15
        return self.args[3]

    @property
    def q(self):
        if False:
            while True:
                i = 10
        '\n        The complex parameter representing the beam.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.q\n        1 + 1.88679245283019*I*pi\n        '
        return self.z + I * self.z_r

    @property
    def radius(self):
        if False:
            print('Hello World!')
        '\n        The radius of curvature of the phase front.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.radius\n        1 + 3.55998576005696*pi**2\n        '
        return self.z * (1 + (self.z_r / self.z) ** 2)

    @property
    def w(self):
        if False:
            i = 10
            return i + 15
        '\n        The radius of the beam w(z), at any position z along the beam.\n        The beam radius at `1/e^2` intensity (axial value).\n\n        See Also\n        ========\n\n        w_0 :\n            The minimal radius of beam.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.w\n        0.001*sqrt(0.2809/pi**2 + 1)\n        '
        return self.w_0 * sqrt(1 + (self.z / self.z_r) ** 2)

    @property
    def w_0(self):
        if False:
            while True:
                i = 10
        '\n         The minimal radius of beam at `1/e^2` intensity (peak value).\n\n        See Also\n        ========\n\n        w : the beam radius at `1/e^2` intensity (axial value).\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.w_0\n        0.00100000000000000\n        '
        return sqrt(self.z_r / (pi * self.n) * self.wavelen)

    @property
    def divergence(self):
        if False:
            print('Hello World!')
        '\n        Half of the total angular spread.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.divergence\n        0.00053/pi\n        '
        return self.wavelen / pi / self.w_0

    @property
    def gouy(self):
        if False:
            return 10
        '\n        The Gouy phase.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.gouy\n        atan(0.53/pi)\n        '
        return atan2(self.z, self.z_r)

    @property
    def waist_approximation_limit(self):
        if False:
            while True:
                i = 10
        '\n        The minimal waist for which the gauss beam approximation is valid.\n\n        Explanation\n        ===========\n\n        The gauss beam is a solution to the paraxial equation. For curvatures\n        that are too great it is not a valid approximation.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.optics import BeamParameter\n        >>> p = BeamParameter(530e-9, 1, w=1e-3)\n        >>> p.waist_approximation_limit\n        1.06e-6/pi\n        '
        return 2 * self.wavelen / pi

def waist2rayleigh(w, wavelen, n=1):
    if False:
        while True:
            i = 10
    "\n    Calculate the rayleigh range from the waist of a gaussian beam.\n\n    See Also\n    ========\n\n    rayleigh2waist, BeamParameter\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics import waist2rayleigh\n    >>> from sympy import symbols\n    >>> w, wavelen = symbols('w wavelen')\n    >>> waist2rayleigh(w, wavelen)\n    pi*w**2/wavelen\n    "
    (w, wavelen) = map(sympify, (w, wavelen))
    return w ** 2 * n * pi / wavelen

def rayleigh2waist(z_r, wavelen):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the waist from the rayleigh range of a gaussian beam.\n\n    See Also\n    ========\n\n    waist2rayleigh, BeamParameter\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics import rayleigh2waist\n    >>> from sympy import symbols\n    >>> z_r, wavelen = symbols('z_r wavelen')\n    >>> rayleigh2waist(z_r, wavelen)\n    sqrt(wavelen*z_r)/sqrt(pi)\n    "
    (z_r, wavelen) = map(sympify, (z_r, wavelen))
    return sqrt(z_r / pi * wavelen)

def geometric_conj_ab(a, b):
    if False:
        i = 10
        return i + 15
    "\n    Conjugation relation for geometrical beams under paraxial conditions.\n\n    Explanation\n    ===========\n\n    Takes the distances to the optical element and returns the needed\n    focal distance.\n\n    See Also\n    ========\n\n    geometric_conj_af, geometric_conj_bf\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics import geometric_conj_ab\n    >>> from sympy import symbols\n    >>> a, b = symbols('a b')\n    >>> geometric_conj_ab(a, b)\n    a*b/(a + b)\n    "
    (a, b) = map(sympify, (a, b))
    if a.is_infinite or b.is_infinite:
        return a if b.is_infinite else b
    else:
        return a * b / (a + b)

def geometric_conj_af(a, f):
    if False:
        for i in range(10):
            print('nop')
    "\n    Conjugation relation for geometrical beams under paraxial conditions.\n\n    Explanation\n    ===========\n\n    Takes the object distance (for geometric_conj_af) or the image distance\n    (for geometric_conj_bf) to the optical element and the focal distance.\n    Then it returns the other distance needed for conjugation.\n\n    See Also\n    ========\n\n    geometric_conj_ab\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf\n    >>> from sympy import symbols\n    >>> a, b, f = symbols('a b f')\n    >>> geometric_conj_af(a, f)\n    a*f/(a - f)\n    >>> geometric_conj_bf(b, f)\n    b*f/(b - f)\n    "
    (a, f) = map(sympify, (a, f))
    return -geometric_conj_ab(a, -f)
geometric_conj_bf = geometric_conj_af

def gaussian_conj(s_in, z_r_in, f):
    if False:
        i = 10
        return i + 15
    "\n    Conjugation relation for gaussian beams.\n\n    Parameters\n    ==========\n\n    s_in :\n        The distance to optical element from the waist.\n    z_r_in :\n        The rayleigh range of the incident beam.\n    f :\n        The focal length of the optical element.\n\n    Returns\n    =======\n\n    a tuple containing (s_out, z_r_out, m)\n    s_out :\n        The distance between the new waist and the optical element.\n    z_r_out :\n        The rayleigh range of the emergent beam.\n    m :\n        The ration between the new and the old waists.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics import gaussian_conj\n    >>> from sympy import symbols\n    >>> s_in, z_r_in, f = symbols('s_in z_r_in f')\n\n    >>> gaussian_conj(s_in, z_r_in, f)[0]\n    1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)\n\n    >>> gaussian_conj(s_in, z_r_in, f)[1]\n    z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)\n\n    >>> gaussian_conj(s_in, z_r_in, f)[2]\n    1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)\n    "
    (s_in, z_r_in, f) = map(sympify, (s_in, z_r_in, f))
    s_out = 1 / (-1 / (s_in + z_r_in ** 2 / (s_in - f)) + 1 / f)
    m = 1 / sqrt(1 - (s_in / f) ** 2 + (z_r_in / f) ** 2)
    z_r_out = z_r_in / (1 - (s_in / f) ** 2 + (z_r_in / f) ** 2)
    return (s_out, z_r_out, m)

def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    if False:
        print('Hello World!')
    "\n    Find the optical setup conjugating the object/image waists.\n\n    Parameters\n    ==========\n\n    wavelen :\n        The wavelength of the beam.\n    waist_in and waist_out :\n        The waists to be conjugated.\n    f :\n        The focal distance of the element used in the conjugation.\n\n    Returns\n    =======\n\n    a tuple containing (s_in, s_out, f)\n    s_in :\n        The distance before the optical element.\n    s_out :\n        The distance after the optical element.\n    f :\n        The focal distance of the optical element.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.optics import conjugate_gauss_beams\n    >>> from sympy import symbols, factor\n    >>> l, w_i, w_o, f = symbols('l w_i w_o f')\n\n    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]\n    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))\n\n    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])\n    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -\n              pi**2*w_i**4/(f**2*l**2)))/w_i**2\n\n    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]\n    f\n    "
    (wavelen, waist_in, waist_out) = map(sympify, (wavelen, waist_in, waist_out))
    m = waist_out / waist_in
    z = waist2rayleigh(waist_in, wavelen)
    if len(kwargs) != 1:
        raise ValueError('The function expects only one named argument')
    elif 'dist' in kwargs:
        raise NotImplementedError(filldedent('\n            Currently only focal length is supported as a parameter'))
    elif 'f' in kwargs:
        f = sympify(kwargs['f'])
        s_in = f * (1 - sqrt(1 / m ** 2 - z ** 2 / f ** 2))
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        raise NotImplementedError(filldedent('\n            Currently only focal length is supported as a parameter'))
    else:
        raise ValueError(filldedent('\n            The functions expects the focal length as a named argument'))
    return (s_in, s_out, f)