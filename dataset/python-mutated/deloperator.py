from sympy.core import Basic
from sympy.vector.operators import gradient, divergence, curl

class Del(Basic):
    """
    Represents the vector differential operator, usually represented in
    mathematical expressions as the 'nabla' symbol.
    """

    def __new__(cls):
        if False:
            for i in range(10):
                print('nop')
        obj = super().__new__(cls)
        obj._name = 'delop'
        return obj

    def gradient(self, scalar_field, doit=False):
        if False:
            return 10
        "\n        Returns the gradient of the given scalar field, as a\n        Vector instance.\n\n        Parameters\n        ==========\n\n        scalar_field : SymPy expression\n            The scalar field to calculate the gradient of.\n\n        doit : bool\n            If True, the result is returned after calling .doit() on\n            each component. Else, the returned expression contains\n            Derivative instances\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, Del\n        >>> C = CoordSys3D('C')\n        >>> delop = Del()\n        >>> delop.gradient(9)\n        0\n        >>> delop(C.x*C.y*C.z).doit()\n        C.y*C.z*C.i + C.x*C.z*C.j + C.x*C.y*C.k\n\n        "
        return gradient(scalar_field, doit=doit)
    __call__ = gradient
    __call__.__doc__ = gradient.__doc__

    def dot(self, vect, doit=False):
        if False:
            print('Hello World!')
        "\n        Represents the dot product between this operator and a given\n        vector - equal to the divergence of the vector field.\n\n        Parameters\n        ==========\n\n        vect : Vector\n            The vector whose divergence is to be calculated.\n\n        doit : bool\n            If True, the result is returned after calling .doit() on\n            each component. Else, the returned expression contains\n            Derivative instances\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, Del\n        >>> delop = Del()\n        >>> C = CoordSys3D('C')\n        >>> delop.dot(C.x*C.i)\n        Derivative(C.x, C.x)\n        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)\n        >>> (delop & v).doit()\n        C.x*C.y + C.x*C.z + C.y*C.z\n\n        "
        return divergence(vect, doit=doit)
    __and__ = dot
    __and__.__doc__ = dot.__doc__

    def cross(self, vect, doit=False):
        if False:
            while True:
                i = 10
        "\n        Represents the cross product between this operator and a given\n        vector - equal to the curl of the vector field.\n\n        Parameters\n        ==========\n\n        vect : Vector\n            The vector whose curl is to be calculated.\n\n        doit : bool\n            If True, the result is returned after calling .doit() on\n            each component. Else, the returned expression contains\n            Derivative instances\n\n        Examples\n        ========\n\n        >>> from sympy.vector import CoordSys3D, Del\n        >>> C = CoordSys3D('C')\n        >>> delop = Del()\n        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)\n        >>> delop.cross(v, doit = True)\n        (-C.x*C.y + C.x*C.z)*C.i + (C.x*C.y - C.y*C.z)*C.j +\n            (-C.x*C.z + C.y*C.z)*C.k\n        >>> (delop ^ C.i).doit()\n        0\n\n        "
        return curl(vect, doit=doit)
    __xor__ = cross
    __xor__.__doc__ = cross.__doc__

    def _sympystr(self, printer):
        if False:
            for i in range(10):
                print('nop')
        return self._name