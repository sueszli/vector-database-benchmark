from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt

class DFT(MatrixExpr):
    """
    Returns a discrete Fourier transform matrix. The matrix is scaled
    with :math:`\\frac{1}{\\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy.matrices.expressions.fourier import DFT
    >>> DFT(3)
    DFT(3)
    >>> DFT(3).as_explicit()
    Matrix([
    [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
    >>> DFT(n).shape
    (n, n)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/DFT_matrix

    """

    def __new__(cls, n):
        if False:
            for i in range(10):
                print('nop')
        n = _sympify(n)
        cls._check_dim(n)
        obj = super().__new__(cls, n)
        return obj
    n = property(lambda self: self.args[0])
    shape = property(lambda self: (self.n, self.n))

    def _entry(self, i, j, **kwargs):
        if False:
            i = 10
            return i + 15
        w = exp(-2 * S.Pi * I / self.n)
        return w ** (i * j) / sqrt(self.n)

    def _eval_inverse(self):
        if False:
            return 10
        return IDFT(self.n)

class IDFT(DFT):
    """
    Returns an inverse discrete Fourier transform matrix. The matrix is scaled
    with :math:`\\frac{1}{\\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform

    Examples
    ========

    >>> from sympy.matrices.expressions.fourier import DFT, IDFT
    >>> IDFT(3)
    IDFT(3)
    >>> IDFT(4)*DFT(4)
    I

    See Also
    ========

    DFT

    """

    def _entry(self, i, j, **kwargs):
        if False:
            while True:
                i = 10
        w = exp(-2 * S.Pi * I / self.n)
        return w ** (-i * j) / sqrt(self.n)

    def _eval_inverse(self):
        if False:
            return 10
        return DFT(self.n)