from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension
__all__ = ['CircularEnsemble', 'CircularUnitaryEnsemble', 'CircularOrthogonalEnsemble', 'CircularSymplecticEnsemble', 'GaussianEnsemble', 'GaussianUnitaryEnsemble', 'GaussianOrthogonalEnsemble', 'GaussianSymplecticEnsemble', 'joint_eigen_distribution', 'JointEigenDistribution', 'level_spacing_distribution']

@is_random.register(RandomMatrixSymbol)
def _(x):
    if False:
        return 10
    return True

class RandomMatrixEnsembleModel(Basic):
    """
    Base class for random matrix ensembles.
    It acts as an umbrella and contains
    the methods common to all the ensembles
    defined in sympy.stats.random_matrix_models.
    """

    def __new__(cls, sym, dim=None):
        if False:
            return 10
        (sym, dim) = (_symbol_converter(sym), _sympify(dim))
        if dim.is_integer == False:
            raise ValueError('Dimension of the random matrices must be integers, received %s instead.' % dim)
        return Basic.__new__(cls, sym, dim)
    symbol = property(lambda self: self.args[0])
    dimension = property(lambda self: self.args[1])

    def density(self, expr):
        if False:
            i = 10
            return i + 15
        return Density(expr)

    def __call__(self, expr):
        if False:
            i = 10
            return i + 15
        return self.density(expr)

class GaussianEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Gaussian ensembles.
    Contains the properties common to all the
    gaussian ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
    .. [2] https://arxiv.org/pdf/1712.07903.pdf
    """

    def _compute_normalization_constant(self, beta, n):
        if False:
            print('Hello World!')
        "\n        Helper function for computing normalization\n        constant for joint probability density of eigen\n        values of Gaussian ensembles.\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Selberg_integral#Mehta's_integral\n        "
        n = S(n)
        prod_term = lambda j: gamma(1 + beta * S(j) / 2) / gamma(S.One + beta / S(2))
        j = Dummy('j', integer=True, positive=True)
        term1 = Product(prod_term(j), (j, 1, n)).doit()
        term2 = (2 / (beta * n)) ** (beta * n * (n - 1) / 4 + n / 2)
        term3 = (2 * pi) ** (n / 2)
        return term1 * term2 * term3

    def _compute_joint_eigen_distribution(self, beta):
        if False:
            return 10
        '\n        Helper function for computing the joint\n        probability distribution of eigen values\n        of the random matrix.\n        '
        n = self.dimension
        Zbn = self._compute_normalization_constant(beta, n)
        l = IndexedBase('l')
        i = Dummy('i', integer=True, positive=True)
        j = Dummy('j', integer=True, positive=True)
        k = Dummy('k', integer=True, positive=True)
        term1 = exp(-S(n) / 2 * Sum(l[k] ** 2, (k, 1, n)).doit())
        sub_term = Lambda(i, Product(Abs(l[j] - l[i]) ** beta, (j, i + 1, n)))
        term2 = Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        syms = ArrayComprehension(l[k], (k, 1, n)).doit()
        return Lambda(tuple(syms), term1 * term2 / Zbn)

class GaussianUnitaryEnsembleModel(GaussianEnsembleModel):

    @property
    def normalization_constant(self):
        if False:
            i = 10
            return i + 15
        n = self.dimension
        return 2 ** (S(n) / 2) * pi ** (S(n ** 2) / 2)

    def density(self, expr):
        if False:
            print('Hello World!')
        (n, ZGUE) = (self.dimension, self.normalization_constant)
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) / 2 * Trace(H ** 2)) / ZGUE)(expr)

    def joint_eigen_distribution(self):
        if False:
            for i in range(10):
                print('nop')
        return self._compute_joint_eigen_distribution(S(2))

    def level_spacing_distribution(self):
        if False:
            return 10
        s = Dummy('s')
        f = 32 / pi ** 2 * s ** 2 * exp(-4 / pi * s ** 2)
        return Lambda(s, f)

class GaussianOrthogonalEnsembleModel(GaussianEnsembleModel):

    @property
    def normalization_constant(self):
        if False:
            return 10
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) / 4 * Trace(_H ** 2)))

    def density(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (n, ZGOE) = (self.dimension, self.normalization_constant)
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) / 4 * Trace(H ** 2)) / ZGOE)(expr)

    def joint_eigen_distribution(self):
        if False:
            i = 10
            return i + 15
        return self._compute_joint_eigen_distribution(S.One)

    def level_spacing_distribution(self):
        if False:
            i = 10
            return i + 15
        s = Dummy('s')
        f = pi / 2 * s * exp(-pi / 4 * s ** 2)
        return Lambda(s, f)

class GaussianSymplecticEnsembleModel(GaussianEnsembleModel):

    @property
    def normalization_constant(self):
        if False:
            i = 10
            return i + 15
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) * Trace(_H ** 2)))

    def density(self, expr):
        if False:
            return 10
        (n, ZGSE) = (self.dimension, self.normalization_constant)
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) * Trace(H ** 2)) / ZGSE)(expr)

    def joint_eigen_distribution(self):
        if False:
            for i in range(10):
                print('nop')
        return self._compute_joint_eigen_distribution(S(4))

    def level_spacing_distribution(self):
        if False:
            print('Hello World!')
        s = Dummy('s')
        f = S(2) ** 18 / (S(3) ** 6 * pi ** 3) * s ** 4 * exp(-64 / (9 * pi) * s ** 2)
        return Lambda(s, f)

def GaussianEnsemble(sym, dim):
    if False:
        i = 10
        return i + 15
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = GaussianEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianUnitaryEnsemble(sym, dim):
    if False:
        return 10
    "\n    Represents Gaussian Unitary Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE, density\n    >>> from sympy import MatrixSymbol\n    >>> G = GUE('U', 2)\n    >>> X = MatrixSymbol('X', 2, 2)\n    >>> density(G)(X)\n    exp(-Trace(X**2))/(2*pi**2)\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = GaussianUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianOrthogonalEnsemble(sym, dim):
    if False:
        i = 10
        return i + 15
    "\n    Represents Gaussian Orthogonal Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianOrthogonalEnsemble as GOE, density\n    >>> from sympy import MatrixSymbol\n    >>> G = GOE('U', 2)\n    >>> X = MatrixSymbol('X', 2, 2)\n    >>> density(G)(X)\n    exp(-Trace(X**2)/2)/Integral(exp(-Trace(_H**2)/2), _H)\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = GaussianOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianSymplecticEnsemble(sym, dim):
    if False:
        i = 10
        return i + 15
    "\n    Represents Gaussian Symplectic Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianSymplecticEnsemble as GSE, density\n    >>> from sympy import MatrixSymbol\n    >>> G = GSE('U', 2)\n    >>> X = MatrixSymbol('X', 2, 2)\n    >>> density(G)(X)\n    exp(-2*Trace(X**2))/Integral(exp(-2*Trace(_H**2)), _H)\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = GaussianSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

class CircularEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Circular ensembles.
    Contains the properties and methods
    common to all the circular ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Circular_ensemble
    """

    def density(self, expr):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("Support for Haar measure hasn't been implemented yet, therefore the density of %s cannot be computed." % self)

    def _compute_joint_eigen_distribution(self, beta):
        if False:
            while True:
                i = 10
        '\n        Helper function to compute the joint distribution of phases\n        of the complex eigen values of matrices belonging to any\n        circular ensembles.\n        '
        n = self.dimension
        Zbn = (2 * pi) ** n * (gamma(beta * n / 2 + 1) / S(gamma(beta / 2 + 1)) ** n)
        t = IndexedBase('t')
        (i, j, k) = (Dummy('i', integer=True), Dummy('j', integer=True), Dummy('k', integer=True))
        syms = ArrayComprehension(t[i], (i, 1, n)).doit()
        f = Product(Product(Abs(exp(I * t[k]) - exp(I * t[j])) ** beta, (j, k + 1, n)).doit(), (k, 1, n - 1)).doit()
        return Lambda(tuple(syms), f / Zbn)

class CircularUnitaryEnsembleModel(CircularEnsembleModel):

    def joint_eigen_distribution(self):
        if False:
            print('Hello World!')
        return self._compute_joint_eigen_distribution(S(2))

class CircularOrthogonalEnsembleModel(CircularEnsembleModel):

    def joint_eigen_distribution(self):
        if False:
            print('Hello World!')
        return self._compute_joint_eigen_distribution(S.One)

class CircularSymplecticEnsembleModel(CircularEnsembleModel):

    def joint_eigen_distribution(self):
        if False:
            print('Hello World!')
        return self._compute_joint_eigen_distribution(S(4))

def CircularEnsemble(sym, dim):
    if False:
        while True:
            i = 10
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = CircularEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularUnitaryEnsemble(sym, dim):
    if False:
        while True:
            i = 10
    "\n    Represents Circular Unitary Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import CircularUnitaryEnsemble as CUE\n    >>> from sympy.stats import joint_eigen_distribution\n    >>> C = CUE('U', 1)\n    >>> joint_eigen_distribution(C)\n    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**2, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))\n\n    Note\n    ====\n\n    As can be seen above in the example, density of CiruclarUnitaryEnsemble\n    is not evaluated because the exact definition is based on haar measure of\n    unitary group which is not unique.\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = CircularUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularOrthogonalEnsemble(sym, dim):
    if False:
        return 10
    "\n    Represents Circular Orthogonal Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import CircularOrthogonalEnsemble as COE\n    >>> from sympy.stats import joint_eigen_distribution\n    >>> C = COE('O', 1)\n    >>> joint_eigen_distribution(C)\n    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k])), (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))\n\n    Note\n    ====\n\n    As can be seen above in the example, density of CiruclarOrthogonalEnsemble\n    is not evaluated because the exact definition is based on haar measure of\n    unitary group which is not unique.\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = CircularOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularSymplecticEnsemble(sym, dim):
    if False:
        for i in range(10):
            print('nop')
    "\n    Represents Circular Symplectic Ensembles.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import CircularSymplecticEnsemble as CSE\n    >>> from sympy.stats import joint_eigen_distribution\n    >>> C = CSE('S', 1)\n    >>> joint_eigen_distribution(C)\n    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**4, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))\n\n    Note\n    ====\n\n    As can be seen above in the example, density of CiruclarSymplecticEnsemble\n    is not evaluated because the exact definition is based on haar measure of\n    unitary group which is not unique.\n    "
    (sym, dim) = (_symbol_converter(sym), _sympify(dim))
    model = CircularSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def joint_eigen_distribution(mat):
    if False:
        for i in range(10):
            print('nop')
    "\n    For obtaining joint probability distribution\n    of eigen values of random matrix.\n\n    Parameters\n    ==========\n\n    mat: RandomMatrixSymbol\n        The matrix symbol whose eigen values are to be considered.\n\n    Returns\n    =======\n\n    Lambda\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE\n    >>> from sympy.stats import joint_eigen_distribution\n    >>> U = GUE('U', 2)\n    >>> joint_eigen_distribution(U)\n    Lambda((l[1], l[2]), exp(-l[1]**2 - l[2]**2)*Product(Abs(l[_i] - l[_j])**2, (_j, _i + 1, 2), (_i, 1, 1))/pi)\n    "
    if not isinstance(mat, RandomMatrixSymbol):
        raise ValueError('%s is not of type, RandomMatrixSymbol.' % mat)
    return mat.pspace.model.joint_eigen_distribution()

def JointEigenDistribution(mat):
    if False:
        while True:
            i = 10
    "\n    Creates joint distribution of eigen values of matrices with random\n    expressions.\n\n    Parameters\n    ==========\n\n    mat: Matrix\n        The matrix under consideration.\n\n    Returns\n    =======\n\n    JointDistributionHandmade\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Normal, JointEigenDistribution\n    >>> from sympy import Matrix\n    >>> A = [[Normal('A00', 0, 1), Normal('A01', 0, 1)],\n    ... [Normal('A10', 0, 1), Normal('A11', 0, 1)]]\n    >>> JointEigenDistribution(Matrix(A))\n    JointDistributionHandmade(-sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2\n    + A00/2 + A11/2, sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2 + A00/2 + A11/2)\n\n    "
    eigenvals = mat.eigenvals(multiple=True)
    if not all((is_random(eigenval) for eigenval in set(eigenvals))):
        raise ValueError('Eigen values do not have any random expression, joint distribution cannot be generated.')
    return JointDistributionHandmade(*eigenvals)

def level_spacing_distribution(mat):
    if False:
        while True:
            i = 10
    "\n    For obtaining distribution of level spacings.\n\n    Parameters\n    ==========\n\n    mat: RandomMatrixSymbol\n        The random matrix symbol whose eigen values are\n        to be considered for finding the level spacings.\n\n    Returns\n    =======\n\n    Lambda\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE\n    >>> from sympy.stats import level_spacing_distribution\n    >>> U = GUE('U', 2)\n    >>> level_spacing_distribution(U)\n    Lambda(_s, 32*_s**2*exp(-4*_s**2/pi)/pi**2)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Distribution_of_level_spacings\n    "
    return mat.pspace.model.level_spacing_distribution()