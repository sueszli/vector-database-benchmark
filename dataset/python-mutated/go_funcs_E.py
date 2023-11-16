from numpy import abs, asarray, cos, exp, arange, pi, sin, sqrt, sum
from .go_benchmark import Benchmark

class Easom(Benchmark):
    """
    Easom objective function.

    This class defines the Easom [1]_ global optimization problem. This is a
    a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Easom}}({x}) = a - \\frac{a}{e^{b \\sqrt{\\frac{\\sum_{i=1}^{n}
        x_i^{2}}{n}}}} + e - e^{\\frac{\\sum_{i=1}^{n} \\cos\\left(c x_i\\right)}
        {n}}


    Where, in this exercise, :math:`a = 20, b = 0.2` and :math:`c = 2 \\pi`.

    Here, :math:`x_i \\in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Gavana website disagrees with Jamil, etc. Gavana equation in docstring is totally wrong.
    """

    def __init__(self, dimensions=2):
        if False:
            return 10
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.global_optimum = [[pi for _ in range(self.N)]]
        self.fglob = -1.0

    def fun(self, x, *args):
        if False:
            i = 10
            return i + 15
        self.nfev += 1
        a = (x[0] - pi) ** 2 + (x[1] - pi) ** 2
        return -cos(x[0]) * cos(x[1]) * exp(-a)

class Eckerle4(Benchmark):
    """
    Eckerle4 objective function.
    Eckerle, K., NIST (1979).
    Circular Interference Transmittance Study.

    ..[1] https://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml

    #TODO, this is a NIST regression standard dataset, docstring needs
    improving
    """

    def __init__(self, dimensions=3):
        if False:
            print('Hello World!')
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0, 1.0, 10.0], [20, 20.0, 600.0]))
        self.global_optimum = [[1.5543827178, 4.0888321754, 451.54121844]]
        self.fglob = 0.0014635887487
        self.a = asarray([0.0001575, 0.0001699, 0.000235, 0.0003102, 0.0004917, 0.000871, 0.0017418, 0.00464, 0.0065895, 0.0097302, 0.0149002, 0.023731, 0.0401683, 0.0712559, 0.1264458, 0.2073413, 0.2902366, 0.3445623, 0.3698049, 0.3668534, 0.3106727, 0.2078154, 0.1164354, 0.0616764, 0.03372, 0.0194023, 0.0117831, 0.0074357, 0.0022732, 0.00088, 0.0004579, 0.0002345, 0.0001586, 0.0001143, 7.1e-05])
        self.b = asarray([400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 444.0, 445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5, 456.0, 457.5, 459.0, 460.5, 462.0, 463.5, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0])

    def fun(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        self.nfev += 1
        vec = x[0] / x[1] * exp(-(self.b - x[2]) ** 2 / (2 * x[1] ** 2))
        return sum((self.a - vec) ** 2)

class EggCrate(Benchmark):
    """
    Egg Crate objective function.

    This class defines the Egg Crate [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{EggCrate}}(x) = x_1^2 + x_2^2 + 25 \\left[ \\sin^2(x_1)
        + \\sin^2(x_2) \\right]


    with :math:`x_i \\in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        if False:
            while True:
                i = 10
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        self.nfev += 1
        return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)

class EggHolder(Benchmark):
    """
    Egg Holder [1]_ objective function.

    This class defines the Egg Holder global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{EggHolder}}=\\sum_{1}^{n - 1}\\left[-\\left(x_{i + 1}
        + 47 \\right ) \\sin\\sqrt{\\lvert x_{i+1} + x_i/2 + 47 \\rvert}
        - x_i \\sin\\sqrt{\\lvert x_i - (x_{i + 1} + 47)\\rvert}\\right ]


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-512, 512]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -959.640662711` for
    :math:`{x} = [512, 404.2319]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: Jamil is missing a minus sign on the fglob value
    """

    def __init__(self, dimensions=2):
        if False:
            i = 10
            return i + 15
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-512.1] * self.N, [512.0] * self.N))
        self.global_optimum = [[512.0, 404.2319]]
        self.fglob = -959.640662711
        self.change_dimensionality = True

    def fun(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        self.nfev += 1
        vec = -(x[1:] + 47) * sin(sqrt(abs(x[1:] + x[:-1] / 2.0 + 47))) - x[:-1] * sin(sqrt(abs(x[:-1] - (x[1:] + 47))))
        return sum(vec)

class ElAttarVidyasagarDutta(Benchmark):
    """
    El-Attar-Vidyasagar-Dutta [1]_ objective function.

    This class defines the El-Attar-Vidyasagar-Dutta function global
    optimization problem. This is a multimodal minimization problem defined as
    follows:

    .. math::

       f_{\\text{ElAttarVidyasagarDutta}}(x) = (x_1^2 + x_2 - 10)^2
       + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2


    with :math:`x_i \\in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 1.712780354` for
    :math:`x= [3.40918683, -2.17143304]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        if False:
            i = 10
            return i + 15
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        self.custom_bounds = [(-4, 4), (-4, 4)]
        self.global_optimum = [[3.40918683, -2.17143304]]
        self.fglob = 1.712780354

    def fun(self, x, *args):
        if False:
            return 10
        self.nfev += 1
        return (x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2

class Exp2(Benchmark):
    """
    Exp2 objective function.

    This class defines the Exp2 global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Exp2}}(x) = \\sum_{i=0}^9 \\left ( e^{-ix_1/10} - 5e^{-ix_2/10}
        - e^{-i/10} + 5e^{-i} \\right )^2


    with :math:`x_i \\in [0, 20]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10.]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        if False:
            return 10
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([0.0] * self.N, [20.0] * self.N))
        self.custom_bounds = [(0, 2), (0, 20)]
        self.global_optimum = [[1.0, 10.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        if False:
            i = 10
            return i + 15
        self.nfev += 1
        i = arange(10.0)
        vec = (exp(-i * x[0] / 10.0) - 5 * exp(-i * x[1] / 10.0) - exp(-i / 10.0) + 5 * exp(-i)) ** 2
        return sum(vec)

class Exponential(Benchmark):
    """
    Exponential [1] objective function.

    This class defines the Exponential global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\\text{Exponential}}(x) = -e^{-0.5 \\sum_{i=1}^n x_i^2}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in
    [-1, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = -1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil are missing a minus sign on fglob
    """

    def __init__(self, dimensions=2):
        if False:
            i = 10
            return i + 15
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = -1.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        if False:
            print('Hello World!')
        self.nfev += 1
        return -exp(-0.5 * sum(x ** 2.0))