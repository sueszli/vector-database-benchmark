from sympy import MatAdd, MatMul, Array
from sympy.algebras.quaternion import Quaternion
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple, Dict
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Derivative, Function, Lambda, Subs, diff
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import AlgebraicNumber, Float, I, Integer, Rational, oo, pi
from sympy.core.parameters import evaluate
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions.combinatorial.factorials import FallingFactorial, RisingFactorial, binomial, factorial, factorial2, subfactorial
from sympy.functions.combinatorial.numbers import bernoulli, bell, catalan, euler, genocchi, lucas, fibonacci, tribonacci
from sympy.functions.elementary.complexes import Abs, arg, conjugate, im, polar_lift, re
from sympy.functions.elementary.exponential import LambertW, exp, log
from sympy.functions.elementary.hyperbolic import asinh, coth
from sympy.functions.elementary.integers import ceiling, floor, frac
from sympy.functions.elementary.miscellaneous import Max, Min, root, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import acsc, asin, cos, cot, sin, tan
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f, elliptic_k, elliptic_pi
from sympy.functions.special.error_functions import Chi, Ci, Ei, Shi, Si, expint
from sympy.functions.special.gamma_functions import gamma, uppergamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.mathieu_functions import mathieuc, mathieucprime, mathieus, mathieusprime
from sympy.functions.special.polynomials import assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.spherical_harmonics import Ynm, Znm
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.functions.special.zeta_functions import dirichlet_eta, lerchphi, polylog, stieltjes, zeta
from sympy.integrals.integrals import Integral
from sympy.integrals.transforms import CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform
from sympy.logic import Implies
from sympy.logic.boolalg import And, Or, Xor, Equivalent, false, Not, true
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.kronecker import KroneckerProduct
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.permutation import PermutationMatrix
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.physics.control.lti import TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.ntheory.factor_ import divisor_sigma, primenu, primeomega, reduced_totient, totient, udivisor_sigma
from sympy.physics.quantum import Commutator, Operator
from sympy.physics.quantum.trace import Tr
from sympy.physics.units import meter, gibibyte, gram, microgram, second, milli, micro
from sympy.polys.domains.integerring import ZZ
from sympy.polys.fields import field
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring
from sympy.polys.rootoftools import RootSum, rootof
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import SeqAdd, SeqFormula, SeqMul, SeqPer
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import ComplexRegion, ImageSet, Range
from sympy.sets.ordinals import Ordinal, OrdinalOmega, OmegaPower
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import Covariance, Expectation, Probability, Variance
from sympy.tensor.array import ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray, MutableDenseNDimArray, tensorproduct
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.tensor.indexed import Idx, Indexed, IndexedBase
from sympy.tensor.toperators import PartialDerivative
from sympy.vector import CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow, warns_deprecated_sympy
from sympy.printing.latex import latex, translate, greek_letters_set, tex_greek_dictionary, multiline_latex, latex_escape, LatexPrinter
import sympy as sym
from sympy.abc import mu, tau

class lowergamma(sym.lowergamma):
    pass
(x, y, z, t, w, a, b, c, s, p) = symbols('x y z t w a b c s p')
(k, m, n) = symbols('k m n', integer=True)

def test_printmethod():
    if False:
        i = 10
        return i + 15

    class R(Abs):

        def _latex(self, printer):
            if False:
                for i in range(10):
                    print('nop')
            return 'foo(%s)' % printer._print(self.args[0])
    assert latex(R(x)) == 'foo(x)'

    class R(Abs):

        def _latex(self, printer):
            if False:
                return 10
            return 'foo'
    assert latex(R(x)) == 'foo'

def test_latex_basic():
    if False:
        print('Hello World!')
    assert latex(1 + x) == 'x + 1'
    assert latex(x ** 2) == 'x^{2}'
    assert latex(x ** (1 + x)) == 'x^{x + 1}'
    assert latex(x ** 3 + x + 1 + x ** 2) == 'x^{3} + x^{2} + x + 1'
    assert latex(2 * x * y) == '2 x y'
    assert latex(2 * x * y, mul_symbol='dot') == '2 \\cdot x \\cdot y'
    assert latex(3 * x ** 2 * y, mul_symbol='\\,') == '3\\,x^{2}\\,y'
    assert latex(1.5 * 3 ** x, mul_symbol='\\,') == '1.5 \\cdot 3^{x}'
    assert latex(x ** S.Half ** 5) == '\\sqrt[32]{x}'
    assert latex(Mul(S.Half, x ** 2, -5, evaluate=False)) == '\\frac{1}{2} x^{2} \\left(-5\\right)'
    assert latex(Mul(S.Half, x ** 2, 5, evaluate=False)) == '\\frac{1}{2} x^{2} \\cdot 5'
    assert latex(Mul(-5, -5, evaluate=False)) == '\\left(-5\\right) \\left(-5\\right)'
    assert latex(Mul(5, -5, evaluate=False)) == '5 \\left(-5\\right)'
    assert latex(Mul(S.Half, -5, S.Half, evaluate=False)) == '\\frac{1}{2} \\left(-5\\right) \\frac{1}{2}'
    assert latex(Mul(5, I, 5, evaluate=False)) == '5 i 5'
    assert latex(Mul(5, I, -5, evaluate=False)) == '5 i \\left(-5\\right)'
    assert latex(Mul(0, 1, evaluate=False)) == '0 \\cdot 1'
    assert latex(Mul(1, 0, evaluate=False)) == '1 \\cdot 0'
    assert latex(Mul(1, 1, evaluate=False)) == '1 \\cdot 1'
    assert latex(Mul(-1, 1, evaluate=False)) == '\\left(-1\\right) 1'
    assert latex(Mul(1, 1, 1, evaluate=False)) == '1 \\cdot 1 \\cdot 1'
    assert latex(Mul(1, 2, evaluate=False)) == '1 \\cdot 2'
    assert latex(Mul(1, S.Half, evaluate=False)) == '1 \\cdot \\frac{1}{2}'
    assert latex(Mul(1, 1, S.Half, evaluate=False)) == '1 \\cdot 1 \\cdot \\frac{1}{2}'
    assert latex(Mul(1, 1, 2, 3, x, evaluate=False)) == '1 \\cdot 1 \\cdot 2 \\cdot 3 x'
    assert latex(Mul(1, -1, evaluate=False)) == '1 \\left(-1\\right)'
    assert latex(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == '4 \\cdot 3 \\cdot 2 \\cdot 1 \\cdot 0 y x'
    assert latex(Mul(4, 3, 2, 1 + z, 0, y, x, evaluate=False)) == '4 \\cdot 3 \\cdot 2 \\left(z + 1\\right) 0 y x'
    assert latex(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == '\\frac{2}{3} \\cdot \\frac{5}{7}'
    assert latex(1 / x) == '\\frac{1}{x}'
    assert latex(1 / x, fold_short_frac=True) == '1 / x'
    assert latex(-S(3) / 2) == '- \\frac{3}{2}'
    assert latex(-S(3) / 2, fold_short_frac=True) == '- 3 / 2'
    assert latex(1 / x ** 2) == '\\frac{1}{x^{2}}'
    assert latex(1 / (x + y) / 2) == '\\frac{1}{2 \\left(x + y\\right)}'
    assert latex(x / 2) == '\\frac{x}{2}'
    assert latex(x / 2, fold_short_frac=True) == 'x / 2'
    assert latex((x + y) / (2 * x)) == '\\frac{x + y}{2 x}'
    assert latex((x + y) / (2 * x), fold_short_frac=True) == '\\left(x + y\\right) / 2 x'
    assert latex((x + y) / (2 * x), long_frac_ratio=0) == '\\frac{1}{2 x} \\left(x + y\\right)'
    assert latex((x + y) / x) == '\\frac{x + y}{x}'
    assert latex((x + y) / x, long_frac_ratio=3) == '\\frac{x + y}{x}'
    assert latex(2 * sqrt(2) * x / 3) == '\\frac{2 \\sqrt{2} x}{3}'
    assert latex(2 * sqrt(2) * x / 3, long_frac_ratio=2) == '\\frac{2 x}{3} \\sqrt{2}'
    assert latex(binomial(x, y)) == '{\\binom{x}{y}}'
    x_star = Symbol('x^*')
    f = Function('f')
    assert latex(x_star ** 2) == '\\left(x^{*}\\right)^{2}'
    assert latex(x_star ** 2, parenthesize_super=False) == '{x^{*}}^{2}'
    assert latex(Derivative(f(x_star), x_star, 2)) == '\\frac{d^{2}}{d \\left(x^{*}\\right)^{2}} f{\\left(x^{*} \\right)}'
    assert latex(Derivative(f(x_star), x_star, 2), parenthesize_super=False) == '\\frac{d^{2}}{d {x^{*}}^{2}} f{\\left(x^{*} \\right)}'
    assert latex(2 * Integral(x, x) / 3) == '\\frac{2 \\int x\\, dx}{3}'
    assert latex(2 * Integral(x, x) / 3, fold_short_frac=True) == '\\left(2 \\int x\\, dx\\right) / 3'
    assert latex(sqrt(x)) == '\\sqrt{x}'
    assert latex(x ** Rational(1, 3)) == '\\sqrt[3]{x}'
    assert latex(x ** Rational(1, 3), root_notation=False) == 'x^{\\frac{1}{3}}'
    assert latex(sqrt(x) ** 3) == 'x^{\\frac{3}{2}}'
    assert latex(sqrt(x), itex=True) == '\\sqrt{x}'
    assert latex(x ** Rational(1, 3), itex=True) == '\\root{3}{x}'
    assert latex(sqrt(x) ** 3, itex=True) == 'x^{\\frac{3}{2}}'
    assert latex(x ** Rational(3, 4)) == 'x^{\\frac{3}{4}}'
    assert latex(x ** Rational(3, 4), fold_frac_powers=True) == 'x^{3/4}'
    assert latex((x + 1) ** Rational(3, 4)) == '\\left(x + 1\\right)^{\\frac{3}{4}}'
    assert latex((x + 1) ** Rational(3, 4), fold_frac_powers=True) == '\\left(x + 1\\right)^{3/4}'
    assert latex(AlgebraicNumber(sqrt(2))) == '\\sqrt{2}'
    assert latex(AlgebraicNumber(sqrt(2), [3, -7])) == '-7 + 3 \\sqrt{2}'
    assert latex(AlgebraicNumber(sqrt(2), alias='alpha')) == '\\alpha'
    assert latex(AlgebraicNumber(sqrt(2), [3, -7], alias='alpha')) == '3 \\alpha - 7'
    assert latex(AlgebraicNumber(2 ** (S(1) / 3), [1, 3, -7], alias='beta')) == '\\beta^{2} + 3 \\beta - 7'
    k = ZZ.cyclotomic_field(5)
    assert latex(k.ext.field_element([1, 2, 3, 4])) == '\\zeta^{3} + 2 \\zeta^{2} + 3 \\zeta + 4'
    assert latex(k.ext.field_element([1, 2, 3, 4]), order='old') == '4 + 3 \\zeta + 2 \\zeta^{2} + \\zeta^{3}'
    assert latex(k.primes_above(19)[0]) == '\\left(19, \\zeta^{2} + 5 \\zeta + 1\\right)'
    assert latex(k.primes_above(19)[0], order='old') == '\\left(19, 1 + 5 \\zeta + \\zeta^{2}\\right)'
    assert latex(k.primes_above(7)[0]) == '\\left(7\\right)'
    assert latex(1.5e+20 * x) == '1.5 \\cdot 10^{20} x'
    assert latex(1.5e+20 * x, mul_symbol='dot') == '1.5 \\cdot 10^{20} \\cdot x'
    assert latex(1.5e+20 * x, mul_symbol='times') == '1.5 \\times 10^{20} \\times x'
    assert latex(1 / sin(x)) == '\\frac{1}{\\sin{\\left(x \\right)}}'
    assert latex(sin(x) ** (-1)) == '\\frac{1}{\\sin{\\left(x \\right)}}'
    assert latex(sin(x) ** Rational(3, 2)) == '\\sin^{\\frac{3}{2}}{\\left(x \\right)}'
    assert latex(sin(x) ** Rational(3, 2), fold_frac_powers=True) == '\\sin^{3/2}{\\left(x \\right)}'
    assert latex(~x) == '\\neg x'
    assert latex(x & y) == 'x \\wedge y'
    assert latex(x & y & z) == 'x \\wedge y \\wedge z'
    assert latex(x | y) == 'x \\vee y'
    assert latex(x | y | z) == 'x \\vee y \\vee z'
    assert latex(x & y | z) == 'z \\vee \\left(x \\wedge y\\right)'
    assert latex(Implies(x, y)) == 'x \\Rightarrow y'
    assert latex(~(x >> ~y)) == 'x \\not\\Rightarrow \\neg y'
    assert latex(Implies(Or(x, y), z)) == '\\left(x \\vee y\\right) \\Rightarrow z'
    assert latex(Implies(z, Or(x, y))) == 'z \\Rightarrow \\left(x \\vee y\\right)'
    assert latex(~(x & y)) == '\\neg \\left(x \\wedge y\\right)'
    assert latex(~x, symbol_names={x: 'x_i'}) == '\\neg x_i'
    assert latex(x & y, symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\wedge y_i'
    assert latex(x & y & z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'x_i \\wedge y_i \\wedge z_i'
    assert latex(x | y, symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\vee y_i'
    assert latex(x | y | z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'x_i \\vee y_i \\vee z_i'
    assert latex(x & y | z, symbol_names={x: 'x_i', y: 'y_i', z: 'z_i'}) == 'z_i \\vee \\left(x_i \\wedge y_i\\right)'
    assert latex(Implies(x, y), symbol_names={x: 'x_i', y: 'y_i'}) == 'x_i \\Rightarrow y_i'
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == '\\frac{1}{\\frac{1}{3}}'
    assert latex(Pow(Rational(1, 3), -2, evaluate=False)) == '\\frac{1}{(\\frac{1}{3})^{2}}'
    assert latex(Pow(Integer(1) / 100, -1, evaluate=False)) == '\\frac{1}{\\frac{1}{100}}'
    p = Symbol('p', positive=True)
    assert latex(exp(-p) * log(p)) == 'e^{- p} \\log{\\left(p \\right)}'

def test_latex_builtins():
    if False:
        print('Hello World!')
    assert latex(True) == '\\text{True}'
    assert latex(False) == '\\text{False}'
    assert latex(None) == '\\text{None}'
    assert latex(true) == '\\text{True}'
    assert latex(false) == '\\text{False}'

def test_latex_SingularityFunction():
    if False:
        i = 10
        return i + 15
    assert latex(SingularityFunction(x, 4, 5)) == '{\\left\\langle x - 4 \\right\\rangle}^{5}'
    assert latex(SingularityFunction(x, -3, 4)) == '{\\left\\langle x + 3 \\right\\rangle}^{4}'
    assert latex(SingularityFunction(x, 0, 4)) == '{\\left\\langle x \\right\\rangle}^{4}'
    assert latex(SingularityFunction(x, a, n)) == '{\\left\\langle - a + x \\right\\rangle}^{n}'
    assert latex(SingularityFunction(x, 4, -2)) == '{\\left\\langle x - 4 \\right\\rangle}^{-2}'
    assert latex(SingularityFunction(x, 4, -1)) == '{\\left\\langle x - 4 \\right\\rangle}^{-1}'
    assert latex(SingularityFunction(x, 4, 5) ** 3) == '{\\left({\\langle x - 4 \\rangle}^{5}\\right)}^{3}'
    assert latex(SingularityFunction(x, -3, 4) ** 3) == '{\\left({\\langle x + 3 \\rangle}^{4}\\right)}^{3}'
    assert latex(SingularityFunction(x, 0, 4) ** 3) == '{\\left({\\langle x \\rangle}^{4}\\right)}^{3}'
    assert latex(SingularityFunction(x, a, n) ** 3) == '{\\left({\\langle - a + x \\rangle}^{n}\\right)}^{3}'
    assert latex(SingularityFunction(x, 4, -2) ** 3) == '{\\left({\\langle x - 4 \\rangle}^{-2}\\right)}^{3}'
    assert latex((SingularityFunction(x, 4, -1) ** 3) ** 3) == '{\\left({\\langle x - 4 \\rangle}^{-1}\\right)}^{9}'

def test_latex_cycle():
    if False:
        print('Hello World!')
    assert latex(Cycle(1, 2, 4)) == '\\left( 1\\; 2\\; 4\\right)'
    assert latex(Cycle(1, 2)(4, 5, 6)) == '\\left( 1\\; 2\\right)\\left( 4\\; 5\\; 6\\right)'
    assert latex(Cycle()) == '\\left( \\right)'

def test_latex_permutation():
    if False:
        for i in range(10):
            print('nop')
    assert latex(Permutation(1, 2, 4)) == '\\left( 1\\; 2\\; 4\\right)'
    assert latex(Permutation(1, 2)(4, 5, 6)) == '\\left( 1\\; 2\\right)\\left( 4\\; 5\\; 6\\right)'
    assert latex(Permutation()) == '\\left( \\right)'
    assert latex(Permutation(2, 4) * Permutation(5)) == '\\left( 2\\; 4\\right)\\left( 5\\right)'
    assert latex(Permutation(5)) == '\\left( 5\\right)'
    assert latex(Permutation(0, 1), perm_cyclic=False) == '\\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}'
    assert latex(Permutation(0, 1)(2, 3), perm_cyclic=False) == '\\begin{pmatrix} 0 & 1 & 2 & 3 \\\\ 1 & 0 & 3 & 2 \\end{pmatrix}'
    assert latex(Permutation(), perm_cyclic=False) == '\\left( \\right)'
    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert latex(Permutation(0, 1)(2, 3)) == '\\begin{pmatrix} 0 & 1 & 2 & 3 \\\\ 1 & 0 & 3 & 2 \\end{pmatrix}'
        Permutation.print_cyclic = old_print_cyclic

def test_latex_Float():
    if False:
        print('Hello World!')
    assert latex(Float(1e+100)) == '1.0 \\cdot 10^{100}'
    assert latex(Float(1e-100)) == '1.0 \\cdot 10^{-100}'
    assert latex(Float(1e-100), mul_symbol='times') == '1.0 \\times 10^{-100}'
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=2) == '1.0 \\cdot 10^{4}'
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=4) == '1.0 \\cdot 10^{4}'
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=5) == '10000.0'
    assert latex(Float('0.099999'), full_prec=True, min=-2, max=5) == '9.99990000000000 \\cdot 10^{-2}'

def test_latex_vector_expressions():
    if False:
        return 10
    A = CoordSys3D('A')
    assert latex(Cross(A.i, A.j * A.x * 3 + A.k)) == '\\mathbf{\\hat{i}_{A}} \\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}} + \\mathbf{\\hat{k}_{A}}\\right)'
    assert latex(Cross(A.i, A.j)) == '\\mathbf{\\hat{i}_{A}} \\times \\mathbf{\\hat{j}_{A}}'
    assert latex(x * Cross(A.i, A.j)) == 'x \\left(\\mathbf{\\hat{i}_{A}} \\times \\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Cross(x * A.i, A.j)) == '- \\mathbf{\\hat{j}_{A}} \\times \\left(\\left(x\\right)\\mathbf{\\hat{i}_{A}}\\right)'
    assert latex(Curl(3 * A.x * A.j)) == '\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Curl(3 * A.x * A.j + A.i)) == '\\nabla\\times \\left(\\mathbf{\\hat{i}_{A}} + \\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Curl(3 * x * A.x * A.j)) == '\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}} x\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(x * Curl(3 * A.x * A.j)) == 'x \\left(\\nabla\\times \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)\\right)'
    assert latex(Divergence(3 * A.x * A.j + A.i)) == '\\nabla\\cdot \\left(\\mathbf{\\hat{i}_{A}} + \\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Divergence(3 * A.x * A.j)) == '\\nabla\\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(x * Divergence(3 * A.x * A.j)) == 'x \\left(\\nabla\\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}}\\right)\\right)'
    assert latex(Dot(A.i, A.j * A.x * 3 + A.k)) == '\\mathbf{\\hat{i}_{A}} \\cdot \\left(\\left(3 \\mathbf{{x}_{A}}\\right)\\mathbf{\\hat{j}_{A}} + \\mathbf{\\hat{k}_{A}}\\right)'
    assert latex(Dot(A.i, A.j)) == '\\mathbf{\\hat{i}_{A}} \\cdot \\mathbf{\\hat{j}_{A}}'
    assert latex(Dot(x * A.i, A.j)) == '\\mathbf{\\hat{j}_{A}} \\cdot \\left(\\left(x\\right)\\mathbf{\\hat{i}_{A}}\\right)'
    assert latex(x * Dot(A.i, A.j)) == 'x \\left(\\mathbf{\\hat{i}_{A}} \\cdot \\mathbf{\\hat{j}_{A}}\\right)'
    assert latex(Gradient(A.x)) == '\\nabla \\mathbf{{x}_{A}}'
    assert latex(Gradient(A.x + 3 * A.y)) == '\\nabla \\left(\\mathbf{{x}_{A}} + 3 \\mathbf{{y}_{A}}\\right)'
    assert latex(x * Gradient(A.x)) == 'x \\left(\\nabla \\mathbf{{x}_{A}}\\right)'
    assert latex(Gradient(x * A.x)) == '\\nabla \\left(\\mathbf{{x}_{A}} x\\right)'
    assert latex(Laplacian(A.x)) == '\\Delta \\mathbf{{x}_{A}}'
    assert latex(Laplacian(A.x + 3 * A.y)) == '\\Delta \\left(\\mathbf{{x}_{A}} + 3 \\mathbf{{y}_{A}}\\right)'
    assert latex(x * Laplacian(A.x)) == 'x \\left(\\Delta \\mathbf{{x}_{A}}\\right)'
    assert latex(Laplacian(x * A.x)) == '\\Delta \\left(\\mathbf{{x}_{A}} x\\right)'

def test_latex_symbols():
    if False:
        i = 10
        return i + 15
    (Gamma, lmbda, rho) = symbols('Gamma, lambda, rho')
    (tau, Tau, TAU, taU) = symbols('tau, Tau, TAU, taU')
    assert latex(tau) == '\\tau'
    assert latex(Tau) == '\\mathrm{T}'
    assert latex(TAU) == '\\tau'
    assert latex(taU) == '\\tau'
    capitalized_letters = {l.capitalize() for l in greek_letters_set}
    assert len(capitalized_letters - set(tex_greek_dictionary.keys())) == 0
    assert latex(Gamma + lmbda) == '\\Gamma + \\lambda'
    assert latex(Gamma * lmbda) == '\\Gamma \\lambda'
    assert latex(Symbol('q1')) == 'q_{1}'
    assert latex(Symbol('q21')) == 'q_{21}'
    assert latex(Symbol('epsilon0')) == '\\epsilon_{0}'
    assert latex(Symbol('omega1')) == '\\omega_{1}'
    assert latex(Symbol('91')) == '91'
    assert latex(Symbol('alpha_new')) == '\\alpha_{new}'
    assert latex(Symbol('C^orig')) == 'C^{orig}'
    assert latex(Symbol('x^alpha')) == 'x^{\\alpha}'
    assert latex(Symbol('beta^alpha')) == '\\beta^{\\alpha}'
    assert latex(Symbol('e^Alpha')) == 'e^{\\mathrm{A}}'
    assert latex(Symbol('omega_alpha^beta')) == '\\omega^{\\beta}_{\\alpha}'
    assert latex(Symbol('omega') ** Symbol('beta')) == '\\omega^{\\beta}'

@XFAIL
def test_latex_symbols_failing():
    if False:
        return 10
    (rho, mass, volume) = symbols('rho, mass, volume')
    assert latex(volume * rho == mass) == '\\rho \\mathrm{volume} = \\mathrm{mass}'
    assert latex(volume / mass * rho == 1) == '\\rho \\mathrm{volume} {\\mathrm{mass}}^{(-1)} = 1'
    assert latex(mass ** 3 * volume ** 3) == '{\\mathrm{mass}}^{3} \\cdot {\\mathrm{volume}}^{3}'

@_both_exp_pow
def test_latex_functions():
    if False:
        print('Hello World!')
    assert latex(exp(x)) == 'e^{x}'
    assert latex(exp(1) + exp(2)) == 'e + e^{2}'
    f = Function('f')
    assert latex(f(x)) == 'f{\\left(x \\right)}'
    assert latex(f) == 'f'
    g = Function('g')
    assert latex(g(x, y)) == 'g{\\left(x,y \\right)}'
    assert latex(g) == 'g'
    h = Function('h')
    assert latex(h(x, y, z)) == 'h{\\left(x,y,z \\right)}'
    assert latex(h) == 'h'
    Li = Function('Li')
    assert latex(Li) == '\\operatorname{Li}'
    assert latex(Li(x)) == '\\operatorname{Li}{\\left(x \\right)}'
    mybeta = Function('beta')
    assert latex(mybeta(x, y, z)) == '\\beta{\\left(x,y,z \\right)}'
    assert latex(beta(x, y)) == '\\operatorname{B}\\left(x, y\\right)'
    assert latex(beta(x, evaluate=False)) == '\\operatorname{B}\\left(x, x\\right)'
    assert latex(beta(x, y) ** 2) == '\\operatorname{B}^{2}\\left(x, y\\right)'
    assert latex(mybeta(x)) == '\\beta{\\left(x \\right)}'
    assert latex(mybeta) == '\\beta'
    g = Function('gamma')
    assert latex(g(x, y, z)) == '\\gamma{\\left(x,y,z \\right)}'
    assert latex(g(x)) == '\\gamma{\\left(x \\right)}'
    assert latex(g) == '\\gamma'
    a_1 = Function('a_1')
    assert latex(a_1) == 'a_{1}'
    assert latex(a_1(x)) == 'a_{1}{\\left(x \\right)}'
    assert latex(Function('a_1')) == 'a_{1}'
    assert latex(Function('ab')) == '\\operatorname{ab}'
    assert latex(Function('ab1')) == '\\operatorname{ab}_{1}'
    assert latex(Function('ab12')) == '\\operatorname{ab}_{12}'
    assert latex(Function('ab_1')) == '\\operatorname{ab}_{1}'
    assert latex(Function('ab_12')) == '\\operatorname{ab}_{12}'
    assert latex(Function('ab_c')) == '\\operatorname{ab}_{c}'
    assert latex(Function('ab_cd')) == '\\operatorname{ab}_{cd}'
    assert latex(Function('ab')(Symbol('x'))) == '\\operatorname{ab}{\\left(x \\right)}'
    assert latex(Function('ab1')(Symbol('x'))) == '\\operatorname{ab}_{1}{\\left(x \\right)}'
    assert latex(Function('ab12')(Symbol('x'))) == '\\operatorname{ab}_{12}{\\left(x \\right)}'
    assert latex(Function('ab_1')(Symbol('x'))) == '\\operatorname{ab}_{1}{\\left(x \\right)}'
    assert latex(Function('ab_c')(Symbol('x'))) == '\\operatorname{ab}_{c}{\\left(x \\right)}'
    assert latex(Function('ab_cd')(Symbol('x'))) == '\\operatorname{ab}_{cd}{\\left(x \\right)}'
    assert latex(Function('ab')() ** 2) == '\\operatorname{ab}^{2}{\\left( \\right)}'
    assert latex(Function('ab1')() ** 2) == '\\operatorname{ab}_{1}^{2}{\\left( \\right)}'
    assert latex(Function('ab12')() ** 2) == '\\operatorname{ab}_{12}^{2}{\\left( \\right)}'
    assert latex(Function('ab_1')() ** 2) == '\\operatorname{ab}_{1}^{2}{\\left( \\right)}'
    assert latex(Function('ab_12')() ** 2) == '\\operatorname{ab}_{12}^{2}{\\left( \\right)}'
    assert latex(Function('ab')(Symbol('x')) ** 2) == '\\operatorname{ab}^{2}{\\left(x \\right)}'
    assert latex(Function('ab1')(Symbol('x')) ** 2) == '\\operatorname{ab}_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('ab12')(Symbol('x')) ** 2) == '\\operatorname{ab}_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('ab_1')(Symbol('x')) ** 2) == '\\operatorname{ab}_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('ab_12')(Symbol('x')) ** 2) == '\\operatorname{ab}_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a')) == 'a'
    assert latex(Function('a1')) == 'a_{1}'
    assert latex(Function('a12')) == 'a_{12}'
    assert latex(Function('a_1')) == 'a_{1}'
    assert latex(Function('a_12')) == 'a_{12}'
    assert latex(Function('a')()) == 'a{\\left( \\right)}'
    assert latex(Function('a1')()) == 'a_{1}{\\left( \\right)}'
    assert latex(Function('a12')()) == 'a_{12}{\\left( \\right)}'
    assert latex(Function('a_1')()) == 'a_{1}{\\left( \\right)}'
    assert latex(Function('a_12')()) == 'a_{12}{\\left( \\right)}'
    assert latex(Function('a')() ** 2) == 'a^{2}{\\left( \\right)}'
    assert latex(Function('a1')() ** 2) == 'a_{1}^{2}{\\left( \\right)}'
    assert latex(Function('a12')() ** 2) == 'a_{12}^{2}{\\left( \\right)}'
    assert latex(Function('a_1')() ** 2) == 'a_{1}^{2}{\\left( \\right)}'
    assert latex(Function('a_12')() ** 2) == 'a_{12}^{2}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** 2) == 'a^{2}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** 2) == 'a_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** 2) == 'a_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** 2) == 'a_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** 2) == 'a_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a')() ** 32) == 'a^{32}{\\left( \\right)}'
    assert latex(Function('a1')() ** 32) == 'a_{1}^{32}{\\left( \\right)}'
    assert latex(Function('a12')() ** 32) == 'a_{12}^{32}{\\left( \\right)}'
    assert latex(Function('a_1')() ** 32) == 'a_{1}^{32}{\\left( \\right)}'
    assert latex(Function('a_12')() ** 32) == 'a_{12}^{32}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** 32) == 'a^{32}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** 32) == 'a_{1}^{32}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** 32) == 'a_{12}^{32}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** 32) == 'a_{1}^{32}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** 32) == 'a_{12}^{32}{\\left(x \\right)}'
    assert latex(Function('a')() ** a) == 'a^{a}{\\left( \\right)}'
    assert latex(Function('a1')() ** a) == 'a_{1}^{a}{\\left( \\right)}'
    assert latex(Function('a12')() ** a) == 'a_{12}^{a}{\\left( \\right)}'
    assert latex(Function('a_1')() ** a) == 'a_{1}^{a}{\\left( \\right)}'
    assert latex(Function('a_12')() ** a) == 'a_{12}^{a}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** a) == 'a^{a}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** a) == 'a_{1}^{a}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** a) == 'a_{12}^{a}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** a) == 'a_{1}^{a}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** a) == 'a_{12}^{a}{\\left(x \\right)}'
    ab = Symbol('ab')
    assert latex(Function('a')() ** ab) == 'a^{ab}{\\left( \\right)}'
    assert latex(Function('a1')() ** ab) == 'a_{1}^{ab}{\\left( \\right)}'
    assert latex(Function('a12')() ** ab) == 'a_{12}^{ab}{\\left( \\right)}'
    assert latex(Function('a_1')() ** ab) == 'a_{1}^{ab}{\\left( \\right)}'
    assert latex(Function('a_12')() ** ab) == 'a_{12}^{ab}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** ab) == 'a^{ab}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** ab) == 'a_{1}^{ab}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** ab) == 'a_{12}^{ab}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** ab) == 'a_{1}^{ab}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** ab) == 'a_{12}^{ab}{\\left(x \\right)}'
    assert latex(Function('a^12')(x)) == 'a^{12}{\\left(x \\right)}'
    assert latex(Function('a^12')(x) ** ab) == '\\left(a^{12}\\right)^{ab}{\\left(x \\right)}'
    assert latex(Function('a__12')(x)) == 'a^{12}{\\left(x \\right)}'
    assert latex(Function('a__12')(x) ** ab) == '\\left(a^{12}\\right)^{ab}{\\left(x \\right)}'
    assert latex(Function('a_1__1_2')(x)) == 'a^{1}_{1 2}{\\left(x \\right)}'
    omega1 = Function('omega1')
    assert latex(omega1) == '\\omega_{1}'
    assert latex(omega1(x)) == '\\omega_{1}{\\left(x \\right)}'
    assert latex(sin(x)) == '\\sin{\\left(x \\right)}'
    assert latex(sin(x), fold_func_brackets=True) == '\\sin {x}'
    assert latex(sin(2 * x ** 2), fold_func_brackets=True) == '\\sin {2 x^{2}}'
    assert latex(sin(x ** 2), fold_func_brackets=True) == '\\sin {x^{2}}'
    assert latex(asin(x) ** 2) == '\\operatorname{asin}^{2}{\\left(x \\right)}'
    assert latex(asin(x) ** 2, inv_trig_style='full') == '\\arcsin^{2}{\\left(x \\right)}'
    assert latex(asin(x) ** 2, inv_trig_style='power') == '\\sin^{-1}{\\left(x \\right)}^{2}'
    assert latex(asin(x ** 2), inv_trig_style='power', fold_func_brackets=True) == '\\sin^{-1} {x^{2}}'
    assert latex(acsc(x), inv_trig_style='full') == '\\operatorname{arccsc}{\\left(x \\right)}'
    assert latex(asinh(x), inv_trig_style='full') == '\\operatorname{arsinh}{\\left(x \\right)}'
    assert latex(factorial(k)) == 'k!'
    assert latex(factorial(-k)) == '\\left(- k\\right)!'
    assert latex(factorial(k) ** 2) == 'k!^{2}'
    assert latex(subfactorial(k)) == '!k'
    assert latex(subfactorial(-k)) == '!\\left(- k\\right)'
    assert latex(subfactorial(k) ** 2) == '\\left(!k\\right)^{2}'
    assert latex(factorial2(k)) == 'k!!'
    assert latex(factorial2(-k)) == '\\left(- k\\right)!!'
    assert latex(factorial2(k) ** 2) == 'k!!^{2}'
    assert latex(binomial(2, k)) == '{\\binom{2}{k}}'
    assert latex(binomial(2, k) ** 2) == '{\\binom{2}{k}}^{2}'
    assert latex(FallingFactorial(3, k)) == '{\\left(3\\right)}_{k}'
    assert latex(RisingFactorial(3, k)) == '{3}^{\\left(k\\right)}'
    assert latex(floor(x)) == '\\left\\lfloor{x}\\right\\rfloor'
    assert latex(ceiling(x)) == '\\left\\lceil{x}\\right\\rceil'
    assert latex(frac(x)) == '\\operatorname{frac}{\\left(x\\right)}'
    assert latex(floor(x) ** 2) == '\\left\\lfloor{x}\\right\\rfloor^{2}'
    assert latex(ceiling(x) ** 2) == '\\left\\lceil{x}\\right\\rceil^{2}'
    assert latex(frac(x) ** 2) == '\\operatorname{frac}{\\left(x\\right)}^{2}'
    assert latex(Min(x, 2, x ** 3)) == '\\min\\left(2, x, x^{3}\\right)'
    assert latex(Min(x, y) ** 2) == '\\min\\left(x, y\\right)^{2}'
    assert latex(Max(x, 2, x ** 3)) == '\\max\\left(2, x, x^{3}\\right)'
    assert latex(Max(x, y) ** 2) == '\\max\\left(x, y\\right)^{2}'
    assert latex(Abs(x)) == '\\left|{x}\\right|'
    assert latex(Abs(x) ** 2) == '\\left|{x}\\right|^{2}'
    assert latex(re(x)) == '\\operatorname{re}{\\left(x\\right)}'
    assert latex(re(x + y)) == '\\operatorname{re}{\\left(x\\right)} + \\operatorname{re}{\\left(y\\right)}'
    assert latex(im(x)) == '\\operatorname{im}{\\left(x\\right)}'
    assert latex(conjugate(x)) == '\\overline{x}'
    assert latex(conjugate(x) ** 2) == '\\overline{x}^{2}'
    assert latex(conjugate(x ** 2)) == '\\overline{x}^{2}'
    assert latex(gamma(x)) == '\\Gamma\\left(x\\right)'
    w = Wild('w')
    assert latex(gamma(w)) == '\\Gamma\\left(w\\right)'
    assert latex(Order(x)) == 'O\\left(x\\right)'
    assert latex(Order(x, x)) == 'O\\left(x\\right)'
    assert latex(Order(x, (x, 0))) == 'O\\left(x\\right)'
    assert latex(Order(x, (x, oo))) == 'O\\left(x; x\\rightarrow \\infty\\right)'
    assert latex(Order(x - y, (x, y))) == 'O\\left(x - y; x\\rightarrow y\\right)'
    assert latex(Order(x, x, y)) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( 0, \\  0\\right)\\right)'
    assert latex(Order(x, x, y)) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( 0, \\  0\\right)\\right)'
    assert latex(Order(x, (x, oo), (y, oo))) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( \\infty, \\  \\infty\\right)\\right)'
    assert latex(lowergamma(x, y)) == '\\gamma\\left(x, y\\right)'
    assert latex(lowergamma(x, y) ** 2) == '\\gamma^{2}\\left(x, y\\right)'
    assert latex(uppergamma(x, y)) == '\\Gamma\\left(x, y\\right)'
    assert latex(uppergamma(x, y) ** 2) == '\\Gamma^{2}\\left(x, y\\right)'
    assert latex(cot(x)) == '\\cot{\\left(x \\right)}'
    assert latex(coth(x)) == '\\coth{\\left(x \\right)}'
    assert latex(re(x)) == '\\operatorname{re}{\\left(x\\right)}'
    assert latex(im(x)) == '\\operatorname{im}{\\left(x\\right)}'
    assert latex(root(x, y)) == 'x^{\\frac{1}{y}}'
    assert latex(arg(x)) == '\\arg{\\left(x \\right)}'
    assert latex(zeta(x)) == '\\zeta\\left(x\\right)'
    assert latex(zeta(x) ** 2) == '\\zeta^{2}\\left(x\\right)'
    assert latex(zeta(x, y)) == '\\zeta\\left(x, y\\right)'
    assert latex(zeta(x, y) ** 2) == '\\zeta^{2}\\left(x, y\\right)'
    assert latex(dirichlet_eta(x)) == '\\eta\\left(x\\right)'
    assert latex(dirichlet_eta(x) ** 2) == '\\eta^{2}\\left(x\\right)'
    assert latex(polylog(x, y)) == '\\operatorname{Li}_{x}\\left(y\\right)'
    assert latex(polylog(x, y) ** 2) == '\\operatorname{Li}_{x}^{2}\\left(y\\right)'
    assert latex(lerchphi(x, y, n)) == '\\Phi\\left(x, y, n\\right)'
    assert latex(lerchphi(x, y, n) ** 2) == '\\Phi^{2}\\left(x, y, n\\right)'
    assert latex(stieltjes(x)) == '\\gamma_{x}'
    assert latex(stieltjes(x) ** 2) == '\\gamma_{x}^{2}'
    assert latex(stieltjes(x, y)) == '\\gamma_{x}\\left(y\\right)'
    assert latex(stieltjes(x, y) ** 2) == '\\gamma_{x}\\left(y\\right)^{2}'
    assert latex(elliptic_k(z)) == 'K\\left(z\\right)'
    assert latex(elliptic_k(z) ** 2) == 'K^{2}\\left(z\\right)'
    assert latex(elliptic_f(x, y)) == 'F\\left(x\\middle| y\\right)'
    assert latex(elliptic_f(x, y) ** 2) == 'F^{2}\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(x, y)) == 'E\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(x, y) ** 2) == 'E^{2}\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(z)) == 'E\\left(z\\right)'
    assert latex(elliptic_e(z) ** 2) == 'E^{2}\\left(z\\right)'
    assert latex(elliptic_pi(x, y, z)) == '\\Pi\\left(x; y\\middle| z\\right)'
    assert latex(elliptic_pi(x, y, z) ** 2) == '\\Pi^{2}\\left(x; y\\middle| z\\right)'
    assert latex(elliptic_pi(x, y)) == '\\Pi\\left(x\\middle| y\\right)'
    assert latex(elliptic_pi(x, y) ** 2) == '\\Pi^{2}\\left(x\\middle| y\\right)'
    assert latex(Ei(x)) == '\\operatorname{Ei}{\\left(x \\right)}'
    assert latex(Ei(x) ** 2) == '\\operatorname{Ei}^{2}{\\left(x \\right)}'
    assert latex(expint(x, y)) == '\\operatorname{E}_{x}\\left(y\\right)'
    assert latex(expint(x, y) ** 2) == '\\operatorname{E}_{x}^{2}\\left(y\\right)'
    assert latex(Shi(x) ** 2) == '\\operatorname{Shi}^{2}{\\left(x \\right)}'
    assert latex(Si(x) ** 2) == '\\operatorname{Si}^{2}{\\left(x \\right)}'
    assert latex(Ci(x) ** 2) == '\\operatorname{Ci}^{2}{\\left(x \\right)}'
    assert latex(Chi(x) ** 2) == '\\operatorname{Chi}^{2}\\left(x\\right)'
    assert latex(Chi(x)) == '\\operatorname{Chi}\\left(x\\right)'
    assert latex(jacobi(n, a, b, x)) == 'P_{n}^{\\left(a,b\\right)}\\left(x\\right)'
    assert latex(jacobi(n, a, b, x) ** 2) == '\\left(P_{n}^{\\left(a,b\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(gegenbauer(n, a, x)) == 'C_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(gegenbauer(n, a, x) ** 2) == '\\left(C_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(chebyshevt(n, x)) == 'T_{n}\\left(x\\right)'
    assert latex(chebyshevt(n, x) ** 2) == '\\left(T_{n}\\left(x\\right)\\right)^{2}'
    assert latex(chebyshevu(n, x)) == 'U_{n}\\left(x\\right)'
    assert latex(chebyshevu(n, x) ** 2) == '\\left(U_{n}\\left(x\\right)\\right)^{2}'
    assert latex(legendre(n, x)) == 'P_{n}\\left(x\\right)'
    assert latex(legendre(n, x) ** 2) == '\\left(P_{n}\\left(x\\right)\\right)^{2}'
    assert latex(assoc_legendre(n, a, x)) == 'P_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(assoc_legendre(n, a, x) ** 2) == '\\left(P_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(laguerre(n, x)) == 'L_{n}\\left(x\\right)'
    assert latex(laguerre(n, x) ** 2) == '\\left(L_{n}\\left(x\\right)\\right)^{2}'
    assert latex(assoc_laguerre(n, a, x)) == 'L_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(assoc_laguerre(n, a, x) ** 2) == '\\left(L_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(hermite(n, x)) == 'H_{n}\\left(x\\right)'
    assert latex(hermite(n, x) ** 2) == '\\left(H_{n}\\left(x\\right)\\right)^{2}'
    theta = Symbol('theta', real=True)
    phi = Symbol('phi', real=True)
    assert latex(Ynm(n, m, theta, phi)) == 'Y_{n}^{m}\\left(\\theta,\\phi\\right)'
    assert latex(Ynm(n, m, theta, phi) ** 3) == '\\left(Y_{n}^{m}\\left(\\theta,\\phi\\right)\\right)^{3}'
    assert latex(Znm(n, m, theta, phi)) == 'Z_{n}^{m}\\left(\\theta,\\phi\\right)'
    assert latex(Znm(n, m, theta, phi) ** 3) == '\\left(Z_{n}^{m}\\left(\\theta,\\phi\\right)\\right)^{3}'
    assert latex(polar_lift(0)) == '\\operatorname{polar\\_lift}{\\left(0 \\right)}'
    assert latex(polar_lift(0) ** 3) == '\\operatorname{polar\\_lift}^{3}{\\left(0 \\right)}'
    assert latex(totient(n)) == '\\phi\\left(n\\right)'
    assert latex(totient(n) ** 2) == '\\left(\\phi\\left(n\\right)\\right)^{2}'
    assert latex(reduced_totient(n)) == '\\lambda\\left(n\\right)'
    assert latex(reduced_totient(n) ** 2) == '\\left(\\lambda\\left(n\\right)\\right)^{2}'
    assert latex(divisor_sigma(x)) == '\\sigma\\left(x\\right)'
    assert latex(divisor_sigma(x) ** 2) == '\\sigma^{2}\\left(x\\right)'
    assert latex(divisor_sigma(x, y)) == '\\sigma_y\\left(x\\right)'
    assert latex(divisor_sigma(x, y) ** 2) == '\\sigma^{2}_y\\left(x\\right)'
    assert latex(udivisor_sigma(x)) == '\\sigma^*\\left(x\\right)'
    assert latex(udivisor_sigma(x) ** 2) == '\\sigma^*^{2}\\left(x\\right)'
    assert latex(udivisor_sigma(x, y)) == '\\sigma^*_y\\left(x\\right)'
    assert latex(udivisor_sigma(x, y) ** 2) == '\\sigma^*^{2}_y\\left(x\\right)'
    assert latex(primenu(n)) == '\\nu\\left(n\\right)'
    assert latex(primenu(n) ** 2) == '\\left(\\nu\\left(n\\right)\\right)^{2}'
    assert latex(primeomega(n)) == '\\Omega\\left(n\\right)'
    assert latex(primeomega(n) ** 2) == '\\left(\\Omega\\left(n\\right)\\right)^{2}'
    assert latex(LambertW(n)) == 'W\\left(n\\right)'
    assert latex(LambertW(n, -1)) == 'W_{-1}\\left(n\\right)'
    assert latex(LambertW(n, k)) == 'W_{k}\\left(n\\right)'
    assert latex(LambertW(n) * LambertW(n)) == 'W^{2}\\left(n\\right)'
    assert latex(Pow(LambertW(n), 2)) == 'W^{2}\\left(n\\right)'
    assert latex(LambertW(n) ** k) == 'W^{k}\\left(n\\right)'
    assert latex(LambertW(n, k) ** p) == 'W^{p}_{k}\\left(n\\right)'
    assert latex(Mod(x, 7)) == 'x \\bmod 7'
    assert latex(Mod(x + 1, 7)) == '\\left(x + 1\\right) \\bmod 7'
    assert latex(Mod(7, x + 1)) == '7 \\bmod \\left(x + 1\\right)'
    assert latex(Mod(2 * x, 7)) == '2 x \\bmod 7'
    assert latex(Mod(7, 2 * x)) == '7 \\bmod 2 x'
    assert latex(Mod(x, 7) + 1) == '\\left(x \\bmod 7\\right) + 1'
    assert latex(2 * Mod(x, 7)) == '2 \\left(x \\bmod 7\\right)'
    assert latex(Mod(7, 2 * x) ** n) == '\\left(7 \\bmod 2 x\\right)^{n}'
    fjlkd = Function('fjlkd')
    assert latex(fjlkd(x)) == '\\operatorname{fjlkd}{\\left(x \\right)}'
    assert latex(fjlkd) == '\\operatorname{fjlkd}'

def test_function_subclass_different_name():
    if False:
        i = 10
        return i + 15

    class mygamma(gamma):
        pass
    assert latex(mygamma) == '\\operatorname{mygamma}'
    assert latex(mygamma(x)) == '\\operatorname{mygamma}{\\left(x \\right)}'

def test_hyper_printing():
    if False:
        for i in range(10):
            print('nop')
    from sympy.abc import x, z
    assert latex(meijerg(Tuple(pi, pi, x), Tuple(1), (0, 1), Tuple(1, 2, 3 / pi), z)) == '{G_{4, 5}^{2, 3}\\left(\\begin{matrix} \\pi, \\pi, x & 1 \\\\0, 1 & 1, 2, \\frac{3}{\\pi} \\end{matrix} \\middle| {z} \\right)}'
    assert latex(meijerg(Tuple(), Tuple(1), (0,), Tuple(), z)) == '{G_{1, 1}^{1, 0}\\left(\\begin{matrix}  & 1 \\\\0 &  \\end{matrix} \\middle| {z} \\right)}'
    assert latex(hyper((x, 2), (3,), z)) == '{{}_{2}F_{1}\\left(\\begin{matrix} 2, x \\\\ 3 \\end{matrix}\\middle| {z} \\right)}'
    assert latex(hyper(Tuple(), Tuple(1), z)) == '{{}_{0}F_{1}\\left(\\begin{matrix}  \\\\ 1 \\end{matrix}\\middle| {z} \\right)}'

def test_latex_bessel():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import besselj, bessely, besseli, besselk, hankel1, hankel2, jn, yn, hn1, hn2
    from sympy.abc import z
    assert latex(besselj(n, z ** 2) ** k) == 'J^{k}_{n}\\left(z^{2}\\right)'
    assert latex(bessely(n, z)) == 'Y_{n}\\left(z\\right)'
    assert latex(besseli(n, z)) == 'I_{n}\\left(z\\right)'
    assert latex(besselk(n, z)) == 'K_{n}\\left(z\\right)'
    assert latex(hankel1(n, z ** 2) ** 2) == '\\left(H^{(1)}_{n}\\left(z^{2}\\right)\\right)^{2}'
    assert latex(hankel2(n, z)) == 'H^{(2)}_{n}\\left(z\\right)'
    assert latex(jn(n, z)) == 'j_{n}\\left(z\\right)'
    assert latex(yn(n, z)) == 'y_{n}\\left(z\\right)'
    assert latex(hn1(n, z)) == 'h^{(1)}_{n}\\left(z\\right)'
    assert latex(hn2(n, z)) == 'h^{(2)}_{n}\\left(z\\right)'

def test_latex_fresnel():
    if False:
        while True:
            i = 10
    from sympy.functions.special.error_functions import fresnels, fresnelc
    from sympy.abc import z
    assert latex(fresnels(z)) == 'S\\left(z\\right)'
    assert latex(fresnelc(z)) == 'C\\left(z\\right)'
    assert latex(fresnels(z) ** 2) == 'S^{2}\\left(z\\right)'
    assert latex(fresnelc(z) ** 2) == 'C^{2}\\left(z\\right)'

def test_latex_brackets():
    if False:
        for i in range(10):
            print('nop')
    assert latex((-1) ** x) == '\\left(-1\\right)^{x}'

def test_latex_indexed():
    if False:
        for i in range(10):
            print('nop')
    Psi_symbol = Symbol('Psi_0', complex=True, real=False)
    Psi_indexed = IndexedBase(Symbol('Psi', complex=True, real=False))
    symbol_latex = latex(Psi_symbol * conjugate(Psi_symbol))
    indexed_latex = latex(Psi_indexed[0] * conjugate(Psi_indexed[0]))
    assert symbol_latex == '\\Psi_{0} \\overline{\\Psi_{0}}'
    assert indexed_latex == '\\overline{{\\Psi}_{0}} {\\Psi}_{0}'
    interval = '\\mathrel{..}\\nobreak '
    assert latex(Indexed('x1', Symbol('i'))) == '{x_{1}}_{i}'
    assert latex(Indexed('x2', Idx('i'))) == '{x_{2}}_{i}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')))) == '{x_{3}}_{{i}_{0' + interval + 'N - 1}}'
    assert latex(Indexed('x3', Idx('i', Symbol('N') + 1))) == '{x_{3}}_{{i}_{0' + interval + 'N}}'
    assert latex(Indexed('x4', Idx('i', (Symbol('a'), Symbol('b'))))) == '{x_{4}}_{{i}_{a' + interval + 'b}}'
    assert latex(IndexedBase('gamma')) == '\\gamma'
    assert latex(IndexedBase('a b')) == 'a b'
    assert latex(IndexedBase('a_b')) == 'a_{b}'

def test_latex_derivatives():
    if False:
        return 10
    assert latex(diff(x ** 3, x, evaluate=False)) == '\\frac{d}{d x} x^{3}'
    assert latex(diff(sin(x) + x ** 2, x, evaluate=False)) == '\\frac{d}{d x} \\left(x^{2} + \\sin{\\left(x \\right)}\\right)'
    assert latex(diff(diff(sin(x) + x ** 2, x, evaluate=False), evaluate=False)) == '\\frac{d^{2}}{d x^{2}} \\left(x^{2} + \\sin{\\left(x \\right)}\\right)'
    assert latex(diff(diff(diff(sin(x) + x ** 2, x, evaluate=False), evaluate=False), evaluate=False)) == '\\frac{d^{3}}{d x^{3}} \\left(x^{2} + \\sin{\\left(x \\right)}\\right)'
    assert latex(diff(sin(x * y), x, evaluate=False)) == '\\frac{\\partial}{\\partial x} \\sin{\\left(x y \\right)}'
    assert latex(diff(sin(x * y) + x ** 2, x, evaluate=False)) == '\\frac{\\partial}{\\partial x} \\left(x^{2} + \\sin{\\left(x y \\right)}\\right)'
    assert latex(diff(diff(sin(x * y) + x ** 2, x, evaluate=False), x, evaluate=False)) == '\\frac{\\partial^{2}}{\\partial x^{2}} \\left(x^{2} + \\sin{\\left(x y \\right)}\\right)'
    assert latex(diff(diff(diff(sin(x * y) + x ** 2, x, evaluate=False), x, evaluate=False), x, evaluate=False)) == '\\frac{\\partial^{3}}{\\partial x^{3}} \\left(x^{2} + \\sin{\\left(x y \\right)}\\right)'
    f = Function('f')
    assert latex(diff(diff(f(x, y), x, evaluate=False), y, evaluate=False)) == '\\frac{\\partial^{2}}{\\partial y\\partial x} ' + latex(f(x, y))
    assert latex(diff(diff(diff(f(x, y), x, evaluate=False), x, evaluate=False), y, evaluate=False)) == '\\frac{\\partial^{3}}{\\partial y\\partial x^{2}} ' + latex(f(x, y))
    assert latex(diff(-diff(y ** 2, x, evaluate=False), x, evaluate=False)) == '\\frac{d}{d x} \\left(- \\frac{d}{d x} y^{2}\\right)'
    assert latex(diff(diff(-diff(diff(y, x, evaluate=False), x, evaluate=False), x, evaluate=False), x, evaluate=False)) == '\\frac{d^{2}}{d x^{2}} \\left(- \\frac{d^{2}}{d x^{2}} y\\right)'
    assert latex(diff(Integral(exp(-x * y), (x, 0, oo)), y, evaluate=False)) == '\\frac{d}{d y} \\int\\limits_{0}^{\\infty} e^{- x y}\\, dx'
    assert latex(diff(x, x, evaluate=False) ** 2) == '\\left(\\frac{d}{d x} x\\right)^{2}'
    assert latex(diff(f(x), x) ** 2) == '\\left(\\frac{d}{d x} f{\\left(x \\right)}\\right)^{2}'
    assert latex(diff(f(x), (x, n))) == '\\frac{d^{n}}{d x^{n}} f{\\left(x \\right)}'
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    assert latex(diff(f(x1, x2), x1)) == '\\frac{\\partial}{\\partial x_{1}} f{\\left(x_{1},x_{2} \\right)}'
    n1 = Symbol('n1')
    assert latex(diff(f(x), (x, n1))) == '\\frac{d^{n_{1}}}{d x^{n_{1}}} f{\\left(x \\right)}'
    n2 = Symbol('n2')
    assert latex(diff(f(x), (x, Max(n1, n2)))) == '\\frac{d^{\\max\\left(n_{1}, n_{2}\\right)}}{d x^{\\max\\left(n_{1}, n_{2}\\right)}} f{\\left(x \\right)}'
    assert latex(diff(f(x), x), diff_operator='rd') == '\\frac{\\mathrm{d}}{\\mathrm{d} x} f{\\left(x \\right)}'

def test_latex_subs():
    if False:
        while True:
            i = 10
    assert latex(Subs(x * y, (x, y), (1, 2))) == '\\left. x y \\right|_{\\substack{ x=1\\\\ y=2 }}'

def test_latex_integrals():
    if False:
        while True:
            i = 10
    assert latex(Integral(log(x), x)) == '\\int \\log{\\left(x \\right)}\\, dx'
    assert latex(Integral(x ** 2, (x, 0, 1))) == '\\int\\limits_{0}^{1} x^{2}\\, dx'
    assert latex(Integral(x ** 2, (x, 10, 20))) == '\\int\\limits_{10}^{20} x^{2}\\, dx'
    assert latex(Integral(y * x ** 2, (x, 0, 1), y)) == '\\int\\int\\limits_{0}^{1} x^{2} y\\, dx\\, dy'
    assert latex(Integral(y * x ** 2, (x, 0, 1), y), mode='equation*') == '\\begin{equation*}\\int\\int\\limits_{0}^{1} x^{2} y\\, dx\\, dy\\end{equation*}'
    assert latex(Integral(y * x ** 2, (x, 0, 1), y), mode='equation*', itex=True) == '$$\\int\\int_{0}^{1} x^{2} y\\, dx\\, dy$$'
    assert latex(Integral(x, (x, 0))) == '\\int\\limits^{0} x\\, dx'
    assert latex(Integral(x * y, x, y)) == '\\iint x y\\, dx\\, dy'
    assert latex(Integral(x * y * z, x, y, z)) == '\\iiint x y z\\, dx\\, dy\\, dz'
    assert latex(Integral(x * y * z * t, x, y, z, t)) == '\\iiiint t x y z\\, dx\\, dy\\, dz\\, dt'
    assert latex(Integral(x, x, x, x, x, x, x)) == '\\int\\int\\int\\int\\int\\int x\\, dx\\, dx\\, dx\\, dx\\, dx\\, dx'
    assert latex(Integral(x, x, y, (z, 0, 1))) == '\\int\\limits_{0}^{1}\\int\\int x\\, dx\\, dy\\, dz'
    assert latex(Integral(-Integral(y ** 2, x), x)) == '\\int \\left(- \\int y^{2}\\, dx\\right)\\, dx'
    assert latex(Integral(-Integral(-Integral(y, x), x), x)) == '\\int \\left(- \\int \\left(- \\int y\\, dx\\right)\\, dx\\right)\\, dx'
    assert latex(Integral(z, z) ** 2) == '\\left(\\int z\\, dz\\right)^{2}'
    assert latex(Integral(x + z, z)) == '\\int \\left(x + z\\right)\\, dz'
    assert latex(Integral(x + z / 2, z)) == '\\int \\left(x + \\frac{z}{2}\\right)\\, dz'
    assert latex(Integral(x ** y, z)) == '\\int x^{y}\\, dz'
    assert latex(Integral(x, x), diff_operator='rd') == '\\int x\\, \\mathrm{d}x'
    assert latex(Integral(x, (x, 0, 1)), diff_operator='rd') == '\\int\\limits_{0}^{1} x\\, \\mathrm{d}x'

def test_latex_sets():
    if False:
        i = 10
        return i + 15
    for s in (frozenset, set):
        assert latex(s([x * y, x ** 2])) == '\\left\\{x^{2}, x y\\right\\}'
        assert latex(s(range(1, 6))) == '\\left\\{1, 2, 3, 4, 5\\right\\}'
        assert latex(s(range(1, 13))) == '\\left\\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\\right\\}'
    s = FiniteSet
    assert latex(s(*[x * y, x ** 2])) == '\\left\\{x^{2}, x y\\right\\}'
    assert latex(s(*range(1, 6))) == '\\left\\{1, 2, 3, 4, 5\\right\\}'
    assert latex(s(*range(1, 13))) == '\\left\\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\\right\\}'

def test_latex_SetExpr():
    if False:
        for i in range(10):
            print('nop')
    iv = Interval(1, 3)
    se = SetExpr(iv)
    assert latex(se) == 'SetExpr\\left(\\left[1, 3\\right]\\right)'

def test_latex_Range():
    if False:
        while True:
            i = 10
    assert latex(Range(1, 51)) == '\\left\\{1, 2, \\ldots, 50\\right\\}'
    assert latex(Range(1, 4)) == '\\left\\{1, 2, 3\\right\\}'
    assert latex(Range(0, 3, 1)) == '\\left\\{0, 1, 2\\right\\}'
    assert latex(Range(0, 30, 1)) == '\\left\\{0, 1, \\ldots, 29\\right\\}'
    assert latex(Range(30, 1, -1)) == '\\left\\{30, 29, \\ldots, 2\\right\\}'
    assert latex(Range(0, oo, 2)) == '\\left\\{0, 2, \\ldots\\right\\}'
    assert latex(Range(oo, -2, -2)) == '\\left\\{\\ldots, 2, 0\\right\\}'
    assert latex(Range(-2, -oo, -1)) == '\\left\\{-2, -3, \\ldots\\right\\}'
    assert latex(Range(-oo, oo)) == '\\left\\{\\ldots, -1, 0, 1, \\ldots\\right\\}'
    assert latex(Range(oo, -oo, -1)) == '\\left\\{\\ldots, 1, 0, -1, \\ldots\\right\\}'
    (a, b, c) = symbols('a:c')
    assert latex(Range(a, b, c)) == '\\text{Range}\\left(a, b, c\\right)'
    assert latex(Range(a, 10, 1)) == '\\text{Range}\\left(a, 10\\right)'
    assert latex(Range(0, b, 1)) == '\\text{Range}\\left(b\\right)'
    assert latex(Range(0, 10, c)) == '\\text{Range}\\left(0, 10, c\\right)'
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)
    assert latex(Range(i, i + 3)) == '\\left\\{i, i + 1, i + 2\\right\\}'
    assert latex(Range(-oo, n, 2)) == '\\left\\{\\ldots, n - 4, n - 2\\right\\}'
    assert latex(Range(p, oo)) == '\\left\\{p, p + 1, \\ldots\\right\\}'
    assert latex(Range(a, a + 3)) == '\\text{Range}\\left(a, a + 3\\right)'

def test_latex_sequences():
    if False:
        print('Hello World!')
    s1 = SeqFormula(a ** 2, (0, oo))
    s2 = SeqPer((1, 2))
    latex_str = '\\left[0, 1, 4, 9, \\ldots\\right]'
    assert latex(s1) == latex_str
    latex_str = '\\left[1, 2, 1, 2, \\ldots\\right]'
    assert latex(s2) == latex_str
    s3 = SeqFormula(a ** 2, (0, 2))
    s4 = SeqPer((1, 2), (0, 2))
    latex_str = '\\left[0, 1, 4\\right]'
    assert latex(s3) == latex_str
    latex_str = '\\left[1, 2, 1\\right]'
    assert latex(s4) == latex_str
    s5 = SeqFormula(a ** 2, (-oo, 0))
    s6 = SeqPer((1, 2), (-oo, 0))
    latex_str = '\\left[\\ldots, 9, 4, 1, 0\\right]'
    assert latex(s5) == latex_str
    latex_str = '\\left[\\ldots, 2, 1, 2, 1\\right]'
    assert latex(s6) == latex_str
    latex_str = '\\left[1, 3, 5, 11, \\ldots\\right]'
    assert latex(SeqAdd(s1, s2)) == latex_str
    latex_str = '\\left[1, 3, 5\\right]'
    assert latex(SeqAdd(s3, s4)) == latex_str
    latex_str = '\\left[\\ldots, 11, 5, 3, 1\\right]'
    assert latex(SeqAdd(s5, s6)) == latex_str
    latex_str = '\\left[0, 2, 4, 18, \\ldots\\right]'
    assert latex(SeqMul(s1, s2)) == latex_str
    latex_str = '\\left[0, 2, 4\\right]'
    assert latex(SeqMul(s3, s4)) == latex_str
    latex_str = '\\left[\\ldots, 18, 4, 2, 0\\right]'
    assert latex(SeqMul(s5, s6)) == latex_str
    s7 = SeqFormula(a ** 2, (a, 0, x))
    latex_str = '\\left\\{a^{2}\\right\\}_{a=0}^{x}'
    assert latex(s7) == latex_str
    b = Symbol('b')
    s8 = SeqFormula(b * a ** 2, (a, 0, 2))
    latex_str = '\\left[0, b, 4 b\\right]'
    assert latex(s8) == latex_str

def test_latex_FourierSeries():
    if False:
        for i in range(10):
            print('nop')
    latex_str = '2 \\sin{\\left(x \\right)} - \\sin{\\left(2 x \\right)} + \\frac{2 \\sin{\\left(3 x \\right)}}{3} + \\ldots'
    assert latex(fourier_series(x, (x, -pi, pi))) == latex_str

def test_latex_FormalPowerSeries():
    if False:
        while True:
            i = 10
    latex_str = '\\sum_{k=1}^{\\infty} - \\frac{\\left(-1\\right)^{- k} x^{k}}{k}'
    assert latex(fps(log(1 + x))) == latex_str

def test_latex_intervals():
    if False:
        for i in range(10):
            print('nop')
    a = Symbol('a', real=True)
    assert latex(Interval(0, 0)) == '\\left\\{0\\right\\}'
    assert latex(Interval(0, a)) == '\\left[0, a\\right]'
    assert latex(Interval(0, a, False, False)) == '\\left[0, a\\right]'
    assert latex(Interval(0, a, True, False)) == '\\left(0, a\\right]'
    assert latex(Interval(0, a, False, True)) == '\\left[0, a\\right)'
    assert latex(Interval(0, a, True, True)) == '\\left(0, a\\right)'

def test_latex_AccumuBounds():
    if False:
        while True:
            i = 10
    a = Symbol('a', real=True)
    assert latex(AccumBounds(0, 1)) == '\\left\\langle 0, 1\\right\\rangle'
    assert latex(AccumBounds(0, a)) == '\\left\\langle 0, a\\right\\rangle'
    assert latex(AccumBounds(a + 1, a + 2)) == '\\left\\langle a + 1, a + 2\\right\\rangle'

def test_latex_emptyset():
    if False:
        i = 10
        return i + 15
    assert latex(S.EmptySet) == '\\emptyset'

def test_latex_universalset():
    if False:
        i = 10
        return i + 15
    assert latex(S.UniversalSet) == '\\mathbb{U}'

def test_latex_commutator():
    if False:
        for i in range(10):
            print('nop')
    A = Operator('A')
    B = Operator('B')
    comm = Commutator(B, A)
    assert latex(comm.doit()) == '- (A B - B A)'

def test_latex_union():
    if False:
        while True:
            i = 10
    assert latex(Union(Interval(0, 1), Interval(2, 3))) == '\\left[0, 1\\right] \\cup \\left[2, 3\\right]'
    assert latex(Union(Interval(1, 1), Interval(2, 2), Interval(3, 4))) == '\\left\\{1, 2\\right\\} \\cup \\left[3, 4\\right]'

def test_latex_intersection():
    if False:
        for i in range(10):
            print('nop')
    assert latex(Intersection(Interval(0, 1), Interval(x, y))) == '\\left[0, 1\\right] \\cap \\left[x, y\\right]'

def test_latex_symmetric_difference():
    if False:
        print('Hello World!')
    assert latex(SymmetricDifference(Interval(2, 5), Interval(4, 7), evaluate=False)) == '\\left[2, 5\\right] \\triangle \\left[4, 7\\right]'

def test_latex_Complement():
    if False:
        print('Hello World!')
    assert latex(Complement(S.Reals, S.Naturals)) == '\\mathbb{R} \\setminus \\mathbb{N}'

def test_latex_productset():
    if False:
        return 10
    line = Interval(0, 1)
    bigline = Interval(0, 10)
    fset = FiniteSet(1, 2, 3)
    assert latex(line ** 2) == '%s^{2}' % latex(line)
    assert latex(line ** 10) == '%s^{10}' % latex(line)
    assert latex((line * bigline * fset).flatten()) == '%s \\times %s \\times %s' % (latex(line), latex(bigline), latex(fset))

def test_latex_powerset():
    if False:
        return 10
    fset = FiniteSet(1, 2, 3)
    assert latex(PowerSet(fset)) == '\\mathcal{P}\\left(\\left\\{1, 2, 3\\right\\}\\right)'

def test_latex_ordinals():
    if False:
        i = 10
        return i + 15
    w = OrdinalOmega()
    assert latex(w) == '\\omega'
    wp = OmegaPower(2, 3)
    assert latex(wp) == '3 \\omega^{2}'
    assert latex(Ordinal(wp, OmegaPower(1, 1))) == '3 \\omega^{2} + \\omega'
    assert latex(Ordinal(OmegaPower(2, 1), OmegaPower(1, 2))) == '\\omega^{2} + 2 \\omega'

def test_set_operators_parenthesis():
    if False:
        for i in range(10):
            print('nop')
    (a, b, c, d) = symbols('a:d')
    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)
    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(A, B, evaluate=False)
    D2 = SymmetricDifference(C, D, evaluate=False)
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)
    assert latex(Intersection(A, U2, evaluate=False)) == '\\left\\{a\\right\\} \\cap \\left(\\left\\{c\\right\\} \\cup \\left\\{d\\right\\}\\right)'
    assert latex(Intersection(U1, U2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cup \\left\\{b\\right\\}\\right) \\cap \\left(\\left\\{c\\right\\} \\cup \\left\\{d\\right\\}\\right)'
    assert latex(Intersection(C1, C2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\setminus \\left\\{b\\right\\}\\right) \\cap \\left(\\left\\{c\\right\\} \\setminus \\left\\{d\\right\\}\\right)'
    assert latex(Intersection(D1, D2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\triangle \\left\\{b\\right\\}\\right) \\cap \\left(\\left\\{c\\right\\} \\triangle \\left\\{d\\right\\}\\right)'
    assert latex(Intersection(P1, P2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\times \\left\\{b\\right\\}\\right) \\cap \\left(\\left\\{c\\right\\} \\times \\left\\{d\\right\\}\\right)'
    assert latex(Union(A, I2, evaluate=False)) == '\\left\\{a\\right\\} \\cup \\left(\\left\\{c\\right\\} \\cap \\left\\{d\\right\\}\\right)'
    assert latex(Union(I1, I2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cap \\left\\{b\\right\\}\\right) \\cup \\left(\\left\\{c\\right\\} \\cap \\left\\{d\\right\\}\\right)'
    assert latex(Union(C1, C2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\setminus \\left\\{b\\right\\}\\right) \\cup \\left(\\left\\{c\\right\\} \\setminus \\left\\{d\\right\\}\\right)'
    assert latex(Union(D1, D2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\triangle \\left\\{b\\right\\}\\right) \\cup \\left(\\left\\{c\\right\\} \\triangle \\left\\{d\\right\\}\\right)'
    assert latex(Union(P1, P2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\times \\left\\{b\\right\\}\\right) \\cup \\left(\\left\\{c\\right\\} \\times \\left\\{d\\right\\}\\right)'
    assert latex(Complement(A, C2, evaluate=False)) == '\\left\\{a\\right\\} \\setminus \\left(\\left\\{c\\right\\} \\setminus \\left\\{d\\right\\}\\right)'
    assert latex(Complement(U1, U2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cup \\left\\{b\\right\\}\\right) \\setminus \\left(\\left\\{c\\right\\} \\cup \\left\\{d\\right\\}\\right)'
    assert latex(Complement(I1, I2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cap \\left\\{b\\right\\}\\right) \\setminus \\left(\\left\\{c\\right\\} \\cap \\left\\{d\\right\\}\\right)'
    assert latex(Complement(D1, D2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\triangle \\left\\{b\\right\\}\\right) \\setminus \\left(\\left\\{c\\right\\} \\triangle \\left\\{d\\right\\}\\right)'
    assert latex(Complement(P1, P2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\times \\left\\{b\\right\\}\\right) \\setminus \\left(\\left\\{c\\right\\} \\times \\left\\{d\\right\\}\\right)'
    assert latex(SymmetricDifference(A, D2, evaluate=False)) == '\\left\\{a\\right\\} \\triangle \\left(\\left\\{c\\right\\} \\triangle \\left\\{d\\right\\}\\right)'
    assert latex(SymmetricDifference(U1, U2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cup \\left\\{b\\right\\}\\right) \\triangle \\left(\\left\\{c\\right\\} \\cup \\left\\{d\\right\\}\\right)'
    assert latex(SymmetricDifference(I1, I2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\cap \\left\\{b\\right\\}\\right) \\triangle \\left(\\left\\{c\\right\\} \\cap \\left\\{d\\right\\}\\right)'
    assert latex(SymmetricDifference(C1, C2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\setminus \\left\\{b\\right\\}\\right) \\triangle \\left(\\left\\{c\\right\\} \\setminus \\left\\{d\\right\\}\\right)'
    assert latex(SymmetricDifference(P1, P2, evaluate=False)) == '\\left(\\left\\{a\\right\\} \\times \\left\\{b\\right\\}\\right) \\triangle \\left(\\left\\{c\\right\\} \\times \\left\\{d\\right\\}\\right)'
    assert latex(ProductSet(A, P2).flatten()) == '\\left\\{a\\right\\} \\times \\left\\{c\\right\\} \\times \\left\\{d\\right\\}'
    assert latex(ProductSet(U1, U2)) == '\\left(\\left\\{a\\right\\} \\cup \\left\\{b\\right\\}\\right) \\times \\left(\\left\\{c\\right\\} \\cup \\left\\{d\\right\\}\\right)'
    assert latex(ProductSet(I1, I2)) == '\\left(\\left\\{a\\right\\} \\cap \\left\\{b\\right\\}\\right) \\times \\left(\\left\\{c\\right\\} \\cap \\left\\{d\\right\\}\\right)'
    assert latex(ProductSet(C1, C2)) == '\\left(\\left\\{a\\right\\} \\setminus \\left\\{b\\right\\}\\right) \\times \\left(\\left\\{c\\right\\} \\setminus \\left\\{d\\right\\}\\right)'
    assert latex(ProductSet(D1, D2)) == '\\left(\\left\\{a\\right\\} \\triangle \\left\\{b\\right\\}\\right) \\times \\left(\\left\\{c\\right\\} \\triangle \\left\\{d\\right\\}\\right)'

def test_latex_Complexes():
    if False:
        return 10
    assert latex(S.Complexes) == '\\mathbb{C}'

def test_latex_Naturals():
    if False:
        for i in range(10):
            print('nop')
    assert latex(S.Naturals) == '\\mathbb{N}'

def test_latex_Naturals0():
    if False:
        i = 10
        return i + 15
    assert latex(S.Naturals0) == '\\mathbb{N}_0'

def test_latex_Integers():
    if False:
        print('Hello World!')
    assert latex(S.Integers) == '\\mathbb{Z}'

def test_latex_ImageSet():
    if False:
        return 10
    x = Symbol('x')
    assert latex(ImageSet(Lambda(x, x ** 2), S.Naturals)) == '\\left\\{x^{2}\\; \\middle|\\; x \\in \\mathbb{N}\\right\\}'
    y = Symbol('y')
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    assert latex(imgset) == '\\left\\{x + y\\; \\middle|\\; x \\in \\left\\{1, 2, 3\\right\\}, y \\in \\left\\{3, 4\\right\\}\\right\\}'
    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    assert latex(imgset) == '\\left\\{x + y\\; \\middle|\\; \\left( x, \\  y\\right) \\in \\left\\{1, 2, 3\\right\\} \\times \\left\\{3, 4\\right\\}\\right\\}'

def test_latex_ConditionSet():
    if False:
        print('Hello World!')
    x = Symbol('x')
    assert latex(ConditionSet(x, Eq(x ** 2, 1), S.Reals)) == '\\left\\{x\\; \\middle|\\; x \\in \\mathbb{R} \\wedge x^{2} = 1 \\right\\}'
    assert latex(ConditionSet(x, Eq(x ** 2, 1), S.UniversalSet)) == '\\left\\{x\\; \\middle|\\; x^{2} = 1 \\right\\}'

def test_latex_ComplexRegion():
    if False:
        print('Hello World!')
    assert latex(ComplexRegion(Interval(3, 5) * Interval(4, 6))) == '\\left\\{x + y i\\; \\middle|\\; x, y \\in \\left[3, 5\\right] \\times \\left[4, 6\\right] \\right\\}'
    assert latex(ComplexRegion(Interval(0, 1) * Interval(0, 2 * pi), polar=True)) == '\\left\\{r \\left(i \\sin{\\left(\\theta \\right)} + \\cos{\\left(\\theta \\right)}\\right)\\; \\middle|\\; r, \\theta \\in \\left[0, 1\\right] \\times \\left[0, 2 \\pi\\right) \\right\\}'

def test_latex_Contains():
    if False:
        while True:
            i = 10
    x = Symbol('x')
    assert latex(Contains(x, S.Naturals)) == 'x \\in \\mathbb{N}'

def test_latex_sum():
    if False:
        print('Hello World!')
    assert latex(Sum(x * y ** 2, (x, -2, 2), (y, -5, 5))) == '\\sum_{\\substack{-2 \\leq x \\leq 2\\\\-5 \\leq y \\leq 5}} x y^{2}'
    assert latex(Sum(x ** 2, (x, -2, 2))) == '\\sum_{x=-2}^{2} x^{2}'
    assert latex(Sum(x ** 2 + y, (x, -2, 2))) == '\\sum_{x=-2}^{2} \\left(x^{2} + y\\right)'
    assert latex(Sum(x ** 2 + y, (x, -2, 2)) ** 2) == '\\left(\\sum_{x=-2}^{2} \\left(x^{2} + y\\right)\\right)^{2}'

def test_latex_product():
    if False:
        return 10
    assert latex(Product(x * y ** 2, (x, -2, 2), (y, -5, 5))) == '\\prod_{\\substack{-2 \\leq x \\leq 2\\\\-5 \\leq y \\leq 5}} x y^{2}'
    assert latex(Product(x ** 2, (x, -2, 2))) == '\\prod_{x=-2}^{2} x^{2}'
    assert latex(Product(x ** 2 + y, (x, -2, 2))) == '\\prod_{x=-2}^{2} \\left(x^{2} + y\\right)'
    assert latex(Product(x, (x, -2, 2)) ** 2) == '\\left(\\prod_{x=-2}^{2} x\\right)^{2}'

def test_latex_limits():
    if False:
        while True:
            i = 10
    assert latex(Limit(x, x, oo)) == '\\lim_{x \\to \\infty} x'
    f = Function('f')
    assert latex(Limit(f(x), x, 0)) == '\\lim_{x \\to 0^+} f{\\left(x \\right)}'
    assert latex(Limit(f(x), x, 0, '-')) == '\\lim_{x \\to 0^-} f{\\left(x \\right)}'
    assert latex(Limit(f(x), x, 0) ** 2) == '\\left(\\lim_{x \\to 0^+} f{\\left(x \\right)}\\right)^{2}'
    assert latex(Limit(f(x), x, 0, dir='+-')) == '\\lim_{x \\to 0} f{\\left(x \\right)}'

def test_latex_log():
    if False:
        i = 10
        return i + 15
    assert latex(log(x)) == '\\log{\\left(x \\right)}'
    assert latex(log(x), ln_notation=True) == '\\ln{\\left(x \\right)}'
    assert latex(log(x) + log(y)) == '\\log{\\left(x \\right)} + \\log{\\left(y \\right)}'
    assert latex(log(x) + log(y), ln_notation=True) == '\\ln{\\left(x \\right)} + \\ln{\\left(y \\right)}'
    assert latex(pow(log(x), x)) == '\\log{\\left(x \\right)}^{x}'
    assert latex(pow(log(x), x), ln_notation=True) == '\\ln{\\left(x \\right)}^{x}'

def test_issue_3568():
    if False:
        while True:
            i = 10
    beta = Symbol('\\beta')
    y = beta + x
    assert latex(y) in ['\\beta + x', 'x + \\beta']
    beta = Symbol('beta')
    y = beta + x
    assert latex(y) in ['\\beta + x', 'x + \\beta']

def test_latex():
    if False:
        return 10
    assert latex((2 * tau) ** Rational(7, 2)) == '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    assert latex((2 * mu) ** Rational(7, 2), mode='equation*') == '\\begin{equation*}8 \\sqrt{2} \\mu^{\\frac{7}{2}}\\end{equation*}'
    assert latex((2 * mu) ** Rational(7, 2), mode='equation', itex=True) == '$$8 \\sqrt{2} \\mu^{\\frac{7}{2}}$$'
    assert latex([2 / x, y]) == '\\left[ \\frac{2}{x}, \\  y\\right]'

def test_latex_dict():
    if False:
        while True:
            i = 10
    d = {Rational(1): 1, x ** 2: 2, x: 3, x ** 3: 4}
    assert latex(d) == '\\left\\{ 1 : 1, \\  x : 3, \\  x^{2} : 2, \\  x^{3} : 4\\right\\}'
    D = Dict(d)
    assert latex(D) == '\\left\\{ 1 : 1, \\  x : 3, \\  x^{2} : 2, \\  x^{3} : 4\\right\\}'

def test_latex_list():
    if False:
        i = 10
        return i + 15
    ll = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
    assert latex(ll) == '\\left[ \\omega_{1}, \\  a, \\  \\alpha\\right]'

def test_latex_NumberSymbols():
    if False:
        print('Hello World!')
    assert latex(S.Catalan) == 'G'
    assert latex(S.EulerGamma) == '\\gamma'
    assert latex(S.Exp1) == 'e'
    assert latex(S.GoldenRatio) == '\\phi'
    assert latex(S.Pi) == '\\pi'
    assert latex(S.TribonacciConstant) == '\\text{TribonacciConstant}'

def test_latex_rational():
    if False:
        return 10
    assert latex(-Rational(1, 2)) == '- \\frac{1}{2}'
    assert latex(Rational(-1, 2)) == '- \\frac{1}{2}'
    assert latex(Rational(1, -2)) == '- \\frac{1}{2}'
    assert latex(-Rational(-1, 2)) == '\\frac{1}{2}'
    assert latex(-Rational(1, 2) * x) == '- \\frac{x}{2}'
    assert latex(-Rational(1, 2) * x + Rational(-2, 3) * y) == '- \\frac{x}{2} - \\frac{2 y}{3}'

def test_latex_inverse():
    if False:
        return 10
    assert latex(1 / x) == '\\frac{1}{x}'
    assert latex(1 / (x + y)) == '\\frac{1}{x + y}'

def test_latex_DiracDelta():
    if False:
        i = 10
        return i + 15
    assert latex(DiracDelta(x)) == '\\delta\\left(x\\right)'
    assert latex(DiracDelta(x) ** 2) == '\\left(\\delta\\left(x\\right)\\right)^{2}'
    assert latex(DiracDelta(x, 0)) == '\\delta\\left(x\\right)'
    assert latex(DiracDelta(x, 5)) == '\\delta^{\\left( 5 \\right)}\\left( x \\right)'
    assert latex(DiracDelta(x, 5) ** 2) == '\\left(\\delta^{\\left( 5 \\right)}\\left( x \\right)\\right)^{2}'

def test_latex_Heaviside():
    if False:
        i = 10
        return i + 15
    assert latex(Heaviside(x)) == '\\theta\\left(x\\right)'
    assert latex(Heaviside(x) ** 2) == '\\left(\\theta\\left(x\\right)\\right)^{2}'

def test_latex_KroneckerDelta():
    if False:
        return 10
    assert latex(KroneckerDelta(x, y)) == '\\delta_{x y}'
    assert latex(KroneckerDelta(x, y + 1)) == '\\delta_{x, y + 1}'
    assert latex(KroneckerDelta(x + 1, y)) == '\\delta_{y, x + 1}'
    assert latex(Pow(KroneckerDelta(x, y), 2, evaluate=False)) == '\\left(\\delta_{x y}\\right)^{2}'

def test_latex_LeviCivita():
    if False:
        i = 10
        return i + 15
    assert latex(LeviCivita(x, y, z)) == '\\varepsilon_{x y z}'
    assert latex(LeviCivita(x, y, z) ** 2) == '\\left(\\varepsilon_{x y z}\\right)^{2}'
    assert latex(LeviCivita(x, y, z + 1)) == '\\varepsilon_{x, y, z + 1}'
    assert latex(LeviCivita(x, y + 1, z)) == '\\varepsilon_{x, y + 1, z}'
    assert latex(LeviCivita(x + 1, y, z)) == '\\varepsilon_{x + 1, y, z}'

def test_mode():
    if False:
        while True:
            i = 10
    expr = x + y
    assert latex(expr) == 'x + y'
    assert latex(expr, mode='plain') == 'x + y'
    assert latex(expr, mode='inline') == '$x + y$'
    assert latex(expr, mode='equation*') == '\\begin{equation*}x + y\\end{equation*}'
    assert latex(expr, mode='equation') == '\\begin{equation}x + y\\end{equation}'
    raises(ValueError, lambda : latex(expr, mode='foo'))

def test_latex_mathieu():
    if False:
        return 10
    assert latex(mathieuc(x, y, z)) == 'C\\left(x, y, z\\right)'
    assert latex(mathieus(x, y, z)) == 'S\\left(x, y, z\\right)'
    assert latex(mathieuc(x, y, z) ** 2) == 'C\\left(x, y, z\\right)^{2}'
    assert latex(mathieus(x, y, z) ** 2) == 'S\\left(x, y, z\\right)^{2}'
    assert latex(mathieucprime(x, y, z)) == 'C^{\\prime}\\left(x, y, z\\right)'
    assert latex(mathieusprime(x, y, z)) == 'S^{\\prime}\\left(x, y, z\\right)'
    assert latex(mathieucprime(x, y, z) ** 2) == 'C^{\\prime}\\left(x, y, z\\right)^{2}'
    assert latex(mathieusprime(x, y, z) ** 2) == 'S^{\\prime}\\left(x, y, z\\right)^{2}'

def test_latex_Piecewise():
    if False:
        while True:
            i = 10
    p = Piecewise((x, x < 1), (x ** 2, True))
    assert latex(p) == '\\begin{cases} x & \\text{for}\\: x < 1 \\\\x^{2} & \\text{otherwise} \\end{cases}'
    assert latex(p, itex=True) == '\\begin{cases} x & \\text{for}\\: x \\lt 1 \\\\x^{2} & \\text{otherwise} \\end{cases}'
    p = Piecewise((x, x < 0), (0, x >= 0))
    assert latex(p) == '\\begin{cases} x & \\text{for}\\: x < 0 \\\\0 & \\text{otherwise} \\end{cases}'
    (A, B) = symbols('A B', commutative=False)
    p = Piecewise((A ** 2, Eq(A, B)), (A * B, True))
    s = '\\begin{cases} A^{2} & \\text{for}\\: A = B \\\\A B & \\text{otherwise} \\end{cases}'
    assert latex(p) == s
    assert latex(A * p) == 'A \\left(%s\\right)' % s
    assert latex(p * A) == '\\left(%s\\right) A' % s
    assert latex(Piecewise((x, x < 1), (x ** 2, x < 2))) == '\\begin{cases} x & \\text{for}\\: x < 1 \\\\x^{2} & \\text{for}\\: x < 2 \\end{cases}'

def test_latex_Matrix():
    if False:
        print('Hello World!')
    M = Matrix([[1 + x, y], [y, x - 1]])
    assert latex(M) == '\\left[\\begin{matrix}x + 1 & y\\\\y & x - 1\\end{matrix}\\right]'
    assert latex(M, mode='inline') == '$\\left[\\begin{smallmatrix}x + 1 & y\\\\y & x - 1\\end{smallmatrix}\\right]$'
    assert latex(M, mat_str='array') == '\\left[\\begin{array}{cc}x + 1 & y\\\\y & x - 1\\end{array}\\right]'
    assert latex(M, mat_str='bmatrix') == '\\left[\\begin{bmatrix}x + 1 & y\\\\y & x - 1\\end{bmatrix}\\right]'
    assert latex(M, mat_delim=None, mat_str='bmatrix') == '\\begin{bmatrix}x + 1 & y\\\\y & x - 1\\end{bmatrix}'
    M2 = Matrix(1, 11, range(11))
    assert latex(M2) == '\\left[\\begin{array}{ccccccccccc}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\\end{array}\\right]'

def test_latex_matrix_with_functions():
    if False:
        while True:
            i = 10
    t = symbols('t')
    theta1 = symbols('theta1', cls=Function)
    M = Matrix([[sin(theta1(t)), cos(theta1(t))], [cos(theta1(t).diff(t)), sin(theta1(t).diff(t))]])
    expected = '\\left[\\begin{matrix}\\sin{\\left(\\theta_{1}{\\left(t \\right)} \\right)} & \\cos{\\left(\\theta_{1}{\\left(t \\right)} \\right)}\\\\\\cos{\\left(\\frac{d}{d t} \\theta_{1}{\\left(t \\right)} \\right)} & \\sin{\\left(\\frac{d}{d t} \\theta_{1}{\\left(t \\right)} \\right)}\\end{matrix}\\right]'
    assert latex(M) == expected

def test_latex_NDimArray():
    if False:
        print('Hello World!')
    (x, y, z, w) = symbols('x y z w')
    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray):
        M = ArrayType(x)
        assert latex(M) == 'x'
        M = ArrayType([[1 / x, y], [z, w]])
        M1 = ArrayType([1 / x, y, z])
        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)
        assert latex(M) == '\\left[\\begin{matrix}\\frac{1}{x} & y\\\\z & w\\end{matrix}\\right]'
        assert latex(M1) == '\\left[\\begin{matrix}\\frac{1}{x} & y & z\\end{matrix}\\right]'
        assert latex(M2) == '\\left[\\begin{matrix}\\left[\\begin{matrix}\\frac{1}{x^{2}} & \\frac{y}{x}\\\\\\frac{z}{x} & \\frac{w}{x}\\end{matrix}\\right] & \\left[\\begin{matrix}\\frac{y}{x} & y^{2}\\\\y z & w y\\end{matrix}\\right] & \\left[\\begin{matrix}\\frac{z}{x} & y z\\\\z^{2} & w z\\end{matrix}\\right]\\end{matrix}\\right]'
        assert latex(M3) == '\\left[\\begin{matrix}\\left[\\begin{matrix}\\frac{1}{x^{2}} & \\frac{y}{x}\\\\\\frac{z}{x} & \\frac{w}{x}\\end{matrix}\\right] & \\left[\\begin{matrix}\\frac{y}{x} & y^{2}\\\\y z & w y\\end{matrix}\\right]\\\\\\left[\\begin{matrix}\\frac{z}{x} & y z\\\\z^{2} & w z\\end{matrix}\\right] & \\left[\\begin{matrix}\\frac{w}{x} & w y\\\\w z & w^{2}\\end{matrix}\\right]\\end{matrix}\\right]'
        Mrow = ArrayType([[x, y, 1 / z]])
        Mcolumn = ArrayType([[x], [y], [1 / z]])
        Mcol2 = ArrayType([Mcolumn.tolist()])
        assert latex(Mrow) == '\\left[\\left[\\begin{matrix}x & y & \\frac{1}{z}\\end{matrix}\\right]\\right]'
        assert latex(Mcolumn) == '\\left[\\begin{matrix}x\\\\y\\\\\\frac{1}{z}\\end{matrix}\\right]'
        assert latex(Mcol2) == '\\left[\\begin{matrix}\\left[\\begin{matrix}x\\\\y\\\\\\frac{1}{z}\\end{matrix}\\right]\\end{matrix}\\right]'

def test_latex_mul_symbol():
    if False:
        i = 10
        return i + 15
    assert latex(4 * 4 ** x, mul_symbol='times') == '4 \\times 4^{x}'
    assert latex(4 * 4 ** x, mul_symbol='dot') == '4 \\cdot 4^{x}'
    assert latex(4 * 4 ** x, mul_symbol='ldot') == '4 \\,.\\, 4^{x}'
    assert latex(4 * x, mul_symbol='times') == '4 \\times x'
    assert latex(4 * x, mul_symbol='dot') == '4 \\cdot x'
    assert latex(4 * x, mul_symbol='ldot') == '4 \\,.\\, x'

def test_latex_issue_4381():
    if False:
        print('Hello World!')
    y = 4 * 4 ** log(2)
    assert latex(y) == '4 \\cdot 4^{\\log{\\left(2 \\right)}}'
    assert latex(1 / y) == '\\frac{1}{4 \\cdot 4^{\\log{\\left(2 \\right)}}}'

def test_latex_issue_4576():
    if False:
        print('Hello World!')
    assert latex(Symbol('beta_13_2')) == '\\beta_{13 2}'
    assert latex(Symbol('beta_132_20')) == '\\beta_{132 20}'
    assert latex(Symbol('beta_13')) == '\\beta_{13}'
    assert latex(Symbol('x_a_b')) == 'x_{a b}'
    assert latex(Symbol('x_1_2_3')) == 'x_{1 2 3}'
    assert latex(Symbol('x_a_b1')) == 'x_{a b1}'
    assert latex(Symbol('x_a_1')) == 'x_{a 1}'
    assert latex(Symbol('x_1_a')) == 'x_{1 a}'
    assert latex(Symbol('x_1^aa')) == 'x^{aa}_{1}'
    assert latex(Symbol('x_1__aa')) == 'x^{aa}_{1}'
    assert latex(Symbol('x_11^a')) == 'x^{a}_{11}'
    assert latex(Symbol('x_11__a')) == 'x^{a}_{11}'
    assert latex(Symbol('x_a_a_a_a')) == 'x_{a a a a}'
    assert latex(Symbol('x_a_a^a^a')) == 'x^{a a}_{a a}'
    assert latex(Symbol('x_a_a__a__a')) == 'x^{a a}_{a a}'
    assert latex(Symbol('alpha_11')) == '\\alpha_{11}'
    assert latex(Symbol('alpha_11_11')) == '\\alpha_{11 11}'
    assert latex(Symbol('alpha_alpha')) == '\\alpha_{\\alpha}'
    assert latex(Symbol('alpha^aleph')) == '\\alpha^{\\aleph}'
    assert latex(Symbol('alpha__aleph')) == '\\alpha^{\\aleph}'

def test_latex_pow_fraction():
    if False:
        while True:
            i = 10
    x = Symbol('x')
    assert 'e^{-x}' in latex(exp(-x) / 2).replace(' ', '')
    assert '3^{-x}' in latex(3 ** (-x) / 2).replace(' ', '')

def test_noncommutative():
    if False:
        print('Hello World!')
    (A, B, C) = symbols('A,B,C', commutative=False)
    assert latex(A * B * C ** (-1)) == 'A B C^{-1}'
    assert latex(C ** (-1) * A * B) == 'C^{-1} A B'
    assert latex(A * C ** (-1) * B) == 'A C^{-1} B'

def test_latex_order():
    if False:
        print('Hello World!')
    expr = x ** 3 + x ** 2 * y + y ** 4 + 3 * x * y ** 3
    assert latex(expr, order='lex') == 'x^{3} + x^{2} y + 3 x y^{3} + y^{4}'
    assert latex(expr, order='rev-lex') == 'y^{4} + 3 x y^{3} + x^{2} y + x^{3}'
    assert latex(expr, order='none') == 'x^{3} + y^{4} + y x^{2} + 3 x y^{3}'

def test_latex_Lambda():
    if False:
        print('Hello World!')
    assert latex(Lambda(x, x + 1)) == '\\left( x \\mapsto x + 1 \\right)'
    assert latex(Lambda((x, y), x + 1)) == '\\left( \\left( x, \\  y\\right) \\mapsto x + 1 \\right)'
    assert latex(Lambda(x, x)) == '\\left( x \\mapsto x \\right)'

def test_latex_PolyElement():
    if False:
        return 10
    (Ruv, u, v) = ring('u,v', ZZ)
    (Rxyz, x, y, z) = ring('x,y,z', Ruv)
    assert latex(x - x) == '0'
    assert latex(x - 1) == 'x - 1'
    assert latex(x + 1) == 'x + 1'
    assert latex((u ** 2 + 3 * u * v + 1) * x ** 2 * y + u + 1) == '\\left({u}^{2} + 3 u v + 1\\right) {x}^{2} y + u + 1'
    assert latex((u ** 2 + 3 * u * v + 1) * x ** 2 * y + (u + 1) * x) == '\\left({u}^{2} + 3 u v + 1\\right) {x}^{2} y + \\left(u + 1\\right) x'
    assert latex((u ** 2 + 3 * u * v + 1) * x ** 2 * y + (u + 1) * x + 1) == '\\left({u}^{2} + 3 u v + 1\\right) {x}^{2} y + \\left(u + 1\\right) x + 1'
    assert latex((-u ** 2 + 3 * u * v - 1) * x ** 2 * y - (u + 1) * x - 1) == '-\\left({u}^{2} - 3 u v + 1\\right) {x}^{2} y - \\left(u + 1\\right) x - 1'
    assert latex(-(v ** 2 + v + 1) * x + 3 * u * v + 1) == '-\\left({v}^{2} + v + 1\\right) x + 3 u v + 1'
    assert latex(-(v ** 2 + v + 1) * x - 3 * u * v + 1) == '-\\left({v}^{2} + v + 1\\right) x - 3 u v + 1'

def test_latex_FracElement():
    if False:
        return 10
    (Fuv, u, v) = field('u,v', ZZ)
    (Fxyzt, x, y, z, t) = field('x,y,z,t', Fuv)
    assert latex(x - x) == '0'
    assert latex(x - 1) == 'x - 1'
    assert latex(x + 1) == 'x + 1'
    assert latex(x / 3) == '\\frac{x}{3}'
    assert latex(x / z) == '\\frac{x}{z}'
    assert latex(x * y / z) == '\\frac{x y}{z}'
    assert latex(x / (z * t)) == '\\frac{x}{z t}'
    assert latex(x * y / (z * t)) == '\\frac{x y}{z t}'
    assert latex((x - 1) / y) == '\\frac{x - 1}{y}'
    assert latex((x + 1) / y) == '\\frac{x + 1}{y}'
    assert latex((-x - 1) / y) == '\\frac{-x - 1}{y}'
    assert latex((x + 1) / (y * z)) == '\\frac{x + 1}{y z}'
    assert latex(-y / (x + 1)) == '\\frac{-y}{x + 1}'
    assert latex(y * z / (x + 1)) == '\\frac{y z}{x + 1}'
    assert latex(((u + 1) * x * y + 1) / ((v - 1) * z - 1)) == '\\frac{\\left(u + 1\\right) x y + 1}{\\left(v - 1\\right) z - 1}'
    assert latex(((u + 1) * x * y + 1) / ((v - 1) * z - t * u * v - 1)) == '\\frac{\\left(u + 1\\right) x y + 1}{\\left(v - 1\\right) z - u v t - 1}'

def test_latex_Poly():
    if False:
        return 10
    assert latex(Poly(x ** 2 + 2 * x, x)) == '\\operatorname{Poly}{\\left( x^{2} + 2 x, x, domain=\\mathbb{Z} \\right)}'
    assert latex(Poly(x / y, x)) == '\\operatorname{Poly}{\\left( \\frac{1}{y} x, x, domain=\\mathbb{Z}\\left(y\\right) \\right)}'
    assert latex(Poly(2.0 * x + y)) == '\\operatorname{Poly}{\\left( 2.0 x + 1.0 y, x, y, domain=\\mathbb{R} \\right)}'

def test_latex_Poly_order():
    if False:
        for i in range(10):
            print('nop')
    assert latex(Poly([a, 1, b, 2, c, 3], x)) == '\\operatorname{Poly}{\\left( a x^{5} + x^{4} + b x^{3} + 2 x^{2} + c x + 3, x, domain=\\mathbb{Z}\\left[a, b, c\\right] \\right)}'
    assert latex(Poly([a, 1, b + c, 2, 3], x)) == '\\operatorname{Poly}{\\left( a x^{4} + x^{3} + \\left(b + c\\right) x^{2} + 2 x + 3, x, domain=\\mathbb{Z}\\left[a, b, c\\right] \\right)}'
    assert latex(Poly(a * x ** 3 + x ** 2 * y - x * y - c * y ** 3 - b * x * y ** 2 + y - a * x + b, (x, y))) == '\\operatorname{Poly}{\\left( a x^{3} + x^{2}y -  b xy^{2} - xy -  a x -  c y^{3} + y + b, x, y, domain=\\mathbb{Z}\\left[a, b, c\\right] \\right)}'

def test_latex_ComplexRootOf():
    if False:
        return 10
    assert latex(rootof(x ** 5 + x + 3, 0)) == '\\operatorname{CRootOf} {\\left(x^{5} + x + 3, 0\\right)}'

def test_latex_RootSum():
    if False:
        for i in range(10):
            print('nop')
    assert latex(RootSum(x ** 5 + x + 3, sin)) == '\\operatorname{RootSum} {\\left(x^{5} + x + 3, \\left( x \\mapsto \\sin{\\left(x \\right)} \\right)\\right)}'

def test_settings():
    if False:
        while True:
            i = 10
    raises(TypeError, lambda : latex(x * y, method='garbage'))

def test_latex_numbers():
    if False:
        i = 10
        return i + 15
    assert latex(catalan(n)) == 'C_{n}'
    assert latex(catalan(n) ** 2) == 'C_{n}^{2}'
    assert latex(bernoulli(n)) == 'B_{n}'
    assert latex(bernoulli(n, x)) == 'B_{n}\\left(x\\right)'
    assert latex(bernoulli(n) ** 2) == 'B_{n}^{2}'
    assert latex(bernoulli(n, x) ** 2) == 'B_{n}^{2}\\left(x\\right)'
    assert latex(genocchi(n)) == 'G_{n}'
    assert latex(genocchi(n, x)) == 'G_{n}\\left(x\\right)'
    assert latex(genocchi(n) ** 2) == 'G_{n}^{2}'
    assert latex(genocchi(n, x) ** 2) == 'G_{n}^{2}\\left(x\\right)'
    assert latex(bell(n)) == 'B_{n}'
    assert latex(bell(n, x)) == 'B_{n}\\left(x\\right)'
    assert latex(bell(n, m, (x, y))) == 'B_{n, m}\\left(x, y\\right)'
    assert latex(bell(n) ** 2) == 'B_{n}^{2}'
    assert latex(bell(n, x) ** 2) == 'B_{n}^{2}\\left(x\\right)'
    assert latex(bell(n, m, (x, y)) ** 2) == 'B_{n, m}^{2}\\left(x, y\\right)'
    assert latex(fibonacci(n)) == 'F_{n}'
    assert latex(fibonacci(n, x)) == 'F_{n}\\left(x\\right)'
    assert latex(fibonacci(n) ** 2) == 'F_{n}^{2}'
    assert latex(fibonacci(n, x) ** 2) == 'F_{n}^{2}\\left(x\\right)'
    assert latex(lucas(n)) == 'L_{n}'
    assert latex(lucas(n) ** 2) == 'L_{n}^{2}'
    assert latex(tribonacci(n)) == 'T_{n}'
    assert latex(tribonacci(n, x)) == 'T_{n}\\left(x\\right)'
    assert latex(tribonacci(n) ** 2) == 'T_{n}^{2}'
    assert latex(tribonacci(n, x) ** 2) == 'T_{n}^{2}\\left(x\\right)'

def test_latex_euler():
    if False:
        return 10
    assert latex(euler(n)) == 'E_{n}'
    assert latex(euler(n, x)) == 'E_{n}\\left(x\\right)'
    assert latex(euler(n, x) ** 2) == 'E_{n}^{2}\\left(x\\right)'

def test_lamda():
    if False:
        print('Hello World!')
    assert latex(Symbol('lamda')) == '\\lambda'
    assert latex(Symbol('Lamda')) == '\\Lambda'

def test_custom_symbol_names():
    if False:
        while True:
            i = 10
    x = Symbol('x')
    y = Symbol('y')
    assert latex(x) == 'x'
    assert latex(x, symbol_names={x: 'x_i'}) == 'x_i'
    assert latex(x + y, symbol_names={x: 'x_i'}) == 'x_i + y'
    assert latex(x ** 2, symbol_names={x: 'x_i'}) == 'x_i^{2}'
    assert latex(x + y, symbol_names={x: 'x_i', y: 'y_j'}) == 'x_i + y_j'

def test_matAdd():
    if False:
        for i in range(10):
            print('nop')
    C = MatrixSymbol('C', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    n = symbols('n')
    h = MatrixSymbol('h', 1, 1)
    assert latex(C - 2 * B) in ['- 2 B + C', 'C -2 B']
    assert latex(C + 2 * B) in ['2 B + C', 'C + 2 B']
    assert latex(B - 2 * C) in ['B - 2 C', '- 2 C + B']
    assert latex(B + 2 * C) in ['B + 2 C', '2 C + B']
    assert latex(n * h - (-h + h.T) * (h + h.T)) == 'n h - \\left(- h + h^{T}\\right) \\left(h + h^{T}\\right)'
    assert latex(MatAdd(MatAdd(h, h), MatAdd(h, h))) == '\\left(h + h\\right) + \\left(h + h\\right)'
    assert latex(MatMul(MatMul(h, h), MatMul(h, h))) == '\\left(h h\\right) \\left(h h\\right)'

def test_matMul():
    if False:
        i = 10
        return i + 15
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    x = Symbol('x')
    assert latex(2 * A) == '2 A'
    assert latex(2 * x * A) == '2 x A'
    assert latex(-2 * A) == '- 2 A'
    assert latex(1.5 * A) == '1.5 A'
    assert latex(sqrt(2) * A) == '\\sqrt{2} A'
    assert latex(-sqrt(2) * A) == '- \\sqrt{2} A'
    assert latex(2 * sqrt(2) * x * A) == '2 \\sqrt{2} x A'
    assert latex(-2 * A * (A + 2 * B)) in ['- 2 A \\left(A + 2 B\\right)', '- 2 A \\left(2 B + A\\right)']

def test_latex_MatrixSlice():
    if False:
        for i in range(10):
            print('nop')
    n = Symbol('n', integer=True)
    (x, y, z, w, t) = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)
    assert latex(MatrixSlice(X, (None, None, None), (None, None, None))) == 'X\\left[:, :\\right]'
    assert latex(X[x:x + 1, y:y + 1]) == 'X\\left[x:x + 1, y:y + 1\\right]'
    assert latex(X[x:x + 1:2, y:y + 1:2]) == 'X\\left[x:x + 1:2, y:y + 1:2\\right]'
    assert latex(X[:x, y:]) == 'X\\left[:x, y:\\right]'
    assert latex(X[:x, y:]) == 'X\\left[:x, y:\\right]'
    assert latex(X[x:, :y]) == 'X\\left[x:, :y\\right]'
    assert latex(X[x:y, z:w]) == 'X\\left[x:y, z:w\\right]'
    assert latex(X[x:y:t, w:t:x]) == 'X\\left[x:y:t, w:t:x\\right]'
    assert latex(X[x::y, t::w]) == 'X\\left[x::y, t::w\\right]'
    assert latex(X[:x:y, :t:w]) == 'X\\left[:x:y, :t:w\\right]'
    assert latex(X[::x, ::y]) == 'X\\left[::x, ::y\\right]'
    assert latex(MatrixSlice(X, (0, None, None), (0, None, None))) == 'X\\left[:, :\\right]'
    assert latex(MatrixSlice(X, (None, n, None), (None, n, None))) == 'X\\left[:, :\\right]'
    assert latex(MatrixSlice(X, (0, n, None), (0, n, None))) == 'X\\left[:, :\\right]'
    assert latex(MatrixSlice(X, (0, n, 2), (0, n, 2))) == 'X\\left[::2, ::2\\right]'
    assert latex(X[1:2:3, 4:5:6]) == 'X\\left[1:2:3, 4:5:6\\right]'
    assert latex(X[1:3:5, 4:6:8]) == 'X\\left[1:3:5, 4:6:8\\right]'
    assert latex(X[1:10:2]) == 'X\\left[1:10:2, :\\right]'
    assert latex(Y[:5, 1:9:2]) == 'Y\\left[:5, 1:9:2\\right]'
    assert latex(Y[:5, 1:10:2]) == 'Y\\left[:5, 1::2\\right]'
    assert latex(Y[5, :5:2]) == 'Y\\left[5:6, :5:2\\right]'
    assert latex(X[0:1, 0:1]) == 'X\\left[:1, :1\\right]'
    assert latex(X[0:1:2, 0:1:2]) == 'X\\left[:1:2, :1:2\\right]'
    assert latex((Y + Z)[2:, 2:]) == '\\left(Y + Z\\right)\\left[2:, 2:\\right]'

def test_latex_RandomDomain():
    if False:
        print('Hello World!')
    from sympy.stats import Normal, Die, Exponential, pspace, where
    from sympy.stats.rv import RandomDomain
    X = Normal('x1', 0, 1)
    assert latex(where(X > 0)) == '\\text{Domain: }0 < x_{1} \\wedge x_{1} < \\infty'
    D = Die('d1', 6)
    assert latex(where(D > 4)) == '\\text{Domain: }d_{1} = 5 \\vee d_{1} = 6'
    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert latex(pspace(Tuple(A, B)).domain) == '\\text{Domain: }0 \\leq a \\wedge 0 \\leq b \\wedge a < \\infty \\wedge b < \\infty'
    assert latex(RandomDomain(FiniteSet(x), FiniteSet(1, 2))) == '\\text{Domain: }\\left\\{x\\right\\} \\in \\left\\{1, 2\\right\\}'

def test_PrettyPoly():
    if False:
        for i in range(10):
            print('nop')
    from sympy.polys.domains import QQ
    F = QQ.frac_field(x, y)
    R = QQ[x, y]
    assert latex(F.convert(x / (x + y))) == latex(x / (x + y))
    assert latex(R.convert(x + y)) == latex(x + y)

def test_integral_transforms():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x')
    k = Symbol('k')
    f = Function('f')
    a = Symbol('a')
    b = Symbol('b')
    assert latex(MellinTransform(f(x), x, k)) == '\\mathcal{M}_{x}\\left[f{\\left(x \\right)}\\right]\\left(k\\right)'
    assert latex(InverseMellinTransform(f(k), k, x, a, b)) == '\\mathcal{M}^{-1}_{k}\\left[f{\\left(k \\right)}\\right]\\left(x\\right)'
    assert latex(LaplaceTransform(f(x), x, k)) == '\\mathcal{L}_{x}\\left[f{\\left(x \\right)}\\right]\\left(k\\right)'
    assert latex(InverseLaplaceTransform(f(k), k, x, (a, b))) == '\\mathcal{L}^{-1}_{k}\\left[f{\\left(k \\right)}\\right]\\left(x\\right)'
    assert latex(FourierTransform(f(x), x, k)) == '\\mathcal{F}_{x}\\left[f{\\left(x \\right)}\\right]\\left(k\\right)'
    assert latex(InverseFourierTransform(f(k), k, x)) == '\\mathcal{F}^{-1}_{k}\\left[f{\\left(k \\right)}\\right]\\left(x\\right)'
    assert latex(CosineTransform(f(x), x, k)) == '\\mathcal{COS}_{x}\\left[f{\\left(x \\right)}\\right]\\left(k\\right)'
    assert latex(InverseCosineTransform(f(k), k, x)) == '\\mathcal{COS}^{-1}_{k}\\left[f{\\left(k \\right)}\\right]\\left(x\\right)'
    assert latex(SineTransform(f(x), x, k)) == '\\mathcal{SIN}_{x}\\left[f{\\left(x \\right)}\\right]\\left(k\\right)'
    assert latex(InverseSineTransform(f(k), k, x)) == '\\mathcal{SIN}^{-1}_{k}\\left[f{\\left(k \\right)}\\right]\\left(x\\right)'

def test_PolynomialRingBase():
    if False:
        while True:
            i = 10
    from sympy.polys.domains import QQ
    assert latex(QQ.old_poly_ring(x, y)) == '\\mathbb{Q}\\left[x, y\\right]'
    assert latex(QQ.old_poly_ring(x, y, order='ilex')) == 'S_<^{-1}\\mathbb{Q}\\left[x, y\\right]'

def test_categories():
    if False:
        print('Hello World!')
    from sympy.categories import Object, IdentityMorphism, NamedMorphism, Category, Diagram, DiagramGrid
    A1 = Object('A1')
    A2 = Object('A2')
    A3 = Object('A3')
    f1 = NamedMorphism(A1, A2, 'f1')
    f2 = NamedMorphism(A2, A3, 'f2')
    id_A1 = IdentityMorphism(A1)
    K1 = Category('K1')
    assert latex(A1) == 'A_{1}'
    assert latex(f1) == 'f_{1}:A_{1}\\rightarrow A_{2}'
    assert latex(id_A1) == 'id:A_{1}\\rightarrow A_{1}'
    assert latex(f2 * f1) == 'f_{2}\\circ f_{1}:A_{1}\\rightarrow A_{3}'
    assert latex(K1) == '\\mathbf{K_{1}}'
    d = Diagram()
    assert latex(d) == '\\emptyset'
    d = Diagram({f1: 'unique', f2: S.EmptySet})
    assert latex(d) == '\\left\\{ f_{2}\\circ f_{1}:A_{1}\\rightarrow A_{3} : \\emptyset, \\  id:A_{1}\\rightarrow A_{1} : \\emptyset, \\  id:A_{2}\\rightarrow A_{2} : \\emptyset, \\  id:A_{3}\\rightarrow A_{3} : \\emptyset, \\  f_{1}:A_{1}\\rightarrow A_{2} : \\left\\{unique\\right\\}, \\  f_{2}:A_{2}\\rightarrow A_{3} : \\emptyset\\right\\}'
    d = Diagram({f1: 'unique', f2: S.EmptySet}, {f2 * f1: 'unique'})
    assert latex(d) == '\\left\\{ f_{2}\\circ f_{1}:A_{1}\\rightarrow A_{3} : \\emptyset, \\  id:A_{1}\\rightarrow A_{1} : \\emptyset, \\  id:A_{2}\\rightarrow A_{2} : \\emptyset, \\  id:A_{3}\\rightarrow A_{3} : \\emptyset, \\  f_{1}:A_{1}\\rightarrow A_{2} : \\left\\{unique\\right\\}, \\  f_{2}:A_{2}\\rightarrow A_{3} : \\emptyset\\right\\}\\Longrightarrow \\left\\{ f_{2}\\circ f_{1}:A_{1}\\rightarrow A_{3} : \\left\\{unique\\right\\}\\right\\}'
    A = Object('A')
    B = Object('B')
    C = Object('C')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    d = Diagram([f, g])
    grid = DiagramGrid(d)
    assert latex(grid) == '\\begin{array}{cc}' + '\nA & B \\\\' + '\n & C ' + '\n\\end{array}' + '\n'

def test_Modules():
    if False:
        print('Hello World!')
    from sympy.polys.domains import QQ
    from sympy.polys.agca import homomorphism
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x ** 2])
    assert latex(F) == '{\\mathbb{Q}\\left[x, y\\right]}^{2}'
    assert latex(M) == '\\left\\langle {\\left[ {x},{y} \\right]},{\\left[ {1},{x^{2}} \\right]} \\right\\rangle'
    I = R.ideal(x ** 2, y)
    assert latex(I) == '\\left\\langle {x^{2}},{y} \\right\\rangle'
    Q = F / M
    assert latex(Q) == '\\frac{{\\mathbb{Q}\\left[x, y\\right]}^{2}}{\\left\\langle {\\left[ {x},{y} \\right]},{\\left[ {1},{x^{2}} \\right]} \\right\\rangle}'
    assert latex(Q.submodule([1, x ** 3 / 2], [2, y])) == '\\left\\langle {{\\left[ {1},{\\frac{x^{3}}{2}} \\right]} + {\\left\\langle {\\left[ {x},{y} \\right]},{\\left[ {1},{x^{2}} \\right]} \\right\\rangle}},{{\\left[ {2},{y} \\right]} + {\\left\\langle {\\left[ {x},{y} \\right]},{\\left[ {1},{x^{2}} \\right]} \\right\\rangle}} \\right\\rangle'
    h = homomorphism(QQ.old_poly_ring(x).free_module(2), QQ.old_poly_ring(x).free_module(2), [0, 0])
    assert latex(h) == '{\\left[\\begin{matrix}0 & 0\\\\0 & 0\\end{matrix}\\right]} : {{\\mathbb{Q}\\left[x\\right]}^{2}} \\to {{\\mathbb{Q}\\left[x\\right]}^{2}}'

def test_QuotientRing():
    if False:
        print('Hello World!')
    from sympy.polys.domains import QQ
    R = QQ.old_poly_ring(x) / [x ** 2 + 1]
    assert latex(R) == '\\frac{\\mathbb{Q}\\left[x\\right]}{\\left\\langle {x^{2} + 1} \\right\\rangle}'
    assert latex(R.one) == '{1} + {\\left\\langle {x^{2} + 1} \\right\\rangle}'

def test_Tr():
    if False:
        i = 10
        return i + 15
    (A, B) = symbols('A B', commutative=False)
    t = Tr(A * B)
    assert latex(t) == '\\operatorname{tr}\\left(A B\\right)'

def test_Determinant():
    if False:
        while True:
            i = 10
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Determinant(m)) == '\\left|{\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}}\\right|'
    assert latex(Determinant(Inverse(m))) == '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{-1}}\\right|'
    X = MatrixSymbol('X', 2, 2)
    assert latex(Determinant(X)) == '\\left|{X}\\right|'
    assert latex(Determinant(X + m)) == '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X}\\right|'
    assert latex(Determinant(BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2)))))) == '\\left|{\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}}\\right|'

def test_Adjoint():
    if False:
        while True:
            i = 10
    from sympy.matrices import Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Adjoint(X)) == 'X^{\\dagger}'
    assert latex(Adjoint(X + Y)) == '\\left(X + Y\\right)^{\\dagger}'
    assert latex(Adjoint(X) + Adjoint(Y)) == 'X^{\\dagger} + Y^{\\dagger}'
    assert latex(Adjoint(X * Y)) == '\\left(X Y\\right)^{\\dagger}'
    assert latex(Adjoint(Y) * Adjoint(X)) == 'Y^{\\dagger} X^{\\dagger}'
    assert latex(Adjoint(X ** 2)) == '\\left(X^{2}\\right)^{\\dagger}'
    assert latex(Adjoint(X) ** 2) == '\\left(X^{\\dagger}\\right)^{2}'
    assert latex(Adjoint(Inverse(X))) == '\\left(X^{-1}\\right)^{\\dagger}'
    assert latex(Inverse(Adjoint(X))) == '\\left(X^{\\dagger}\\right)^{-1}'
    assert latex(Adjoint(Transpose(X))) == '\\left(X^{T}\\right)^{\\dagger}'
    assert latex(Transpose(Adjoint(X))) == '\\left(X^{\\dagger}\\right)^{T}'
    assert latex(Transpose(Adjoint(X) + Y)) == '\\left(X^{\\dagger} + Y\\right)^{T}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Adjoint(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{\\dagger}'
    assert latex(Adjoint(m + X)) == '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{\\dagger}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Adjoint(BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2)))))) == '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{\\dagger}'
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Adjoint(Mx)) == '\\left(M^{x}\\right)^{\\dagger}'

def test_Transpose():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices import Transpose, MatPow, HadamardPower
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Transpose(X)) == 'X^{T}'
    assert latex(Transpose(X + Y)) == '\\left(X + Y\\right)^{T}'
    assert latex(Transpose(HadamardPower(X, 2))) == '\\left(X^{\\circ {2}}\\right)^{T}'
    assert latex(HadamardPower(Transpose(X), 2)) == '\\left(X^{T}\\right)^{\\circ {2}}'
    assert latex(Transpose(MatPow(X, 2))) == '\\left(X^{2}\\right)^{T}'
    assert latex(MatPow(Transpose(X), 2)) == '\\left(X^{T}\\right)^{2}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Transpose(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{T}'
    assert latex(Transpose(m + X)) == '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{T}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Transpose(BlockMatrix(((OneMatrix(2, 2), X), (m, ZeroMatrix(2, 2)))))) == '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{T}'
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Transpose(Mx)) == '\\left(M^{x}\\right)^{T}'

def test_Hadamard():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices import HadamardProduct, HadamardPower
    from sympy.matrices.expressions import MatAdd, MatMul, MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(HadamardProduct(X, Y * Y)) == 'X \\circ Y^{2}'
    assert latex(HadamardProduct(X, Y) * Y) == '\\left(X \\circ Y\\right) Y'
    assert latex(HadamardPower(X, 2)) == 'X^{\\circ {2}}'
    assert latex(HadamardPower(X, -1)) == 'X^{\\circ \\left({-1}\\right)}'
    assert latex(HadamardPower(MatAdd(X, Y), 2)) == '\\left(X + Y\\right)^{\\circ {2}}'
    assert latex(HadamardPower(MatMul(X, Y), 2)) == '\\left(X Y\\right)^{\\circ {2}}'
    assert latex(HadamardPower(MatPow(X, -1), -1)) == '\\left(X^{-1}\\right)^{\\circ \\left({-1}\\right)}'
    assert latex(MatPow(HadamardPower(X, -1), -1)) == '\\left(X^{\\circ \\left({-1}\\right)}\\right)^{-1}'
    assert latex(HadamardPower(X, n + 1)) == 'X^{\\circ \\left({n + 1}\\right)}'

def test_MatPow():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions import MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(MatPow(X, 2)) == 'X^{2}'
    assert latex(MatPow(X * X, 2)) == '\\left(X^{2}\\right)^{2}'
    assert latex(MatPow(X * Y, 2)) == '\\left(X Y\\right)^{2}'
    assert latex(MatPow(X + Y, 2)) == '\\left(X + Y\\right)^{2}'
    assert latex(MatPow(X + X, 2)) == '\\left(2 X\\right)^{2}'
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(MatPow(Mx, 2)) == '\\left(M^{x}\\right)^{2}'

def test_ElementwiseApplyFunction():
    if False:
        while True:
            i = 10
    X = MatrixSymbol('X', 2, 2)
    expr = (X.T * X).applyfunc(sin)
    assert latex(expr) == '{\\left( d \\mapsto \\sin{\\left(d \\right)} \\right)}_{\\circ}\\left({X^{T} X}\\right)'
    expr = X.applyfunc(Lambda(x, 1 / x))
    assert latex(expr) == '{\\left( x \\mapsto \\frac{1}{x} \\right)}_{\\circ}\\left({X}\\right)'

def test_ZeroMatrix():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.special import ZeroMatrix
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='plain') == '0'
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='bold') == '\\mathbf{0}'

def test_OneMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.special import OneMatrix
    assert latex(OneMatrix(3, 4), mat_symbol_style='plain') == '1'
    assert latex(OneMatrix(3, 4), mat_symbol_style='bold') == '\\mathbf{1}'

def test_Identity():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.special import Identity
    assert latex(Identity(1), mat_symbol_style='plain') == '\\mathbb{I}'
    assert latex(Identity(1), mat_symbol_style='bold') == '\\mathbf{I}'

def test_latex_DFT_IDFT():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.fourier import DFT, IDFT
    assert latex(DFT(13)) == '\\text{DFT}_{13}'
    assert latex(IDFT(x)) == '\\text{IDFT}_{x}'

def test_boolean_args_order():
    if False:
        while True:
            i = 10
    syms = symbols('a:f')
    expr = And(*syms)
    assert latex(expr) == 'a \\wedge b \\wedge c \\wedge d \\wedge e \\wedge f'
    expr = Or(*syms)
    assert latex(expr) == 'a \\vee b \\vee c \\vee d \\vee e \\vee f'
    expr = Equivalent(*syms)
    assert latex(expr) == 'a \\Leftrightarrow b \\Leftrightarrow c \\Leftrightarrow d \\Leftrightarrow e \\Leftrightarrow f'
    expr = Xor(*syms)
    assert latex(expr) == 'a \\veebar b \\veebar c \\veebar d \\veebar e \\veebar f'

def test_imaginary():
    if False:
        i = 10
        return i + 15
    i = sqrt(-1)
    assert latex(i) == 'i'

def test_builtins_without_args():
    if False:
        return 10
    assert latex(sin) == '\\sin'
    assert latex(cos) == '\\cos'
    assert latex(tan) == '\\tan'
    assert latex(log) == '\\log'
    assert latex(Ei) == '\\operatorname{Ei}'
    assert latex(zeta) == '\\zeta'

def test_latex_greek_functions():
    if False:
        return 10
    s = Function('Alpha')
    assert latex(s) == '\\mathrm{A}'
    assert latex(s(x)) == '\\mathrm{A}{\\left(x \\right)}'
    s = Function('Beta')
    assert latex(s) == '\\mathrm{B}'
    s = Function('Eta')
    assert latex(s) == '\\mathrm{H}'
    assert latex(s(x)) == '\\mathrm{H}{\\left(x \\right)}'
    p = Function('Pi')
    assert latex(p) == '\\Pi'
    c = Function('chi')
    assert latex(c(x)) == '\\chi{\\left(x \\right)}'
    assert latex(c) == '\\chi'

def test_translate():
    if False:
        i = 10
        return i + 15
    s = 'Alpha'
    assert translate(s) == '\\mathrm{A}'
    s = 'Beta'
    assert translate(s) == '\\mathrm{B}'
    s = 'Eta'
    assert translate(s) == '\\mathrm{H}'
    s = 'omicron'
    assert translate(s) == 'o'
    s = 'Pi'
    assert translate(s) == '\\Pi'
    s = 'pi'
    assert translate(s) == '\\pi'
    s = 'LamdaHatDOT'
    assert translate(s) == '\\dot{\\hat{\\Lambda}}'

def test_other_symbols():
    if False:
        i = 10
        return i + 15
    from sympy.printing.latex import other_symbols
    for s in other_symbols:
        assert latex(symbols(s)) == '\\' + s

def test_modifiers():
    if False:
        return 10
    assert latex(symbols('xMathring')) == '\\mathring{x}'
    assert latex(symbols('xCheck')) == '\\check{x}'
    assert latex(symbols('xBreve')) == '\\breve{x}'
    assert latex(symbols('xAcute')) == '\\acute{x}'
    assert latex(symbols('xGrave')) == '\\grave{x}'
    assert latex(symbols('xTilde')) == '\\tilde{x}'
    assert latex(symbols('xPrime')) == "{x}'"
    assert latex(symbols('xddDDot')) == '\\ddddot{x}'
    assert latex(symbols('xDdDot')) == '\\dddot{x}'
    assert latex(symbols('xDDot')) == '\\ddot{x}'
    assert latex(symbols('xBold')) == '\\boldsymbol{x}'
    assert latex(symbols('xnOrM')) == '\\left\\|{x}\\right\\|'
    assert latex(symbols('xAVG')) == '\\left\\langle{x}\\right\\rangle'
    assert latex(symbols('xHat')) == '\\hat{x}'
    assert latex(symbols('xDot')) == '\\dot{x}'
    assert latex(symbols('xBar')) == '\\bar{x}'
    assert latex(symbols('xVec')) == '\\vec{x}'
    assert latex(symbols('xAbs')) == '\\left|{x}\\right|'
    assert latex(symbols('xMag')) == '\\left|{x}\\right|'
    assert latex(symbols('xPrM')) == "{x}'"
    assert latex(symbols('xBM')) == '\\boldsymbol{x}'
    assert latex(symbols('Mathring')) == 'Mathring'
    assert latex(symbols('Check')) == 'Check'
    assert latex(symbols('Breve')) == 'Breve'
    assert latex(symbols('Acute')) == 'Acute'
    assert latex(symbols('Grave')) == 'Grave'
    assert latex(symbols('Tilde')) == 'Tilde'
    assert latex(symbols('Prime')) == 'Prime'
    assert latex(symbols('DDot')) == '\\dot{D}'
    assert latex(symbols('Bold')) == 'Bold'
    assert latex(symbols('NORm')) == 'NORm'
    assert latex(symbols('AVG')) == 'AVG'
    assert latex(symbols('Hat')) == 'Hat'
    assert latex(symbols('Dot')) == 'Dot'
    assert latex(symbols('Bar')) == 'Bar'
    assert latex(symbols('Vec')) == 'Vec'
    assert latex(symbols('Abs')) == 'Abs'
    assert latex(symbols('Mag')) == 'Mag'
    assert latex(symbols('PrM')) == 'PrM'
    assert latex(symbols('BM')) == 'BM'
    assert latex(symbols('hbar')) == '\\hbar'
    assert latex(symbols('xvecdot')) == '\\dot{\\vec{x}}'
    assert latex(symbols('xDotVec')) == '\\vec{\\dot{x}}'
    assert latex(symbols('xHATNorm')) == '\\left\\|{\\hat{x}}\\right\\|'
    assert latex(symbols('xMathringBm_yCheckPRM__zbreveAbs')) == "\\boldsymbol{\\mathring{x}}^{\\left|{\\breve{z}}\\right|}_{{\\check{y}}'}"
    assert latex(symbols('alphadothat_nVECDOT__tTildePrime')) == "\\hat{\\dot{\\alpha}}^{{\\tilde{t}}'}_{\\dot{\\vec{n}}}"

def test_greek_symbols():
    if False:
        print('Hello World!')
    assert latex(Symbol('alpha')) == '\\alpha'
    assert latex(Symbol('beta')) == '\\beta'
    assert latex(Symbol('gamma')) == '\\gamma'
    assert latex(Symbol('delta')) == '\\delta'
    assert latex(Symbol('epsilon')) == '\\epsilon'
    assert latex(Symbol('zeta')) == '\\zeta'
    assert latex(Symbol('eta')) == '\\eta'
    assert latex(Symbol('theta')) == '\\theta'
    assert latex(Symbol('iota')) == '\\iota'
    assert latex(Symbol('kappa')) == '\\kappa'
    assert latex(Symbol('lambda')) == '\\lambda'
    assert latex(Symbol('mu')) == '\\mu'
    assert latex(Symbol('nu')) == '\\nu'
    assert latex(Symbol('xi')) == '\\xi'
    assert latex(Symbol('omicron')) == 'o'
    assert latex(Symbol('pi')) == '\\pi'
    assert latex(Symbol('rho')) == '\\rho'
    assert latex(Symbol('sigma')) == '\\sigma'
    assert latex(Symbol('tau')) == '\\tau'
    assert latex(Symbol('upsilon')) == '\\upsilon'
    assert latex(Symbol('phi')) == '\\phi'
    assert latex(Symbol('chi')) == '\\chi'
    assert latex(Symbol('psi')) == '\\psi'
    assert latex(Symbol('omega')) == '\\omega'
    assert latex(Symbol('Alpha')) == '\\mathrm{A}'
    assert latex(Symbol('Beta')) == '\\mathrm{B}'
    assert latex(Symbol('Gamma')) == '\\Gamma'
    assert latex(Symbol('Delta')) == '\\Delta'
    assert latex(Symbol('Epsilon')) == '\\mathrm{E}'
    assert latex(Symbol('Zeta')) == '\\mathrm{Z}'
    assert latex(Symbol('Eta')) == '\\mathrm{H}'
    assert latex(Symbol('Theta')) == '\\Theta'
    assert latex(Symbol('Iota')) == '\\mathrm{I}'
    assert latex(Symbol('Kappa')) == '\\mathrm{K}'
    assert latex(Symbol('Lambda')) == '\\Lambda'
    assert latex(Symbol('Mu')) == '\\mathrm{M}'
    assert latex(Symbol('Nu')) == '\\mathrm{N}'
    assert latex(Symbol('Xi')) == '\\Xi'
    assert latex(Symbol('Omicron')) == '\\mathrm{O}'
    assert latex(Symbol('Pi')) == '\\Pi'
    assert latex(Symbol('Rho')) == '\\mathrm{P}'
    assert latex(Symbol('Sigma')) == '\\Sigma'
    assert latex(Symbol('Tau')) == '\\mathrm{T}'
    assert latex(Symbol('Upsilon')) == '\\Upsilon'
    assert latex(Symbol('Phi')) == '\\Phi'
    assert latex(Symbol('Chi')) == '\\mathrm{X}'
    assert latex(Symbol('Psi')) == '\\Psi'
    assert latex(Symbol('Omega')) == '\\Omega'
    assert latex(Symbol('varepsilon')) == '\\varepsilon'
    assert latex(Symbol('varkappa')) == '\\varkappa'
    assert latex(Symbol('varphi')) == '\\varphi'
    assert latex(Symbol('varpi')) == '\\varpi'
    assert latex(Symbol('varrho')) == '\\varrho'
    assert latex(Symbol('varsigma')) == '\\varsigma'
    assert latex(Symbol('vartheta')) == '\\vartheta'

def test_fancyset_symbols():
    if False:
        for i in range(10):
            print('nop')
    assert latex(S.Rationals) == '\\mathbb{Q}'
    assert latex(S.Naturals) == '\\mathbb{N}'
    assert latex(S.Naturals0) == '\\mathbb{N}_0'
    assert latex(S.Integers) == '\\mathbb{Z}'
    assert latex(S.Reals) == '\\mathbb{R}'
    assert latex(S.Complexes) == '\\mathbb{C}'

@XFAIL
def test_builtin_without_args_mismatched_names():
    if False:
        return 10
    assert latex(CosineTransform) == '\\mathcal{COS}'

def test_builtin_no_args():
    if False:
        for i in range(10):
            print('nop')
    assert latex(Chi) == '\\operatorname{Chi}'
    assert latex(beta) == '\\operatorname{B}'
    assert latex(gamma) == '\\Gamma'
    assert latex(KroneckerDelta) == '\\delta'
    assert latex(DiracDelta) == '\\delta'
    assert latex(lowergamma) == '\\gamma'

def test_issue_6853():
    if False:
        for i in range(10):
            print('nop')
    p = Function('Pi')
    assert latex(p(x)) == '\\Pi{\\left(x \\right)}'

def test_Mul():
    if False:
        for i in range(10):
            print('nop')
    e = Mul(-2, x + 1, evaluate=False)
    assert latex(e) == '- 2 \\left(x + 1\\right)'
    e = Mul(2, x + 1, evaluate=False)
    assert latex(e) == '2 \\left(x + 1\\right)'
    e = Mul(S.Half, x + 1, evaluate=False)
    assert latex(e) == '\\frac{x + 1}{2}'
    e = Mul(y, x + 1, evaluate=False)
    assert latex(e) == 'y \\left(x + 1\\right)'
    e = Mul(-y, x + 1, evaluate=False)
    assert latex(e) == '- y \\left(x + 1\\right)'
    e = Mul(-2, x + 1)
    assert latex(e) == '- 2 x - 2'
    e = Mul(2, x + 1)
    assert latex(e) == '2 x + 2'

def test_Pow():
    if False:
        for i in range(10):
            print('nop')
    e = Pow(2, 2, evaluate=False)
    assert latex(e) == '2^{2}'
    assert latex(x ** Rational(-1, 3)) == '\\frac{1}{\\sqrt[3]{x}}'
    x2 = Symbol('x^2')
    assert latex(x2 ** 2) == '\\left(x^{2}\\right)^{2}'

def test_issue_7180():
    if False:
        return 10
    assert latex(Equivalent(x, y)) == 'x \\Leftrightarrow y'
    assert latex(Not(Equivalent(x, y))) == 'x \\not\\Leftrightarrow y'

def test_issue_8409():
    if False:
        while True:
            i = 10
    assert latex(S.Half ** n) == '\\left(\\frac{1}{2}\\right)^{n}'

def test_issue_8470():
    if False:
        for i in range(10):
            print('nop')
    from sympy.parsing.sympy_parser import parse_expr
    e = parse_expr('-B*A', evaluate=False)
    assert latex(e) == 'A \\left(- B\\right)'

def test_issue_15439():
    if False:
        return 10
    x = MatrixSymbol('x', 2, 2)
    y = MatrixSymbol('y', 2, 2)
    assert latex((x * y).subs(y, -y)) == 'x \\left(- y\\right)'
    assert latex((x * y).subs(y, -2 * y)) == 'x \\left(- 2 y\\right)'
    assert latex((x * y).subs(x, -x)) == '\\left(- x\\right) y'

def test_issue_2934():
    if False:
        print('Hello World!')
    assert latex(Symbol('\\frac{a_1}{b_1}')) == '\\frac{a_1}{b_1}'

def test_issue_10489():
    if False:
        while True:
            i = 10
    latexSymbolWithBrace = 'C_{x_{0}}'
    s = Symbol(latexSymbolWithBrace)
    assert latex(s) == latexSymbolWithBrace
    assert latex(cos(s)) == '\\cos{\\left(C_{x_{0}} \\right)}'

def test_issue_12886():
    if False:
        for i in range(10):
            print('nop')
    (m__1, l__1) = symbols('m__1, l__1')
    assert latex(m__1 ** 2 + l__1 ** 2) == '\\left(l^{1}\\right)^{2} + \\left(m^{1}\\right)^{2}'

def test_issue_13559():
    if False:
        print('Hello World!')
    from sympy.parsing.sympy_parser import parse_expr
    expr = parse_expr('5/1', evaluate=False)
    assert latex(expr) == '\\frac{5}{1}'

def test_issue_13651():
    if False:
        for i in range(10):
            print('nop')
    expr = c + Mul(-1, a + b, evaluate=False)
    assert latex(expr) == 'c - \\left(a + b\\right)'

def test_latex_UnevaluatedExpr():
    if False:
        return 10
    x = symbols('x')
    he = UnevaluatedExpr(1 / x)
    assert latex(he) == latex(1 / x) == '\\frac{1}{x}'
    assert latex(he ** 2) == '\\left(\\frac{1}{x}\\right)^{2}'
    assert latex(he + 1) == '1 + \\frac{1}{x}'
    assert latex(x * he) == 'x \\frac{1}{x}'

def test_MatrixElement_printing():
    if False:
        while True:
            i = 10
    A = MatrixSymbol('A', 1, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    assert latex(A[0, 0]) == 'A_{0, 0}'
    assert latex(3 * A[0, 0]) == '3 A_{0, 0}'
    F = C[0, 0].subs(C, A - B)
    assert latex(F) == '\\left(A - B\\right)_{0, 0}'
    (i, j, k) = symbols('i j k')
    M = MatrixSymbol('M', k, k)
    N = MatrixSymbol('N', k, k)
    assert latex((M * N)[i, j]) == '\\sum_{i_{1}=0}^{k - 1} M_{i, i_{1}} N_{i_{1}, j}'

def test_MatrixSymbol_printing():
    if False:
        i = 10
        return i + 15
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    C = MatrixSymbol('C', 3, 3)
    assert latex(-A) == '- A'
    assert latex(A - A * B - B) == 'A - A B - B'
    assert latex(-A * B - A * B * C - B) == '- A B - A B C - B'

def test_KroneckerProduct_printing():
    if False:
        while True:
            i = 10
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 2, 2)
    assert latex(KroneckerProduct(A, B)) == 'A \\otimes B'

def test_Series_printing():
    if False:
        i = 10
        return i + 15
    tf1 = TransferFunction(x * y ** 2 - z, y ** 3 - t ** 3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t * x ** 2 - t ** w * x + w, t - y, y)
    assert latex(Series(tf1, tf2)) == '\\left(\\frac{x y^{2} - z}{- t^{3} + y^{3}}\\right) \\left(\\frac{x - y}{x + y}\\right)'
    assert latex(Series(tf1, tf2, tf3)) == '\\left(\\frac{x y^{2} - z}{- t^{3} + y^{3}}\\right) \\left(\\frac{x - y}{x + y}\\right) \\left(\\frac{t x^{2} - t^{w} x + w}{t - y}\\right)'
    assert latex(Series(-tf2, tf1)) == '\\left(\\frac{- x + y}{x + y}\\right) \\left(\\frac{x y^{2} - z}{- t^{3} + y^{3}}\\right)'
    M_1 = Matrix([[5 / s], [5 / (2 * s)]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5, 6 * s ** 3]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    assert latex(T_1 * (T_2 + T_2)) == '\\left[\\begin{matrix}\\frac{5}{s}\\\\\\frac{5}{2 s}\\end{matrix}\\right]_\\tau\\cdot\\left(\\left[\\begin{matrix}\\frac{5}{1} & \\frac{6 s^{3}}{1}\\end{matrix}\\right]_\\tau + \\left[\\begin{matrix}\\frac{5}{1} & \\frac{6 s^{3}}{1}\\end{matrix}\\right]_\\tau\\right)' == latex(MIMOSeries(MIMOParallel(T_2, T_2), T_1))
    M_3 = Matrix([[5, 6], [6, 5 / s]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1 * T_2 + T_3) == '\\left[\\begin{matrix}\\frac{5}{s}\\\\\\frac{5}{2 s}\\end{matrix}\\right]_\\tau\\cdot\\left[\\begin{matrix}\\frac{5}{1} & \\frac{6 s^{3}}{1}\\end{matrix}\\right]_\\tau + \\left[\\begin{matrix}\\frac{5}{1} & \\frac{6}{1}\\\\\\frac{6}{1} & \\frac{5}{s}\\end{matrix}\\right]_\\tau' == latex(MIMOParallel(MIMOSeries(T_2, T_1), T_3))

def test_TransferFunction_printing():
    if False:
        print('Hello World!')
    tf1 = TransferFunction(x - 1, x + 1, x)
    assert latex(tf1) == '\\frac{x - 1}{x + 1}'
    tf2 = TransferFunction(x + 1, 2 - y, x)
    assert latex(tf2) == '\\frac{x + 1}{2 - y}'
    tf3 = TransferFunction(y, y ** 2 + 2 * y + 3, y)
    assert latex(tf3) == '\\frac{y}{y^{2} + 2 y + 3}'

def test_Parallel_printing():
    if False:
        for i in range(10):
            print('nop')
    tf1 = TransferFunction(x * y ** 2 - z, y ** 3 - t ** 3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    assert latex(Parallel(tf1, tf2)) == '\\frac{x y^{2} - z}{- t^{3} + y^{3}} + \\frac{x - y}{x + y}'
    assert latex(Parallel(-tf2, tf1)) == '\\frac{- x + y}{x + y} + \\frac{x y^{2} - z}{- t^{3} + y^{3}}'
    M_1 = Matrix([[5, 6], [6, 5 / s]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5 / s, 6], [6, 5 / (s - 1)]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    M_3 = Matrix([[6, 5 / (s * (s - 1))], [5, 6]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1 + T_2 + T_3) == '\\left[\\begin{matrix}\\frac{5}{1} & \\frac{6}{1}\\\\\\frac{6}{1} & \\frac{5}{s}\\end{matrix}\\right]_\\tau + \\left[\\begin{matrix}\\frac{5}{s} & \\frac{6}{1}\\\\\\frac{6}{1} & \\frac{5}{s - 1}\\end{matrix}\\right]_\\tau + \\left[\\begin{matrix}\\frac{6}{1} & \\frac{5}{s \\left(s - 1\\right)}\\\\\\frac{5}{1} & \\frac{6}{1}\\end{matrix}\\right]_\\tau' == latex(MIMOParallel(T_1, T_2, T_3)) == latex(MIMOParallel(T_1, MIMOParallel(T_2, T_3))) == latex(MIMOParallel(MIMOParallel(T_1, T_2), T_3))

def test_TransferFunctionMatrix_printing():
    if False:
        for i in range(10):
            print('nop')
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    tf3 = TransferFunction(p, y ** 2 + 2 * y + 3, p)
    assert latex(TransferFunctionMatrix([[tf1], [tf2]])) == '\\left[\\begin{matrix}\\frac{p}{p + x}\\\\\\frac{p - s}{p + s}\\end{matrix}\\right]_\\tau'
    assert latex(TransferFunctionMatrix([[tf1, tf2], [tf3, -tf1]])) == '\\left[\\begin{matrix}\\frac{p}{p + x} & \\frac{p - s}{p + s}\\\\\\frac{p}{y^{2} + 2 y + 3} & \\frac{\\left(-1\\right) p}{p + x}\\end{matrix}\\right]_\\tau'

def test_Feedback_printing():
    if False:
        return 10
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    assert latex(Feedback(tf1, tf2)) == '\\frac{\\frac{p}{p + x}}{\\frac{1}{1} + \\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}'
    assert latex(Feedback(tf1 * tf2, TransferFunction(1, 1, p))) == '\\frac{\\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}{\\frac{1}{1} + \\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}'
    assert latex(Feedback(tf1, tf2, 1)) == '\\frac{\\frac{p}{p + x}}{\\frac{1}{1} - \\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}'
    assert latex(Feedback(tf1 * tf2, sign=1)) == '\\frac{\\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}{\\frac{1}{1} - \\left(\\frac{p}{p + x}\\right) \\left(\\frac{p - s}{p + s}\\right)}'

def test_MIMOFeedback_printing():
    if False:
        for i in range(10):
            print('nop')
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s ** 2 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s ** 2, s ** 2 - 1, s)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf4, tf3], [tf2, tf1]])
    assert latex(MIMOFeedback(tfm_1, tfm_2)) == '\\left(I_{\\tau} + \\left[\\begin{matrix}\\frac{1}{s} & \\frac{s}{s^{2} - 1}\\\\\\frac{s}{s - 1} & \\frac{s^{2}}{s^{2} - 1}\\end{matrix}\\right]_\\tau\\cdot\\left[\\begin{matrix}\\frac{s^{2}}{s^{2} - 1} & \\frac{s}{s - 1}\\\\\\frac{s}{s^{2} - 1} & \\frac{1}{s}\\end{matrix}\\right]_\\tau\\right)^{-1} \\cdot \\left[\\begin{matrix}\\frac{1}{s} & \\frac{s}{s^{2} - 1}\\\\\\frac{s}{s - 1} & \\frac{s^{2}}{s^{2} - 1}\\end{matrix}\\right]_\\tau'
    assert latex(MIMOFeedback(tfm_1 * tfm_2, tfm_1, 1)) == '\\left(I_{\\tau} - \\left[\\begin{matrix}\\frac{1}{s} & \\frac{s}{s^{2} - 1}\\\\\\frac{s}{s - 1} & \\frac{s^{2}}{s^{2} - 1}\\end{matrix}\\right]_\\tau\\cdot\\left[\\begin{matrix}\\frac{s^{2}}{s^{2} - 1} & \\frac{s}{s - 1}\\\\\\frac{s}{s^{2} - 1} & \\frac{1}{s}\\end{matrix}\\right]_\\tau\\cdot\\left[\\begin{matrix}\\frac{1}{s} & \\frac{s}{s^{2} - 1}\\\\\\frac{s}{s - 1} & \\frac{s^{2}}{s^{2} - 1}\\end{matrix}\\right]_\\tau\\right)^{-1} \\cdot \\left[\\begin{matrix}\\frac{1}{s} & \\frac{s}{s^{2} - 1}\\\\\\frac{s}{s - 1} & \\frac{s^{2}}{s^{2} - 1}\\end{matrix}\\right]_\\tau\\cdot\\left[\\begin{matrix}\\frac{s^{2}}{s^{2} - 1} & \\frac{s}{s - 1}\\\\\\frac{s}{s^{2} - 1} & \\frac{1}{s}\\end{matrix}\\right]_\\tau'

def test_Quaternion_latex_printing():
    if False:
        return 10
    q = Quaternion(x, y, z, t)
    assert latex(q) == 'x + y i + z j + t k'
    q = Quaternion(x, y, z, x * t)
    assert latex(q) == 'x + y i + z j + t x k'
    q = Quaternion(x, y, z, x + t)
    assert latex(q) == 'x + y i + z j + \\left(t + x\\right) k'

def test_TensorProduct_printing():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    assert latex(TensorProduct(A, B)) == 'A \\otimes B'

def test_WedgeProduct_printing():
    if False:
        while True:
            i = 10
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    wp = WedgeProduct(R2.dx, R2.dy)
    assert latex(wp) == '\\operatorname{d}x \\wedge \\operatorname{d}y'

def test_issue_9216():
    if False:
        while True:
            i = 10
    expr_1 = Pow(1, -1, evaluate=False)
    assert latex(expr_1) == '1^{-1}'
    expr_2 = Pow(1, Pow(1, -1, evaluate=False), evaluate=False)
    assert latex(expr_2) == '1^{1^{-1}}'
    expr_3 = Pow(3, -2, evaluate=False)
    assert latex(expr_3) == '\\frac{1}{9}'
    expr_4 = Pow(1, -2, evaluate=False)
    assert latex(expr_4) == '1^{-2}'

def test_latex_printer_tensor():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensor_heads
    L = TensorIndexType('L')
    (i, j, k, l) = tensor_indices('i j k l', L)
    i0 = tensor_indices('i_0', L)
    (A, B, C, D) = tensor_heads('A B C D', [L])
    H = TensorHead('H', [L, L])
    K = TensorHead('K', [L, L, L, L])
    assert latex(i) == '{}^{i}'
    assert latex(-i) == '{}_{i}'
    expr = A(i)
    assert latex(expr) == 'A{}^{i}'
    expr = A(i0)
    assert latex(expr) == 'A{}^{i_{0}}'
    expr = A(-i)
    assert latex(expr) == 'A{}_{i}'
    expr = -3 * A(i)
    assert latex(expr) == '-3A{}^{i}'
    expr = K(i, j, -k, -i0)
    assert latex(expr) == 'K{}^{ij}{}_{ki_{0}}'
    expr = K(i, -j, -k, i0)
    assert latex(expr) == 'K{}^{i}{}_{jk}{}^{i_{0}}'
    expr = K(i, -j, k, -i0)
    assert latex(expr) == 'K{}^{i}{}_{j}{}^{k}{}_{i_{0}}'
    expr = H(i, -j)
    assert latex(expr) == 'H{}^{i}{}_{j}'
    expr = H(i, j)
    assert latex(expr) == 'H{}^{ij}'
    expr = H(-i, -j)
    assert latex(expr) == 'H{}_{ij}'
    expr = (1 + x) * A(i)
    assert latex(expr) == '\\left(x + 1\\right)A{}^{i}'
    expr = H(i, -i)
    assert latex(expr) == 'H{}^{L_{0}}{}_{L_{0}}'
    expr = H(i, -j) * A(j) * B(k)
    assert latex(expr) == 'H{}^{i}{}_{L_{0}}A{}^{L_{0}}B{}^{k}'
    expr = A(i) + 3 * B(i)
    assert latex(expr) == '3B{}^{i} + A{}^{i}'
    from sympy.tensor.tensor import TensorElement
    expr = TensorElement(K(i, j, k, l), {i: 3, k: 2})
    assert latex(expr) == 'K{}^{i=3,j,k=2,l}'
    expr = TensorElement(K(i, j, k, l), {i: 3})
    assert latex(expr) == 'K{}^{i=3,jkl}'
    expr = TensorElement(K(i, -j, k, l), {i: 3, k: 2})
    assert latex(expr) == 'K{}^{i=3}{}_{j}{}^{k=2,l}'
    expr = TensorElement(K(i, -j, k, -l), {i: 3, k: 2})
    assert latex(expr) == 'K{}^{i=3}{}_{j}{}^{k=2}{}_{l}'
    expr = TensorElement(K(i, j, -k, -l), {i: 3, -k: 2})
    assert latex(expr) == 'K{}^{i=3,j}{}_{k=2,l}'
    expr = TensorElement(K(i, j, -k, -l), {i: 3})
    assert latex(expr) == 'K{}^{i=3,j}{}_{kl}'
    expr = PartialDerivative(A(i), A(i))
    assert latex(expr) == '\\frac{\\partial}{\\partial {A{}^{L_{0}}}}{A{}^{L_{0}}}'
    expr = PartialDerivative(A(-i), A(-j))
    assert latex(expr) == '\\frac{\\partial}{\\partial {A{}_{j}}}{A{}_{i}}'
    expr = PartialDerivative(K(i, j, -k, -l), A(m), A(-n))
    assert latex(expr) == '\\frac{\\partial^{2}}{\\partial {A{}^{m}} \\partial {A{}_{n}}}{K{}^{ij}{}_{kl}}'
    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    assert latex(expr) == '\\frac{\\partial^{2}}{\\partial {A{}_{j}} \\partial {A{}_{n}}}{\\left(A{}_{i} + B{}_{i}\\right)}'
    expr = PartialDerivative(3 * A(-i), A(-j), A(-n))
    assert latex(expr) == '\\frac{\\partial^{2}}{\\partial {A{}_{j}} \\partial {A{}_{n}}}{\\left(3A{}_{i}\\right)}'

def test_multiline_latex():
    if False:
        i = 10
        return i + 15
    (a, b, c, d, e, f) = symbols('a b c d e f')
    expr = -a + 2 * b - 3 * c + 4 * d - 5 * e
    expected = '\\begin{eqnarray}' + '\nf & = &- a \\nonumber\\\\' + '\n& & + 2 b \\nonumber\\\\' + '\n& & - 3 c \\nonumber\\\\' + '\n& & + 4 d \\nonumber\\\\' + '\n& & - 5 e ' + '\n\\end{eqnarray}'
    assert multiline_latex(f, expr, environment='eqnarray') == expected
    expected2 = '\\begin{eqnarray}' + '\nf & = &- a + 2 b \\nonumber\\\\' + '\n& & - 3 c + 4 d \\nonumber\\\\' + '\n& & - 5 e ' + '\n\\end{eqnarray}'
    assert multiline_latex(f, expr, 2, environment='eqnarray') == expected2
    expected3 = '\\begin{eqnarray}' + '\nf & = &- a + 2 b - 3 c \\nonumber\\\\' + '\n& & + 4 d - 5 e ' + '\n\\end{eqnarray}'
    assert multiline_latex(f, expr, 3, environment='eqnarray') == expected3
    expected3dots = '\\begin{eqnarray}' + '\nf & = &- a + 2 b - 3 c \\dots\\nonumber\\\\' + '\n& & + 4 d - 5 e ' + '\n\\end{eqnarray}'
    assert multiline_latex(f, expr, 3, environment='eqnarray', use_dots=True) == expected3dots
    expected3align = '\\begin{align*}' + '\nf = &- a + 2 b - 3 c \\\\' + '\n& + 4 d - 5 e ' + '\n\\end{align*}'
    assert multiline_latex(f, expr, 3) == expected3align
    assert multiline_latex(f, expr, 3, environment='align*') == expected3align
    expected2ieee = '\\begin{IEEEeqnarray}{rCl}' + '\nf & = &- a + 2 b \\nonumber\\\\' + '\n& & - 3 c + 4 d \\nonumber\\\\' + '\n& & - 5 e ' + '\n\\end{IEEEeqnarray}'
    assert multiline_latex(f, expr, 2, environment='IEEEeqnarray') == expected2ieee
    raises(ValueError, lambda : multiline_latex(f, expr, environment='foo'))

def test_issue_15353():
    if False:
        return 10
    (a, x) = symbols('a x')
    sol = ConditionSet(Tuple(x, a), Eq(sin(a * x), 0) & Eq(cos(a * x), 0), S.Complexes ** 2)
    assert latex(sol) == '\\left\\{\\left( x, \\  a\\right)\\; \\middle|\\; \\left( x, \\  a\\right) \\in \\mathbb{C}^{2} \\wedge \\sin{\\left(a x \\right)} = 0 \\wedge \\cos{\\left(a x \\right)} = 0 \\right\\}'

def test_latex_symbolic_probability():
    if False:
        i = 10
        return i + 15
    mu = symbols('mu')
    sigma = symbols('sigma', positive=True)
    X = Normal('X', mu, sigma)
    assert latex(Expectation(X)) == '\\operatorname{E}\\left[X\\right]'
    assert latex(Variance(X)) == '\\operatorname{Var}\\left(X\\right)'
    assert latex(Probability(X > 0)) == '\\operatorname{P}\\left(X > 0\\right)'
    Y = Normal('Y', mu, sigma)
    assert latex(Covariance(X, Y)) == '\\operatorname{Cov}\\left(X, Y\\right)'

def test_trace():
    if False:
        return 10
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol('A', 2, 2)
    assert latex(trace(A)) == '\\operatorname{tr}\\left(A \\right)'
    assert latex(trace(A ** 2)) == '\\operatorname{tr}\\left(A^{2} \\right)'

def test_print_basic():
    if False:
        while True:
            i = 10
    from sympy.core.basic import Basic
    from sympy.core.expr import Expr

    class UnimplementedExpr(Expr):

        def __new__(cls, e):
            if False:
                for i in range(10):
                    print('nop')
            return Basic.__new__(cls, e)

    def unimplemented_expr(expr):
        if False:
            return 10
        return UnimplementedExpr(expr).doit()

    def unimplemented_expr_sup_sub(expr):
        if False:
            for i in range(10):
                print('nop')
        result = UnimplementedExpr(expr)
        result.__class__.__name__ = 'UnimplementedExpr_x^1'
        return result
    assert latex(unimplemented_expr(x)) == '\\operatorname{UnimplementedExpr}\\left(x\\right)'
    assert latex(unimplemented_expr(x ** 2)) == '\\operatorname{UnimplementedExpr}\\left(x^{2}\\right)'
    assert latex(unimplemented_expr_sup_sub(x)) == '\\operatorname{UnimplementedExpr^{1}_{x}}\\left(x\\right)'

def test_MatrixSymbol_bold():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol('A', 2, 2)
    assert latex(trace(A), mat_symbol_style='bold') == '\\operatorname{tr}\\left(\\mathbf{A} \\right)'
    assert latex(trace(A), mat_symbol_style='plain') == '\\operatorname{tr}\\left(A \\right)'
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    C = MatrixSymbol('C', 3, 3)
    assert latex(-A, mat_symbol_style='bold') == '- \\mathbf{A}'
    assert latex(A - A * B - B, mat_symbol_style='bold') == '\\mathbf{A} - \\mathbf{A} \\mathbf{B} - \\mathbf{B}'
    assert latex(-A * B - A * B * C - B, mat_symbol_style='bold') == '- \\mathbf{A} \\mathbf{B} - \\mathbf{A} \\mathbf{B} \\mathbf{C} - \\mathbf{B}'
    A_k = MatrixSymbol('A_k', 3, 3)
    assert latex(A_k, mat_symbol_style='bold') == '\\mathbf{A}_{k}'
    A = MatrixSymbol('\\nabla_k', 3, 3)
    assert latex(A, mat_symbol_style='bold') == '\\mathbf{\\nabla}_{k}'

def test_AppliedPermutation():
    if False:
        for i in range(10):
            print('nop')
    p = Permutation(0, 1, 2)
    x = Symbol('x')
    assert latex(AppliedPermutation(p, x)) == '\\sigma_{\\left( 0\\; 1\\; 2\\right)}(x)'

def test_PermutationMatrix():
    if False:
        return 10
    p = Permutation(0, 1, 2)
    assert latex(PermutationMatrix(p)) == 'P_{\\left( 0\\; 1\\; 2\\right)}'
    p = Permutation(0, 3)(1, 2)
    assert latex(PermutationMatrix(p)) == 'P_{\\left( 0\\; 3\\right)\\left( 1\\; 2\\right)}'

def test_issue_21758():
    if False:
        return 10
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.series.fourier import FourierSeries
    x = Symbol('x')
    (k, n) = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(Piecewise((-2 * pi * cos(n * pi) / n + 2 * sin(n * pi) / n ** 2, (n > -oo) & (n < oo) & Ne(n, 0)), (0, True)) * sin(n * x) / pi, (n, 1, oo))))
    assert latex(piecewise_fold(fo)) == '\\begin{cases} 2 \\sin{\\left(x \\right)} - \\sin{\\left(2 x \\right)} + \\frac{2 \\sin{\\left(3 x \\right)}}{3} + \\ldots & \\text{for}\\: n > -\\infty \\wedge n < \\infty \\wedge n \\neq 0 \\\\0 & \\text{otherwise} \\end{cases}'
    assert latex(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(0, (n, 1, oo))))) == '0'

def test_imaginary_unit():
    if False:
        print('Hello World!')
    assert latex(1 + I) == '1 + i'
    assert latex(1 + I, imaginary_unit='i') == '1 + i'
    assert latex(1 + I, imaginary_unit='j') == '1 + j'
    assert latex(1 + I, imaginary_unit='foo') == '1 + foo'
    assert latex(I, imaginary_unit='ti') == '\\text{i}'
    assert latex(I, imaginary_unit='tj') == '\\text{j}'

def test_text_re_im():
    if False:
        return 10
    assert latex(im(x), gothic_re_im=True) == '\\Im{\\left(x\\right)}'
    assert latex(im(x), gothic_re_im=False) == '\\operatorname{im}{\\left(x\\right)}'
    assert latex(re(x), gothic_re_im=True) == '\\Re{\\left(x\\right)}'
    assert latex(re(x), gothic_re_im=False) == '\\operatorname{re}{\\left(x\\right)}'

def test_latex_diffgeom():
    if False:
        while True:
            i = 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    from sympy.diffgeom.rn import R2
    (x, y) = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert latex(m) == '\\text{M}'
    p = Patch('P', m)
    assert latex(p) == '\\text{P}_{\\text{M}}'
    rect = CoordSystem('rect', p, [x, y])
    assert latex(rect) == '\\text{rect}^{\\text{P}}_{\\text{M}}'
    b = BaseScalarField(rect, 0)
    assert latex(b) == '\\mathbf{x}'
    g = Function('g')
    s_field = g(R2.x, R2.y)
    assert latex(Differential(s_field)) == '\\operatorname{d}\\left(g{\\left(\\mathbf{x},\\mathbf{y} \\right)}\\right)'

def test_unit_printing():
    if False:
        return 10
    assert latex(5 * meter) == '5 \\text{m}'
    assert latex(3 * gibibyte) == '3 \\text{gibibyte}'
    assert latex(4 * microgram / second) == '\\frac{4 \\mu\\text{g}}{\\text{s}}'
    assert latex(4 * micro * gram / second) == '\\frac{4 \\mu \\text{g}}{\\text{s}}'
    assert latex(5 * milli * meter) == '5 \\text{m} \\text{m}'
    assert latex(milli) == '\\text{m}'

def test_issue_17092():
    if False:
        for i in range(10):
            print('nop')
    x_star = Symbol('x^*')
    assert latex(Derivative(x_star, x_star, 2)) == '\\frac{d^{2}}{d \\left(x^{*}\\right)^{2}} x^{*}'

def test_latex_decimal_separator():
    if False:
        i = 10
        return i + 15
    (x, y, z, t) = symbols('x y z t')
    (k, m, n) = symbols('k m n', integer=True)
    (f, g, h) = symbols('f g h', cls=Function)
    assert latex([1, 2.3, 4.5], decimal_separator='comma') == '\\left[ 1; \\  2{,}3; \\  4{,}5\\right]'
    assert latex(FiniteSet(1, 2.3, 4.5), decimal_separator='comma') == '\\left\\{1; 2{,}3; 4{,}5\\right\\}'
    assert latex((1, 2.3, 4.6), decimal_separator='comma') == '\\left( 1; \\  2{,}3; \\  4{,}6\\right)'
    assert latex((1,), decimal_separator='comma') == '\\left( 1;\\right)'
    assert latex([1, 2.3, 4.5], decimal_separator='period') == '\\left[ 1, \\  2.3, \\  4.5\\right]'
    assert latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == '\\left\\{1, 2.3, 4.5\\right\\}'
    assert latex((1, 2.3, 4.6), decimal_separator='period') == '\\left( 1, \\  2.3, \\  4.6\\right)'
    assert latex((1,), decimal_separator='period') == '\\left( 1,\\right)'
    assert latex([1, 2.3, 4.5]) == '\\left[ 1, \\  2.3, \\  4.5\\right]'
    assert latex(FiniteSet(1, 2.3, 4.5)) == '\\left\\{1, 2.3, 4.5\\right\\}'
    assert latex((1, 2.3, 4.6)) == '\\left( 1, \\  2.3, \\  4.6\\right)'
    assert latex((1,)) == '\\left( 1,\\right)'
    assert latex(Mul(3.4, 5.3), decimal_separator='comma') == '18{,}02'
    assert latex(3.4 * 5.3, decimal_separator='comma') == '18{,}02'
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    assert latex(x * 5.3 + 2 ** y ** 3.4 + 4.5 + z, decimal_separator='comma') == '2^{y^{3{,}4}} + 5{,}3 x + z + 4{,}5'
    assert latex(0.987, decimal_separator='comma') == '0{,}987'
    assert latex(S(0.987), decimal_separator='comma') == '0{,}987'
    assert latex(0.3, decimal_separator='comma') == '0{,}3'
    assert latex(S(0.3), decimal_separator='comma') == '0{,}3'
    assert latex(5.8 * 10 ** (-7), decimal_separator='comma') == '5{,}8 \\cdot 10^{-7}'
    assert latex(S(5.7) * 10 ** (-7), decimal_separator='comma') == '5{,}7 \\cdot 10^{-7}'
    assert latex(S(5.7 * 10 ** (-7)), decimal_separator='comma') == '5{,}7 \\cdot 10^{-7}'
    x = symbols('x')
    assert latex(1.2 * x + 3.4, decimal_separator='comma') == '1{,}2 x + 3{,}4'
    assert latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == '\\left\\{1, 2.3, 4.5\\right\\}'
    raises(ValueError, lambda : latex([1, 2.3, 4.5], decimal_separator='non_existing_decimal_separator_in_list'))
    raises(ValueError, lambda : latex(FiniteSet(1, 2.3, 4.5), decimal_separator='non_existing_decimal_separator_in_set'))
    raises(ValueError, lambda : latex((1, 2.3, 4.5), decimal_separator='non_existing_decimal_separator_in_tuple'))

def test_Str():
    if False:
        return 10
    from sympy.core.symbol import Str
    assert str(Str('x')) == 'x'

def test_latex_escape():
    if False:
        while True:
            i = 10
    assert latex_escape('~^\\&%$#_{}') == ''.join(['\\textasciitilde', '\\textasciicircum', '\\textbackslash', '\\&', '\\%', '\\$', '\\#', '\\_', '\\{', '\\}'])

def test_emptyPrinter():
    if False:
        return 10

    class MyObject:

        def __repr__(self):
            if False:
                i = 10
                return i + 15
            return '<MyObject with {...}>'
    assert latex(MyObject()) == '\\mathtt{\\text{<MyObject with \\{...\\}>}}'
    assert latex((MyObject(),)) == '\\left( \\mathtt{\\text{<MyObject with \\{...\\}>}},\\right)'

def test_global_settings():
    if False:
        i = 10
        return i + 15
    import inspect
    assert inspect.signature(latex).parameters['imaginary_unit'].default == 'i'
    assert latex(I) == 'i'
    try:
        LatexPrinter.set_global_settings(imaginary_unit='j')
        assert inspect.signature(latex).parameters['imaginary_unit'].default == 'j'
        assert latex(I) == 'j'
    finally:
        del LatexPrinter._global_settings['imaginary_unit']
    assert inspect.signature(latex).parameters['imaginary_unit'].default == 'i'
    assert latex(I) == 'i'

def test_pickleable():
    if False:
        for i in range(10):
            print('nop')
    import pickle
    assert pickle.loads(pickle.dumps(latex)) is latex

def test_printing_latex_array_expressions():
    if False:
        for i in range(10):
            print('nop')
    assert latex(ArraySymbol('A', (2, 3, 4))) == 'A'
    assert latex(ArrayElement('A', (2, 1 / (1 - x), 0))) == '{{A}_{2, \\frac{1}{1 - x}, 0}}'
    M = MatrixSymbol('M', 3, 3)
    N = MatrixSymbol('N', 3, 3)
    assert latex(ArrayElement(M * N, [x, 0])) == '{{\\left(M N\\right)}_{x, 0}}'

def test_Array():
    if False:
        for i in range(10):
            print('nop')
    arr = Array(range(10))
    assert latex(arr) == '\\left[\\begin{matrix}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\\end{matrix}\\right]'
    arr = Array(range(11))
    assert latex(arr) == '\\left[\\begin{array}{}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\\end{array}\\right]'

def test_latex_with_unevaluated():
    if False:
        for i in range(10):
            print('nop')
    with evaluate(False):
        assert latex(a * a) == 'a a'