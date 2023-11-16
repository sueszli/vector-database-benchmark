"""
This module contains :py:meth:`~sympy.solvers.ode.dsolve` and different helper
functions that it uses.

:py:meth:`~sympy.solvers.ode.dsolve` solves ordinary differential equations.
See the docstring on the various functions for their uses.  Note that partial
differential equations support is in ``pde.py``.  Note that hint functions
have docstrings describing their various methods, but they are intended for
internal use.  Use ``dsolve(ode, func, hint=hint)`` to solve an ODE using a
specific hint.  See also the docstring on
:py:meth:`~sympy.solvers.ode.dsolve`.

**Functions in this module**

    These are the user functions in this module:

    - :py:meth:`~sympy.solvers.ode.dsolve` - Solves ODEs.
    - :py:meth:`~sympy.solvers.ode.classify_ode` - Classifies ODEs into
      possible hints for :py:meth:`~sympy.solvers.ode.dsolve`.
    - :py:meth:`~sympy.solvers.ode.checkodesol` - Checks if an equation is the
      solution to an ODE.
    - :py:meth:`~sympy.solvers.ode.homogeneous_order` - Returns the
      homogeneous order of an expression.
    - :py:meth:`~sympy.solvers.ode.infinitesimals` - Returns the infinitesimals
      of the Lie group of point transformations of an ODE, such that it is
      invariant.
    - :py:meth:`~sympy.solvers.ode.checkinfsol` - Checks if the given infinitesimals
      are the actual infinitesimals of a first order ODE.

    These are the non-solver helper functions that are for internal use.  The
    user should use the various options to
    :py:meth:`~sympy.solvers.ode.dsolve` to obtain the functionality provided
    by these functions:

    - :py:meth:`~sympy.solvers.ode.ode.odesimp` - Does all forms of ODE
      simplification.
    - :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity` - A key function for
      comparing solutions by simplicity.
    - :py:meth:`~sympy.solvers.ode.constantsimp` - Simplifies arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode.constant_renumber` - Renumber arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode._handle_Integral` - Evaluate unevaluated
      Integrals.

    See also the docstrings of these functions.

**Currently implemented solver methods**

The following methods are implemented for solving ordinary differential
equations.  See the docstrings of the various hint functions for more
information on each (run ``help(ode)``):

  - 1st order separable differential equations.
  - 1st order differential equations whose coefficients or `dx` and `dy` are
    functions homogeneous of the same order.
  - 1st order exact differential equations.
  - 1st order linear differential equations.
  - 1st order Bernoulli differential equations.
  - Power series solutions for first order differential equations.
  - Lie Group method of solving first order differential equations.
  - 2nd order Liouville differential equations.
  - Power series solutions for second order differential equations
    at ordinary and regular singular points.
  - `n`\\th order differential equation that can be solved with algebraic
    rearrangement and integration.
  - `n`\\th order linear homogeneous differential equation with constant
    coefficients.
  - `n`\\th order linear inhomogeneous differential equation with constant
    coefficients using the method of undetermined coefficients.
  - `n`\\th order linear inhomogeneous differential equation with constant
    coefficients using the method of variation of parameters.

**Philosophy behind this module**

This module is designed to make it easy to add new ODE solving methods without
having to mess with the solving code for other methods.  The idea is that
there is a :py:meth:`~sympy.solvers.ode.classify_ode` function, which takes in
an ODE and tells you what hints, if any, will solve the ODE.  It does this
without attempting to solve the ODE, so it is fast.  Each solving method is a
hint, and it has its own function, named ``ode_<hint>``.  That function takes
in the ODE and any match expression gathered by
:py:meth:`~sympy.solvers.ode.classify_ode` and returns a solved result.  If
this result has any integrals in it, the hint function will return an
unevaluated :py:class:`~sympy.integrals.integrals.Integral` class.
:py:meth:`~sympy.solvers.ode.dsolve`, which is the user wrapper function
around all of this, will then call :py:meth:`~sympy.solvers.ode.ode.odesimp` on
the result, which, among other things, will attempt to solve the equation for
the dependent variable (the function we are solving for), simplify the
arbitrary constants in the expression, and evaluate any integrals, if the hint
allows it.

**How to add new solution methods**

If you have an ODE that you want :py:meth:`~sympy.solvers.ode.dsolve` to be
able to solve, try to avoid adding special case code here.  Instead, try
finding a general method that will solve your ODE, as well as others.  This
way, the :py:mod:`~sympy.solvers.ode` module will become more robust, and
unhindered by special case hacks.  WolphramAlpha and Maple's
DETools[odeadvisor] function are two resources you can use to classify a
specific ODE.  It is also better for a method to work with an `n`\\th order ODE
instead of only with specific orders, if possible.

To add a new method, there are a few things that you need to do.  First, you
need a hint name for your method.  Try to name your hint so that it is
unambiguous with all other methods, including ones that may not be implemented
yet.  If your method uses integrals, also include a ``hint_Integral`` hint.
If there is more than one way to solve ODEs with your method, include a hint
for each one, as well as a ``<hint>_best`` hint.  Your ``ode_<hint>_best()``
function should choose the best using min with ``ode_sol_simplicity`` as the
key argument.  See
:obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest`, for example.
The function that uses your method will be called ``ode_<hint>()``, so the
hint must only use characters that are allowed in a Python function name
(alphanumeric characters and the underscore '``_``' character).  Include a
function for every hint, except for ``_Integral`` hints
(:py:meth:`~sympy.solvers.ode.dsolve` takes care of those automatically).
Hint names should be all lowercase, unless a word is commonly capitalized
(such as Integral or Bernoulli).  If you have a hint that you do not want to
run with ``all_Integral`` that does not have an ``_Integral`` counterpart (such
as a best hint that would defeat the purpose of ``all_Integral``), you will
need to remove it manually in the :py:meth:`~sympy.solvers.ode.dsolve` code.
See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
guidelines on writing a hint name.

Determine *in general* how the solutions returned by your method compare with
other methods that can potentially solve the same ODEs.  Then, put your hints
in the :py:data:`~sympy.solvers.ode.allhints` tuple in the order that they
should be called.  The ordering of this tuple determines which hints are
default.  Note that exceptions are ok, because it is easy for the user to
choose individual hints with :py:meth:`~sympy.solvers.ode.dsolve`.  In
general, ``_Integral`` variants should go at the end of the list, and
``_best`` variants should go before the various hints they apply to.  For
example, the ``undetermined_coefficients`` hint comes before the
``variation_of_parameters`` hint because, even though variation of parameters
is more general than undetermined coefficients, undetermined coefficients
generally returns cleaner results for the ODEs that it can solve than
variation of parameters does, and it does not require integration, so it is
much faster.

Next, you need to have a match expression or a function that matches the type
of the ODE, which you should put in :py:meth:`~sympy.solvers.ode.classify_ode`
(if the match function is more than just a few lines.  It should match the
ODE without solving for it as much as possible, so that
:py:meth:`~sympy.solvers.ode.classify_ode` remains fast and is not hindered by
bugs in solving code.  Be sure to consider corner cases.  For example, if your
solution method involves dividing by something, make sure you exclude the case
where that division will be 0.

In most cases, the matching of the ODE will also give you the various parts
that you need to solve it.  You should put that in a dictionary (``.match()``
will do this for you), and add that as ``matching_hints['hint'] = matchdict``
in the relevant part of :py:meth:`~sympy.solvers.ode.classify_ode`.
:py:meth:`~sympy.solvers.ode.classify_ode` will then send this to
:py:meth:`~sympy.solvers.ode.dsolve`, which will send it to your function as
the ``match`` argument.  Your function should be named ``ode_<hint>(eq, func,
order, match)`.  If you need to send more information, put it in the ``match``
dictionary.  For example, if you had to substitute in a dummy variable in
:py:meth:`~sympy.solvers.ode.classify_ode` to match the ODE, you will need to
pass it to your function using the `match` dict to access it.  You can access
the independent variable using ``func.args[0]``, and the dependent variable
(the function you are trying to solve for) as ``func.func``.  If, while trying
to solve the ODE, you find that you cannot, raise ``NotImplementedError``.
:py:meth:`~sympy.solvers.ode.dsolve` will catch this error with the ``all``
meta-hint, rather than causing the whole routine to fail.

Add a docstring to your function that describes the method employed.  Like
with anything else in SymPy, you will need to add a doctest to the docstring,
in addition to real tests in ``test_ode.py``.  Try to maintain consistency
with the other hint functions' docstrings.  Add your method to the list at the
top of this docstring.  Also, add your method to ``ode.rst`` in the
``docs/src`` directory, so that the Sphinx docs will pull its docstring into
the main SymPy documentation.  Be sure to make the Sphinx documentation by
running ``make html`` from within the doc directory to verify that the
docstring formats correctly.

If your solution method involves integrating, use :py:obj:`~.Integral` instead of
:py:meth:`~sympy.core.expr.Expr.integrate`.  This allows the user to bypass
hard/slow integration by using the ``_Integral`` variant of your hint.  In
most cases, calling :py:meth:`sympy.core.basic.Basic.doit` will integrate your
solution.  If this is not the case, you will need to write special code in
:py:meth:`~sympy.solvers.ode.ode._handle_Integral`.  Arbitrary constants should be
symbols named ``C1``, ``C2``, and so on.  All solution methods should return
an equality instance.  If you need an arbitrary number of arbitrary constants,
you can use ``constants = numbered_symbols(prefix='C', cls=Symbol, start=1)``.
If it is possible to solve for the dependent function in a general way, do so.
Otherwise, do as best as you can, but do not call solve in your
``ode_<hint>()`` function.  :py:meth:`~sympy.solvers.ode.ode.odesimp` will attempt
to solve the solution for you, so you do not need to do that.  Lastly, if your
ODE has a common simplification that can be applied to your solutions, you can
add a special case in :py:meth:`~sympy.solvers.ode.ode.odesimp` for it.  For
example, solutions returned from the ``1st_homogeneous_coeff`` hints often
have many :obj:`~sympy.functions.elementary.exponential.log` terms, so
:py:meth:`~sympy.solvers.ode.ode.odesimp` calls
:py:meth:`~sympy.simplify.simplify.logcombine` on them (it also helps to write
the arbitrary constant as ``log(C1)`` instead of ``C1`` in this case).  Also
consider common ways that you can rearrange your solution to have
:py:meth:`~sympy.solvers.ode.constantsimp` take better advantage of it.  It is
better to put simplification in :py:meth:`~sympy.solvers.ode.ode.odesimp` than in
your method, because it can then be turned off with the simplify flag in
:py:meth:`~sympy.solvers.ode.dsolve`.  If you have any extraneous
simplification in your function, be sure to only run it using ``if
match.get('simplify', True):``, especially if it can be slow or if it can
reduce the domain of the solution.

Finally, as with every contribution to SymPy, your method will need to be
tested.  Add a test for each method in ``test_ode.py``.  Follow the
conventions there, i.e., test the solver using ``dsolve(eq, f(x),
hint=your_hint)``, and also test the solution using
:py:meth:`~sympy.solvers.ode.checkodesol` (you can put these in a separate
tests and skip/XFAIL if it runs too slow/does not work).  Be sure to call your
hint specifically in :py:meth:`~sympy.solvers.ode.dsolve`, that way the test
will not be broken simply by the introduction of another matching hint.  If your
method works for higher order (>1) ODEs, you will need to run ``sol =
constant_renumber(sol, 'C', 1, order)`` for each solution, where ``order`` is
the order of the ODE.  This is because ``constant_renumber`` renumbers the
arbitrary constants by printing order, which is platform dependent.  Try to
test every corner case of your solver, including a range of orders if it is a
`n`\\th order solver, but if your solver is slow, such as if it involves hard
integration, try to keep the test run time down.

Feel free to refactor existing hints to avoid duplicating code or creating
inconsistencies.  If you can show that your method exactly duplicates an
existing method, including in the simplicity and speed of obtaining the
solutions, then you can remove the old, less general method.  The existing
code is tested extensively in ``test_ode.py``, so if anything is broken, one
of those tests will surely fail.

"""
from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import Function, Derivative, AppliedUndef, diff, expand, expand_mul, Subs
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import BooleanAtom, BooleanTrue, BooleanFalse
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import Poly, terms_gcd, PolynomialError, lcm
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import collect, logcombine, powsimp, separatevars, simplify, cse
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
allhints = ('factorable', 'nth_algebraic', 'separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_rational_riccati', 'Riccati_special_minus2', '1st_homogeneous_coeff_best', '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep', 'almost_linear', 'linear_coefficients', 'separable_reduced', '1st_power_series', 'lie_group', 'nth_linear_constant_coeff_homogeneous', 'nth_linear_euler_eq_homogeneous', 'nth_linear_constant_coeff_undetermined_coefficients', 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients', 'nth_linear_constant_coeff_variation_of_parameters', 'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters', 'Liouville', '2nd_linear_airy', '2nd_linear_bessel', '2nd_hypergeometric', '2nd_hypergeometric_Integral', 'nth_order_reducible', '2nd_power_series_ordinary', '2nd_power_series_regular', 'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral', '1st_linear_Integral', 'Bernoulli_Integral', '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_homogeneous_coeff_subs_dep_div_indep_Integral', 'almost_linear_Integral', 'linear_coefficients_Integral', 'separable_reduced_Integral', 'nth_linear_constant_coeff_variation_of_parameters_Integral', 'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral', 'Liouville_Integral', '2nd_nonlinear_autonomous_conserved', '2nd_nonlinear_autonomous_conserved_Integral')

def get_numbered_constants(eq, num=1, start=1, prefix='C'):
    if False:
        return 10
    '\n    Returns a list of constants that do not occur\n    in eq already.\n    '
    ncs = iter_numbered_constants(eq, start, prefix)
    Cs = [next(ncs) for i in range(num)]
    return Cs[0] if num == 1 else tuple(Cs)

def iter_numbered_constants(eq, start=1, prefix='C'):
    if False:
        return 10
    '\n    Returns an iterator of constants that do not occur\n    in eq already.\n    '
    if isinstance(eq, (Expr, Eq)):
        eq = [eq]
    elif not iterable(eq):
        raise ValueError('Expected Expr or iterable but got %s' % eq)
    atom_set = set().union(*[i.free_symbols for i in eq])
    func_set = set().union(*[i.atoms(Function) for i in eq])
    if func_set:
        atom_set |= {Symbol(str(f.func)) for f in func_set}
    return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)

def dsolve(eq, func=None, hint='default', simplify=True, ics=None, xi=None, eta=None, x0=0, n=6, **kwargs):
    if False:
        print('Hello World!')
    '\n    Solves any (supported) kind of ordinary differential equation and\n    system of ordinary differential equations.\n\n    For single ordinary differential equation\n    =========================================\n\n    It is classified under this when number of equation in ``eq`` is one.\n    **Usage**\n\n        ``dsolve(eq, f(x), hint)`` -> Solve ordinary differential equation\n        ``eq`` for function ``f(x)``, using method ``hint``.\n\n    **Details**\n\n        ``eq`` can be any supported ordinary differential equation (see the\n            :py:mod:`~sympy.solvers.ode` docstring for supported methods).\n            This can either be an :py:class:`~sympy.core.relational.Equality`,\n            or an expression, which is assumed to be equal to ``0``.\n\n        ``f(x)`` is a function of one variable whose derivatives in that\n            variable make up the ordinary differential equation ``eq``.  In\n            many cases it is not necessary to provide this; it will be\n            autodetected (and an error raised if it could not be detected).\n\n        ``hint`` is the solving method that you want dsolve to use.  Use\n            ``classify_ode(eq, f(x))`` to get all of the possible hints for an\n            ODE.  The default hint, ``default``, will use whatever hint is\n            returned first by :py:meth:`~sympy.solvers.ode.classify_ode`.  See\n            Hints below for more options that you can use for hint.\n\n        ``simplify`` enables simplification by\n            :py:meth:`~sympy.solvers.ode.ode.odesimp`.  See its docstring for more\n            information.  Turn this off, for example, to disable solving of\n            solutions for ``func`` or simplification of arbitrary constants.\n            It will still integrate with this hint. Note that the solution may\n            contain more arbitrary constants than the order of the ODE with\n            this option enabled.\n\n        ``xi`` and ``eta`` are the infinitesimal functions of an ordinary\n            differential equation. They are the infinitesimals of the Lie group\n            of point transformations for which the differential equation is\n            invariant. The user can specify values for the infinitesimals. If\n            nothing is specified, ``xi`` and ``eta`` are calculated using\n            :py:meth:`~sympy.solvers.ode.infinitesimals` with the help of various\n            heuristics.\n\n        ``ics`` is the set of initial/boundary conditions for the differential equation.\n          It should be given in the form of ``{f(x0): x1, f(x).diff(x).subs(x, x2):\n          x3}`` and so on.  For power series solutions, if no initial\n          conditions are specified ``f(0)`` is assumed to be ``C0`` and the power\n          series solution is calculated about 0.\n\n        ``x0`` is the point about which the power series solution of a differential\n          equation is to be evaluated.\n\n        ``n`` gives the exponent of the dependent variable up to which the power series\n          solution of a differential equation is to be evaluated.\n\n    **Hints**\n\n        Aside from the various solving methods, there are also some meta-hints\n        that you can pass to :py:meth:`~sympy.solvers.ode.dsolve`:\n\n        ``default``:\n                This uses whatever hint is returned first by\n                :py:meth:`~sympy.solvers.ode.classify_ode`. This is the\n                default argument to :py:meth:`~sympy.solvers.ode.dsolve`.\n\n        ``all``:\n                To make :py:meth:`~sympy.solvers.ode.dsolve` apply all\n                relevant classification hints, use ``dsolve(ODE, func,\n                hint="all")``.  This will return a dictionary of\n                ``hint:solution`` terms.  If a hint causes dsolve to raise the\n                ``NotImplementedError``, value of that hint\'s key will be the\n                exception object raised.  The dictionary will also include\n                some special keys:\n\n                - ``order``: The order of the ODE.  See also\n                  :py:meth:`~sympy.solvers.deutils.ode_order` in\n                  ``deutils.py``.\n                - ``best``: The simplest hint; what would be returned by\n                  ``best`` below.\n                - ``best_hint``: The hint that would produce the solution\n                  given by ``best``.  If more than one hint produces the best\n                  solution, the first one in the tuple returned by\n                  :py:meth:`~sympy.solvers.ode.classify_ode` is chosen.\n                - ``default``: The solution that would be returned by default.\n                  This is the one produced by the hint that appears first in\n                  the tuple returned by\n                  :py:meth:`~sympy.solvers.ode.classify_ode`.\n\n        ``all_Integral``:\n                This is the same as ``all``, except if a hint also has a\n                corresponding ``_Integral`` hint, it only returns the\n                ``_Integral`` hint.  This is useful if ``all`` causes\n                :py:meth:`~sympy.solvers.ode.dsolve` to hang because of a\n                difficult or impossible integral.  This meta-hint will also be\n                much faster than ``all``, because\n                :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive\n                routine.\n\n        ``best``:\n                To have :py:meth:`~sympy.solvers.ode.dsolve` try all methods\n                and return the simplest one.  This takes into account whether\n                the solution is solvable in the function, whether it contains\n                any Integral classes (i.e.  unevaluatable integrals), and\n                which one is the shortest in size.\n\n        See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for\n        more info on hints, and the :py:mod:`~sympy.solvers.ode` docstring for\n        a list of all supported hints.\n\n    **Tips**\n\n        - You can declare the derivative of an unknown function this way:\n\n            >>> from sympy import Function, Derivative\n            >>> from sympy.abc import x # x is the independent variable\n            >>> f = Function("f")(x) # f is a function of x\n            >>> # f_ will be the derivative of f with respect to x\n            >>> f_ = Derivative(f, x)\n\n        - See ``test_ode.py`` for many tests, which serves also as a set of\n          examples for how to use :py:meth:`~sympy.solvers.ode.dsolve`.\n        - :py:meth:`~sympy.solvers.ode.dsolve` always returns an\n          :py:class:`~sympy.core.relational.Equality` class (except for the\n          case when the hint is ``all`` or ``all_Integral``).  If possible, it\n          solves the solution explicitly for the function being solved for.\n          Otherwise, it returns an implicit solution.\n        - Arbitrary constants are symbols named ``C1``, ``C2``, and so on.\n        - Because all solutions should be mathematically equivalent, some\n          hints may return the exact same result for an ODE. Often, though,\n          two different hints will return the same solution formatted\n          differently.  The two should be equivalent. Also note that sometimes\n          the values of the arbitrary constants in two different solutions may\n          not be the same, because one constant may have "absorbed" other\n          constants into it.\n        - Do ``help(ode.ode_<hintname>)`` to get help more information on a\n          specific hint, where ``<hintname>`` is the name of a hint without\n          ``_Integral``.\n\n    For system of ordinary differential equations\n    =============================================\n\n    **Usage**\n        ``dsolve(eq, func)`` -> Solve a system of ordinary differential\n        equations ``eq`` for ``func`` being list of functions including\n        `x(t)`, `y(t)`, `z(t)` where number of functions in the list depends\n        upon the number of equations provided in ``eq``.\n\n    **Details**\n\n        ``eq`` can be any supported system of ordinary differential equations\n        This can either be an :py:class:`~sympy.core.relational.Equality`,\n        or an expression, which is assumed to be equal to ``0``.\n\n        ``func`` holds ``x(t)`` and ``y(t)`` being functions of one variable which\n        together with some of their derivatives make up the system of ordinary\n        differential equation ``eq``. It is not necessary to provide this; it\n        will be autodetected (and an error raised if it could not be detected).\n\n    **Hints**\n\n        The hints are formed by parameters returned by classify_sysode, combining\n        them give hints name used later for forming method name.\n\n    Examples\n    ========\n\n    >>> from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols\n    >>> from sympy.abc import x\n    >>> f = Function(\'f\')\n    >>> dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))\n    Eq(f(x), C1*sin(3*x) + C2*cos(3*x))\n\n    >>> eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)\n    >>> dsolve(eq, hint=\'1st_exact\')\n    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]\n    >>> dsolve(eq, hint=\'almost_linear\')\n    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]\n    >>> t = symbols(\'t\')\n    >>> x, y = symbols(\'x, y\', cls=Function)\n    >>> eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))\n    >>> dsolve(eq)\n    [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t)),\n    Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t) +\n    exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)))]\n    >>> eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))\n    >>> dsolve(eq)\n    {Eq(x(t), -exp(C1)/(C2*exp(C1) - cos(t))), Eq(y(t), -1/(C1 - cos(t)))}\n    '
    if iterable(eq):
        from sympy.solvers.ode.systems import dsolve_system
        try:
            sol = dsolve_system(eq, funcs=func, ics=ics, doit=True)
            return sol[0] if len(sol) == 1 else sol
        except NotImplementedError:
            pass
        match = classify_sysode(eq, func)
        eq = match['eq']
        order = match['order']
        func = match['func']
        t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
        for i in range(len(eq)):
            for func_ in func:
                if isinstance(func_, list):
                    pass
                elif eq[i].coeff(diff(func[i], t, ode_order(eq[i], func[i]))).is_negative:
                    eq[i] = -eq[i]
        match['eq'] = eq
        if len(set(order.values())) != 1:
            raise ValueError('It solves only those systems of equations whose orders are equal')
        match['order'] = list(order.values())[0]

        def recur_len(l):
            if False:
                i = 10
                return i + 15
            return sum((recur_len(item) if isinstance(item, list) else 1 for item in l))
        if recur_len(func) != len(eq):
            raise ValueError('dsolve() and classify_sysode() work with number of functions being equal to number of equations')
        if match['type_of_equation'] is None:
            raise NotImplementedError
        else:
            if match['is_linear'] == True:
                solvefunc = globals()['sysode_linear_%(no_of_equation)seq_order%(order)s' % match]
            else:
                solvefunc = globals()['sysode_nonlinear_%(no_of_equation)seq_order%(order)s' % match]
            sols = solvefunc(match)
            if ics:
                constants = Tuple(*sols).free_symbols - Tuple(*eq).free_symbols
                solved_constants = solve_ics(sols, func, constants, ics)
                return [sol.subs(solved_constants) for sol in sols]
            return sols
    else:
        given_hint = hint
        hints = _desolve(eq, func=func, hint=hint, simplify=True, xi=xi, eta=eta, type='ode', ics=ics, x0=x0, n=n, **kwargs)
        eq = hints.pop('eq', eq)
        all_ = hints.pop('all', False)
        if all_:
            retdict = {}
            failed_hints = {}
            gethints = classify_ode(eq, dict=True, hint='all')
            orderedhints = gethints['ordered_hints']
            for hint in hints:
                try:
                    rv = _helper_simplify(eq, hint, hints[hint], simplify)
                except NotImplementedError as detail:
                    failed_hints[hint] = detail
                else:
                    retdict[hint] = rv
            func = hints[hint]['func']
            retdict['best'] = min(list(retdict.values()), key=lambda x: ode_sol_simplicity(x, func, trysolving=not simplify))
            if given_hint == 'best':
                return retdict['best']
            for i in orderedhints:
                if retdict['best'] == retdict.get(i, None):
                    retdict['best_hint'] = i
                    break
            retdict['default'] = gethints['default']
            retdict['order'] = gethints['order']
            retdict.update(failed_hints)
            return retdict
        else:
            hint = hints['hint']
            return _helper_simplify(eq, hint, hints, simplify, ics=ics)

def _helper_simplify(eq, hint, match, simplify=True, ics=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function of dsolve that calls the respective\n    :py:mod:`~sympy.solvers.ode` functions to solve for the ordinary\n    differential equations. This minimizes the computation in calling\n    :py:meth:`~sympy.solvers.deutils._desolve` multiple times.\n    '
    r = match
    func = r['func']
    order = r['order']
    match = r[hint]
    if isinstance(match, SingleODESolver):
        solvefunc = match
    elif hint.endswith('_Integral'):
        solvefunc = globals()['ode_' + hint[:-len('_Integral')]]
    else:
        solvefunc = globals()['ode_' + hint]
    free = eq.free_symbols
    cons = lambda s: s.free_symbols.difference(free)
    if simplify:
        if isinstance(solvefunc, SingleODESolver):
            sols = solvefunc.get_general_solution()
        else:
            sols = solvefunc(eq, func, order, match)
        if iterable(sols):
            rv = []
            for s in sols:
                simp = odesimp(eq, s, func, hint)
                if iterable(simp):
                    rv.extend(simp)
                else:
                    rv.append(simp)
        else:
            rv = odesimp(eq, sols, func, hint)
    else:
        if isinstance(solvefunc, SingleODESolver):
            exprs = solvefunc.get_general_solution(simplify=False)
        else:
            match['simplify'] = False
            exprs = solvefunc(eq, func, order, match)
        if isinstance(exprs, list):
            rv = [_handle_Integral(expr, func, hint) for expr in exprs]
        else:
            rv = _handle_Integral(exprs, func, hint)
    if isinstance(rv, list):
        assert all((isinstance(i, Eq) for i in rv)), rv
        if simplify:
            rv = _remove_redundant_solutions(eq, rv, order, func.args[0])
        if len(rv) == 1:
            rv = rv[0]
    if ics and 'power_series' not in hint:
        if isinstance(rv, (Expr, Eq)):
            solved_constants = solve_ics([rv], [r['func']], cons(rv), ics)
            rv = rv.subs(solved_constants)
        else:
            rv1 = []
            for s in rv:
                try:
                    solved_constants = solve_ics([s], [r['func']], cons(s), ics)
                except ValueError:
                    continue
                rv1.append(s.subs(solved_constants))
            if len(rv1) == 1:
                return rv1[0]
            rv = rv1
    return rv

def solve_ics(sols, funcs, constants, ics):
    if False:
        for i in range(10):
            print('nop')
    "\n    Solve for the constants given initial conditions\n\n    ``sols`` is a list of solutions.\n\n    ``funcs`` is a list of functions.\n\n    ``constants`` is a list of constants.\n\n    ``ics`` is the set of initial/boundary conditions for the differential\n    equation. It should be given in the form of ``{f(x0): x1,\n    f(x).diff(x).subs(x, x2):  x3}`` and so on.\n\n    Returns a dictionary mapping constants to values.\n    ``solution.subs(constants)`` will replace the constants in ``solution``.\n\n    Example\n    =======\n    >>> # From dsolve(f(x).diff(x) - f(x), f(x))\n    >>> from sympy import symbols, Eq, exp, Function\n    >>> from sympy.solvers.ode.ode import solve_ics\n    >>> f = Function('f')\n    >>> x, C1 = symbols('x C1')\n    >>> sols = [Eq(f(x), C1*exp(x))]\n    >>> funcs = [f(x)]\n    >>> constants = [C1]\n    >>> ics = {f(0): 2}\n    >>> solved_constants = solve_ics(sols, funcs, constants, ics)\n    >>> solved_constants\n    {C1: 2}\n    >>> sols[0].subs(solved_constants)\n    Eq(f(x), 2*exp(x))\n\n    "
    x = funcs[0].args[0]
    diff_sols = []
    subs_sols = []
    diff_variables = set()
    for (funcarg, value) in ics.items():
        if isinstance(funcarg, AppliedUndef):
            x0 = funcarg.args[0]
            matching_func = [f for f in funcs if f.func == funcarg.func][0]
            S = sols
        elif isinstance(funcarg, (Subs, Derivative)):
            if isinstance(funcarg, Subs):
                funcarg = funcarg.doit()
            if isinstance(funcarg, Subs):
                deriv = funcarg.expr
                x0 = funcarg.point[0]
                variables = funcarg.expr.variables
                matching_func = deriv
            elif isinstance(funcarg, Derivative):
                deriv = funcarg
                x0 = funcarg.variables[0]
                variables = (x,) * len(funcarg.variables)
                matching_func = deriv.subs(x0, x)
            for sol in sols:
                if sol.has(deriv.expr.func):
                    diff_sols.append(Eq(sol.lhs.diff(*variables), sol.rhs.diff(*variables)))
            diff_variables.add(variables)
            S = diff_sols
        else:
            raise NotImplementedError('Unrecognized initial condition')
        for sol in S:
            if sol.has(matching_func):
                sol2 = sol
                sol2 = sol2.subs(x, x0)
                sol2 = sol2.subs(funcarg, value)
                if not isinstance(sol2, BooleanAtom) or not subs_sols:
                    subs_sols = [s for s in subs_sols if not isinstance(s, BooleanAtom)]
                    subs_sols.append(sol2)
    try:
        solved_constants = solve(subs_sols, constants, dict=True)
    except NotImplementedError:
        solved_constants = []
    if not solved_constants:
        raise ValueError("Couldn't solve for initial conditions")
    if solved_constants == True:
        raise ValueError('Initial conditions did not produce any solutions for constants. Perhaps they are degenerate.')
    if len(solved_constants) > 1:
        raise NotImplementedError('Initial conditions produced too many solutions for constants')
    return solved_constants[0]

def classify_ode(eq, func=None, dict=False, ics=None, *, prep=True, xi=None, eta=None, n=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a tuple of possible :py:meth:`~sympy.solvers.ode.dsolve`\n    classifications for an ODE.\n\n    The tuple is ordered so that first item is the classification that\n    :py:meth:`~sympy.solvers.ode.dsolve` uses to solve the ODE by default.  In\n    general, classifications at the near the beginning of the list will\n    produce better solutions faster than those near the end, thought there are\n    always exceptions.  To make :py:meth:`~sympy.solvers.ode.dsolve` use a\n    different classification, use ``dsolve(ODE, func,\n    hint=<classification>)``.  See also the\n    :py:meth:`~sympy.solvers.ode.dsolve` docstring for different meta-hints\n    you can use.\n\n    If ``dict`` is true, :py:meth:`~sympy.solvers.ode.classify_ode` will\n    return a dictionary of ``hint:match`` expression terms. This is intended\n    for internal use by :py:meth:`~sympy.solvers.ode.dsolve`.  Note that\n    because dictionaries are ordered arbitrarily, this will most likely not be\n    in the same order as the tuple.\n\n    You can get help on different hints by executing\n    ``help(ode.ode_hintname)``, where ``hintname`` is the name of the hint\n    without ``_Integral``.\n\n    See :py:data:`~sympy.solvers.ode.allhints` or the\n    :py:mod:`~sympy.solvers.ode` docstring for a list of all supported hints\n    that can be returned from :py:meth:`~sympy.solvers.ode.classify_ode`.\n\n    Notes\n    =====\n\n    These are remarks on hint names.\n\n    ``_Integral``\n\n        If a classification has ``_Integral`` at the end, it will return the\n        expression with an unevaluated :py:class:`~.Integral`\n        class in it.  Note that a hint may do this anyway if\n        :py:meth:`~sympy.core.expr.Expr.integrate` cannot do the integral,\n        though just using an ``_Integral`` will do so much faster.  Indeed, an\n        ``_Integral`` hint will always be faster than its corresponding hint\n        without ``_Integral`` because\n        :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive routine.\n        If :py:meth:`~sympy.solvers.ode.dsolve` hangs, it is probably because\n        :py:meth:`~sympy.core.expr.Expr.integrate` is hanging on a tough or\n        impossible integral.  Try using an ``_Integral`` hint or\n        ``all_Integral`` to get it return something.\n\n        Note that some hints do not have ``_Integral`` counterparts. This is\n        because :py:func:`~sympy.integrals.integrals.integrate` is not used in\n        solving the ODE for those method. For example, `n`\\th order linear\n        homogeneous ODEs with constant coefficients do not require integration\n        to solve, so there is no\n        ``nth_linear_homogeneous_constant_coeff_Integrate`` hint. You can\n        easily evaluate any unevaluated\n        :py:class:`~sympy.integrals.integrals.Integral`\\s in an expression by\n        doing ``expr.doit()``.\n\n    Ordinals\n\n        Some hints contain an ordinal such as ``1st_linear``.  This is to help\n        differentiate them from other hints, as well as from other methods\n        that may not be implemented yet. If a hint has ``nth`` in it, such as\n        the ``nth_linear`` hints, this means that the method used to applies\n        to ODEs of any order.\n\n    ``indep`` and ``dep``\n\n        Some hints contain the words ``indep`` or ``dep``.  These reference\n        the independent variable and the dependent function, respectively. For\n        example, if an ODE is in terms of `f(x)`, then ``indep`` will refer to\n        `x` and ``dep`` will refer to `f`.\n\n    ``subs``\n\n        If a hints has the word ``subs`` in it, it means that the ODE is solved\n        by substituting the expression given after the word ``subs`` for a\n        single dummy variable.  This is usually in terms of ``indep`` and\n        ``dep`` as above.  The substituted expression will be written only in\n        characters allowed for names of Python objects, meaning operators will\n        be spelled out.  For example, ``indep``/``dep`` will be written as\n        ``indep_div_dep``.\n\n    ``coeff``\n\n        The word ``coeff`` in a hint refers to the coefficients of something\n        in the ODE, usually of the derivative terms.  See the docstring for\n        the individual methods for more info (``help(ode)``).  This is\n        contrast to ``coefficients``, as in ``undetermined_coefficients``,\n        which refers to the common name of a method.\n\n    ``_best``\n\n        Methods that have more than one fundamental way to solve will have a\n        hint for each sub-method and a ``_best`` meta-classification. This\n        will evaluate all hints and return the best, using the same\n        considerations as the normal ``best`` meta-hint.\n\n\n    Examples\n    ========\n\n    >>> from sympy import Function, classify_ode, Eq\n    >>> from sympy.abc import x\n    >>> f = Function('f')\n    >>> classify_ode(Eq(f(x).diff(x), 0), f(x))\n    ('nth_algebraic',\n    'separable',\n    '1st_exact',\n    '1st_linear',\n    'Bernoulli',\n    '1st_homogeneous_coeff_best',\n    '1st_homogeneous_coeff_subs_indep_div_dep',\n    '1st_homogeneous_coeff_subs_dep_div_indep',\n    '1st_power_series', 'lie_group', 'nth_linear_constant_coeff_homogeneous',\n    'nth_linear_euler_eq_homogeneous',\n    'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral',\n    '1st_linear_Integral', 'Bernoulli_Integral',\n    '1st_homogeneous_coeff_subs_indep_div_dep_Integral',\n    '1st_homogeneous_coeff_subs_dep_div_indep_Integral')\n    >>> classify_ode(f(x).diff(x, 2) + 3*f(x).diff(x) + 2*f(x) - 4)\n    ('factorable', 'nth_linear_constant_coeff_undetermined_coefficients',\n    'nth_linear_constant_coeff_variation_of_parameters',\n    'nth_linear_constant_coeff_variation_of_parameters_Integral')\n\n    "
    ics = sympify(ics)
    if func and len(func.args) != 1:
        raise ValueError('dsolve() and classify_ode() only work with functions of one variable, not %s' % func)
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    eq_orig = eq
    if prep or func is None:
        (eq, func_) = _preprocess(eq, func)
        if func is None:
            func = func_
    x = func.args[0]
    f = func.func
    y = Dummy('y')
    terms = 5 if n is None else n
    order = ode_order(eq, f(x))
    matching_hints = {'order': order}
    df = f(x).diff(x)
    a = Wild('a', exclude=[f(x)])
    d = Wild('d', exclude=[df, f(x).diff(x, 2)])
    e = Wild('e', exclude=[df])
    n = Wild('n', exclude=[x, f(x), df])
    c1 = Wild('c1', exclude=[x])
    a3 = Wild('a3', exclude=[f(x), df, f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), df, f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), df, f(x).diff(x, 2)])
    boundary = {}
    C1 = Symbol('C1')
    if ics is not None:
        for funcarg in ics:
            if isinstance(funcarg, (Subs, Derivative)):
                if isinstance(funcarg, Subs):
                    deriv = funcarg.expr
                    old = funcarg.variables[0]
                    new = funcarg.point[0]
                elif isinstance(funcarg, Derivative):
                    deriv = funcarg
                    old = x
                    new = funcarg.variables[0]
                if isinstance(deriv, Derivative) and isinstance(deriv.args[0], AppliedUndef) and (deriv.args[0].func == f) and (len(deriv.args[0].args) == 1) and (old == x) and (not new.has(x)) and all((i == deriv.variables[0] for i in deriv.variables)) and (x not in ics[funcarg].free_symbols):
                    dorder = ode_order(deriv, x)
                    temp = 'f' + str(dorder)
                    boundary.update({temp: new, temp + 'val': ics[funcarg]})
                else:
                    raise ValueError('Invalid boundary conditions for Derivatives')
            elif isinstance(funcarg, AppliedUndef):
                if funcarg.func == f and len(funcarg.args) == 1 and (not funcarg.args[0].has(x)) and (x not in ics[funcarg].free_symbols):
                    boundary.update({'f0': funcarg.args[0], 'f0val': ics[funcarg]})
                else:
                    raise ValueError('Invalid boundary conditions for Function')
            else:
                raise ValueError('Enter boundary conditions of the form ics={f(point): value, f(x).diff(x, order).subs(x, point): value}')
    ode = SingleODEProblem(eq_orig, func, x, prep=prep, xi=xi, eta=eta)
    user_hint = kwargs.get('hint', 'default')
    early_exit = user_hint == 'default'
    if user_hint.endswith('_Integral'):
        user_hint = user_hint[:-len('_Integral')]
    user_map = solver_map
    if user_hint not in ['default', 'all', 'all_Integral', 'best'] and user_hint in solver_map:
        user_map = {user_hint: solver_map[user_hint]}
    for hint in user_map:
        solver = user_map[hint](ode)
        if solver.matches():
            matching_hints[hint] = solver
            if user_map[hint].has_integral:
                matching_hints[hint + '_Integral'] = solver
            if dict and early_exit:
                matching_hints['default'] = hint
                return matching_hints
    eq = expand(eq)
    reduced_eq = None
    if eq.is_Add:
        deriv_coef = eq.coeff(f(x).diff(x, order))
        if deriv_coef not in (1, 0):
            r = deriv_coef.match(a * f(x) ** c1)
            if r and r[c1]:
                den = f(x) ** r[c1]
                reduced_eq = Add(*[arg / den for arg in eq.args])
    if not reduced_eq:
        reduced_eq = eq
    if order == 1:
        r = collect(eq, df, exact=True).match(d + e * df)
        if r:
            r['d'] = d
            r['e'] = e
            r['y'] = y
            r[d] = r[d].subs(f(x), y)
            r[e] = r[e].subs(f(x), y)
            point = boundary.get('f0', 0)
            value = boundary.get('f0val', C1)
            check = cancel(r[d] / r[e])
            check1 = check.subs({x: point, y: value})
            if not check1.has(oo) and (not check1.has(zoo)) and (not check1.has(nan)) and (not check1.has(-oo)):
                check2 = check1.diff(x).subs({x: point, y: value})
                if not check2.has(oo) and (not check2.has(zoo)) and (not check2.has(nan)) and (not check2.has(-oo)):
                    rseries = r.copy()
                    rseries.update({'terms': terms, 'f0': point, 'f0val': value})
                    matching_hints['1st_power_series'] = rseries
    elif order == 2:
        deq = a3 * f(x).diff(x, 2) + b3 * df + c3 * f(x)
        r = collect(reduced_eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        ordinary = False
        if r:
            if not all((r[key].is_polynomial() for key in r)):
                (n, d) = reduced_eq.as_numer_denom()
                reduced_eq = expand(n)
                r = collect(reduced_eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        if r and r[a3] != 0:
            p = cancel(r[b3] / r[a3])
            q = cancel(r[c3] / r[a3])
            point = kwargs.get('x0', 0)
            check = p.subs(x, point)
            if not check.has(oo, nan, zoo, -oo):
                check = q.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    ordinary = True
                    r.update({'a3': a3, 'b3': b3, 'c3': c3, 'x0': point, 'terms': terms})
                    matching_hints['2nd_power_series_ordinary'] = r
            if not ordinary:
                p = cancel((x - point) * p)
                check = p.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    q = cancel((x - point) ** 2 * q)
                    check = q.subs(x, point)
                    if not check.has(oo, nan, zoo, -oo):
                        coeff_dict = {'p': p, 'q': q, 'x0': point, 'terms': terms}
                        matching_hints['2nd_power_series_regular'] = coeff_dict
    retlist = [i for i in allhints if i in matching_hints]
    if dict:
        matching_hints['default'] = retlist[0] if retlist else None
        matching_hints['ordered_hints'] = tuple(retlist)
        return matching_hints
    else:
        return tuple(retlist)

def classify_sysode(eq, funcs=None, **kwargs):
    if False:
        return 10
    '\n    Returns a dictionary of parameter names and values that define the system\n    of ordinary differential equations in ``eq``.\n    The parameters are further used in\n    :py:meth:`~sympy.solvers.ode.dsolve` for solving that system.\n\n    Some parameter names and values are:\n\n    \'is_linear\' (boolean), which tells whether the given system is linear.\n    Note that "linear" here refers to the operator: terms such as ``x*diff(x,t)`` are\n    nonlinear, whereas terms like ``sin(t)*diff(x,t)`` are still linear operators.\n\n    \'func\' (list) contains the :py:class:`~sympy.core.function.Function`s that\n    appear with a derivative in the ODE, i.e. those that we are trying to solve\n    the ODE for.\n\n    \'order\' (dict) with the maximum derivative for each element of the \'func\'\n    parameter.\n\n    \'func_coeff\' (dict or Matrix) with the coefficient for each triple ``(equation number,\n    function, order)```. The coefficients are those subexpressions that do not\n    appear in \'func\', and hence can be considered constant for purposes of ODE\n    solving. The value of this parameter can also be a  Matrix if the system of ODEs are\n    linear first order of the form X\' = AX where X is the vector of dependent variables.\n    Here, this function returns the coefficient matrix A.\n\n    \'eq\' (list) with the equations from ``eq``, sympified and transformed into\n    expressions (we are solving for these expressions to be zero).\n\n    \'no_of_equations\' (int) is the number of equations (same as ``len(eq)``).\n\n    \'type_of_equation\' (string) is an internal classification of the type of\n    ODE.\n\n    \'is_constant\' (boolean), which tells if the system of ODEs is constant coefficient\n    or not. This key is temporary addition for now and is in the match dict only when\n    the system of ODEs is linear first order constant coefficient homogeneous. So, this\n    key\'s value is True for now if it is available else it does not exist.\n\n    \'is_homogeneous\' (boolean), which tells if the system of ODEs is homogeneous. Like the\n    key \'is_constant\', this key is a temporary addition and it is True since this key value\n    is available only when the system is linear first order constant coefficient homogeneous.\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode-toc1.htm\n    -A. D. Polyanin and A. V. Manzhirov, Handbook of Mathematics for Engineers and Scientists\n\n    Examples\n    ========\n\n    >>> from sympy import Function, Eq, symbols, diff\n    >>> from sympy.solvers.ode.ode import classify_sysode\n    >>> from sympy.abc import t\n    >>> f, x, y = symbols(\'f, x, y\', cls=Function)\n    >>> k, l, m, n = symbols(\'k, l, m, n\', Integer=True)\n    >>> x1 = diff(x(t), t) ; y1 = diff(y(t), t)\n    >>> x2 = diff(x(t), t, t) ; y2 = diff(y(t), t, t)\n    >>> eq = (Eq(x1, 12*x(t) - 6*y(t)), Eq(y1, 11*x(t) + 3*y(t)))\n    >>> classify_sysode(eq)\n    {\'eq\': [-12*x(t) + 6*y(t) + Derivative(x(t), t), -11*x(t) - 3*y(t) + Derivative(y(t), t)], \'func\': [x(t), y(t)],\n     \'func_coeff\': {(0, x(t), 0): -12, (0, x(t), 1): 1, (0, y(t), 0): 6, (0, y(t), 1): 0, (1, x(t), 0): -11, (1, x(t), 1): 0, (1, y(t), 0): -3, (1, y(t), 1): 1}, \'is_linear\': True, \'no_of_equation\': 2, \'order\': {x(t): 1, y(t): 1}, \'type_of_equation\': None}\n    >>> eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t) + 2), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))\n    >>> classify_sysode(eq)\n    {\'eq\': [-t**2*y(t) - 5*t*x(t) + Derivative(x(t), t) - 2, t**2*x(t) - 5*t*y(t) + Derivative(y(t), t)],\n     \'func\': [x(t), y(t)], \'func_coeff\': {(0, x(t), 0): -5*t, (0, x(t), 1): 1, (0, y(t), 0): -t**2, (0, y(t), 1): 0,\n     (1, x(t), 0): t**2, (1, x(t), 1): 0, (1, y(t), 0): -5*t, (1, y(t), 1): 1}, \'is_linear\': True, \'no_of_equation\': 2,\n      \'order\': {x(t): 1, y(t): 1}, \'type_of_equation\': None}\n\n    '

    def _sympify(eq):
        if False:
            while True:
                i = 10
        return list(map(sympify, eq if iterable(eq) else [eq]))
    (eq, funcs) = (_sympify(w) for w in [eq, funcs])
    for (i, fi) in enumerate(eq):
        if isinstance(fi, Equality):
            eq[i] = fi.lhs - fi.rhs
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    matching_hints = {'no_of_equation': i + 1}
    matching_hints['eq'] = eq
    if i == 0:
        raise ValueError('classify_sysode() works for systems of ODEs. For scalar ODEs, classify_ode should be used')
    order = {}
    if funcs == [None]:
        funcs = _extract_funcs(eq)
    funcs = list(set(funcs))
    if len(funcs) != len(eq):
        raise ValueError('Number of functions given is not equal to the number of equations %s' % funcs)
    func_dict = {}
    for func in funcs:
        if not order.get(func, False):
            max_order = 0
            for (i, eqs_) in enumerate(eq):
                order_ = ode_order(eqs_, func)
                if max_order < order_:
                    max_order = order_
                    eq_no = i
            if eq_no in func_dict:
                func_dict[eq_no] = [func_dict[eq_no], func]
            else:
                func_dict[eq_no] = func
            order[func] = max_order
    funcs = [func_dict[i] for i in range(len(func_dict))]
    matching_hints['func'] = funcs
    for func in funcs:
        if isinstance(func, list):
            for func_elem in func:
                if len(func_elem.args) != 1:
                    raise ValueError('dsolve() and classify_sysode() work with functions of one variable only, not %s' % func)
        elif func and len(func.args) != 1:
            raise ValueError('dsolve() and classify_sysode() work with functions of one variable only, not %s' % func)
    matching_hints['order'] = order

    def linearity_check(eqs, j, func, is_linear_):
        if False:
            for i in range(10):
                print('nop')
        for k in range(order[func] + 1):
            func_coef[j, func, k] = collect(eqs.expand(), [diff(func, t, k)]).coeff(diff(func, t, k))
            if is_linear_ == True:
                if func_coef[j, func, k] == 0:
                    if k == 0:
                        coef = eqs.as_independent(func, as_Add=True)[1]
                        for xr in range(1, ode_order(eqs, func) + 1):
                            coef -= eqs.as_independent(diff(func, t, xr), as_Add=True)[1]
                        if coef != 0:
                            is_linear_ = False
                    elif eqs.as_independent(diff(func, t, k), as_Add=True)[1]:
                        is_linear_ = False
                else:
                    for func_ in funcs:
                        if isinstance(func_, list):
                            for elem_func_ in func_:
                                dep = func_coef[j, func, k].as_independent(elem_func_, as_Add=True)[1]
                                if dep != 0:
                                    is_linear_ = False
                        else:
                            dep = func_coef[j, func, k].as_independent(func_, as_Add=True)[1]
                            if dep != 0:
                                is_linear_ = False
        return is_linear_
    func_coef = {}
    is_linear = True
    for (j, eqs) in enumerate(eq):
        for func in funcs:
            if isinstance(func, list):
                for func_elem in func:
                    is_linear = linearity_check(eqs, j, func_elem, is_linear)
            else:
                is_linear = linearity_check(eqs, j, func, is_linear)
    matching_hints['func_coeff'] = func_coef
    matching_hints['is_linear'] = is_linear
    if len(set(order.values())) == 1:
        order_eq = list(matching_hints['order'].values())[0]
        if matching_hints['is_linear'] == True:
            if matching_hints['no_of_equation'] == 2:
                if order_eq == 1:
                    type_of_equation = check_linear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            else:
                type_of_equation = None
        elif matching_hints['no_of_equation'] == 2:
            if order_eq == 1:
                type_of_equation = check_nonlinear_2eq_order1(eq, funcs, func_coef)
            else:
                type_of_equation = None
        elif matching_hints['no_of_equation'] == 3:
            if order_eq == 1:
                type_of_equation = check_nonlinear_3eq_order1(eq, funcs, func_coef)
            else:
                type_of_equation = None
        else:
            type_of_equation = None
    else:
        type_of_equation = None
    matching_hints['type_of_equation'] = type_of_equation
    return matching_hints

def check_linear_2eq_order1(eq, func, func_coef):
    if False:
        return 10
    x = func[0].func
    y = func[1].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    r = {}
    r['a1'] = fc[0, x(t), 1]
    r['a2'] = fc[1, y(t), 1]
    r['b1'] = -fc[0, x(t), 0] / fc[0, x(t), 1]
    r['b2'] = -fc[1, x(t), 0] / fc[1, y(t), 1]
    r['c1'] = -fc[0, y(t), 0] / fc[0, x(t), 1]
    r['c2'] = -fc[1, y(t), 0] / fc[1, y(t), 1]
    forcing = [S.Zero, S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['d1'] = forcing[0]
        r['d2'] = forcing[1]
    else:
        return None
    p = 0
    q = 0
    p1 = cancel(r['b2'] / cancel(r['b2'] / r['c2']).as_numer_denom()[0])
    p2 = cancel(r['b1'] / cancel(r['b1'] / r['c1']).as_numer_denom()[0])
    for (n, i) in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q and n == 0:
                if (r['b2'] / j - r['b1']) / (r['c1'] - r['c2'] / j) == j:
                    p = 1
            elif q and n == 1:
                if (r['b1'] / j - r['b2']) / (r['c2'] - r['c1'] / j) == j:
                    p = 2
    if r['d1'] != 0 or r['d2'] != 0:
        return None
    elif not any((r[k].has(t) for k in 'a1 a2 b1 b2 c1 c2'.split())):
        return None
    else:
        r['b1'] = r['b1'] / r['a1']
        r['b2'] = r['b2'] / r['a2']
        r['c1'] = r['c1'] / r['a1']
        r['c2'] = r['c2'] / r['a2']
        if p:
            return 'type6'
        else:
            return 'type7'

def check_nonlinear_2eq_order1(eq, func, func_coef):
    if False:
        for i in range(10):
            print('nop')
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    f = Wild('f')
    g = Wild('g')
    (u, v) = symbols('u, v', cls=Dummy)

    def check_type(x, y):
        if False:
            i = 10
            return i + 15
        r1 = eq[0].match(t * diff(x(t), t) - x(t) + f)
        r2 = eq[1].match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t), t) - x(t) / t + f / t)
            r2 = eq[1].match(diff(y(t), t) - y(t) / t + g / t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t * diff(x(t), t) - x(t) + f)
            r2 = (-eq[1]).match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t), t) - x(t) / t + f / t)
            r2 = (-eq[1]).match(diff(y(t), t) - y(t) / t + g / t)
        if r1 and r2 and (not (r1[f].subs(diff(x(t), t), u).subs(diff(y(t), t), v).has(t) or r2[g].subs(diff(x(t), t), u).subs(diff(y(t), t), v).has(t))):
            return 'type5'
        else:
            return None
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            eq_type = check_type(x, y)
            if not eq_type:
                eq_type = check_type(y, x)
            return eq_type
    x = func[0].func
    y = func[1].func
    fc = func_coef
    n = Wild('n', exclude=[x(t), y(t)])
    f1 = Wild('f1', exclude=[v, t])
    f2 = Wild('f2', exclude=[v, t])
    g1 = Wild('g1', exclude=[u, t])
    g2 = Wild('g2', exclude=[u, t])
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms / fc[i, func[i], 1]
        eq[i] = eqs
    r = eq[0].match(diff(x(t), t) - x(t) ** n * f)
    if r:
        g = (diff(y(t), t) - eq[1]) / r[f]
    if r and (not (g.has(x(t)) or g.subs(y(t), v).has(t) or r[f].subs(x(t), u).subs(y(t), v).has(t))):
        return 'type1'
    r = eq[0].match(diff(x(t), t) - exp(n * x(t)) * f)
    if r:
        g = (diff(y(t), t) - eq[1]) / r[f]
    if r and (not (g.has(x(t)) or g.subs(y(t), v).has(t) or r[f].subs(x(t), u).subs(y(t), v).has(t))):
        return 'type2'
    g = Wild('g')
    r1 = eq[0].match(diff(x(t), t) - f)
    r2 = eq[1].match(diff(y(t), t) - g)
    if r1 and r2 and (not (r1[f].subs(x(t), u).subs(y(t), v).has(t) or r2[g].subs(x(t), u).subs(y(t), v).has(t))):
        return 'type3'
    r1 = eq[0].match(diff(x(t), t) - f)
    r2 = eq[1].match(diff(y(t), t) - g)
    (num, den) = (r1[f].subs(x(t), u).subs(y(t), v) / r2[g].subs(x(t), u).subs(y(t), v)).as_numer_denom()
    R1 = num.match(f1 * g1)
    R2 = den.match(f2 * g2)
    if R1 and R2:
        return 'type4'
    return None

def check_nonlinear_2eq_order2(eq, func, func_coef):
    if False:
        while True:
            i = 10
    return None

def check_nonlinear_3eq_order1(eq, func, func_coef):
    if False:
        print('Hello World!')
    x = func[0].func
    y = func[1].func
    z = func[2].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    (u, v, w) = symbols('u, v, w', cls=Dummy)
    a = Wild('a', exclude=[x(t), y(t), z(t), t])
    b = Wild('b', exclude=[x(t), y(t), z(t), t])
    c = Wild('c', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    F1 = Wild('F1')
    F2 = Wild('F2')
    F3 = Wild('F3')
    for i in range(3):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms / fc[i, func[i], 1]
        eq[i] = eqs
    r1 = eq[0].match(diff(x(t), t) - a * y(t) * z(t))
    r2 = eq[1].match(diff(y(t), t) - b * z(t) * x(t))
    r3 = eq[2].match(diff(z(t), t) - c * x(t) * y(t))
    if r1 and r2 and r3:
        (num1, den1) = r1[a].as_numer_denom()
        (num2, den2) = r2[b].as_numer_denom()
        (num3, den3) = r3[c].as_numer_denom()
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type1'
    r = eq[0].match(diff(x(t), t) - y(t) * z(t) * f)
    if r:
        r1 = collect_const(r[f]).match(a * f)
        r2 = ((diff(y(t), t) - eq[1]) / r1[f]).match(b * z(t) * x(t))
        r3 = ((diff(z(t), t) - eq[2]) / r1[f]).match(c * x(t) * y(t))
    if r1 and r2 and r3:
        (num1, den1) = r1[a].as_numer_denom()
        (num2, den2) = r2[b].as_numer_denom()
        (num3, den3) = r3[c].as_numer_denom()
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type2'
    r = eq[0].match(diff(x(t), t) - (F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                (r1[F2], r1[F3]) = (r1[F3], r1[F2])
                (r1[c], r1[b]) = (-r1[b], -r1[c])
            r2 = eq[1].match(diff(y(t), t) - a * r1[F3] + r1[c] * F1)
        if r2:
            r3 = eq[2] == diff(z(t), t) - r1[b] * r2[F1] + r2[a] * r1[F2]
        if r1 and r2 and r3:
            return 'type3'
    r = eq[0].match(diff(x(t), t) - z(t) * F2 + y(t) * F3)
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                (r1[F2], r1[F3]) = (r1[F3], r1[F2])
                (r1[c], r1[b]) = (-r1[b], -r1[c])
            r2 = (diff(y(t), t) - eq[1]).match(a * x(t) * r1[F3] - r1[c] * z(t) * F1)
        if r2:
            r3 = diff(z(t), t) - eq[2] == r1[b] * y(t) * r2[F1] - r2[a] * x(t) * r1[F2]
        if r1 and r2 and r3:
            return 'type4'
    r = (diff(x(t), t) - eq[0]).match(x(t) * (F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                (r1[F2], r1[F3]) = (r1[F3], r1[F2])
                (r1[c], r1[b]) = (-r1[b], -r1[c])
            r2 = (diff(y(t), t) - eq[1]).match(y(t) * (a * r1[F3] - r1[c] * F1))
        if r2:
            r3 = diff(z(t), t) - eq[2] == z(t) * (r1[b] * r2[F1] - r2[a] * r1[F2])
        if r1 and r2 and r3:
            return 'type5'
    return None

def check_nonlinear_3eq_order2(eq, func, func_coef):
    if False:
        i = 10
        return i + 15
    return None

@vectorize(0)
def odesimp(ode, eq, func, hint):
    if False:
        i = 10
        return i + 15
    "\n    Simplifies solutions of ODEs, including trying to solve for ``func`` and\n    running :py:meth:`~sympy.solvers.ode.constantsimp`.\n\n    It may use knowledge of the type of solution that the hint returns to\n    apply additional simplifications.\n\n    It also attempts to integrate any :py:class:`~sympy.integrals.integrals.Integral`\\s\n    in the expression, if the hint is not an ``_Integral`` hint.\n\n    This function should have no effect on expressions returned by\n    :py:meth:`~sympy.solvers.ode.dsolve`, as\n    :py:meth:`~sympy.solvers.ode.dsolve` already calls\n    :py:meth:`~sympy.solvers.ode.ode.odesimp`, but the individual hint functions\n    do not call :py:meth:`~sympy.solvers.ode.ode.odesimp` (because the\n    :py:meth:`~sympy.solvers.ode.dsolve` wrapper does).  Therefore, this\n    function is designed for mainly internal use.\n\n    Examples\n    ========\n\n    >>> from sympy import sin, symbols, dsolve, pprint, Function\n    >>> from sympy.solvers.ode.ode import odesimp\n    >>> x, u2, C1= symbols('x,u2,C1')\n    >>> f = Function('f')\n\n    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),\n    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',\n    ... simplify=False)\n    >>> pprint(eq, wrap_line=False)\n                            x\n                           ----\n                           f(x)\n                             /\n                            |\n                            |   /        1   \\\n                            |  -|u1 + -------|\n                            |   |        /1 \\|\n                            |   |     sin|--||\n                            |   \\        \\u1//\n    log(f(x)) = log(C1) +   |  ---------------- d(u1)\n                            |          2\n                            |        u1\n                            |\n                           /\n\n    >>> pprint(odesimp(eq, f(x), 1, {C1},\n    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'\n    ... )) #doctest: +SKIP\n        x\n    --------- = C1\n       /f(x)\\\n    tan|----|\n       \\2*x /\n\n    "
    x = func.args[0]
    f = func.func
    C1 = get_numbered_constants(eq, num=1)
    constants = eq.free_symbols - ode.free_symbols
    eq = _handle_Integral(eq, func, hint)
    if hint.startswith('nth_linear_euler_eq_nonhomogeneous'):
        eq = simplify(eq)
    if not isinstance(eq, Equality):
        raise TypeError('eq should be an instance of Equality')
    eq = constantsimp(eq, constants)
    if eq.rhs == func and (not eq.lhs.has(func)):
        eq = [Eq(eq.rhs, eq.lhs)]
    if eq.lhs == func and (not eq.rhs.has(func)):
        eq = [eq]
    else:
        try:
            floats = any((i.is_Float for i in eq.atoms(Number)))
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            eq = [eq]
        else:

            def _expand(expr):
                if False:
                    return 10
                (numer, denom) = expr.as_numer_denom()
                if denom.is_Add:
                    return expr
                else:
                    return powsimp(expr.expand(), combine='exp', deep=True)
            eq = [Eq(f(x), _expand(t)) for t in eqsol]
        if hint.startswith('1st_homogeneous_coeff'):
            for (j, eqi) in enumerate(eq):
                newi = logcombine(eqi, force=True)
                if isinstance(newi.lhs, log) and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0] / C1, C1)
                eq[j] = newi
    for (i, eqi) in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        eq[i] = constant_renumber(eq[i], ode.free_symbols)
    if len(eq) == 1:
        eq = eq[0]
    return eq

def ode_sol_simplicity(sol, func, trysolving=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns an extended integer representing how simple a solution to an ODE\n    is.\n\n    The following things are considered, in order from most simple to least:\n\n    - ``sol`` is solved for ``func``.\n    - ``sol`` is not solved for ``func``, but can be if passed to solve (e.g.,\n      a solution returned by ``dsolve(ode, func, simplify=False``).\n    - If ``sol`` is not solved for ``func``, then base the result on the\n      length of ``sol``, as computed by ``len(str(sol))``.\n    - If ``sol`` has any unevaluated :py:class:`~sympy.integrals.integrals.Integral`\\s,\n      this will automatically be considered less simple than any of the above.\n\n    This function returns an integer such that if solution A is simpler than\n    solution B by above metric, then ``ode_sol_simplicity(sola, func) <\n    ode_sol_simplicity(solb, func)``.\n\n    Currently, the following are the numbers returned, but if the heuristic is\n    ever improved, this may change.  Only the ordering is guaranteed.\n\n    +----------------------------------------------+-------------------+\n    | Simplicity                                   | Return            |\n    +==============================================+===================+\n    | ``sol`` solved for ``func``                  | ``-2``            |\n    +----------------------------------------------+-------------------+\n    | ``sol`` not solved for ``func`` but can be   | ``-1``            |\n    +----------------------------------------------+-------------------+\n    | ``sol`` is not solved nor solvable for       | ``len(str(sol))`` |\n    | ``func``                                     |                   |\n    +----------------------------------------------+-------------------+\n    | ``sol`` contains an                          | ``oo``            |\n    | :obj:`~sympy.integrals.integrals.Integral`   |                   |\n    +----------------------------------------------+-------------------+\n\n    ``oo`` here means the SymPy infinity, which should compare greater than\n    any integer.\n\n    If you already know :py:meth:`~sympy.solvers.solvers.solve` cannot solve\n    ``sol``, you can use ``trysolving=False`` to skip that step, which is the\n    only potentially slow step.  For example,\n    :py:meth:`~sympy.solvers.ode.dsolve` with the ``simplify=False`` flag\n    should do this.\n\n    If ``sol`` is a list of solutions, if the worst solution in the list\n    returns ``oo`` it returns that, otherwise it returns ``len(str(sol))``,\n    that is, the length of the string representation of the whole list.\n\n    Examples\n    ========\n\n    This function is designed to be passed to ``min`` as the key argument,\n    such as ``min(listofsolutions, key=lambda i: ode_sol_simplicity(i,\n    f(x)))``.\n\n    >>> from sympy import symbols, Function, Eq, tan, Integral\n    >>> from sympy.solvers.ode.ode import ode_sol_simplicity\n    >>> x, C1, C2 = symbols('x, C1, C2')\n    >>> f = Function('f')\n\n    >>> ode_sol_simplicity(Eq(f(x), C1*x**2), f(x))\n    -2\n    >>> ode_sol_simplicity(Eq(x**2 + f(x), C1), f(x))\n    -1\n    >>> ode_sol_simplicity(Eq(f(x), C1*Integral(2*x, x)), f(x))\n    oo\n    >>> eq1 = Eq(f(x)/tan(f(x)/(2*x)), C1)\n    >>> eq2 = Eq(f(x)/tan(f(x)/(2*x) + f(x)), C2)\n    >>> [ode_sol_simplicity(eq, f(x)) for eq in [eq1, eq2]]\n    [28, 35]\n    >>> min([eq1, eq2], key=lambda i: ode_sol_simplicity(i, f(x)))\n    Eq(f(x)/tan(f(x)/(2*x)), C1)\n\n    "
    if iterable(sol):
        for i in sol:
            if ode_sol_simplicity(i, func, trysolving=trysolving) == oo:
                return oo
        return len(str(sol))
    if sol.has(Integral):
        return oo
    if sol.lhs == func and (not sol.rhs.has(func)) or (sol.rhs == func and (not sol.lhs.has(func))):
        return -2
    if trysolving:
        try:
            sols = solve(sol, func)
            if not sols:
                raise NotImplementedError
        except NotImplementedError:
            pass
        else:
            return -1
    return len(str(sol))

def _extract_funcs(eqs):
    if False:
        i = 10
        return i + 15
    funcs = []
    for eq in eqs:
        derivs = [node for node in preorder_traversal(eq) if isinstance(node, Derivative)]
        func = []
        for d in derivs:
            func += list(d.atoms(AppliedUndef))
        for func_ in func:
            funcs.append(func_)
    funcs = list(uniq(funcs))
    return funcs

def _get_constant_subexpressions(expr, Cs):
    if False:
        i = 10
        return i + 15
    Cs = set(Cs)
    Ces = []

    def _recursive_walk(expr):
        if False:
            return 10
        expr_syms = expr.free_symbols
        if expr_syms and expr_syms.issubset(Cs):
            Ces.append(expr)
        else:
            if expr.func == exp:
                expr = expr.expand(mul=True)
            if expr.func in (Add, Mul):
                d = sift(expr.args, lambda i: i.free_symbols.issubset(Cs))
                if len(d[True]) > 1:
                    x = expr.func(*d[True])
                    if not x.is_number:
                        Ces.append(x)
            elif isinstance(expr, Integral):
                if expr.free_symbols.issubset(Cs) and all((len(x) == 3 for x in expr.limits)):
                    Ces.append(expr)
            for i in expr.args:
                _recursive_walk(i)
        return
    _recursive_walk(expr)
    return Ces

def __remove_linear_redundancies(expr, Cs):
    if False:
        for i in range(10):
            print('nop')
    cnts = {i: expr.count(i) for i in Cs}
    Cs = [i for i in Cs if cnts[i] > 0]

    def _linear(expr):
        if False:
            print('Hello World!')
        if isinstance(expr, Add):
            xs = [i for i in Cs if expr.count(i) == cnts[i] and 0 == expr.diff(i, 2)]
            d = {}
            for x in xs:
                y = expr.diff(x)
                if y not in d:
                    d[y] = []
                d[y].append(x)
            for y in d:
                if len(d[y]) > 1:
                    d[y].sort(key=str)
                    for x in d[y][1:]:
                        expr = expr.subs(x, 0)
        return expr

    def _recursive_walk(expr):
        if False:
            for i in range(10):
                print('nop')
        if len(expr.args) != 0:
            expr = expr.func(*[_recursive_walk(i) for i in expr.args])
        expr = _linear(expr)
        return expr
    if isinstance(expr, Equality):
        (lhs, rhs) = [_recursive_walk(i) for i in expr.args]
        f = lambda i: isinstance(i, Number) or i in Cs
        if isinstance(lhs, Symbol) and lhs in Cs:
            (rhs, lhs) = (lhs, rhs)
        if lhs.func in (Add, Symbol) and rhs.func in (Add, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            drhs = sift([rhs] if isinstance(rhs, AtomicExpr) else rhs.args, f)
            for i in [True, False]:
                for hs in [dlhs, drhs]:
                    if i not in hs:
                        hs[i] = [0]
            lhs = Add(*dlhs[False]) - Add(*drhs[False])
            rhs = Add(*drhs[True]) - Add(*dlhs[True])
        elif lhs.func in (Mul, Symbol) and rhs.func in (Mul, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            if True in dlhs:
                if False not in dlhs:
                    dlhs[False] = [1]
                lhs = Mul(*dlhs[False])
                rhs = rhs / Mul(*dlhs[True])
        return Eq(lhs, rhs)
    else:
        return _recursive_walk(expr)

@vectorize(0)
def constantsimp(expr, constants):
    if False:
        while True:
            i = 10
    '\n    Simplifies an expression with arbitrary constants in it.\n\n    This function is written specifically to work with\n    :py:meth:`~sympy.solvers.ode.dsolve`, and is not intended for general use.\n\n    Simplification is done by "absorbing" the arbitrary constants into other\n    arbitrary constants, numbers, and symbols that they are not independent\n    of.\n\n    The symbols must all have the same name with numbers after it, for\n    example, ``C1``, ``C2``, ``C3``.  The ``symbolname`` here would be\n    \'``C``\', the ``startnumber`` would be 1, and the ``endnumber`` would be 3.\n    If the arbitrary constants are independent of the variable ``x``, then the\n    independent symbol would be ``x``.  There is no need to specify the\n    dependent function, such as ``f(x)``, because it already has the\n    independent symbol, ``x``, in it.\n\n    Because terms are "absorbed" into arbitrary constants and because\n    constants are renumbered after simplifying, the arbitrary constants in\n    expr are not necessarily equal to the ones of the same name in the\n    returned result.\n\n    If two or more arbitrary constants are added, multiplied, or raised to the\n    power of each other, they are first absorbed together into a single\n    arbitrary constant.  Then the new constant is combined into other terms if\n    necessary.\n\n    Absorption of constants is done with limited assistance:\n\n    1. terms of :py:class:`~sympy.core.add.Add`\\s are collected to try join\n       constants so `e^x (C_1 \\cos(x) + C_2 \\cos(x))` will simplify to `e^x\n       C_1 \\cos(x)`;\n\n    2. powers with exponents that are :py:class:`~sympy.core.add.Add`\\s are\n       expanded so `e^{C_1 + x}` will be simplified to `C_1 e^x`.\n\n    Use :py:meth:`~sympy.solvers.ode.ode.constant_renumber` to renumber constants\n    after simplification or else arbitrary numbers on constants may appear,\n    e.g. `C_1 + C_3 x`.\n\n    In rare cases, a single constant can be "simplified" into two constants.\n    Every differential equation solution should have as many arbitrary\n    constants as the order of the differential equation.  The result here will\n    be technically correct, but it may, for example, have `C_1` and `C_2` in\n    an expression, when `C_1` is actually equal to `C_2`.  Use your discretion\n    in such situations, and also take advantage of the ability to use hints in\n    :py:meth:`~sympy.solvers.ode.dsolve`.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.solvers.ode.ode import constantsimp\n    >>> C1, C2, C3, x, y = symbols(\'C1, C2, C3, x, y\')\n    >>> constantsimp(2*C1*x, {C1, C2, C3})\n    C1*x\n    >>> constantsimp(C1 + 2 + x, {C1, C2, C3})\n    C1 + x\n    >>> constantsimp(C1*C2 + 2 + C2 + C3*x, {C1, C2, C3})\n    C1 + C3*x\n\n    '
    Cs = constants
    orig_expr = expr
    constant_subexprs = _get_constant_subexpressions(expr, Cs)
    for xe in constant_subexprs:
        xes = list(xe.free_symbols)
        if not xes:
            continue
        if all((expr.count(c) == xe.count(c) for c in xes)):
            xes.sort(key=str)
            expr = expr.subs(xe, xes[0])
    try:
        (commons, rexpr) = cse(expr)
        commons.reverse()
        rexpr = rexpr[0]
        for s in commons:
            cs = list(s[1].atoms(Symbol))
            if len(cs) == 1 and cs[0] in Cs and (cs[0] not in rexpr.atoms(Symbol)) and (not any((cs[0] in ex for ex in commons if ex != s))):
                rexpr = rexpr.subs(s[0], cs[0])
            else:
                rexpr = rexpr.subs(*s)
        expr = rexpr
    except IndexError:
        pass
    expr = __remove_linear_redundancies(expr, Cs)

    def _conditional_term_factoring(expr):
        if False:
            while True:
                i = 10
        new_expr = terms_gcd(expr, clear=False, deep=True, expand=False)
        if new_expr.is_Mul:
            infac = False
            asfac = False
            for m in new_expr.args:
                if isinstance(m, exp):
                    asfac = True
                elif m.is_Add:
                    infac = any((isinstance(fi, exp) for t in m.args for fi in Mul.make_args(t)))
                if asfac and infac:
                    new_expr = expr
                    break
        return new_expr
    expr = _conditional_term_factoring(expr)
    if orig_expr != expr:
        return constantsimp(expr, Cs)
    return expr

def constant_renumber(expr, variables=None, newconstants=None):
    if False:
        i = 10
        return i + 15
    "\n    Renumber arbitrary constants in ``expr`` to use the symbol names as given\n    in ``newconstants``. In the process, this reorders expression terms in a\n    standard way.\n\n    If ``newconstants`` is not provided then the new constant names will be\n    ``C1``, ``C2`` etc. Otherwise ``newconstants`` should be an iterable\n    giving the new symbols to use for the constants in order.\n\n    The ``variables`` argument is a list of non-constant symbols. All other\n    free symbols found in ``expr`` are assumed to be constants and will be\n    renumbered. If ``variables`` is not given then any numbered symbol\n    beginning with ``C`` (e.g. ``C1``) is assumed to be a constant.\n\n    Symbols are renumbered based on ``.sort_key()``, so they should be\n    numbered roughly in the order that they appear in the final, printed\n    expression.  Note that this ordering is based in part on hashes, so it can\n    produce different results on different machines.\n\n    The structure of this function is very similar to that of\n    :py:meth:`~sympy.solvers.ode.constantsimp`.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.solvers.ode.ode import constant_renumber\n    >>> x, C1, C2, C3 = symbols('x,C1:4')\n    >>> expr = C3 + C2*x + C1*x**2\n    >>> expr\n    C1*x**2  + C2*x + C3\n    >>> constant_renumber(expr)\n    C1 + C2*x + C3*x**2\n\n    The ``variables`` argument specifies which are constants so that the\n    other symbols will not be renumbered:\n\n    >>> constant_renumber(expr, [C1, x])\n    C1*x**2  + C2 + C3*x\n\n    The ``newconstants`` argument is used to specify what symbols to use when\n    replacing the constants:\n\n    >>> constant_renumber(expr, [x], newconstants=symbols('E1:4'))\n    E1 + E2*x + E3*x**2\n\n    "
    if isinstance(expr, (set, list, tuple)):
        return type(expr)(constant_renumber(Tuple(*expr), variables=variables, newconstants=newconstants))
    if variables is not None:
        variables = set(variables)
        free_symbols = expr.free_symbols
        constantsymbols = list(free_symbols - variables)
    else:
        variables = set()
        isconstant = lambda s: s.startswith('C') and s[1:].isdigit()
        constantsymbols = [sym for sym in expr.free_symbols if isconstant(sym.name)]
    if newconstants is None:
        iter_constants = numbered_symbols(start=1, prefix='C', exclude=variables)
    else:
        iter_constants = (sym for sym in newconstants if sym not in variables)
    constants_found = []
    C_1 = [(ci, S.One) for ci in constantsymbols]
    sort_key = lambda arg: default_sort_key(arg.subs(C_1))

    def _constant_renumber(expr):
        if False:
            print('Hello World!')
        '\n        We need to have an internal recursive function\n        '
        if isinstance(expr, Tuple):
            renumbered = [_constant_renumber(e) for e in expr]
            return Tuple(*renumbered)
        if isinstance(expr, Equality):
            return Eq(_constant_renumber(expr.lhs), _constant_renumber(expr.rhs))
        if type(expr) not in (Mul, Add, Pow) and (not expr.is_Function) and (not expr.has(*constantsymbols)):
            return expr
        elif expr.is_Piecewise:
            return expr
        elif expr in constantsymbols:
            if expr not in constants_found:
                constants_found.append(expr)
            return expr
        elif expr.is_Function or expr.is_Pow:
            return expr.func(*[_constant_renumber(x) for x in expr.args])
        else:
            sortedargs = list(expr.args)
            sortedargs.sort(key=sort_key)
            return expr.func(*[_constant_renumber(x) for x in sortedargs])
    expr = _constant_renumber(expr)
    constants_found = [c for c in constants_found if c not in variables]
    subs_dict = dict(zip(constants_found, iter_constants))
    expr = expr.subs(subs_dict, simultaneous=True)
    return expr

def _handle_Integral(expr, func, hint):
    if False:
        i = 10
        return i + 15
    '\n    Converts a solution with Integrals in it into an actual solution.\n\n    For most hints, this simply runs ``expr.doit()``.\n\n    '
    if hint == 'nth_linear_constant_coeff_homogeneous':
        sol = expr
    elif not hint.endswith('_Integral'):
        sol = expr.doit()
    else:
        sol = expr
    return sol

def homogeneous_order(eq, *symbols):
    if False:
        return 10
    "\n    Returns the order `n` if `g` is homogeneous and ``None`` if it is not\n    homogeneous.\n\n    Determines if a function is homogeneous and if so of what order.  A\n    function `f(x, y, \\cdots)` is homogeneous of order `n` if `f(t x, t y,\n    \\cdots) = t^n f(x, y, \\cdots)`.\n\n    If the function is of two variables, `F(x, y)`, then `f` being homogeneous\n    of any order is equivalent to being able to rewrite `F(x, y)` as `G(x/y)`\n    or `H(y/x)`.  This fact is used to solve 1st order ordinary differential\n    equations whose coefficients are homogeneous of the same order (see the\n    docstrings of\n    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep` and\n    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`).\n\n    Symbols can be functions, but every argument of the function must be a\n    symbol, and the arguments of the function that appear in the expression\n    must match those given in the list of symbols.  If a declared function\n    appears with different arguments than given in the list of symbols,\n    ``None`` is returned.\n\n    Examples\n    ========\n\n    >>> from sympy import Function, homogeneous_order, sqrt\n    >>> from sympy.abc import x, y\n    >>> f = Function('f')\n    >>> homogeneous_order(f(x), f(x)) is None\n    True\n    >>> homogeneous_order(f(x,y), f(y, x), x, y) is None\n    True\n    >>> homogeneous_order(f(x), f(x), x)\n    1\n    >>> homogeneous_order(x**2*f(x)/sqrt(x**2+f(x)**2), x, f(x))\n    2\n    >>> homogeneous_order(x**2+f(x), x, f(x)) is None\n    True\n\n    "
    if not symbols:
        raise ValueError('homogeneous_order: no symbols were given.')
    symset = set(symbols)
    eq = sympify(eq)
    if eq.has(Order, Derivative):
        return None
    if eq.is_Number or eq.is_NumberSymbol or eq.is_number:
        return S.Zero
    dum = numbered_symbols(prefix='d', cls=Dummy)
    newsyms = set()
    for i in [j for j in symset if getattr(j, 'is_Function')]:
        iargs = set(i.args)
        if iargs.difference(symset):
            return None
        else:
            dummyvar = next(dum)
            eq = eq.subs(i, dummyvar)
            symset.remove(i)
            newsyms.add(dummyvar)
    symset.update(newsyms)
    if not eq.free_symbols & symset:
        return None
    if isinstance(eq, Function):
        return None if homogeneous_order(eq.args[0], *tuple(symset)) != 0 else S.Zero
    t = Dummy('t', positive=True)
    eqs = separatevars(eq.subs([(i, t * i) for i in symset]), [t], dict=True)[t]
    if eqs is S.One:
        return S.Zero
    (i, d) = eqs.as_independent(t, as_Add=False)
    (b, e) = d.as_base_exp()
    if b == t:
        return e

def ode_2nd_power_series_ordinary(eq, func, order, match):
    if False:
        i = 10
        return i + 15
    '\n    Gives a power series solution to a second order homogeneous differential\n    equation with polynomial coefficients at an ordinary point. A homogeneous\n    differential equation is of the form\n\n    .. math :: P(x)\\frac{d^2y}{dx^2} + Q(x)\\frac{dy}{dx} + R(x) y(x) = 0\n\n    For simplicity it is assumed that `P(x)`, `Q(x)` and `R(x)` are polynomials,\n    it is sufficient that `\\frac{Q(x)}{P(x)}` and `\\frac{R(x)}{P(x)}` exists at\n    `x_{0}`. A recurrence relation is obtained by substituting `y` as `\\sum_{n=0}^\\infty a_{n}x^{n}`,\n    in the differential equation, and equating the nth term. Using this relation\n    various terms can be generated.\n\n\n    Examples\n    ========\n\n    >>> from sympy import dsolve, Function, pprint\n    >>> from sympy.abc import x\n    >>> f = Function("f")\n    >>> eq = f(x).diff(x, 2) + f(x)\n    >>> pprint(dsolve(eq, hint=\'2nd_power_series_ordinary\'))\n              / 4    2    \\        /     2\\\n              |x    x     |        |    x |    / 6\\\n    f(x) = C2*|-- - -- + 1| + C1*x*|1 - --| + O\\x /\n              \\24   2     /        \\    6 /\n\n\n    References\n    ==========\n    - https://tutorial.math.lamar.edu/Classes/DE/SeriesSolutions.aspx\n    - George E. Simmons, "Differential Equations with Applications and\n      Historical Notes", p.p 176 - 184\n\n    '
    x = func.args[0]
    f = func.func
    (C0, C1) = get_numbered_constants(eq, num=2)
    n = Dummy('n', integer=True)
    s = Wild('s')
    k = Wild('k', exclude=[x])
    x0 = match['x0']
    terms = match['terms']
    p = match[match['a3']]
    q = match[match['b3']]
    r = match[match['c3']]
    seriesdict = {}
    recurr = Function('r')
    coefflist = [(recurr(n), r), (n * recurr(n), q), (n * (n - 1) * recurr(n), p)]
    for (index, coeff) in enumerate(coefflist):
        if coeff[1]:
            f2 = powsimp(expand((coeff[1] * (x - x0) ** (n - index)).subs(x, x + x0)))
            if f2.is_Add:
                addargs = f2.args
            else:
                addargs = [f2]
            for arg in addargs:
                powm = arg.match(s * x ** k)
                term = coeff[0] * powm[s]
                if not powm[k].is_Symbol:
                    term = term.subs(n, n - powm[k].as_independent(n)[0])
                startind = powm[k].subs(n, index)
                if startind:
                    for i in reversed(range(startind)):
                        if not term.subs(n, i):
                            seriesdict[term] = i
                        else:
                            seriesdict[term] = i + 1
                            break
                else:
                    seriesdict[term] = S.Zero
    teq = S.Zero
    suminit = seriesdict.values()
    rkeys = seriesdict.keys()
    req = Add(*rkeys)
    if any(suminit):
        maxval = max(suminit)
        for term in seriesdict:
            val = seriesdict[term]
            if val != maxval:
                for i in range(val, maxval):
                    teq += term.subs(n, val)
    finaldict = {}
    if teq:
        fargs = teq.atoms(AppliedUndef)
        if len(fargs) == 1:
            finaldict[fargs.pop()] = 0
        else:
            maxf = max(fargs, key=lambda x: x.args[0])
            sol = solve(teq, maxf)
            if isinstance(sol, list):
                sol = sol[0]
            finaldict[maxf] = sol
    fargs = req.atoms(AppliedUndef)
    maxf = max(fargs, key=lambda x: x.args[0])
    minf = min(fargs, key=lambda x: x.args[0])
    if minf.args[0].is_Symbol:
        startiter = 0
    else:
        startiter = -minf.args[0].as_independent(n)[0]
    lhs = maxf
    rhs = solve(req, maxf)
    if isinstance(rhs, list):
        rhs = rhs[0]
    tcounter = len([t for t in finaldict.values() if t])
    for _ in range(tcounter, terms - 3):
        check = rhs.subs(n, startiter)
        nlhs = lhs.subs(n, startiter)
        nrhs = check.subs(finaldict)
        finaldict[nlhs] = nrhs
        startiter += 1
    series = C0 + C1 * (x - x0)
    for term in finaldict:
        if finaldict[term]:
            fact = term.args[0]
            series += finaldict[term].subs([(recurr(0), C0), (recurr(1), C1)]) * (x - x0) ** fact
    series = collect(expand_mul(series), [C0, C1]) + Order(x ** terms)
    return Eq(f(x), series)

def ode_2nd_power_series_regular(eq, func, order, match):
    if False:
        while True:
            i = 10
    '\n    Gives a power series solution to a second order homogeneous differential\n    equation with polynomial coefficients at a regular point. A second order\n    homogeneous differential equation is of the form\n\n    .. math :: P(x)\\frac{d^2y}{dx^2} + Q(x)\\frac{dy}{dx} + R(x) y(x) = 0\n\n    A point is said to regular singular at `x0` if `x - x0\\frac{Q(x)}{P(x)}`\n    and `(x - x0)^{2}\\frac{R(x)}{P(x)}` are analytic at `x0`. For simplicity\n    `P(x)`, `Q(x)` and `R(x)` are assumed to be polynomials. The algorithm for\n    finding the power series solutions is:\n\n    1.  Try expressing `(x - x0)P(x)` and `((x - x0)^{2})Q(x)` as power series\n        solutions about x0. Find `p0` and `q0` which are the constants of the\n        power series expansions.\n    2.  Solve the indicial equation `f(m) = m(m - 1) + m*p0 + q0`, to obtain the\n        roots `m1` and `m2` of the indicial equation.\n    3.  If `m1 - m2` is a non integer there exists two series solutions. If\n        `m1 = m2`, there exists only one solution. If `m1 - m2` is an integer,\n        then the existence of one solution is confirmed. The other solution may\n        or may not exist.\n\n    The power series solution is of the form `x^{m}\\sum_{n=0}^\\infty a_{n}x^{n}`. The\n    coefficients are determined by the following recurrence relation.\n    `a_{n} = -\\frac{\\sum_{k=0}^{n-1} q_{n-k} + (m + k)p_{n-k}}{f(m + n)}`. For the case\n    in which `m1 - m2` is an integer, it can be seen from the recurrence relation\n    that for the lower root `m`, when `n` equals the difference of both the\n    roots, the denominator becomes zero. So if the numerator is not equal to zero,\n    a second series solution exists.\n\n\n    Examples\n    ========\n\n    >>> from sympy import dsolve, Function, pprint\n    >>> from sympy.abc import x\n    >>> f = Function("f")\n    >>> eq = x*(f(x).diff(x, 2)) + 2*(f(x).diff(x)) + x*f(x)\n    >>> pprint(dsolve(eq, hint=\'2nd_power_series_regular\'))\n                                  /   6     4    2    \\\n                                  |  x     x    x     |\n              / 4     2    \\   C1*|- --- + -- - -- + 1|\n              |x     x     |      \\  720   24   2     /    / 6\\\n    f(x) = C2*|--- - -- + 1| + ------------------------ + O\\x /\n              \\120   6     /              x\n\n\n    References\n    ==========\n    - George E. Simmons, "Differential Equations with Applications and\n      Historical Notes", p.p 176 - 184\n\n    '
    x = func.args[0]
    f = func.func
    (C0, C1) = get_numbered_constants(eq, num=2)
    m = Dummy('m')
    x0 = match['x0']
    terms = match['terms']
    p = match['p']
    q = match['q']
    indicial = []
    for term in [p, q]:
        if not term.has(x):
            indicial.append(term)
        else:
            term = series(term, x=x, n=1, x0=x0)
            if isinstance(term, Order):
                indicial.append(S.Zero)
            else:
                for arg in term.args:
                    if not arg.has(x):
                        indicial.append(arg)
                        break
    (p0, q0) = indicial
    sollist = solve(m * (m - 1) + m * p0 + q0, m)
    if sollist and isinstance(sollist, list) and all((sol.is_real for sol in sollist)):
        serdict1 = {}
        serdict2 = {}
        if len(sollist) == 1:
            m1 = m2 = sollist.pop()
            if terms - m1 - 1 <= 0:
                return Eq(f(x), Order(terms))
            serdict1 = _frobenius(terms - m1 - 1, m1, p0, q0, p, q, x0, x, C0)
        else:
            m1 = sollist[0]
            m2 = sollist[1]
            if m1 < m2:
                (m1, m2) = (m2, m1)
            serdict1 = _frobenius(terms - m1 - 1, m1, p0, q0, p, q, x0, x, C0)
            if not (m1 - m2).is_integer:
                serdict2 = _frobenius(terms - m2 - 1, m2, p0, q0, p, q, x0, x, C1)
            else:
                serdict2 = _frobenius(terms - m2 - 1, m2, p0, q0, p, q, x0, x, C1, check=m1)
        if serdict1:
            finalseries1 = C0
            for key in serdict1:
                power = int(key.name[1:])
                finalseries1 += serdict1[key] * (x - x0) ** power
            finalseries1 = (x - x0) ** m1 * finalseries1
            finalseries2 = S.Zero
            if serdict2:
                for key in serdict2:
                    power = int(key.name[1:])
                    finalseries2 += serdict2[key] * (x - x0) ** power
                finalseries2 += C1
                finalseries2 = (x - x0) ** m2 * finalseries2
            return Eq(f(x), collect(finalseries1 + finalseries2, [C0, C1]) + Order(x ** terms))

def _frobenius(n, m, p0, q0, p, q, x0, x, c, check=None):
    if False:
        while True:
            i = 10
    '\n    Returns a dict with keys as coefficients and values as their values in terms of C0\n    '
    n = int(n)
    m2 = check
    d = Dummy('d')
    numsyms = numbered_symbols('C', start=0)
    numsyms = [next(numsyms) for i in range(n + 1)]
    serlist = []
    for ser in [p, q]:
        if ser.is_polynomial(x) and Poly(ser, x).degree() <= n:
            if x0:
                ser = ser.subs(x, x + x0)
            dict_ = Poly(ser, x).as_dict()
        else:
            tseries = series(ser, x=x0, n=n + 1)
            dict_ = Poly(list(ordered(tseries.args))[:-1], x).as_dict()
        for i in range(n + 1):
            if (i,) not in dict_:
                dict_[i,] = S.Zero
        serlist.append(dict_)
    pseries = serlist[0]
    qseries = serlist[1]
    indicial = d * (d - 1) + d * p0 + q0
    frobdict = {}
    for i in range(1, n + 1):
        num = c * (m * pseries[i,] + qseries[i,])
        for j in range(1, i):
            sym = Symbol('C' + str(j))
            num += frobdict[sym] * ((m + j) * pseries[i - j,] + qseries[i - j,])
        if m2 is not None and i == m2 - m:
            if num:
                return False
            else:
                frobdict[numsyms[i]] = S.Zero
        else:
            frobdict[numsyms[i]] = -num / indicial.subs(d, m + i)
    return frobdict

def _remove_redundant_solutions(eq, solns, order, var):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove redundant solutions from the set of solutions.\n\n    This function is needed because otherwise dsolve can return\n    redundant solutions. As an example consider:\n\n        eq = Eq((f(x).diff(x, 2))*f(x).diff(x), 0)\n\n    There are two ways to find solutions to eq. The first is to solve f(x).diff(x, 2) = 0\n    leading to solution f(x)=C1 + C2*x. The second is to solve the equation f(x).diff(x) = 0\n    leading to the solution f(x) = C1. In this particular case we then see\n    that the second solution is a special case of the first and we do not\n    want to return it.\n\n    This does not always happen. If we have\n\n        eq = Eq((f(x)**2-4)*(f(x).diff(x)-4), 0)\n\n    then we get the algebraic solution f(x) = [-2, 2] and the integral solution\n    f(x) = x + C1 and in this case the two solutions are not equivalent wrt\n    initial conditions so both should be returned.\n    '

    def is_special_case_of(soln1, soln2):
        if False:
            print('Hello World!')
        return _is_special_case_of(soln1, soln2, eq, order, var)
    unique_solns = []
    for soln1 in solns:
        for soln2 in unique_solns[:]:
            if is_special_case_of(soln1, soln2):
                break
            elif is_special_case_of(soln2, soln1):
                unique_solns.remove(soln2)
        else:
            unique_solns.append(soln1)
    return unique_solns

def _is_special_case_of(soln1, soln2, eq, order, var):
    if False:
        while True:
            i = 10
    '\n    True if soln1 is found to be a special case of soln2 wrt some value of the\n    constants that appear in soln2. False otherwise.\n    '
    soln1 = soln1.rhs - soln1.lhs
    soln2 = soln2.rhs - soln2.lhs
    if soln1.has(Order) and soln2.has(Order):
        if soln1.getO() == soln2.getO():
            soln1 = soln1.removeO()
            soln2 = soln2.removeO()
        else:
            return False
    elif soln1.has(Order) or soln2.has(Order):
        return False
    constants1 = soln1.free_symbols.difference(eq.free_symbols)
    constants2 = soln2.free_symbols.difference(eq.free_symbols)
    constants1_new = get_numbered_constants(Tuple(soln1, soln2), len(constants1))
    if len(constants1) == 1:
        constants1_new = {constants1_new}
    for (c_old, c_new) in zip(constants1, constants1_new):
        soln1 = soln1.subs(c_old, c_new)
    lhs = soln1
    rhs = soln2
    eqns = [Eq(lhs, rhs)]
    for n in range(1, order):
        lhs = lhs.diff(var)
        rhs = rhs.diff(var)
        eq = Eq(lhs, rhs)
        eqns.append(eq)
    if any((isinstance(eq, BooleanFalse) for eq in eqns)):
        return False
    eqns = [eq for eq in eqns if not isinstance(eq, BooleanTrue)]
    try:
        constant_solns = solve(eqns, constants2)
    except NotImplementedError:
        return False
    if isinstance(constant_solns, dict):
        constant_solns = [constant_solns]
    for constant_soln in constant_solns:
        for eq in eqns:
            eq = eq.rhs - eq.lhs
            if checksol(eq, constant_soln) is not True:
                return False
    for constant_soln in constant_solns:
        if not any((c.has(var) for c in constant_soln.values())):
            return True
    return False

def ode_1st_power_series(eq, func, order, match):
    if False:
        i = 10
        return i + 15
    "\n    The power series solution is a method which gives the Taylor series expansion\n    to the solution of a differential equation.\n\n    For a first order differential equation `\\frac{dy}{dx} = h(x, y)`, a power\n    series solution exists at a point `x = x_{0}` if `h(x, y)` is analytic at `x_{0}`.\n    The solution is given by\n\n    .. math:: y(x) = y(x_{0}) + \\sum_{n = 1}^{\\infty} \\frac{F_{n}(x_{0},b)(x - x_{0})^n}{n!},\n\n    where `y(x_{0}) = b` is the value of y at the initial value of `x_{0}`.\n    To compute the values of the `F_{n}(x_{0},b)` the following algorithm is\n    followed, until the required number of terms are generated.\n\n    1. `F_1 = h(x_{0}, b)`\n    2. `F_{n+1} = \\frac{\\partial F_{n}}{\\partial x} + \\frac{\\partial F_{n}}{\\partial y}F_{1}`\n\n    Examples\n    ========\n\n    >>> from sympy import Function, pprint, exp, dsolve\n    >>> from sympy.abc import x\n    >>> f = Function('f')\n    >>> eq = exp(x)*(f(x).diff(x)) - f(x)\n    >>> pprint(dsolve(eq, hint='1st_power_series'))\n                           3       4       5\n                       C1*x    C1*x    C1*x     / 6\\\n    f(x) = C1 + C1*x - ----- + ----- + ----- + O\\x /\n                         6       24      60\n\n\n    References\n    ==========\n\n    - Travis W. Walker, Analytic power series technique for solving first-order\n      differential equations, p.p 17, 18\n\n    "
    x = func.args[0]
    y = match['y']
    f = func.func
    h = -match[match['d']] / match[match['e']]
    point = match['f0']
    value = match['f0val']
    terms = match['terms']
    F = h
    if not h:
        return Eq(f(x), value)
    series = value
    if terms > 1:
        hc = h.subs({x: point, y: value})
        if hc.has(oo) or hc.has(nan) or hc.has(zoo):
            return Eq(f(x), oo)
        elif hc:
            series += hc * (x - point)
    for factcount in range(2, terms):
        Fnew = F.diff(x) + F.diff(y) * h
        Fnewc = Fnew.subs({x: point, y: value})
        if Fnewc.has(oo) or Fnewc.has(nan) or Fnewc.has(-oo) or Fnewc.has(zoo):
            return Eq(f(x), oo)
        series += Fnewc * (x - point) ** factcount / factorial(factcount)
        F = Fnew
    series += Order(x ** terms)
    return Eq(f(x), series)

def checkinfsol(eq, infinitesimals, func=None, order=None):
    if False:
        print('Hello World!')
    '\n    This function is used to check if the given infinitesimals are the\n    actual infinitesimals of the given first order differential equation.\n    This method is specific to the Lie Group Solver of ODEs.\n\n    As of now, it simply checks, by substituting the infinitesimals in the\n    partial differential equation.\n\n\n    .. math:: \\frac{\\partial \\eta}{\\partial x} + \\left(\\frac{\\partial \\eta}{\\partial y}\n                - \\frac{\\partial \\xi}{\\partial x}\\right)*h\n                - \\frac{\\partial \\xi}{\\partial y}*h^{2}\n                - \\xi\\frac{\\partial h}{\\partial x} - \\eta\\frac{\\partial h}{\\partial y} = 0\n\n\n    where `\\eta`, and `\\xi` are the infinitesimals and `h(x,y) = \\frac{dy}{dx}`\n\n    The infinitesimals should be given in the form of a list of dicts\n    ``[{xi(x, y): inf, eta(x, y): inf}]``, corresponding to the\n    output of the function infinitesimals. It returns a list\n    of values of the form ``[(True/False, sol)]`` where ``sol`` is the value\n    obtained after substituting the infinitesimals in the PDE. If it\n    is ``True``, then ``sol`` would be 0.\n\n    '
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        (eq, func) = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError('Lie groups solver has been implemented only for first order differential equations')
        else:
            df = func.diff(x)
            a = Wild('a', exclude=[df])
            b = Wild('b', exclude=[df])
            match = collect(expand(eq), df).match(a * df + b)
            if match:
                h = -simplify(match[b] / match[a])
            else:
                try:
                    sol = solve(eq, df)
                except NotImplementedError:
                    raise NotImplementedError('Infinitesimals for the first order ODE could not be found')
                else:
                    h = sol[0]
            y = Dummy('y')
            h = h.subs(func, y)
            xi = Function('xi')(x, y)
            eta = Function('eta')(x, y)
            dxi = Function('xi')(x, func)
            deta = Function('eta')(x, func)
            pde = eta.diff(x) + (eta.diff(y) - xi.diff(x)) * h - xi.diff(y) * h ** 2 - xi * h.diff(x) - eta * h.diff(y)
            soltup = []
            for sol in infinitesimals:
                tsol = {xi: S(sol[dxi]).subs(func, y), eta: S(sol[deta]).subs(func, y)}
                sol = simplify(pde.subs(tsol).doit())
                if sol:
                    soltup.append((False, sol.subs(y, func)))
                else:
                    soltup.append((True, 0))
            return soltup

def sysode_linear_2eq_order1(match_):
    if False:
        i = 10
        return i + 15
    x = match_['func'][0].func
    y = match_['func'][1].func
    func = match_['func']
    fc = match_['func_coeff']
    eq = match_['eq']
    r = {}
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    for i in range(2):
        eq[i] = Add(*[terms / fc[i, func[i], 1] for terms in Add.make_args(eq[i])])
    r['a'] = -fc[0, x(t), 0] / fc[0, x(t), 1]
    r['c'] = -fc[1, x(t), 0] / fc[1, y(t), 1]
    r['b'] = -fc[0, y(t), 0] / fc[0, x(t), 1]
    r['d'] = -fc[1, y(t), 0] / fc[1, y(t), 1]
    forcing = [S.Zero, S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['k1'] = forcing[0]
        r['k2'] = forcing[1]
    else:
        raise NotImplementedError('Only homogeneous problems are supported' + ' (and constant inhomogeneity)')
    if match_['type_of_equation'] == 'type6':
        sol = _linear_2eq_order1_type6(x, y, t, r, eq)
    if match_['type_of_equation'] == 'type7':
        sol = _linear_2eq_order1_type7(x, y, t, r, eq)
    return sol

def _linear_2eq_order1_type6(x, y, t, r, eq):
    if False:
        while True:
            i = 10
    "\n    The equations of this type of ode are .\n\n    .. math:: x' = f(t) x + g(t) y\n\n    .. math:: y' = a [f(t) + a h(t)] x + a [g(t) - h(t)] y\n\n    This is solved by first multiplying the first equation by `-a` and adding\n    it to the second equation to obtain\n\n    .. math:: y' - a x' = -a h(t) (y - a x)\n\n    Setting `U = y - ax` and integrating the equation we arrive at\n\n    .. math:: y - ax = C_1 e^{-a \\int h(t) \\,dt}\n\n    and on substituting the value of y in first equation give rise to first order ODEs. After solving for\n    `x`, we can obtain `y` by substituting the value of `x` in second equation.\n\n    "
    (C1, C2, C3, C4) = get_numbered_constants(eq, num=4)
    p = 0
    q = 0
    p1 = cancel(r['c'] / cancel(r['c'] / r['d']).as_numer_denom()[0])
    p2 = cancel(r['a'] / cancel(r['a'] / r['b']).as_numer_denom()[0])
    for (n, i) in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q != 0 and n == 0:
                if (r['c'] / j - r['a']) / (r['b'] - r['d'] / j) == j:
                    p = 1
                    s = j
                    break
            if q != 0 and n == 1:
                if (r['a'] / j - r['c']) / (r['d'] - r['b'] / j) == j:
                    p = 2
                    s = j
                    break
    if p == 1:
        equ = diff(x(t), t) - r['a'] * x(t) - r['b'] * (s * x(t) + C1 * exp(-s * Integral(r['b'] - r['d'] / s, t)))
        hint1 = classify_ode(equ)[1]
        sol1 = dsolve(equ, hint=hint1 + '_Integral').rhs
        sol2 = s * sol1 + C1 * exp(-s * Integral(r['b'] - r['d'] / s, t))
    elif p == 2:
        equ = diff(y(t), t) - r['c'] * y(t) - r['d'] * s * y(t) + C1 * exp(-s * Integral(r['d'] - r['b'] / s, t))
        hint1 = classify_ode(equ)[1]
        sol2 = dsolve(equ, hint=hint1 + '_Integral').rhs
        sol1 = s * sol2 + C1 * exp(-s * Integral(r['d'] - r['b'] / s, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]

def _linear_2eq_order1_type7(x, y, t, r, eq):
    if False:
        while True:
            i = 10
    "\n    The equations of this type of ode are .\n\n    .. math:: x' = f(t) x + g(t) y\n\n    .. math:: y' = h(t) x + p(t) y\n\n    Differentiating the first equation and substituting the value of `y`\n    from second equation will give a second-order linear equation\n\n    .. math:: g x'' - (fg + gp + g') x' + (fgp - g^{2} h + f g' - f' g) x = 0\n\n    This above equation can be easily integrated if following conditions are satisfied.\n\n    1. `fgp - g^{2} h + f g' - f' g = 0`\n\n    2. `fgp - g^{2} h + f g' - f' g = ag, fg + gp + g' = bg`\n\n    If first condition is satisfied then it is solved by current dsolve solver and in second case it becomes\n    a constant coefficient differential equation which is also solved by current solver.\n\n    Otherwise if the above condition fails then,\n    a particular solution is assumed as `x = x_0(t)` and `y = y_0(t)`\n    Then the general solution is expressed as\n\n    .. math:: x = C_1 x_0(t) + C_2 x_0(t) \\int \\frac{g(t) F(t) P(t)}{x_0^{2}(t)} \\,dt\n\n    .. math:: y = C_1 y_0(t) + C_2 [\\frac{F(t) P(t)}{x_0(t)} + y_0(t) \\int \\frac{g(t) F(t) P(t)}{x_0^{2}(t)} \\,dt]\n\n    where C1 and C2 are arbitrary constants and\n\n    .. math:: F(t) = e^{\\int f(t) \\,dt}, P(t) = e^{\\int p(t) \\,dt}\n\n    "
    (C1, C2, C3, C4) = get_numbered_constants(eq, num=4)
    e1 = r['a'] * r['b'] * r['c'] - r['b'] ** 2 * r['c'] + r['a'] * diff(r['b'], t) - diff(r['a'], t) * r['b']
    e2 = r['a'] * r['c'] * r['d'] - r['b'] * r['c'] ** 2 + diff(r['c'], t) * r['d'] - r['c'] * diff(r['d'], t)
    m1 = r['a'] * r['b'] + r['b'] * r['d'] + diff(r['b'], t)
    m2 = r['a'] * r['c'] + r['c'] * r['d'] + diff(r['c'], t)
    if e1 == 0:
        sol1 = dsolve(r['b'] * diff(x(t), t, t) - m1 * diff(x(t), t)).rhs
        sol2 = dsolve(diff(y(t), t) - r['c'] * sol1 - r['d'] * y(t)).rhs
    elif e2 == 0:
        sol2 = dsolve(r['c'] * diff(y(t), t, t) - m2 * diff(y(t), t)).rhs
        sol1 = dsolve(diff(x(t), t) - r['a'] * x(t) - r['b'] * sol2).rhs
    elif not (e1 / r['b']).has(t) and (not (m1 / r['b']).has(t)):
        sol1 = dsolve(diff(x(t), t, t) - m1 / r['b'] * diff(x(t), t) - e1 / r['b'] * x(t)).rhs
        sol2 = dsolve(diff(y(t), t) - r['c'] * sol1 - r['d'] * y(t)).rhs
    elif not (e2 / r['c']).has(t) and (not (m2 / r['c']).has(t)):
        sol2 = dsolve(diff(y(t), t, t) - m2 / r['c'] * diff(y(t), t) - e2 / r['c'] * y(t)).rhs
        sol1 = dsolve(diff(x(t), t) - r['a'] * x(t) - r['b'] * sol2).rhs
    else:
        x0 = Function('x0')(t)
        y0 = Function('y0')(t)
        F = exp(Integral(r['a'], t))
        P = exp(Integral(r['d'], t))
        sol1 = C1 * x0 + C2 * x0 * Integral(r['b'] * F * P / x0 ** 2, t)
        sol2 = C1 * y0 + C2 * (F * P / x0 + y0 * Integral(r['b'] * F * P / x0 ** 2, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]

def sysode_nonlinear_2eq_order1(match_):
    if False:
        return 10
    func = match_['func']
    eq = match_['eq']
    fc = match_['func_coeff']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_2eq_order1_type5(func, t, eq)
        return sol
    x = func[0].func
    y = func[1].func
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms / fc[i, func[i], 1]
        eq[i] = eqs
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_2eq_order1_type1(x, y, t, eq)
    elif match_['type_of_equation'] == 'type2':
        sol = _nonlinear_2eq_order1_type2(x, y, t, eq)
    elif match_['type_of_equation'] == 'type3':
        sol = _nonlinear_2eq_order1_type3(x, y, t, eq)
    elif match_['type_of_equation'] == 'type4':
        sol = _nonlinear_2eq_order1_type4(x, y, t, eq)
    return sol

def _nonlinear_2eq_order1_type1(x, y, t, eq):
    if False:
        i = 10
        return i + 15
    "\n    Equations:\n\n    .. math:: x' = x^n F(x,y)\n\n    .. math:: y' = g(y) F(x,y)\n\n    Solution:\n\n    .. math:: x = \\varphi(y), \\int \\frac{1}{g(y) F(\\varphi(y),y)} \\,dy = t + C_2\n\n    where\n\n    if `n \\neq 1`\n\n    .. math:: \\varphi = [C_1 + (1-n) \\int \\frac{1}{g(y)} \\,dy]^{\\frac{1}{1-n}}\n\n    if `n = 1`\n\n    .. math:: \\varphi = C_1 e^{\\int \\frac{1}{g(y)} \\,dy}\n\n    where `C_1` and `C_2` are arbitrary constants.\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t), y(t)])
    f = Wild('f')
    (u, v) = symbols('u, v')
    r = eq[0].match(diff(x(t), t) - x(t) ** n * f)
    g = ((diff(y(t), t) - eq[1]) / r[f]).subs(y(t), v)
    F = r[f].subs(x(t), u).subs(y(t), v)
    n = r[n]
    if n != 1:
        phi = (C1 + (1 - n) * Integral(1 / g, v)) ** (1 / (1 - n))
    else:
        phi = C1 * exp(Integral(1 / g, v))
    phi = phi.doit()
    sol2 = solve(Integral(1 / (g * F.subs(u, phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t), phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type2(x, y, t, eq):
    if False:
        i = 10
        return i + 15
    "\n    Equations:\n\n    .. math:: x' = e^{\\lambda x} F(x,y)\n\n    .. math:: y' = g(y) F(x,y)\n\n    Solution:\n\n    .. math:: x = \\varphi(y), \\int \\frac{1}{g(y) F(\\varphi(y),y)} \\,dy = t + C_2\n\n    where\n\n    if `\\lambda \\neq 0`\n\n    .. math:: \\varphi = -\\frac{1}{\\lambda} log(C_1 - \\lambda \\int \\frac{1}{g(y)} \\,dy)\n\n    if `\\lambda = 0`\n\n    .. math:: \\varphi = C_1 + \\int \\frac{1}{g(y)} \\,dy\n\n    where `C_1` and `C_2` are arbitrary constants.\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t), y(t)])
    f = Wild('f')
    (u, v) = symbols('u, v')
    r = eq[0].match(diff(x(t), t) - exp(n * x(t)) * f)
    g = ((diff(y(t), t) - eq[1]) / r[f]).subs(y(t), v)
    F = r[f].subs(x(t), u).subs(y(t), v)
    n = r[n]
    if n:
        phi = -1 / n * log(C1 - n * Integral(1 / g, v))
    else:
        phi = C1 + Integral(1 / g, v)
    phi = phi.doit()
    sol2 = solve(Integral(1 / (g * F.subs(u, phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t), phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type3(x, y, t, eq):
    if False:
        return 10
    "\n    Autonomous system of general form\n\n    .. math:: x' = F(x,y)\n\n    .. math:: y' = G(x,y)\n\n    Assuming `y = y(x, C_1)` where `C_1` is an arbitrary constant is the general\n    solution of the first-order equation\n\n    .. math:: F(x,y) y'_x = G(x,y)\n\n    Then the general solution of the original system of equations has the form\n\n    .. math:: \\int \\frac{1}{F(x,y(x,C_1))} \\,dx = t + C_1\n\n    "
    (C1, C2, C3, C4) = get_numbered_constants(eq, num=4)
    v = Function('v')
    u = Symbol('u')
    f = Wild('f')
    g = Wild('g')
    r1 = eq[0].match(diff(x(t), t) - f)
    r2 = eq[1].match(diff(y(t), t) - g)
    F = r1[f].subs(x(t), u).subs(y(t), v(u))
    G = r2[g].subs(x(t), u).subs(y(t), v(u))
    sol2r = dsolve(Eq(diff(v(u), u), G / F))
    if isinstance(sol2r, Equality):
        sol2r = [sol2r]
    for sol2s in sol2r:
        sol1 = solve(Integral(1 / F.subs(v(u), sol2s.rhs), u).doit() - t - C2, u)
    sol = []
    for sols in sol1:
        sol.append(Eq(x(t), sols))
        sol.append(Eq(y(t), sol2s.rhs.subs(u, sols)))
    return sol

def _nonlinear_2eq_order1_type4(x, y, t, eq):
    if False:
        print('Hello World!')
    "\n    Equation:\n\n    .. math:: x' = f_1(x) g_1(y) \\phi(x,y,t)\n\n    .. math:: y' = f_2(x) g_2(y) \\phi(x,y,t)\n\n    First integral:\n\n    .. math:: \\int \\frac{f_2(x)}{f_1(x)} \\,dx - \\int \\frac{g_1(y)}{g_2(y)} \\,dy = C\n\n    where `C` is an arbitrary constant.\n\n    On solving the first integral for `x` (resp., `y` ) and on substituting the\n    resulting expression into either equation of the original solution, one\n    arrives at a first-order equation for determining `y` (resp., `x` ).\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    (u, v) = symbols('u, v')
    (U, V) = symbols('U, V', cls=Function)
    f = Wild('f')
    g = Wild('g')
    f1 = Wild('f1', exclude=[v, t])
    f2 = Wild('f2', exclude=[v, t])
    g1 = Wild('g1', exclude=[u, t])
    g2 = Wild('g2', exclude=[u, t])
    r1 = eq[0].match(diff(x(t), t) - f)
    r2 = eq[1].match(diff(y(t), t) - g)
    (num, den) = (r1[f].subs(x(t), u).subs(y(t), v) / r2[g].subs(x(t), u).subs(y(t), v)).as_numer_denom()
    R1 = num.match(f1 * g1)
    R2 = den.match(f2 * g2)
    phi = r1[f].subs(x(t), u).subs(y(t), v) / num
    F1 = R1[f1]
    F2 = R2[f2]
    G1 = R1[g1]
    G2 = R2[g2]
    sol1r = solve(Integral(F2 / F1, u).doit() - Integral(G1 / G2, v).doit() - C1, u)
    sol2r = solve(Integral(F2 / F1, u).doit() - Integral(G1 / G2, v).doit() - C1, v)
    sol = []
    for sols in sol1r:
        sol.append(Eq(y(t), dsolve(diff(V(t), t) - F2.subs(u, sols).subs(v, V(t)) * G2.subs(v, V(t)) * phi.subs(u, sols).subs(v, V(t))).rhs))
    for sols in sol2r:
        sol.append(Eq(x(t), dsolve(diff(U(t), t) - F1.subs(u, U(t)) * G1.subs(v, sols).subs(u, U(t)) * phi.subs(v, sols).subs(u, U(t))).rhs))
    return set(sol)

def _nonlinear_2eq_order1_type5(func, t, eq):
    if False:
        while True:
            i = 10
    "\n    Clairaut system of ODEs\n\n    .. math:: x = t x' + F(x',y')\n\n    .. math:: y = t y' + G(x',y')\n\n    The following are solutions of the system\n\n    `(i)` straight lines:\n\n    .. math:: x = C_1 t + F(C_1, C_2), y = C_2 t + G(C_1, C_2)\n\n    where `C_1` and `C_2` are arbitrary constants;\n\n    `(ii)` envelopes of the above lines;\n\n    `(iii)` continuously differentiable lines made up from segments of the lines\n    `(i)` and `(ii)`.\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    f = Wild('f')
    g = Wild('g')

    def check_type(x, y):
        if False:
            for i in range(10):
                print('nop')
        r1 = eq[0].match(t * diff(x(t), t) - x(t) + f)
        r2 = eq[1].match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t), t) - x(t) / t + f / t)
            r2 = eq[1].match(diff(y(t), t) - y(t) / t + g / t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t * diff(x(t), t) - x(t) + f)
            r2 = (-eq[1]).match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t), t) - x(t) / t + f / t)
            r2 = (-eq[1]).match(diff(y(t), t) - y(t) / t + g / t)
        return [r1, r2]
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            [r1, r2] = check_type(x, y)
            if not (r1 and r2):
                [r1, r2] = check_type(y, x)
                (x, y) = (y, x)
    x1 = diff(x(t), t)
    y1 = diff(y(t), t)
    return {Eq(x(t), C1 * t + r1[f].subs(x1, C1).subs(y1, C2)), Eq(y(t), C2 * t + r2[g].subs(x1, C1).subs(y1, C2))}

def sysode_nonlinear_3eq_order1(match_):
    if False:
        for i in range(10):
            print('nop')
    x = match_['func'][0].func
    y = match_['func'][1].func
    z = match_['func'][2].func
    eq = match_['eq']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_3eq_order1_type1(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type2':
        sol = _nonlinear_3eq_order1_type2(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type3':
        sol = _nonlinear_3eq_order1_type3(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type4':
        sol = _nonlinear_3eq_order1_type4(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_3eq_order1_type5(x, y, z, t, eq)
    return sol

def _nonlinear_3eq_order1_type1(x, y, z, t, eq):
    if False:
        print('Hello World!')
    "\n    Equations:\n\n    .. math:: a x' = (b - c) y z, \\enspace b y' = (c - a) z x, \\enspace c z' = (a - b) x y\n\n    First Integrals:\n\n    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1\n\n    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2\n\n    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and\n    `z` and on substituting the resulting expressions into the first equation of the\n    system, we arrives at a separable first-order equation on `x`. Similarly doing that\n    for other two equations, we will arrive at first order equation on `y` and `z` too.\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0401.pdf\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    (u, v, w) = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    r = (diff(x(t), t) - eq[0]).match(p * y(t) * z(t))
    r.update((diff(y(t), t) - eq[1]).match(q * z(t) * x(t)))
    r.update((diff(z(t), t) - eq[2]).match(s * x(t) * y(t)))
    (n1, d1) = r[p].as_numer_denom()
    (n2, d2) = r[q].as_numer_denom()
    (n3, d3) = r[s].as_numer_denom()
    val = solve([n1 * u - d1 * v + d1 * w, d2 * u + n2 * v - d2 * w, d3 * u - d3 * v - n3 * w], [u, v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    b = vals[0].subs(w, c)
    a = vals[1].subs(w, c)
    y_x = sqrt((c * C1 - C2 - a * (c - a) * x(t) ** 2) / (b * (c - b)))
    z_x = sqrt((b * C1 - C2 - a * (b - a) * x(t) ** 2) / (c * (b - c)))
    z_y = sqrt((a * C1 - C2 - b * (a - b) * y(t) ** 2) / (c * (a - c)))
    x_y = sqrt((c * C1 - C2 - b * (c - b) * y(t) ** 2) / (a * (c - a)))
    x_z = sqrt((b * C1 - C2 - c * (b - c) * z(t) ** 2) / (a * (b - a)))
    y_z = sqrt((a * C1 - C2 - c * (a - c) * z(t) ** 2) / (b * (a - b)))
    sol1 = dsolve(a * diff(x(t), t) - (b - c) * y_x * z_x)
    sol2 = dsolve(b * diff(y(t), t) - (c - a) * z_y * x_y)
    sol3 = dsolve(c * diff(z(t), t) - (a - b) * x_z * y_z)
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type2(x, y, z, t, eq):
    if False:
        return 10
    "\n    Equations:\n\n    .. math:: a x' = (b - c) y z f(x, y, z, t)\n\n    .. math:: b y' = (c - a) z x f(x, y, z, t)\n\n    .. math:: c z' = (a - b) x y f(x, y, z, t)\n\n    First Integrals:\n\n    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1\n\n    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2\n\n    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and\n    `z` and on substituting the resulting expressions into the first equation of the\n    system, we arrives at a first-order differential equations on `x`. Similarly doing\n    that for other two equations we will arrive at first order equation on `y` and `z`.\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0402.pdf\n\n    "
    (C1, C2) = get_numbered_constants(eq, num=2)
    (u, v, w) = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    r1 = (diff(x(t), t) - eq[0]).match(y(t) * z(t) * f)
    r = collect_const(r1[f]).match(p * f)
    r.update(((diff(y(t), t) - eq[1]) / r[f]).match(q * z(t) * x(t)))
    r.update(((diff(z(t), t) - eq[2]) / r[f]).match(s * x(t) * y(t)))
    (n1, d1) = r[p].as_numer_denom()
    (n2, d2) = r[q].as_numer_denom()
    (n3, d3) = r[s].as_numer_denom()
    val = solve([n1 * u - d1 * v + d1 * w, d2 * u + n2 * v - d2 * w, -d3 * u + d3 * v + n3 * w], [u, v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    a = vals[0].subs(w, c)
    b = vals[1].subs(w, c)
    y_x = sqrt((c * C1 - C2 - a * (c - a) * x(t) ** 2) / (b * (c - b)))
    z_x = sqrt((b * C1 - C2 - a * (b - a) * x(t) ** 2) / (c * (b - c)))
    z_y = sqrt((a * C1 - C2 - b * (a - b) * y(t) ** 2) / (c * (a - c)))
    x_y = sqrt((c * C1 - C2 - b * (c - b) * y(t) ** 2) / (a * (c - a)))
    x_z = sqrt((b * C1 - C2 - c * (b - c) * z(t) ** 2) / (a * (b - a)))
    y_z = sqrt((a * C1 - C2 - c * (a - c) * z(t) ** 2) / (b * (a - b)))
    sol1 = dsolve(a * diff(x(t), t) - (b - c) * y_x * z_x * r[f])
    sol2 = dsolve(b * diff(y(t), t) - (c - a) * z_y * x_y * r[f])
    sol3 = dsolve(c * diff(z(t), t) - (a - b) * x_z * y_z * r[f])
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type3(x, y, z, t, eq):
    if False:
        for i in range(10):
            print('nop')
    "\n    Equations:\n\n    .. math:: x' = c F_2 - b F_3, \\enspace y' = a F_3 - c F_1, \\enspace z' = b F_1 - a F_2\n\n    where `F_n = F_n(x, y, z, t)`.\n\n    1. First Integral:\n\n    .. math:: a x + b y + c z = C_1,\n\n    where C is an arbitrary constant.\n\n    2. If we assume function `F_n` to be independent of `t`,i.e, `F_n` = `F_n (x, y, z)`\n    Then, on eliminating `t` and `z` from the first two equation of the system, one\n    arrives at the first-order equation\n\n    .. math:: \\frac{dy}{dx} = \\frac{a F_3 (x, y, z) - c F_1 (x, y, z)}{c F_2 (x, y, z) -\n                b F_3 (x, y, z)}\n\n    where `z = \\frac{1}{c} (C_1 - a x - b y)`\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0404.pdf\n\n    "
    C1 = get_numbered_constants(eq, num=1)
    (u, v, w) = symbols('u, v, w')
    (fu, fv, fw) = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    (F1, F2, F3) = symbols('F1, F2, F3', cls=Wild)
    r1 = (diff(x(t), t) - eq[0]).match(F2 - F3)
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    if eq[1].has(r[F2]) and (not eq[1].has(r[F3])):
        (r[F2], r[F3]) = (r[F3], r[F2])
        (r[s], r[q]) = (-r[q], -r[s])
    r.update((diff(y(t), t) - eq[1]).match(p * r[F3] - r[s] * F1))
    a = r[p]
    b = r[q]
    c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    z_xy = (C1 - a * u - b * v) / c
    y_zx = (C1 - a * u - c * w) / b
    x_yz = (C1 - b * v - c * w) / a
    y_x = dsolve(diff(fv(u), u) - ((a * F3 - c * F1) / (c * F2 - b * F3)).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((b * F1 - a * F2) / (c * F2 - b * F3)).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - ((b * F1 - a * F2) / (a * F3 - c * F1)).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - ((c * F2 - b * F3) / (a * F3 - c * F1)).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - ((a * F3 - c * F1) / (b * F1 - a * F2)).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - ((c * F2 - b * F3) / (b * F1 - a * F2)).subs(v, y_zx).subs(u, fu(w))).rhs
    sol1 = dsolve(diff(fu(t), t) - (c * F2 - b * F3).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (a * F3 - c * F1).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (b * F1 - a * F2).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type4(x, y, z, t, eq):
    if False:
        i = 10
        return i + 15
    "\n    Equations:\n\n    .. math:: x' = c z F_2 - b y F_3, \\enspace y' = a x F_3 - c z F_1, \\enspace z' = b y F_1 - a x F_2\n\n    where `F_n = F_n (x, y, z, t)`\n\n    1. First integral:\n\n    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1\n\n    where `C` is an arbitrary constant.\n\n    2. Assuming the function `F_n` is independent of `t`: `F_n = F_n (x, y, z)`. Then on\n    eliminating `t` and `z` from the first two equations of the system, one arrives at\n    the first-order equation\n\n    .. math:: \\frac{dy}{dx} = \\frac{a x F_3 (x, y, z) - c z F_1 (x, y, z)}\n                {c z F_2 (x, y, z) - b y F_3 (x, y, z)}\n\n    where `z = \\pm \\sqrt{\\frac{1}{c} (C_1 - a x^{2} - b y^{2})}`\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0405.pdf\n\n    "
    C1 = get_numbered_constants(eq, num=1)
    (u, v, w) = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    (F1, F2, F3) = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t), t) - z(t) * F2 + y(t) * F3)
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    if eq[1].has(r[F2]) and (not eq[1].has(r[F3])):
        (r[F2], r[F3]) = (r[F3], r[F2])
        (r[s], r[q]) = (-r[q], -r[s])
    r.update((diff(y(t), t) - eq[1]).match(p * x(t) * r[F3] - r[s] * z(t) * F1))
    a = r[p]
    b = r[q]
    c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    x_yz = sqrt((C1 - b * v ** 2 - c * w ** 2) / a)
    y_zx = sqrt((C1 - c * w ** 2 - a * u ** 2) / b)
    z_xy = sqrt((C1 - a * u ** 2 - b * v ** 2) / c)
    y_x = dsolve(diff(v(u), u) - ((a * u * F3 - c * w * F1) / (c * w * F2 - b * v * F3)).subs(w, z_xy).subs(v, v(u))).rhs
    z_x = dsolve(diff(w(u), u) - ((b * v * F1 - a * u * F2) / (c * w * F2 - b * v * F3)).subs(v, y_zx).subs(w, w(u))).rhs
    z_y = dsolve(diff(w(v), v) - ((b * v * F1 - a * u * F2) / (a * u * F3 - c * w * F1)).subs(u, x_yz).subs(w, w(v))).rhs
    x_y = dsolve(diff(u(v), v) - ((c * w * F2 - b * v * F3) / (a * u * F3 - c * w * F1)).subs(w, z_xy).subs(u, u(v))).rhs
    y_z = dsolve(diff(v(w), w) - ((a * u * F3 - c * w * F1) / (b * v * F1 - a * u * F2)).subs(u, x_yz).subs(v, v(w))).rhs
    x_z = dsolve(diff(u(w), w) - ((c * w * F2 - b * v * F3) / (b * v * F1 - a * u * F2)).subs(v, y_zx).subs(u, u(w))).rhs
    sol1 = dsolve(diff(u(t), t) - (c * w * F2 - b * v * F3).subs(v, y_x).subs(w, z_x).subs(u, u(t))).rhs
    sol2 = dsolve(diff(v(t), t) - (a * u * F3 - c * w * F1).subs(u, x_y).subs(w, z_y).subs(v, v(t))).rhs
    sol3 = dsolve(diff(w(t), t) - (b * v * F1 - a * u * F2).subs(u, x_z).subs(v, y_z).subs(w, w(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type5(x, y, z, t, eq):
    if False:
        while True:
            i = 10
    "\n    .. math:: x' = x (c F_2 - b F_3), \\enspace y' = y (a F_3 - c F_1), \\enspace z' = z (b F_1 - a F_2)\n\n    where `F_n = F_n (x, y, z, t)` and are arbitrary functions.\n\n    First Integral:\n\n    .. math:: \\left|x\\right|^{a} \\left|y\\right|^{b} \\left|z\\right|^{c} = C_1\n\n    where `C` is an arbitrary constant. If the function `F_n` is independent of `t`,\n    then, by eliminating `t` and `z` from the first two equations of the system, one\n    arrives at a first-order equation.\n\n    References\n    ==========\n    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0406.pdf\n\n    "
    C1 = get_numbered_constants(eq, num=1)
    (u, v, w) = symbols('u, v, w')
    (fu, fv, fw) = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    (F1, F2, F3) = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t), t) - x(t) * F2 + x(t) * F3)
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    if eq[1].has(r[F2]) and (not eq[1].has(r[F3])):
        (r[F2], r[F3]) = (r[F3], r[F2])
        (r[s], r[q]) = (-r[q], -r[s])
    r.update((diff(y(t), t) - eq[1]).match(y(t) * (p * r[F3] - r[s] * F1)))
    a = r[p]
    b = r[q]
    c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    x_yz = (C1 * v ** (-b) * w ** (-c)) ** (-a)
    y_zx = (C1 * w ** (-c) * u ** (-a)) ** (-b)
    z_xy = (C1 * u ** (-a) * v ** (-b)) ** (-c)
    y_x = dsolve(diff(fv(u), u) - (v * (a * F3 - c * F1) / (u * (c * F2 - b * F3))).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - (w * (b * F1 - a * F2) / (u * (c * F2 - b * F3))).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - (w * (b * F1 - a * F2) / (v * (a * F3 - c * F1))).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - (u * (c * F2 - b * F3) / (v * (a * F3 - c * F1))).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - (v * (a * F3 - c * F1) / (w * (b * F1 - a * F2))).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - (u * (c * F2 - b * F3) / (w * (b * F1 - a * F2))).subs(v, y_zx).subs(u, fu(w))).rhs
    sol1 = dsolve(diff(fu(t), t) - (u * (c * F2 - b * F3)).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (v * (a * F3 - c * F1)).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (w * (b * F1 - a * F2)).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    return [sol1, sol2, sol3]
from .single import SingleODEProblem, SingleODESolver, solver_map