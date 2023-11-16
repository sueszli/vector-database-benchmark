"""
Unified interfaces to minimization algorithms.

Functions
---------
- minimize : minimization of a function of several variables.
- minimize_scalar : minimization of a function of one variable.
"""
__all__ = ['minimize', 'minimize_scalar']
from warnings import warn
import numpy as np
from ._optimize import _minimize_neldermead, _minimize_powell, _minimize_cg, _minimize_bfgs, _minimize_newtoncg, _minimize_scalar_brent, _minimize_scalar_bounded, _minimize_scalar_golden, MemoizeJac, OptimizeResult, _wrap_callback, _recover_from_bracket_error
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._slsqp_py import _minimize_slsqp
from ._constraints import old_bound_to_new, new_bounds_to_old, old_constraint_to_new, new_constraint_to_old, NonlinearConstraint, LinearConstraint, Bounds, PreparedConstraint
from ._differentiable_functions import FD_METHODS
MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
MINIMIZE_SCALAR_METHODS = ['brent', 'bounded', 'golden']

def minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    if False:
        i = 10
        return i + 15
    'Minimization of scalar function of one or more variables.\n\n    Parameters\n    ----------\n    fun : callable\n        The objective function to be minimized.\n\n            ``fun(x, *args) -> float``\n\n        where ``x`` is a 1-D array with shape (n,) and ``args``\n        is a tuple of the fixed parameters needed to completely\n        specify the function.\n    x0 : ndarray, shape (n,)\n        Initial guess. Array of real elements of size (n,),\n        where ``n`` is the number of independent variables.\n    args : tuple, optional\n        Extra arguments passed to the objective function and its\n        derivatives (`fun`, `jac` and `hess` functions).\n    method : str or callable, optional\n        Type of solver.  Should be one of\n\n            - \'Nelder-Mead\' :ref:`(see here) <optimize.minimize-neldermead>`\n            - \'Powell\'      :ref:`(see here) <optimize.minimize-powell>`\n            - \'CG\'          :ref:`(see here) <optimize.minimize-cg>`\n            - \'BFGS\'        :ref:`(see here) <optimize.minimize-bfgs>`\n            - \'Newton-CG\'   :ref:`(see here) <optimize.minimize-newtoncg>`\n            - \'L-BFGS-B\'    :ref:`(see here) <optimize.minimize-lbfgsb>`\n            - \'TNC\'         :ref:`(see here) <optimize.minimize-tnc>`\n            - \'COBYLA\'      :ref:`(see here) <optimize.minimize-cobyla>`\n            - \'SLSQP\'       :ref:`(see here) <optimize.minimize-slsqp>`\n            - \'trust-constr\':ref:`(see here) <optimize.minimize-trustconstr>`\n            - \'dogleg\'      :ref:`(see here) <optimize.minimize-dogleg>`\n            - \'trust-ncg\'   :ref:`(see here) <optimize.minimize-trustncg>`\n            - \'trust-exact\' :ref:`(see here) <optimize.minimize-trustexact>`\n            - \'trust-krylov\' :ref:`(see here) <optimize.minimize-trustkrylov>`\n            - custom - a callable object, see below for description.\n\n        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,\n        depending on whether or not the problem has constraints or bounds.\n    jac : {callable,  \'2-point\', \'3-point\', \'cs\', bool}, optional\n        Method for computing the gradient vector. Only for CG, BFGS,\n        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,\n        trust-exact and trust-constr.\n        If it is a callable, it should be a function that returns the gradient\n        vector:\n\n            ``jac(x, *args) -> array_like, shape (n,)``\n\n        where ``x`` is an array with shape (n,) and ``args`` is a tuple with\n        the fixed parameters. If `jac` is a Boolean and is True, `fun` is\n        assumed to return a tuple ``(f, g)`` containing the objective\n        function and the gradient.\n        Methods \'Newton-CG\', \'trust-ncg\', \'dogleg\', \'trust-exact\', and\n        \'trust-krylov\' require that either a callable be supplied, or that\n        `fun` return the objective and gradient.\n        If None or False, the gradient will be estimated using 2-point finite\n        difference estimation with an absolute step size.\n        Alternatively, the keywords  {\'2-point\', \'3-point\', \'cs\'} can be used\n        to select a finite difference scheme for numerical estimation of the\n        gradient with a relative step size. These finite difference schemes\n        obey any specified `bounds`.\n    hess : {callable, \'2-point\', \'3-point\', \'cs\', HessianUpdateStrategy}, optional\n        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,\n        trust-ncg, trust-krylov, trust-exact and trust-constr.\n        If it is callable, it should return the Hessian matrix:\n\n            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``\n\n        where ``x`` is a (n,) ndarray and ``args`` is a tuple with the fixed\n        parameters.\n        The keywords {\'2-point\', \'3-point\', \'cs\'} can also be used to select\n        a finite difference scheme for numerical estimation of the hessian.\n        Alternatively, objects implementing the `HessianUpdateStrategy`\n        interface can be used to approximate the Hessian. Available\n        quasi-Newton methods implementing this interface are:\n\n            - `BFGS`;\n            - `SR1`.\n\n        Not all of the options are available for each of the methods; for\n        availability refer to the notes.\n    hessp : callable, optional\n        Hessian of objective function times an arbitrary vector p. Only for\n        Newton-CG, trust-ncg, trust-krylov, trust-constr.\n        Only one of `hessp` or `hess` needs to be given. If `hess` is\n        provided, then `hessp` will be ignored. `hessp` must compute the\n        Hessian times an arbitrary vector:\n\n            ``hessp(x, p, *args) ->  ndarray shape (n,)``\n\n        where ``x`` is a (n,) ndarray, ``p`` is an arbitrary vector with\n        dimension (n,) and ``args`` is a tuple with the fixed\n        parameters.\n    bounds : sequence or `Bounds`, optional\n        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,\n        trust-constr, and COBYLA methods. There are two ways to specify the\n        bounds:\n\n            1. Instance of `Bounds` class.\n            2. Sequence of ``(min, max)`` pairs for each element in `x`. None\n               is used to specify no bound.\n\n    constraints : {Constraint, dict} or List of {Constraint, dict}, optional\n        Constraints definition. Only for COBYLA, SLSQP and trust-constr.\n\n        Constraints for \'trust-constr\' are defined as a single object or a\n        list of objects specifying constraints to the optimization problem.\n        Available constraints are:\n\n            - `LinearConstraint`\n            - `NonlinearConstraint`\n\n        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.\n        Each dictionary with fields:\n\n            type : str\n                Constraint type: \'eq\' for equality, \'ineq\' for inequality.\n            fun : callable\n                The function defining the constraint.\n            jac : callable, optional\n                The Jacobian of `fun` (only for SLSQP).\n            args : sequence, optional\n                Extra arguments to be passed to the function and Jacobian.\n\n        Equality constraint means that the constraint function result is to\n        be zero whereas inequality means that it is to be non-negative.\n        Note that COBYLA only supports inequality constraints.\n    tol : float, optional\n        Tolerance for termination. When `tol` is specified, the selected\n        minimization algorithm sets some relevant solver-specific tolerance(s)\n        equal to `tol`. For detailed control, use solver-specific\n        options.\n    options : dict, optional\n        A dictionary of solver options. All methods except `TNC` accept the\n        following generic options:\n\n            maxiter : int\n                Maximum number of iterations to perform. Depending on the\n                method each iteration may use several function evaluations.\n\n                For `TNC` use `maxfun` instead of `maxiter`.\n            disp : bool\n                Set to True to print convergence messages.\n\n        For method-specific options, see :func:`show_options()`.\n    callback : callable, optional\n        A callable called after each iteration.\n\n        All methods except TNC, SLSQP, and COBYLA support a callable with\n        the signature:\n\n            ``callback(intermediate_result: OptimizeResult)``\n\n        where ``intermediate_result`` is a keyword parameter containing an\n        `OptimizeResult` with attributes ``x`` and ``fun``, the present values\n        of the parameter vector and objective function. Note that the name\n        of the parameter must be ``intermediate_result`` for the callback\n        to be passed an `OptimizeResult`. These methods will also terminate if\n        the callback raises ``StopIteration``.\n\n        All methods except trust-constr (also) support a signature like:\n\n            ``callback(xk)``\n\n        where ``xk`` is the current parameter vector.\n\n        Introspection is used to determine which of the signatures above to\n        invoke.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    See also\n    --------\n    minimize_scalar : Interface to minimization algorithms for scalar\n        univariate functions\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    \'method\' parameter. The default method is *BFGS*.\n\n    **Unconstrained minimization**\n\n    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate\n    gradient algorithm by Polak and Ribiere, a variant of the\n    Fletcher-Reeves method described in [5]_ pp.120-122. Only the\n    first derivatives are used.\n\n    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton\n    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_\n    pp. 136. It uses the first derivatives only. BFGS has proven good\n    performance even for non-smooth optimizations. This method also\n    returns an approximation of the Hessian inverse, stored as\n    `hess_inv` in the OptimizeResult object.\n\n    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a\n    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated\n    Newton method). It uses a CG method to the compute the search\n    direction. See also *TNC* method for a box-constrained\n    minimization with a similar algorithm. Suitable for large-scale\n    problems.\n\n    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg\n    trust-region algorithm [5]_ for unconstrained minimization. This\n    algorithm requires the gradient and Hessian; furthermore the\n    Hessian is required to be positive definite.\n\n    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the\n    Newton conjugate gradient trust-region algorithm [5]_ for\n    unconstrained minimization. This algorithm requires the gradient\n    and either the Hessian or a function that computes the product of\n    the Hessian with a given vector. Suitable for large-scale problems.\n\n    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses\n    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained\n    minimization. This algorithm requires the gradient\n    and either the Hessian or a function that computes the product of\n    the Hessian with a given vector. Suitable for large-scale problems.\n    On indefinite problems it requires usually less iterations than the\n    `trust-ncg` method and is recommended for medium and large-scale problems.\n\n    Method :ref:`trust-exact <optimize.minimize-trustexact>`\n    is a trust-region method for unconstrained minimization in which\n    quadratic subproblems are solved almost exactly [13]_. This\n    algorithm requires the gradient and the Hessian (which is\n    *not* required to be positive definite). It is, in many\n    situations, the Newton method to converge in fewer iterations\n    and the most recommended for small and medium-size problems.\n\n    **Bound-Constrained minimization**\n\n    Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the\n    Simplex algorithm [1]_, [2]_. This algorithm is robust in many\n    applications. However, if numerical computation of derivative can be\n    trusted, other algorithms using the first and/or second derivatives\n    information might be preferred for their better performance in\n    general.\n\n    Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B\n    algorithm [6]_, [7]_ for bound constrained minimization.\n\n    Method :ref:`Powell <optimize.minimize-powell>` is a modification\n    of Powell\'s method [3]_, [4]_ which is a conjugate direction\n    method. It performs sequential one-dimensional minimizations along\n    each vector of the directions set (`direc` field in `options` and\n    `info`), which is updated at each iteration of the main\n    minimization loop. The function need not be differentiable, and no\n    derivatives are taken. If bounds are not provided, then an\n    unbounded line search will be used. If bounds are provided and\n    the initial guess is within the bounds, then every function\n    evaluation throughout the minimization procedure will be within\n    the bounds. If bounds are provided, the initial guess is outside\n    the bounds, and `direc` is full rank (default has full rank), then\n    some function evaluations during the first iteration may be\n    outside the bounds, but every function evaluation after the first\n    iteration will be within the bounds. If `direc` is not full rank,\n    then some parameters may not be optimized and the solution is not\n    guaranteed to be within the bounds.\n\n    Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton\n    algorithm [5]_, [8]_ to minimize a function with variables subject\n    to bounds. This algorithm uses gradient information; it is also\n    called Newton Conjugate-Gradient. It differs from the *Newton-CG*\n    method described above as it wraps a C implementation and allows\n    each variable to be given upper and lower bounds.\n\n    **Constrained Minimization**\n\n    Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the\n    Constrained Optimization BY Linear Approximation (COBYLA) method\n    [9]_, [10]_, [11]_. The algorithm is based on linear\n    approximations to the objective function and each constraint. The\n    method wraps a FORTRAN implementation of the algorithm. The\n    constraints functions \'fun\' may return either a single number\n    or an array or list of numbers.\n\n    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential\n    Least SQuares Programming to minimize a function of several\n    variables with any combination of bounds, equality and inequality\n    constraints. The method wraps the SLSQP Optimization subroutine\n    originally implemented by Dieter Kraft [12]_. Note that the\n    wrapper handles infinite values in bounds by converting them into\n    large floating values.\n\n    Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a\n    trust-region algorithm for constrained optimization. It switches\n    between two implementations depending on the problem definition.\n    It is the most versatile constrained minimization algorithm\n    implemented in SciPy and the most appropriate for large-scale problems.\n    For equality constrained problems it is an implementation of Byrd-Omojokun\n    Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When\n    inequality constraints are imposed as well, it switches to the trust-region\n    interior point method described in [16]_. This interior point algorithm,\n    in turn, solves inequality constraints by introducing slack variables\n    and solving a sequence of equality-constrained barrier problems\n    for progressively smaller values of the barrier parameter.\n    The previously described equality constrained SQP method is\n    used to solve the subproblems with increasing levels of accuracy\n    as the iterate gets closer to a solution.\n\n    **Finite-Difference Options**\n\n    For Method :ref:`trust-constr <optimize.minimize-trustconstr>`\n    the gradient and the Hessian may be approximated using\n    three finite-difference schemes: {\'2-point\', \'3-point\', \'cs\'}.\n    The scheme \'cs\' is, potentially, the most accurate but it\n    requires the function to correctly handle complex inputs and to\n    be differentiable in the complex plane. The scheme \'3-point\' is more\n    accurate than \'2-point\' but requires twice as many operations. If the\n    gradient is estimated via finite-differences the Hessian must be\n    estimated using one of the quasi-Newton strategies.\n\n    **Method specific options for the** `hess` **keyword**\n\n    +--------------+------+----------+-------------------------+-----+\n    | method/Hess  | None | callable | \'2-point/\'3-point\'/\'cs\' | HUS |\n    +==============+======+==========+=========================+=====+\n    | Newton-CG    | x    | (n, n)   | x                       | x   |\n    |              |      | LO       |                         |     |\n    +--------------+------+----------+-------------------------+-----+\n    | dogleg       |      | (n, n)   |                         |     |\n    +--------------+------+----------+-------------------------+-----+\n    | trust-ncg    |      | (n, n)   | x                       | x   |\n    +--------------+------+----------+-------------------------+-----+\n    | trust-krylov |      | (n, n)   | x                       | x   |\n    +--------------+------+----------+-------------------------+-----+\n    | trust-exact  |      | (n, n)   |                         |     |\n    +--------------+------+----------+-------------------------+-----+\n    | trust-constr | x    | (n, n)   |  x                      | x   |\n    |              |      | LO       |                         |     |\n    |              |      | sp       |                         |     |\n    +--------------+------+----------+-------------------------+-----+\n\n    where LO=LinearOperator, sp=Sparse matrix, HUS=HessianUpdateStrategy\n\n    **Custom minimizers**\n\n    It may be useful to pass a custom minimization method, for example\n    when using a frontend to this method such as `scipy.optimize.basinhopping`\n    or a different library.  You can simply pass a callable as the ``method``\n    parameter.\n\n    The callable is called as ``method(fun, x0, args, **kwargs, **options)``\n    where ``kwargs`` corresponds to any other parameters passed to `minimize`\n    (such as `callback`, `hess`, etc.), except the `options` dict, which has\n    its contents also passed as `method` parameters pair by pair.  Also, if\n    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that\n    `fun` returns just the function values and `jac` is converted to a function\n    returning the Jacobian.  The method shall return an `OptimizeResult`\n    object.\n\n    The provided `method` callable must be able to accept (and possibly ignore)\n    arbitrary parameters; the set of parameters accepted by `minimize` may\n    expand in future versions and then these parameters will be passed to\n    the method.  You can find an example in the scipy.optimize tutorial.\n\n    References\n    ----------\n    .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function\n        Minimization. The Computer Journal 7: 308-13.\n    .. [2] Wright M H. 1996. Direct search methods: Once scorned, now\n        respectable, in Numerical Analysis 1995: Proceedings of the 1995\n        Dundee Biennial Conference in Numerical Analysis (Eds. D F\n        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.\n        191-208.\n    .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of\n       a function of several variables without calculating derivatives. The\n       Computer Journal 7: 155-162.\n    .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.\n       Numerical Recipes (any edition), Cambridge University Press.\n    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.\n       Springer New York.\n    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory\n       Algorithm for Bound Constrained Optimization. SIAM Journal on\n       Scientific and Statistical Computing 16 (5): 1190-1208.\n    .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm\n       778: L-BFGS-B, FORTRAN routines for large scale bound constrained\n       optimization. ACM Transactions on Mathematical Software 23 (4):\n       550-560.\n    .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.\n       1984. SIAM Journal of Numerical Analysis 21: 770-778.\n    .. [9] Powell, M J D. A direct search optimization method that models\n       the objective and constraint functions by linear interpolation.\n       1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez\n       and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.\n    .. [10] Powell M J D. Direct search algorithms for optimization\n       calculations. 1998. Acta Numerica 7: 287-336.\n    .. [11] Powell M J D. A view of algorithms for optimization without\n       derivatives. 2007.Cambridge University Technical Report DAMTP\n       2007/NA03\n    .. [12] Kraft, D. A software package for sequential quadratic\n       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace\n       Center -- Institute for Flight Mechanics, Koln, Germany.\n    .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.\n       Trust region methods. 2000. Siam. pp. 169-200.\n    .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free\n       implementation of the GLTR method for iterative solution of\n       the trust region problem", :arxiv:`1611.04718`\n    .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the\n       Trust-Region Subproblem using the Lanczos Method",\n       SIAM J. Optim., 9(2), 504--525, (1999).\n    .. [16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.\n        An interior point algorithm for large-scale nonlinear  programming.\n        SIAM Journal on Optimization 9.4: 877-900.\n    .. [17] Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the\n        implementation of an algorithm for large-scale equality constrained\n        optimization. SIAM Journal on Optimization 8.3: 682-706.\n\n    Examples\n    --------\n    Let us consider the problem of minimizing the Rosenbrock function. This\n    function (and its respective derivatives) is implemented in `rosen`\n    (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.\n\n    >>> from scipy.optimize import minimize, rosen, rosen_der\n\n    A simple application of the *Nelder-Mead* method is:\n\n    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]\n    >>> res = minimize(rosen, x0, method=\'Nelder-Mead\', tol=1e-6)\n    >>> res.x\n    array([ 1.,  1.,  1.,  1.,  1.])\n\n    Now using the *BFGS* algorithm, using the first derivative and a few\n    options:\n\n    >>> res = minimize(rosen, x0, method=\'BFGS\', jac=rosen_der,\n    ...                options={\'gtol\': 1e-6, \'disp\': True})\n    Optimization terminated successfully.\n             Current function value: 0.000000\n             Iterations: 26\n             Function evaluations: 31\n             Gradient evaluations: 31\n    >>> res.x\n    array([ 1.,  1.,  1.,  1.,  1.])\n    >>> print(res.message)\n    Optimization terminated successfully.\n    >>> res.hess_inv\n    array([[ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary\n           [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],\n           [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],\n           [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],\n           [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]])\n\n\n    Next, consider a minimization problem with several constraints (namely\n    Example 16.4 from [5]_). The objective function is:\n\n    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2\n\n    There are three constraints defined as:\n\n    >>> cons = ({\'type\': \'ineq\', \'fun\': lambda x:  x[0] - 2 * x[1] + 2},\n    ...         {\'type\': \'ineq\', \'fun\': lambda x: -x[0] - 2 * x[1] + 6},\n    ...         {\'type\': \'ineq\', \'fun\': lambda x: -x[0] + 2 * x[1] + 2})\n\n    And variables must be positive, hence the following bounds:\n\n    >>> bnds = ((0, None), (0, None))\n\n    The optimization problem is solved using the SLSQP method as:\n\n    >>> res = minimize(fun, (2, 0), method=\'SLSQP\', bounds=bnds,\n    ...                constraints=cons)\n\n    It should converge to the theoretical solution (1.4 ,1.7).\n\n    '
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")
    if x0.dtype.kind in np.typecodes['AllInteger']:
        x0 = np.asarray(x0, dtype=float)
    if not isinstance(args, tuple):
        args = (args,)
    if method is None:
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'
    if callable(method):
        meth = '_custom'
    else:
        meth = method.lower()
    if options is None:
        options = {}
    if meth in ('nelder-mead', 'powell', 'cobyla') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method, RuntimeWarning)
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr', 'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method, RuntimeWarning)
    if meth not in ('newton-cg', 'trust-ncg', 'trust-constr', 'trust-krylov', '_custom') and hessp is not None:
        warn('Method %s does not use Hessian-vector product information (hessp).' % method, RuntimeWarning)
    if meth not in ('cobyla', 'slsqp', 'trust-constr', '_custom') and np.any(constraints):
        warn('Method %s cannot handle constraints.' % method, RuntimeWarning)
    if meth not in ('nelder-mead', 'powell', 'l-bfgs-b', 'cobyla', 'slsqp', 'tnc', 'trust-constr', '_custom') and bounds is not None:
        warn('Method %s cannot handle bounds.' % method, RuntimeWarning)
    if meth in ('l-bfgs-b', 'tnc', 'cobyla', 'slsqp') and options.get('return_all', False):
        warn('Method %s does not support the return_all option.' % method, RuntimeWarning)
    if callable(jac):
        pass
    elif jac is True:
        fun = MemoizeJac(fun)
        jac = fun.derivative
    elif jac in FD_METHODS and meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']:
        pass
    elif meth in ['trust-constr']:
        jac = '2-point'
    elif jac is None or bool(jac) is False:
        jac = None
    else:
        jac = None
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)
    if meth == '_custom':
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, callback=callback, **options)
    constraints = standardize_constraints(constraints, x0, meth)
    remove_vars = False
    if bounds is not None:
        bounds = standardize_bounds(bounds, x0, 'new')
        bounds = _validate_bounds(bounds, x0, meth)
        if meth in {'tnc', 'slsqp', 'l-bfgs-b'}:
            i_fixed = bounds.lb == bounds.ub
            if np.all(i_fixed):
                return _optimize_result_for_equal_bounds(fun, bounds, meth, args=args, constraints=constraints)
            fd_needed = not callable(jac)
            for con in constraints:
                if not callable(con.get('jac', None)):
                    fd_needed = True
            remove_vars = i_fixed.any() and (fd_needed or meth == 'tnc')
            if remove_vars:
                x_fixed = bounds.lb[i_fixed]
                x0 = x0[~i_fixed]
                bounds = _remove_from_bounds(bounds, i_fixed)
                fun = _remove_from_func(fun, i_fixed, x_fixed)
                if callable(callback):
                    callback = _remove_from_func(callback, i_fixed, x_fixed)
                if callable(jac):
                    jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)
                constraints = [con.copy() for con in constraints]
                for con in constraints:
                    con['fun'] = _remove_from_func(con['fun'], i_fixed, x_fixed, min_dim=1, remove=0)
                    if callable(con.get('jac', None)):
                        con['jac'] = _remove_from_func(con['jac'], i_fixed, x_fixed, min_dim=2, remove=1)
        bounds = standardize_bounds(bounds, x0, meth)
    callback = _wrap_callback(callback, meth)
    if meth == 'nelder-mead':
        res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds, **options)
    elif meth == 'powell':
        res = _minimize_powell(fun, x0, args, callback, bounds, **options)
    elif meth == 'cg':
        res = _minimize_cg(fun, x0, args, jac, callback, **options)
    elif meth == 'bfgs':
        res = _minimize_bfgs(fun, x0, args, jac, callback, **options)
    elif meth == 'newton-cg':
        res = _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback, **options)
    elif meth == 'l-bfgs-b':
        res = _minimize_lbfgsb(fun, x0, args, jac, bounds, callback=callback, **options)
    elif meth == 'tnc':
        res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback, **options)
    elif meth == 'cobyla':
        res = _minimize_cobyla(fun, x0, args, constraints, callback=callback, bounds=bounds, **options)
    elif meth == 'slsqp':
        res = _minimize_slsqp(fun, x0, args, jac, bounds, constraints, callback=callback, **options)
    elif meth == 'trust-constr':
        res = _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp, bounds, constraints, callback=callback, **options)
    elif meth == 'dogleg':
        res = _minimize_dogleg(fun, x0, args, jac, hess, callback=callback, **options)
    elif meth == 'trust-ncg':
        res = _minimize_trust_ncg(fun, x0, args, jac, hess, hessp, callback=callback, **options)
    elif meth == 'trust-krylov':
        res = _minimize_trust_krylov(fun, x0, args, jac, hess, hessp, callback=callback, **options)
    elif meth == 'trust-exact':
        res = _minimize_trustregion_exact(fun, x0, args, jac, hess, callback=callback, **options)
    else:
        raise ValueError('Unknown solver %s' % method)
    if remove_vars:
        res.x = _add_to_array(res.x, i_fixed, x_fixed)
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        if 'hess_inv' in res:
            res.hess_inv = None
    if getattr(callback, 'stop_iteration', False):
        res.success = False
        res.status = 99
        res.message = '`callback` raised `StopIteration`.'
    return res

def minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None, tol=None, options=None):
    if False:
        print('Hello World!')
    'Local minimization of scalar function of one variable.\n\n    Parameters\n    ----------\n    fun : callable\n        Objective function.\n        Scalar function, must return a scalar.\n    bracket : sequence, optional\n        For methods \'brent\' and \'golden\', `bracket` defines the bracketing\n        interval and is required.\n        Either a triple ``(xa, xb, xc)`` satisfying ``xa < xb < xc`` and\n        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair\n        ``(xa, xb)`` to be used as initial points for a downhill bracket search\n        (see `scipy.optimize.bracket`).\n        The minimizer ``res.x`` will not necessarily satisfy\n        ``xa <= res.x <= xb``.\n    bounds : sequence, optional\n        For method \'bounded\', `bounds` is mandatory and must have two finite\n        items corresponding to the optimization bounds.\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    method : str or callable, optional\n        Type of solver.  Should be one of:\n\n            - :ref:`Brent <optimize.minimize_scalar-brent>`\n            - :ref:`Bounded <optimize.minimize_scalar-bounded>`\n            - :ref:`Golden <optimize.minimize_scalar-golden>`\n            - custom - a callable object (added in version 0.14.0), see below\n\n        Default is "Bounded" if bounds are provided and "Brent" otherwise.\n        See the \'Notes\' section for details of each solver.\n\n    tol : float, optional\n        Tolerance for termination. For detailed control, use solver-specific\n        options.\n    options : dict, optional\n        A dictionary of solver options.\n\n            maxiter : int\n                Maximum number of iterations to perform.\n            disp : bool\n                Set to True to print convergence messages.\n\n        See :func:`show_options()` for solver-specific options.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    See also\n    --------\n    minimize : Interface to minimization algorithms for scalar multivariate\n        functions\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    \'method\' parameter. The default method is the ``"Bounded"`` Brent method if\n    `bounds` are passed and unbounded ``"Brent"`` otherwise.\n\n    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent\'s\n    algorithm [1]_ to find a local minimum.  The algorithm uses inverse\n    parabolic interpolation when possible to speed up convergence of\n    the golden section method.\n\n    Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the\n    golden section search technique [1]_. It uses analog of the bisection\n    method to decrease the bracketed interval. It is usually\n    preferable to use the *Brent* method.\n\n    Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can\n    perform bounded minimization [2]_ [3]_. It uses the Brent method to find a\n    local minimum in the interval x1 < xopt < x2.\n\n    Note that the Brent and Golden methods do not guarantee success unless a\n    valid ``bracket`` triple is provided. If a three-point bracket cannot be\n    found, consider `scipy.optimize.minimize`. Also, all methods are intended\n    only for local minimization. When the function of interest has more than\n    one local minimum, consider :ref:`global_optimization`.\n\n    **Custom minimizers**\n\n    It may be useful to pass a custom minimization method, for example\n    when using some library frontend to minimize_scalar. You can simply\n    pass a callable as the ``method`` parameter.\n\n    The callable is called as ``method(fun, args, **kwargs, **options)``\n    where ``kwargs`` corresponds to any other parameters passed to `minimize`\n    (such as `bracket`, `tol`, etc.), except the `options` dict, which has\n    its contents also passed as `method` parameters pair by pair.  The method\n    shall return an `OptimizeResult` object.\n\n    The provided `method` callable must be able to accept (and possibly ignore)\n    arbitrary parameters; the set of parameters accepted by `minimize` may\n    expand in future versions and then these parameters will be passed to\n    the method. You can find an example in the scipy.optimize tutorial.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Press, W., S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery.\n           Numerical Recipes in C. Cambridge University Press.\n    .. [2] Forsythe, G.E., M. A. Malcolm, and C. B. Moler. "Computer Methods\n           for Mathematical Computations." Prentice-Hall Series in Automatic\n           Computation 259 (1977).\n    .. [3] Brent, Richard P. Algorithms for Minimization Without Derivatives.\n           Courier Corporation, 2013.\n\n    Examples\n    --------\n    Consider the problem of minimizing the following function.\n\n    >>> def f(x):\n    ...     return (x - 2) * x * (x + 2)**2\n\n    Using the *Brent* method, we find the local minimum as:\n\n    >>> from scipy.optimize import minimize_scalar\n    >>> res = minimize_scalar(f)\n    >>> res.fun\n    -9.9149495908\n\n    The minimizer is:\n\n    >>> res.x\n    1.28077640403\n\n    Using the *Bounded* method, we find a local minimum with specified\n    bounds as:\n\n    >>> res = minimize_scalar(f, bounds=(-3, -1), method=\'bounded\')\n    >>> res.fun  # minimum\n    3.28365179850e-13\n    >>> res.x  # minimizer\n    -2.0000002026\n\n    '
    if not isinstance(args, tuple):
        args = (args,)
    if callable(method):
        meth = '_custom'
    elif method is None:
        meth = 'brent' if bounds is None else 'bounded'
    else:
        meth = method.lower()
    if options is None:
        options = {}
    if bounds is not None and meth in {'brent', 'golden'}:
        message = f"Use of `bounds` is incompatible with 'method={method}'."
        raise ValueError(message)
    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.", RuntimeWarning)
            options['xatol'] = tol
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            options.setdefault('xtol', tol)
    disp = options.get('disp')
    if isinstance(disp, bool):
        options['disp'] = 2 * int(disp)
    if meth == '_custom':
        res = method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        res = _recover_from_bracket_error(_minimize_scalar_brent, fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for method `bounded`.')
        res = _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        res = _recover_from_bracket_error(_minimize_scalar_golden, fun, bracket, args, **options)
    else:
        raise ValueError('Unknown solver %s' % method)
    res.fun = np.asarray(res.fun)[()]
    res.x = np.reshape(res.x, res.fun.shape)[()]
    return res

def _remove_from_bounds(bounds, i_fixed):
    if False:
        for i in range(10):
            print('nop')
    'Removes fixed variables from a `Bounds` instance'
    lb = bounds.lb[~i_fixed]
    ub = bounds.ub[~i_fixed]
    return Bounds(lb, ub)

def _remove_from_func(fun_in, i_fixed, x_fixed, min_dim=None, remove=0):
    if False:
        i = 10
        return i + 15
    'Wraps a function such that fixed variables need not be passed in'

    def fun_out(x_in, *args, **kwargs):
        if False:
            while True:
                i = 10
        x_out = np.zeros_like(i_fixed, dtype=x_in.dtype)
        x_out[i_fixed] = x_fixed
        x_out[~i_fixed] = x_in
        y_out = fun_in(x_out, *args, **kwargs)
        y_out = np.array(y_out)
        if min_dim == 1:
            y_out = np.atleast_1d(y_out)
        elif min_dim == 2:
            y_out = np.atleast_2d(y_out)
        if remove == 1:
            y_out = y_out[..., ~i_fixed]
        elif remove == 2:
            y_out = y_out[~i_fixed, ~i_fixed]
        return y_out
    return fun_out

def _add_to_array(x_in, i_fixed, x_fixed):
    if False:
        print('Hello World!')
    'Adds fixed variables back to an array'
    i_free = ~i_fixed
    if x_in.ndim == 2:
        i_free = i_free[:, None] @ i_free[None, :]
    x_out = np.zeros_like(i_free, dtype=x_in.dtype)
    x_out[~i_free] = x_fixed
    x_out[i_free] = x_in.ravel()
    return x_out

def _validate_bounds(bounds, x0, meth):
    if False:
        while True:
            i = 10
    'Check that bounds are valid.'
    msg = 'An upper bound is less than the corresponding lower bound.'
    if np.any(bounds.ub < bounds.lb):
        raise ValueError(msg)
    msg = 'The number of bounds is not compatible with the length of `x0`.'
    try:
        bounds.lb = np.broadcast_to(bounds.lb, x0.shape)
        bounds.ub = np.broadcast_to(bounds.ub, x0.shape)
    except Exception as e:
        raise ValueError(msg) from e
    return bounds

def standardize_bounds(bounds, x0, meth):
    if False:
        for i in range(10):
            print('nop')
    'Converts bounds to the form required by the solver.'
    if meth in {'trust-constr', 'powell', 'nelder-mead', 'cobyla', 'new'}:
        if not isinstance(bounds, Bounds):
            (lb, ub) = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
    elif meth in ('l-bfgs-b', 'tnc', 'slsqp', 'old'):
        if isinstance(bounds, Bounds):
            bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
    return bounds

def standardize_constraints(constraints, x0, meth):
    if False:
        return 10
    'Converts constraints to the form required by the solver.'
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    new_constraint_types = all_constraint_types[:-1]
    if constraints is None:
        constraints = []
    elif isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    else:
        constraints = list(constraints)
    if meth in ['trust-constr', 'new']:
        for (i, con) in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)
    else:
        for (i, con) in enumerate(list(constraints)):
            if isinstance(con, new_constraint_types):
                old_constraints = new_constraint_to_old(con, x0)
                constraints[i] = old_constraints[0]
                constraints.extend(old_constraints[1:])
    return constraints

def _optimize_result_for_equal_bounds(fun, bounds, method, args=(), constraints=()):
    if False:
        while True:
            i = 10
    '\n    Provides a default OptimizeResult for when a bounded minimization method\n    has (lb == ub).all().\n\n    Parameters\n    ----------\n    fun: callable\n    bounds: Bounds\n    method: str\n    constraints: Constraint\n    '
    success = True
    message = 'All independent variables were fixed by bounds.'
    x0 = bounds.lb
    if constraints:
        message = 'All independent variables were fixed by bounds at values that satisfy the constraints.'
        constraints = standardize_constraints(constraints, x0, 'new')
    maxcv = 0
    for c in constraints:
        pc = PreparedConstraint(c, x0)
        violation = pc.violation(x0)
        if np.sum(violation):
            maxcv = max(maxcv, np.max(violation))
            success = False
            message = f'All independent variables were fixed by bounds, but the independent variables do not satisfy the constraints exactly. (Maximum violation: {maxcv}).'
    return OptimizeResult(x=x0, fun=fun(x0, *args), success=success, message=message, nfev=1, njev=0, nhev=0)