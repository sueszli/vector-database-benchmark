"""shgo: The simplicial homology global optimisation algorithm."""
from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
__all__ = ['shgo']

def shgo(func, bounds, args=(), constraints=None, n=100, iters=1, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial', *, workers=1):
    if False:
        return 10
    '\n    Finds the global minimum of a function using SHG optimization.\n\n    SHGO stands for "simplicial homology global optimization".\n\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized.  Must be in the form\n        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array\n        and ``args`` is a tuple of any additional fixed parameters needed to\n        completely specify the function.\n    bounds : sequence or `Bounds`\n        Bounds for variables. There are two ways to specify the bounds:\n\n        1. Instance of `Bounds` class.\n        2. Sequence of ``(min, max)`` pairs for each element in `x`.\n\n    args : tuple, optional\n        Any additional fixed parameters needed to completely specify the\n        objective function.\n    constraints : {Constraint, dict} or List of {Constraint, dict}, optional\n        Constraints definition. Only for COBYLA, SLSQP and trust-constr.\n        See the tutorial [5]_ for further details on specifying constraints.\n\n        .. note::\n\n           Only COBYLA, SLSQP, and trust-constr local minimize methods\n           currently support constraint arguments. If the ``constraints``\n           sequence used in the local optimization problem is not defined in\n           ``minimizer_kwargs`` and a constrained method is used then the\n           global ``constraints`` will be used.\n           (Defining a ``constraints`` sequence in ``minimizer_kwargs``\n           means that ``constraints`` will not be added so if equality\n           constraints and so forth need to be added then the inequality\n           functions in ``constraints`` need to be added to\n           ``minimizer_kwargs`` too).\n           COBYLA only supports inequality constraints.\n\n        .. versionchanged:: 1.11.0\n\n           ``constraints`` accepts `NonlinearConstraint`, `LinearConstraint`.\n\n    n : int, optional\n        Number of sampling points used in the construction of the simplicial\n        complex. For the default ``simplicial`` sampling method 2**dim + 1\n        sampling points are generated instead of the default `n=100`. For all\n        other specified values `n` sampling points are generated. For\n        ``sobol``, ``halton`` and other arbitrary `sampling_methods` `n=100` or\n        another specified number of sampling points are generated.\n    iters : int, optional\n        Number of iterations used in the construction of the simplicial\n        complex. Default is 1.\n    callback : callable, optional\n        Called after each iteration, as ``callback(xk)``, where ``xk`` is the\n        current parameter vector.\n    minimizer_kwargs : dict, optional\n        Extra keyword arguments to be passed to the minimizer\n        ``scipy.optimize.minimize`` Some important options could be:\n\n            * method : str\n                The minimization method. If not given, chosen to be one of\n                BFGS, L-BFGS-B, SLSQP, depending on whether or not the\n                problem has constraints or bounds.\n            * args : tuple\n                Extra arguments passed to the objective function (``func``) and\n                its derivatives (Jacobian, Hessian).\n            * options : dict, optional\n                Note that by default the tolerance is specified as\n                ``{ftol: 1e-12}``\n\n    options : dict, optional\n        A dictionary of solver options. Many of the options specified for the\n        global routine are also passed to the scipy.optimize.minimize routine.\n        The options that are also passed to the local routine are marked with\n        "(L)".\n\n        Stopping criteria, the algorithm will terminate if any of the specified\n        criteria are met. However, the default algorithm does not require any\n        to be specified:\n\n        * maxfev : int (L)\n            Maximum number of function evaluations in the feasible domain.\n            (Note only methods that support this option will terminate\n            the routine at precisely exact specified value. Otherwise the\n            criterion will only terminate during a global iteration)\n        * f_min\n            Specify the minimum objective function value, if it is known.\n        * f_tol : float\n            Precision goal for the value of f in the stopping\n            criterion. Note that the global routine will also\n            terminate if a sampling point in the global routine is\n            within this tolerance.\n        * maxiter : int\n            Maximum number of iterations to perform.\n        * maxev : int\n            Maximum number of sampling evaluations to perform (includes\n            searching in infeasible points).\n        * maxtime : float\n            Maximum processing runtime allowed\n        * minhgrd : int\n            Minimum homology group rank differential. The homology group of the\n            objective function is calculated (approximately) during every\n            iteration. The rank of this group has a one-to-one correspondence\n            with the number of locally convex subdomains in the objective\n            function (after adequate sampling points each of these subdomains\n            contain a unique global minimum). If the difference in the hgr is 0\n            between iterations for ``maxhgrd`` specified iterations the\n            algorithm will terminate.\n\n        Objective function knowledge:\n\n        * symmetry : list or bool\n            Specify if the objective function contains symmetric variables.\n            The search space (and therefore performance) is decreased by up to\n            O(n!) times in the fully symmetric case. If `True` is specified\n            then all variables will be set symmetric to the first variable.\n            Default\n            is set to False.\n\n            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2\n\n            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and\n            x_6 are symmetric to x_4, this can be specified to the solver as:\n\n            symmetry = [0,  # Variable 1\n                        0,  # symmetric to variable 1\n                        0,  # symmetric to variable 1\n                        3,  # Variable 4\n                        3,  # symmetric to variable 4\n                        3,  # symmetric to variable 4\n                        ]\n\n        * jac : bool or callable, optional\n            Jacobian (gradient) of objective function. Only for CG, BFGS,\n            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If ``jac`` is a\n            boolean and is True, ``fun`` is assumed to return the gradient\n            along with the objective function. If False, the gradient will be\n            estimated numerically. ``jac`` can also be a callable returning the\n            gradient of the objective. In this case, it must accept the same\n            arguments as ``fun``. (Passed to `scipy.optimize.minimize`\n            automatically)\n\n        * hess, hessp : callable, optional\n            Hessian (matrix of second-order derivatives) of objective function\n            or Hessian of objective function times an arbitrary vector p.\n            Only for Newton-CG, dogleg, trust-ncg. Only one of ``hessp`` or\n            ``hess`` needs to be given. If ``hess`` is provided, then\n            ``hessp`` will be ignored. If neither ``hess`` nor ``hessp`` is\n            provided, then the Hessian product will be approximated using\n            finite differences on ``jac``. ``hessp`` must compute the Hessian\n            times an arbitrary vector. (Passed to `scipy.optimize.minimize`\n            automatically)\n\n        Algorithm settings:\n\n        * minimize_every_iter : bool\n            If True then promising global sampling points will be passed to a\n            local minimization routine every iteration. If True then only the\n            final minimizer pool will be run. Defaults to True.\n        * local_iter : int\n            Only evaluate a few of the best minimizer pool candidates every\n            iteration. If False all potential points are passed to the local\n            minimization routine.\n        * infty_constraints : bool\n            If True then any sampling points generated which are outside will\n            the feasible domain will be saved and given an objective function\n            value of ``inf``. If False then these points will be discarded.\n            Using this functionality could lead to higher performance with\n            respect to function evaluations before the global minimum is found,\n            specifying False will use less memory at the cost of a slight\n            decrease in performance. Defaults to True.\n\n        Feedback:\n\n        * disp : bool (L)\n            Set to True to print convergence messages.\n\n    sampling_method : str or function, optional\n        Current built in sampling method options are ``halton``, ``sobol`` and\n        ``simplicial``. The default ``simplicial`` provides\n        the theoretical guarantee of convergence to the global minimum in\n        finite time. ``halton`` and ``sobol`` method are faster in terms of\n        sampling point generation at the cost of the loss of\n        guaranteed convergence. It is more appropriate for most "easier"\n        problems where the convergence is relatively fast.\n        User defined sampling functions must accept two arguments of ``n``\n        sampling points of dimension ``dim`` per call and output an array of\n        sampling points with shape `n x dim`.\n\n    workers : int or map-like callable, optional\n        Sample and run the local serial minimizations in parallel.\n        Supply -1 to use all available CPU cores, or an int to use\n        that many Processes (uses `multiprocessing.Pool <multiprocessing>`).\n\n        Alternatively supply a map-like callable, such as\n        `multiprocessing.Pool.map` for parallel evaluation.\n        This evaluation is carried out as ``workers(func, iterable)``.\n        Requires that `func` be pickleable.\n\n        .. versionadded:: 1.11.0\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a `OptimizeResult` object.\n        Important attributes are:\n        ``x`` the solution array corresponding to the global minimum,\n        ``fun`` the function output at the global solution,\n        ``xl`` an ordered list of local minima solutions,\n        ``funl`` the function output at the corresponding local solutions,\n        ``success`` a Boolean flag indicating if the optimizer exited\n        successfully,\n        ``message`` which describes the cause of the termination,\n        ``nfev`` the total number of objective function evaluations including\n        the sampling calls,\n        ``nlfev`` the total number of objective function evaluations\n        culminating from all local search optimizations,\n        ``nit`` number of iterations performed by the global routine.\n\n    Notes\n    -----\n    Global optimization using simplicial homology global optimization [1]_.\n    Appropriate for solving general purpose NLP and blackbox optimization\n    problems to global optimality (low-dimensional problems).\n\n    In general, the optimization problems are of the form::\n\n        minimize f(x) subject to\n\n        g_i(x) >= 0,  i = 1,...,m\n        h_j(x)  = 0,  j = 1,...,p\n\n    where x is a vector of one or more variables. ``f(x)`` is the objective\n    function ``R^n -> R``, ``g_i(x)`` are the inequality constraints, and\n    ``h_j(x)`` are the equality constraints.\n\n    Optionally, the lower and upper bounds for each element in x can also be\n    specified using the `bounds` argument.\n\n    While most of the theoretical advantages of SHGO are only proven for when\n    ``f(x)`` is a Lipschitz smooth function, the algorithm is also proven to\n    converge to the global optimum for the more general case where ``f(x)`` is\n    non-continuous, non-convex and non-smooth, if the default sampling method\n    is used [1]_.\n\n    The local search method may be specified using the ``minimizer_kwargs``\n    parameter which is passed on to ``scipy.optimize.minimize``. By default,\n    the ``SLSQP`` method is used. In general, it is recommended to use the\n    ``SLSQP`` or ``COBYLA`` local minimization if inequality constraints\n    are defined for the problem since the other methods do not use constraints.\n\n    The ``halton`` and ``sobol`` method points are generated using\n    `scipy.stats.qmc`. Any other QMC method could be used.\n\n    References\n    ----------\n    .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology\n           algorithm for lipschitz optimisation", Journal of Global\n           Optimization.\n    .. [2] Joe, SW and Kuo, FY (2008) "Constructing Sobol\' sequences with\n           better  two-dimensional projections", SIAM J. Sci. Comput. 30,\n           2635-2654.\n    .. [3] Hock, W and Schittkowski, K (1981) "Test examples for nonlinear\n           programming codes", Lecture Notes in Economics and Mathematical\n           Systems, 187. Springer-Verlag, New York.\n           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf\n    .. [4] Wales, DJ (2015) "Perspective: Insight into reaction coordinates and\n           dynamics from the potential energy landscape",\n           Journal of Chemical Physics, 142(13), 2015.\n    .. [5] https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize\n\n    Examples\n    --------\n    First consider the problem of minimizing the Rosenbrock function, `rosen`:\n\n    >>> from scipy.optimize import rosen, shgo\n    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]\n    >>> result = shgo(rosen, bounds)\n    >>> result.x, result.fun\n    (array([1., 1., 1., 1., 1.]), 2.920392374190081e-18)\n\n    Note that bounds determine the dimensionality of the objective\n    function and is therefore a required input, however you can specify\n    empty bounds using ``None`` or objects like ``np.inf`` which will be\n    converted to large float numbers.\n\n    >>> bounds = [(None, None), ]*4\n    >>> result = shgo(rosen, bounds)\n    >>> result.x\n    array([0.99999851, 0.99999704, 0.99999411, 0.9999882 ])\n\n    Next, we consider the Eggholder function, a problem with several local\n    minima and one global minimum. We will demonstrate the use of arguments and\n    the capabilities of `shgo`.\n    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)\n\n    >>> import numpy as np\n    >>> def eggholder(x):\n    ...     return (-(x[1] + 47.0)\n    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))\n    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))\n    ...             )\n    ...\n    >>> bounds = [(-512, 512), (-512, 512)]\n\n    `shgo` has built-in low discrepancy sampling sequences. First, we will\n    input 64 initial sampling points of the *Sobol\'* sequence:\n\n    >>> result = shgo(eggholder, bounds, n=64, sampling_method=\'sobol\')\n    >>> result.x, result.fun\n    (array([512.        , 404.23180824]), -959.6406627208397)\n\n    `shgo` also has a return for any other local minima that was found, these\n    can be called using:\n\n    >>> result.xl\n    array([[ 512.        ,  404.23180824],\n           [ 283.0759062 , -487.12565635],\n           [-294.66820039, -462.01964031],\n           [-105.87688911,  423.15323845],\n           [-242.97926   ,  274.38030925],\n           [-506.25823477,    6.3131022 ],\n           [-408.71980731, -156.10116949],\n           [ 150.23207937,  301.31376595],\n           [  91.00920901, -391.283763  ],\n           [ 202.89662724, -269.38043241],\n           [ 361.66623976, -106.96493868],\n           [-219.40612786, -244.06020508]])\n\n    >>> result.funl\n    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097,\n           -559.78685655, -557.36868733, -507.87385942, -493.9605115 ,\n           -426.48799655, -421.15571437, -419.31194957, -410.98477763])\n\n    These results are useful in applications where there are many global minima\n    and the values of other global minima are desired or where the local minima\n    can provide insight into the system (for example morphologies\n    in physical chemistry [4]_).\n\n    If we want to find a larger number of local minima, we can increase the\n    number of sampling points or the number of iterations. We\'ll increase the\n    number of sampling points to 64 and the number of iterations from the\n    default of 1 to 3. Using ``simplicial`` this would have given us\n    64 x 3 = 192 initial sampling points.\n\n    >>> result_2 = shgo(eggholder,\n    ...                 bounds, n=64, iters=3, sampling_method=\'sobol\')\n    >>> len(result.xl), len(result_2.xl)\n    (12, 23)\n\n    Note the difference between, e.g., ``n=192, iters=1`` and ``n=64,\n    iters=3``.\n    In the first case the promising points contained in the minimiser pool\n    are processed only once. In the latter case it is processed every 64\n    sampling points for a total of 3 times.\n\n    To demonstrate solving problems with non-linear constraints consider the\n    following example from Hock and Schittkowski problem 73 (cattle-feed)\n    [3]_::\n\n        minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4\n\n        subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5    >= 0,\n\n                    12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21\n                        -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +\n                                      20.5 * x_3**2 + 0.62 * x_4**2)      >= 0,\n\n                    x_1 + x_2 + x_3 + x_4 - 1                             == 0,\n\n                    1 >= x_i >= 0 for all i\n\n    The approximate answer given in [3]_ is::\n\n        f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) = 29.894378\n\n    >>> def f(x):  # (cattle-feed)\n    ...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]\n    ...\n    >>> def g1(x):\n    ...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0\n    ...\n    >>> def g2(x):\n    ...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21\n    ...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2\n    ...                             + 20.5*x[2]**2 + 0.62*x[3]**2)\n    ...             ) # >=0\n    ...\n    >>> def h1(x):\n    ...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0\n    ...\n    >>> cons = ({\'type\': \'ineq\', \'fun\': g1},\n    ...         {\'type\': \'ineq\', \'fun\': g2},\n    ...         {\'type\': \'eq\', \'fun\': h1})\n    >>> bounds = [(0, 1.0),]*4\n    >>> res = shgo(f, bounds, n=150, constraints=cons)\n    >>> res\n     message: Optimization terminated successfully.\n     success: True\n         fun: 29.894378159142136\n        funl: [ 2.989e+01]\n           x: [ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]\n          xl: [[ 6.355e-01  1.137e-13  3.127e-01  5.178e-02]]\n         nit: 1\n        nfev: 142\n       nlfev: 35\n       nljev: 5\n       nlhev: 0\n\n    >>> g1(res.x), g2(res.x), h1(res.x)\n    (-5.062616992290714e-14, -2.9594104944408173e-12, 0.0)\n\n    '
    if isinstance(bounds, Bounds):
        bounds = new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb))
    with SHGO(func, bounds, args=args, constraints=constraints, n=n, iters=iters, callback=callback, minimizer_kwargs=minimizer_kwargs, options=options, sampling_method=sampling_method, workers=workers) as shc:
        shc.iterate_all()
    if not shc.break_routine:
        if shc.disp:
            logging.info('Successfully completed construction of complex.')
    if len(shc.LMC.xl_maps) == 0:
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.fail_routine(mes='Failed to find a feasible minimizer point. Lowest sampling point = {}'.format(shc.f_lowest))
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn
        shc.res.tnev = shc.n_sampled
    else:
        pass
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True
    return shc.res

class SHGO:

    def __init__(self, func, bounds, args=(), constraints=None, n=None, iters=None, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial', workers=1):
        if False:
            print('Hello World!')
        from scipy.stats import qmc
        methods = ['halton', 'sobol', 'simplicial']
        if isinstance(sampling_method, str) and sampling_method not in methods:
            raise ValueError('Unknown sampling_method specified. Valid methods: {}'.format(', '.join(methods)))
        try:
            if minimizer_kwargs['jac'] is True and (not callable(minimizer_kwargs['jac'])):
                self.func = MemoizeJac(func)
                jac = self.func.derivative
                minimizer_kwargs['jac'] = jac
                func = self.func
            else:
                self.func = func
        except (TypeError, KeyError):
            self.func = func
        self.func = _FunctionWrapper(func, args)
        self.bounds = bounds
        self.args = args
        self.callback = callback
        abound = np.array(bounds, float)
        self.dim = np.shape(abound)[0]
        infind = ~np.isfinite(abound)
        abound[infind[:, 0], 0] = -1e+50
        abound[infind[:, 1], 1] = 1e+50
        bnderr = abound[:, 0] > abound[:, 1]
        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds {}.'.format(', '.join((str(b) for b in bnderr))))
        self.bounds = abound
        self.constraints = constraints
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []
            self.constraints = standardize_constraints(constraints, np.empty(self.dim, float), 'old')
            for cons in self.constraints:
                if cons['type'] in 'ineq':
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None
        self.minimizer_kwargs = {'method': 'SLSQP', 'bounds': self.bounds, 'options': {}, 'callback': self.callback}
        if minimizer_kwargs is not None:
            self.minimizer_kwargs.update(minimizer_kwargs)
        else:
            self.minimizer_kwargs['options'] = {'ftol': 1e-12}
        if self.minimizer_kwargs['method'].lower() in ('slsqp', 'cobyla', 'trust-constr') and (minimizer_kwargs is not None and 'constraints' not in minimizer_kwargs and (constraints is not None)) or self.g_cons is not None:
            self.minimizer_kwargs['constraints'] = self.min_cons
        if options is not None:
            self.init_options(options)
        else:
            self.f_min_true = None
            self.minimize_every_iter = True
            self.maxiter = None
            self.maxfev = None
            self.maxev = None
            self.maxtime = None
            self.f_min_true = None
            self.minhgrd = None
            self.symmetry = None
            self.infty_cons_sampl = True
            self.local_iter = False
            self.disp = False
        self.min_solver_args = ['fun', 'x0', 'args', 'callback', 'options', 'method']
        solver_args = {'_custom': ['jac', 'hess', 'hessp', 'bounds', 'constraints'], 'nelder-mead': [], 'powell': [], 'cg': ['jac'], 'bfgs': ['jac'], 'newton-cg': ['jac', 'hess', 'hessp'], 'l-bfgs-b': ['jac', 'bounds'], 'tnc': ['jac', 'bounds'], 'cobyla': ['constraints', 'catol'], 'slsqp': ['jac', 'bounds', 'constraints'], 'dogleg': ['jac', 'hess'], 'trust-ncg': ['jac', 'hess', 'hessp'], 'trust-krylov': ['jac', 'hess', 'hessp'], 'trust-exact': ['jac', 'hess'], 'trust-constr': ['jac', 'hess', 'hessp', 'constraints']}
        method = self.minimizer_kwargs['method']
        self.min_solver_args += solver_args[method.lower()]

        def _restrict_to_keys(dictionary, goodkeys):
            if False:
                while True:
                    i = 10
            'Remove keys from dictionary if not in goodkeys - inplace'
            existingkeys = set(dictionary)
            for key in existingkeys - set(goodkeys):
                dictionary.pop(key, None)
        _restrict_to_keys(self.minimizer_kwargs, self.min_solver_args)
        _restrict_to_keys(self.minimizer_kwargs['options'], self.min_solver_args + ['ftol'])
        self.stop_global = False
        self.break_routine = False
        self.iters = iters
        self.iters_done = 0
        self.n = n
        self.nc = 0
        self.n_prc = 0
        self.n_sampled = 0
        self.fn = 0
        self.hgr = 0
        self.qhull_incremental = True
        if self.n is None and self.iters is None and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1
            self.nc = 0
        if self.iters is None:
            self.iters = 1
        if self.n is None and (not sampling_method == 'simplicial'):
            self.n = self.n = 100
            self.nc = 0
        if self.n == 100 and sampling_method == 'simplicial':
            self.n = 2 ** self.dim + 1
        if not (self.maxiter is None and self.maxfev is None and (self.maxev is None) and (self.minhgrd is None) and (self.f_min_true is None)):
            self.iters = None
        self.HC = Complex(dim=self.dim, domain=self.bounds, sfield=self.func, sfield_args=(), symmetry=self.symmetry, constraints=self.constraints, workers=workers)
        if sampling_method == 'simplicial':
            self.iterate_complex = self.iterate_hypercube
            self.sampling_method = sampling_method
        elif sampling_method in ['halton', 'sobol'] or not isinstance(sampling_method, str):
            self.iterate_complex = self.iterate_delaunay
            if sampling_method in ['halton', 'sobol']:
                if sampling_method == 'sobol':
                    self.n = int(2 ** np.ceil(np.log2(self.n)))
                    self.nc = 0
                    self.sampling_method = 'sobol'
                    self.qmc_engine = qmc.Sobol(d=self.dim, scramble=False, seed=0)
                else:
                    self.sampling_method = 'halton'
                    self.qmc_engine = qmc.Halton(d=self.dim, scramble=True, seed=0)

                def sampling_method(n, d):
                    if False:
                        return 10
                    return self.qmc_engine.random(n)
            else:
                self.sampling_method = 'custom'
            self.sampling = self.sampling_custom
            self.sampling_function = sampling_method
        self.stop_l_iter = False
        self.stop_complex_iter = False
        self.minimizer_pool = []
        self.LMC = LMapCache()
        self.res = OptimizeResult()
        self.res.nfev = 0
        self.res.nlfev = 0
        self.res.nljev = 0
        self.res.nlhev = 0

    def init_options(self, options):
        if False:
            while True:
                i = 10
        '\n        Initiates the options.\n\n        Can also be useful to change parameters after class initiation.\n\n        Parameters\n        ----------\n        options : dict\n\n        Returns\n        -------\n        None\n\n        '
        self.minimizer_kwargs['options'].update(options)
        for opt in ['jac', 'hess', 'hessp']:
            if opt in self.minimizer_kwargs['options']:
                self.minimizer_kwargs[opt] = self.minimizer_kwargs['options'].pop(opt)
        self.minimize_every_iter = options.get('minimize_every_iter', True)
        self.maxiter = options.get('maxiter', None)
        self.maxfev = options.get('maxfev', None)
        self.maxev = options.get('maxev', None)
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)
        if 'f_min' in options:
            self.f_min_true = options['f_min']
            self.f_tol = options.get('f_tol', 0.0001)
        else:
            self.f_min_true = None
        self.minhgrd = options.get('minhgrd', None)
        self.symmetry = options.get('symmetry', False)
        if self.symmetry:
            self.symmetry = [0] * len(self.bounds)
        else:
            self.symmetry = None
        self.local_iter = options.get('local_iter', False)
        self.infty_cons_sampl = options.get('infty_constraints', True)
        self.disp = options.get('disp', False)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        return self.HC.V._mapwrapper.__exit__(*args)

    def iterate_all(self):
        if False:
            while True:
                i = 10
        "\n        Construct for `iters` iterations.\n\n        If uniform sampling is used, every iteration adds 'n' sampling points.\n\n        Iterations if a stopping criteria (e.g., sampling points or\n        processing time) has been met.\n\n        "
        if self.disp:
            logging.info('Splitting first generation')
        while not self.stop_global:
            if self.break_routine:
                break
            self.iterate()
            self.stopping_criteria()
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.res.nit = self.iters_done
        self.fn = self.HC.V.nfev

    def find_minima(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct the minimizer pool, map the minimizers to local minima\n        and sort the results into a global return object.\n        '
        if self.disp:
            logging.info('Searching for minimizer pool...')
        self.minimizers()
        if len(self.X_min) != 0:
            self.minimise_pool(self.local_iter)
            self.sort_result()
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Minimiser pool = SHGO.X_min = {self.X_min}')

    def find_lowest_vertex(self):
        if False:
            i = 10
            return i + 15
        self.f_lowest = np.inf
        for x in self.HC.V.cache:
            if self.HC.V[x].f < self.f_lowest:
                if self.disp:
                    logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
                self.f_lowest = self.HC.V[x].f
                self.x_lowest = self.HC.V[x].x_a
        for lmc in self.LMC.cache:
            if self.LMC[lmc].f_min < self.f_lowest:
                self.f_lowest = self.LMC[lmc].f_min
                self.x_lowest = self.LMC[lmc].x_l
        if self.f_lowest == np.inf:
            self.f_lowest = None
            self.x_lowest = None

    def finite_iterations(self):
        if False:
            i = 10
            return i + 15
        mi = min((x for x in [self.iters, self.maxiter] if x is not None))
        if self.disp:
            logging.info(f'Iterations done = {self.iters_done} / {mi}')
        if self.iters is not None:
            if self.iters_done >= self.iters:
                self.stop_global = True
        if self.maxiter is not None:
            if self.iters_done >= self.maxiter:
                self.stop_global = True
        return self.stop_global

    def finite_fev(self):
        if False:
            i = 10
            return i + 15
        if self.disp:
            logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
        if self.fn >= self.maxfev:
            self.stop_global = True
        return self.stop_global

    def finite_ev(self):
        if False:
            while True:
                i = 10
        if self.disp:
            logging.info(f'Sampling evaluations done = {self.n_sampled} / {self.maxev}')
        if self.n_sampled >= self.maxev:
            self.stop_global = True

    def finite_time(self):
        if False:
            print('Hello World!')
        if self.disp:
            logging.info(f'Time elapsed = {time.time() - self.init} / {self.maxtime}')
        if time.time() - self.init >= self.maxtime:
            self.stop_global = True

    def finite_precision(self):
        if False:
            i = 10
            return i + 15
        "\n        Stop the algorithm if the final function value is known\n\n        Specify in options (with ``self.f_min_true = options['f_min']``)\n        and the tolerance with ``f_tol = options['f_tol']``\n        "
        self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Lowest function evaluation = {self.f_lowest}')
            logging.info(f'Specified minimum = {self.f_min_true}')
        if self.f_lowest is None:
            return self.stop_global
        if self.f_min_true == 0.0:
            if self.f_lowest <= self.f_tol:
                self.stop_global = True
        else:
            pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
            if self.f_lowest <= self.f_min_true:
                self.stop_global = True
                if abs(pe) >= 2 * self.f_tol:
                    warnings.warn('A much lower value than expected f* =' + f' {self.f_min_true} than' + ' the was found f_lowest =' + f'{self.f_lowest} ')
            if pe <= self.f_tol:
                self.stop_global = True
        return self.stop_global

    def finite_homology_growth(self):
        if False:
            print('Hello World!')
        '\n        Stop the algorithm if homology group rank did not grow in iteration.\n        '
        if self.LMC.size == 0:
            return
        self.hgrd = self.LMC.size - self.hgr
        self.hgr = self.LMC.size
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        if self.disp:
            logging.info(f'Current homology growth = {self.hgrd}  (minimum growth = {self.minhgrd})')
        return self.stop_global

    def stopping_criteria(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Various stopping criteria ran every iteration\n\n        Returns\n        -------\n        stop : bool\n        '
        if self.maxiter is not None:
            self.finite_iterations()
        if self.iters is not None:
            self.finite_iterations()
        if self.maxfev is not None:
            self.finite_fev()
        if self.maxev is not None:
            self.finite_ev()
        if self.maxtime is not None:
            self.finite_time()
        if self.f_min_true is not None:
            self.finite_precision()
        if self.minhgrd is not None:
            self.finite_homology_growth()
        return self.stop_global

    def iterate(self):
        if False:
            return 10
        self.iterate_complex()
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.iters_done += 1

    def iterate_hypercube(self):
        if False:
            while True:
                i = 10
        '\n        Iterate a subdivision of the complex\n\n        Note: called with ``self.iterate_complex()`` after class initiation\n        '
        if self.disp:
            logging.info('Constructing and refining simplicial complex graph structure')
        if self.n is None:
            self.HC.refine_all()
            self.n_sampled = self.HC.V.size()
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints and objective function values.')
        if len(self.LMC.xl_maps) > 0:
            for xl in self.LMC.cache:
                v = self.HC.V[xl]
                v_near = v.star()
                for v in v.nn:
                    v_near = v_near.union(v.nn)
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')
        self.fn = self.HC.V.nfev
        return

    def iterate_delaunay(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a complex of Delaunay triangulated points\n\n        Note: called with ``self.iterate_complex()`` after class initiation\n        '
        self.nc += self.n
        self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)
        if self.disp:
            logging.info(f'self.n = {self.n}')
            logging.info(f'self.nc = {self.nc}')
            logging.info('Constructing and refining simplicial complex graph structure from sampling points.')
        if self.dim < 2:
            self.Ind_sorted = np.argsort(self.C, axis=0)
            self.Ind_sorted = self.Ind_sorted.flatten()
            tris = []
            for (ind, ind_s) in enumerate(self.Ind_sorted):
                if ind > 0:
                    tris.append(self.Ind_sorted[ind - 1:ind + 1])
            tris = np.array(tris)
            self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
            self.points = {}
        else:
            if self.C.shape[0] > self.dim + 1:
                self.delaunay_triangulation(n_prc=self.n_prc)
            self.n_prc = self.C.shape[0]
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints and objective function values.')
        if hasattr(self, 'Tri'):
            self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)
        if self.disp:
            logging.info('Triangulation completed, evaluating all contraints and objective function values.')
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')
        self.fn = self.HC.V.nfev
        self.n_sampled = self.nc
        return

    def minimizers(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the indexes of all minimizers\n        '
        self.minimizer_pool = []
        for x in self.HC.V.cache:
            in_LMC = False
            if len(self.LMC.xl_maps) > 0:
                for xlmi in self.LMC.xl_maps:
                    if np.all(np.array(x) == np.array(xlmi)):
                        in_LMC = True
            if in_LMC:
                continue
            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                    logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                    logging.info('=' * 30)
                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])
                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info(f'x = {vn.x} || f = {vn.f}')
                    logging.info('=' * 60)
        self.minimizer_pool_F = []
        self.X_min = []
        self.X_min_cache = {}
        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            self.X_min_cache[tuple(v.x_a)] = v.x
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        self.X_min = np.array(self.X_min)
        self.sort_min_pool()
        return self.X_min

    def minimise_pool(self, force_iter=False):
        if False:
            while True:
                i = 10
        '\n        This processing method can optionally minimise only the best candidate\n        solutions in the minimiser pool\n\n        Parameters\n        ----------\n        force_iter : int\n                     Number of starting minimizers to process (can be specified\n                     globally or locally)\n\n        '
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])
        self.trim_min_pool(0)
        while not self.stop_l_iter:
            self.stopping_criteria()
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break
            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break
            self.g_topograph(lres_f_min.x, self.X_min)
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])
            self.trim_min_pool(ind_xmin_l)
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        if False:
            for i in range(10):
                print('nop')
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        if False:
            for i in range(10):
                print('nop')
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        if False:
            while True:
                i = 10
        '\n        Returns the topographical vector stemming from the specified value\n        ``x_min`` for the current feasible set ``X_min`` with True boolean\n        values indicating positive entries and False values indicating\n        negative entries.\n\n        '
        x_min = np.array([x_min])
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        self.Z = np.argsort(self.Y, axis=-1)
        self.Ss = X_min[self.Z][0]
        self.minimizer_pool = self.minimizer_pool[self.Z]
        self.minimizer_pool = self.minimizer_pool[0]
        return self.Ss

    def construct_lcb_simplicial(self, v_min):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct locally (approximately) convex bounds\n\n        Parameters\n        ----------\n        v_min : Vertex object\n                The minimizer vertex\n\n        Returns\n        -------\n        cbounds : list of lists\n            List of size dimension with length-2 list of bounds for each\n            dimension.\n\n        '
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        for vn in v_min.nn:
            for (i, x_i) in enumerate(vn.x_a):
                if x_i < v_min.x_a[i] and x_i > cbounds[i][0]:
                    cbounds[i][0] = x_i
                if x_i > v_min.x_a[i] and x_i < cbounds[i][1]:
                    cbounds[i][1] = x_i
        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a}')
            logging.info(f'cbounds = {cbounds}')
        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        if False:
            while True:
                i = 10
        '\n        Construct locally (approximately) convex bounds\n\n        Parameters\n        ----------\n        v_min : Vertex object\n                The minimizer vertex\n\n        Returns\n        -------\n        cbounds : list of lists\n            List of size dimension with length-2 list of bounds for each\n            dimension.\n        '
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        return cbounds

    def minimize(self, x_min, ind=None):
        if False:
            i = 10
            return i + 15
        '\n        This function is used to calculate the local minima using the specified\n        sampling point as a starting value.\n\n        Parameters\n        ----------\n        x_min : vector of floats\n            Current starting point to minimize.\n\n        Returns\n        -------\n        lres : OptimizeResult\n            The local optimization result represented as a `OptimizeResult`\n            object.\n        '
        if self.disp:
            logging.info(f'Vertex minimiser maps = {self.LMC.v_maps}')
        if self.LMC[x_min].lres is not None:
            logging.info(f'Found self.LMC[x_min].lres = {self.LMC[x_min].lres}')
            return self.LMC[x_min].lres
        if self.callback is not None:
            logging.info('Callback for minimizer starting at {}:'.format(x_min))
        if self.disp:
            logging.info('Starting minimization at {}...'.format(x_min))
        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]
            x_min_t_norm = tuple(x_min_t_norm)
            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])
        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])
        if self.disp and 'bounds' in self.minimizer_kwargs:
            logging.info('bounds in kwarg:')
            logging.info(self.minimizer_kwargs['bounds'])
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)
        if self.disp:
            logging.info(f'lres = {lres}')
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev
        try:
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)
        return lres

    def sort_result(self):
        if False:
            i = 10
            return i + 15
        '\n        Sort results and build the global return object\n        '
        results = self.LMC.sort_cache_result()
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']
        self.res.nfev = self.fn + self.res.nlfev
        return self.res

    def fail_routine(self, mes='Failed to converge'):
        if False:
            return 10
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        if False:
            print('Hello World!')
        '\n        Sample the function surface.\n\n        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled\n        points that are generated outside the feasible domain will be\n        assigned an ``inf`` value in accordance with SHGO rules.\n        This guarantees convergence and usually requires less objective\n        function evaluations at the computational costs of more Delaunay\n        triangulation points.\n\n        If ``infty_cons_sampl`` is False, then the infeasible points are\n        discarded and only a subspace of the sampled points are used. This\n        comes at the cost of the loss of guaranteed convergence and usually\n        requires more objective function evaluations.\n        '
        if self.disp:
            logging.info('Generating sampling points')
        self.sampling(self.nc, self.dim)
        if len(self.LMC.xl_maps) > 0:
            self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
        if not infty_cons_sampl:
            if self.g_cons is not None:
                self.sampling_subspace()
        self.sorted_samples()
        self.n_sampled = self.nc

    def sampling_custom(self, n, dim):
        if False:
            while True:
                i = 10
        '\n        Generates uniform sampling points in a hypercube and scales the points\n        to the bound limits.\n        '
        if self.n_sampled == 0:
            self.C = self.sampling_function(n, dim)
        else:
            self.C = self.sampling_function(n, dim)
        for i in range(len(self.bounds)):
            self.C[:, i] = self.C[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
        return self.C

    def sampling_subspace(self):
        if False:
            while True:
                i = 10
        'Find subspace of feasible points from g_func definition'
        for (ind, g) in enumerate(self.g_cons):
            feasible = np.array([np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C], dtype=bool)
            self.C = self.C[feasible]
            if self.C.size == 0:
                self.res.message = 'No sampling point found within the ' + 'feasible set. Increasing sampling ' + 'size.'
                if self.disp:
                    logging.info(self.res.message)

    def sorted_samples(self):
        if False:
            for i in range(10):
                print('nop')
        'Find indexes of the sorted sampling points'
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return (self.Ind_sorted, self.Xs)

    def delaunay_triangulation(self, n_prc=0):
        if False:
            print('Hello World!')
        if hasattr(self, 'Tri') and self.qhull_incremental:
            self.Tri.add_points(self.C[n_prc:, :])
        else:
            try:
                self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
            except spatial.QhullError:
                if str(sys.exc_info()[1])[:6] == 'QH6239':
                    logging.warning('QH6239 Qhull precision error detected, this usually occurs when no bounds are specified, Qhull can only run with handling cocircular/cospherical points and in this case incremental mode is switched off. The performance of shgo will be reduced in this mode.')
                    self.qhull_incremental = False
                    self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
                else:
                    raise
        return self.Tri

class LMap:

    def __init__(self, v):
        if False:
            i = 10
            return i + 15
        self.v = v
        self.x_l = None
        self.lres = None
        self.f_min = None
        self.lbounds = []

class LMapCache:

    def __init__(self):
        if False:
            return 10
        self.cache = {}
        self.v_maps = []
        self.xl_maps = []
        self.xl_maps_set = set()
        self.f_maps = []
        self.lbound_maps = []
        self.size = 0

    def __getitem__(self, v):
        if False:
            for i in range(10):
                print('nop')
        try:
            v = np.ndarray.tolist(v)
        except TypeError:
            pass
        v = tuple(v)
        try:
            return self.cache[v]
        except KeyError:
            xval = LMap(v)
            self.cache[v] = xval
            return self.cache[v]

    def add_res(self, v, lres, bounds=None):
        if False:
            print('Hello World!')
        v = np.ndarray.tolist(v)
        v = tuple(v)
        self.cache[v].x_l = lres.x
        self.cache[v].lres = lres
        self.cache[v].f_min = lres.fun
        self.cache[v].lbounds = bounds
        self.size += 1
        self.v_maps.append(v)
        self.xl_maps.append(lres.x)
        self.xl_maps_set.add(tuple(lres.x))
        self.f_maps.append(lres.fun)
        self.lbound_maps.append(bounds)

    def sort_cache_result(self):
        if False:
            i = 10
            return i + 15
        '\n        Sort results and build the global return object\n        '
        results = {}
        self.xl_maps = np.array(self.xl_maps)
        self.f_maps = np.array(self.f_maps)
        ind_sorted = np.argsort(self.f_maps)
        results['xl'] = self.xl_maps[ind_sorted]
        self.f_maps = np.array(self.f_maps)
        results['funl'] = self.f_maps[ind_sorted]
        results['funl'] = results['funl'].T
        results['x'] = self.xl_maps[ind_sorted[0]]
        results['fun'] = self.f_maps[ind_sorted[0]]
        self.xl_maps = np.ndarray.tolist(self.xl_maps)
        self.f_maps = np.ndarray.tolist(self.f_maps)
        return results