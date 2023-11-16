"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import Bounds, new_bounds_to_old, NonlinearConstraint, LinearConstraint
from scipy.sparse import issparse
__all__ = ['differential_evolution']
_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, *, integrality=None, vectorized=False):
    if False:
        i = 10
        return i + 15
    "Finds the global minimum of a multivariate function.\n\n    The differential evolution method [1]_ is stochastic in nature. It does\n    not use gradient methods to find the minimum, and can search large areas\n    of candidate space, but often requires larger numbers of function\n    evaluations than conventional gradient-based techniques.\n\n    The algorithm is due to Storn and Price [2]_.\n\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized. Must be in the form\n        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array\n        and ``args`` is a tuple of any additional fixed parameters needed to\n        completely specify the function. The number of parameters, N, is equal\n        to ``len(x)``.\n    bounds : sequence or `Bounds`\n        Bounds for variables. There are two ways to specify the bounds:\n\n            1. Instance of `Bounds` class.\n            2. ``(min, max)`` pairs for each element in ``x``, defining the\n               finite lower and upper bounds for the optimizing argument of\n               `func`.\n\n        The total number of bounds is used to determine the number of\n        parameters, N. If there are parameters whose bounds are equal the total\n        number of free parameters is ``N - N_equal``.\n\n    args : tuple, optional\n        Any additional fixed parameters needed to\n        completely specify the objective function.\n    strategy : {str, callable}, optional\n        The differential evolution strategy to use. Should be one of:\n\n            - 'best1bin'\n            - 'best1exp'\n            - 'rand1bin'\n            - 'rand1exp'\n            - 'rand2bin'\n            - 'rand2exp'\n            - 'randtobest1bin'\n            - 'randtobest1exp'\n            - 'currenttobest1bin'\n            - 'currenttobest1exp'\n            - 'best2exp'\n            - 'best2bin'\n\n        The default is 'best1bin'. Strategies that may be implemented are\n        outlined in 'Notes'.\n        Alternatively the differential evolution strategy can be customized by\n        providing a callable that constructs a trial vector. The callable must\n        have the form ``strategy(candidate: int, population: np.ndarray, rng=None)``,\n        where ``candidate`` is an integer specifying which entry of the\n        population is being evolved, ``population`` is an array of shape\n        ``(S, N)`` containing all the population members (where S is the\n        total population size), and ``rng`` is the random number generator\n        being used within the solver.\n        ``candidate`` will be in the range ``[0, S)``.\n        ``strategy`` must return a trial vector with shape `(N,)`. The\n        fitness of this trial vector is compared against the fitness of\n        ``population[candidate]``.\n\n        .. versionchanged:: 1.12.0\n            Customization of evolution strategy via a callable.\n\n    maxiter : int, optional\n        The maximum number of generations over which the entire population is\n        evolved. The maximum number of function evaluations (with no polishing)\n        is: ``(maxiter + 1) * popsize * (N - N_equal)``\n    popsize : int, optional\n        A multiplier for setting the total population size. The population has\n        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if\n        an initial population is supplied via the `init` keyword. When using\n        ``init='sobol'`` the population size is calculated as the next power\n        of 2 after ``popsize * (N - N_equal)``.\n    tol : float, optional\n        Relative tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    mutation : float or tuple(float, float), optional\n        The mutation constant. In the literature this is also known as\n        differential weight, being denoted by F.\n        If specified as a float it should be in the range [0, 2].\n        If specified as a tuple ``(min, max)`` dithering is employed. Dithering\n        randomly changes the mutation constant on a generation by generation\n        basis. The mutation constant for that generation is taken from\n        ``U[min, max)``. Dithering can help speed convergence significantly.\n        Increasing the mutation constant increases the search radius, but will\n        slow down convergence.\n    recombination : float, optional\n        The recombination constant, should be in the range [0, 1]. In the\n        literature this is also known as the crossover probability, being\n        denoted by CR. Increasing this value allows a larger number of mutants\n        to progress into the next generation, but at the risk of population\n        stability.\n    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional\n        If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n        singleton is used.\n        If `seed` is an int, a new ``RandomState`` instance is used,\n        seeded with `seed`.\n        If `seed` is already a ``Generator`` or ``RandomState`` instance then\n        that instance is used.\n        Specify `seed` for repeatable minimizations.\n    disp : bool, optional\n        Prints the evaluated `func` at every iteration.\n    callback : callable, optional\n        A callable called after each iteration. Has the signature:\n\n            ``callback(intermediate_result: OptimizeResult)``\n\n        where ``intermediate_result`` is a keyword parameter containing an\n        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution\n        found so far and the objective function. Note that the name\n        of the parameter must be ``intermediate_result`` for the callback\n        to be passed an `OptimizeResult`.\n\n        The callback also supports a signature like:\n\n            ``callback(x, convergence: float=val)``\n\n        ``val`` represents the fractional value of the population convergence.\n        When ``val`` is greater than ``1.0``, the function halts.\n\n        Introspection is used to determine which of the signatures is invoked.\n\n        Global minimization will halt if the callback raises ``StopIteration``\n        or returns ``True``; any polishing is still carried out.\n\n        .. versionchanged:: 1.12.0\n            callback accepts the ``intermediate_result`` keyword.\n\n    polish : bool, optional\n        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`\n        method is used to polish the best population member at the end, which\n        can improve the minimization slightly. If a constrained problem is\n        being studied then the `trust-constr` method is used instead. For large\n        problems with many constraints, polishing can take a long time due to\n        the Jacobian computations.\n    init : str or array-like, optional\n        Specify which type of population initialization is performed. Should be\n        one of:\n\n            - 'latinhypercube'\n            - 'sobol'\n            - 'halton'\n            - 'random'\n            - array specifying the initial population. The array should have\n              shape ``(S, N)``, where S is the total population size and N is\n              the number of parameters.\n              `init` is clipped to `bounds` before use.\n\n        The default is 'latinhypercube'. Latin Hypercube sampling tries to\n        maximize coverage of the available parameter space.\n\n        'sobol' and 'halton' are superior alternatives and maximize even more\n        the parameter space. 'sobol' will enforce an initial population\n        size which is calculated as the next power of 2 after\n        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit\n        less efficient. See `scipy.stats.qmc` for more details.\n\n        'random' initializes the population randomly - this has the drawback\n        that clustering can occur, preventing the whole of parameter space\n        being covered. Use of an array to specify a population could be used,\n        for example, to create a tight bunch of initial guesses in an location\n        where the solution is known to exist, thereby reducing time for\n        convergence.\n    atol : float, optional\n        Absolute tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    updating : {'immediate', 'deferred'}, optional\n        If ``'immediate'``, the best solution vector is continuously updated\n        within a single generation [4]_. This can lead to faster convergence as\n        trial vectors can take advantage of continuous improvements in the best\n        solution.\n        With ``'deferred'``, the best solution vector is updated once per\n        generation. Only ``'deferred'`` is compatible with parallelization or\n        vectorization, and the `workers` and `vectorized` keywords can\n        over-ride this option.\n\n        .. versionadded:: 1.2.0\n\n    workers : int or map-like callable, optional\n        If `workers` is an int the population is subdivided into `workers`\n        sections and evaluated in parallel\n        (uses `multiprocessing.Pool <multiprocessing>`).\n        Supply -1 to use all available CPU cores.\n        Alternatively supply a map-like callable, such as\n        `multiprocessing.Pool.map` for evaluating the population in parallel.\n        This evaluation is carried out as ``workers(func, iterable)``.\n        This option will override the `updating` keyword to\n        ``updating='deferred'`` if ``workers != 1``.\n        This option overrides the `vectorized` keyword if ``workers != 1``.\n        Requires that `func` be pickleable.\n\n        .. versionadded:: 1.2.0\n\n    constraints : {NonLinearConstraint, LinearConstraint, Bounds}\n        Constraints on the solver, over and above those applied by the `bounds`\n        kwd. Uses the approach by Lampinen [5]_.\n\n        .. versionadded:: 1.4.0\n\n    x0 : None or array-like, optional\n        Provides an initial guess to the minimization. Once the population has\n        been initialized this vector replaces the first (best) member. This\n        replacement is done even if `init` is given an initial population.\n        ``x0.shape == (N,)``.\n\n        .. versionadded:: 1.7.0\n\n    integrality : 1-D array, optional\n        For each decision variable, a boolean value indicating whether the\n        decision variable is constrained to integer values. The array is\n        broadcast to ``(N,)``.\n        If any decision variables are constrained to be integral, they will not\n        be changed during polishing.\n        Only integer values lying between the lower and upper bounds are used.\n        If there are no integer values lying between the bounds then a\n        `ValueError` is raised.\n\n        .. versionadded:: 1.9.0\n\n    vectorized : bool, optional\n        If ``vectorized is True``, `func` is sent an `x` array with\n        ``x.shape == (N, S)``, and is expected to return an array of shape\n        ``(S,)``, where `S` is the number of solution vectors to be calculated.\n        If constraints are applied, each of the functions used to construct\n        a `Constraint` object should accept an `x` array with\n        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where\n        `M` is the number of constraint components.\n        This option is an alternative to the parallelization offered by\n        `workers`, and may help in optimization speed by reducing interpreter\n        overhead from multiple function calls. This keyword is ignored if\n        ``workers != 1``.\n        This option will override the `updating` keyword to\n        ``updating='deferred'``.\n        See the notes section for further discussion on when to use\n        ``'vectorized'``, and when to use ``'workers'``.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a `OptimizeResult` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully,\n        ``message`` which describes the cause of the termination,\n        ``population`` the solution vectors present in the population, and\n        ``population_energies`` the value of the objective function for each\n        entry in ``population``.\n        See `OptimizeResult` for a description of other attributes. If `polish`\n        was employed, and a lower minimum was obtained by the polishing, then\n        OptimizeResult also contains the ``jac`` attribute.\n        If the eventual solution does not satisfy the applied constraints\n        ``success`` will be `False`.\n\n    Notes\n    -----\n    Differential evolution is a stochastic population based method that is\n    useful for global optimization problems. At each pass through the\n    population the algorithm mutates each candidate solution by mixing with\n    other candidate solutions to create a trial candidate. There are several\n    strategies [3]_ for creating trial candidates, which suit some problems\n    more than others. The 'best1bin' strategy is a good starting point for\n    many systems. In this strategy two members of the population are randomly\n    chosen. Their difference is used to mutate the best member (the 'best' in\n    'best1bin'), :math:`x_0`, so far:\n\n    .. math::\n\n        b' = x_0 + mutation * (x_{r_0} - x_{r_1})\n\n    A trial vector is then constructed. Starting with a randomly chosen ith\n    parameter the trial is sequentially filled (in modulo) with parameters\n    from ``b'`` or the original candidate. The choice of whether to use ``b'``\n    or the original candidate is made with a binomial distribution (the 'bin'\n    in 'best1bin') - a random number in [0, 1) is generated. If this number is\n    less than the `recombination` constant then the parameter is loaded from\n    ``b'``, otherwise it is loaded from the original candidate. The final\n    parameter is always loaded from ``b'``. Once the trial candidate is built\n    its fitness is assessed. If the trial is better than the original candidate\n    then it takes its place. If it is also better than the best overall\n    candidate it also replaces that.\n\n    The other strategies available are outlined in Qiang and\n    Mitchell (2014) [3]_.\n\n    .. math::\n            rand1* : b' = x_{r_0} + mutation*(x_{r_1} - x_{r_2})\n\n            rand2* : b' = x_{r_0} + mutation*(x_{r_1} + x_{r_2}\n                                                - x_{r_3} - x_{r_4})\n\n            best1* : b' = x_0 + mutation*(x_{r_0} - x_{r_1})\n\n            best2* : b' = x_0 + mutation*(x_{r_0} + x_{r_1}\n                                            - x_{r_2} - x_{r_3})\n\n            currenttobest1* : b' = x_i + mutation*(x_0 - x_i\n                                                     + x_{r_0} - x_{r_1})\n\n            randtobest1* : b' = x_{r_0} + mutation*(x_0 - x_{r_0}\n                                                      + x_{r_1} - x_{r_2})\n\n    where the integers :math:`r_0, r_1, r_2, r_3, r_4` are chosen randomly\n    from the interval [0, NP) with `NP` being the total population size and\n    the original candidate having index `i`. The user can fully customize the\n    generation of the trial candidates by supplying a callable to ``strategy``.\n\n    To improve your chances of finding a global minimum use higher `popsize`\n    values, with higher `mutation` and (dithering), but lower `recombination`\n    values. This has the effect of widening the search radius, but slowing\n    convergence.\n\n    By default the best solution vector is updated continuously within a single\n    iteration (``updating='immediate'``). This is a modification [4]_ of the\n    original differential evolution algorithm which can lead to faster\n    convergence as trial vectors can immediately benefit from improved\n    solutions. To use the original Storn and Price behaviour, updating the best\n    solution once per iteration, set ``updating='deferred'``.\n    The ``'deferred'`` approach is compatible with both parallelization and\n    vectorization (``'workers'`` and ``'vectorized'`` keywords). These may\n    improve minimization speed by using computer resources more efficiently.\n    The ``'workers'`` distribute calculations over multiple processors. By\n    default the Python `multiprocessing` module is used, but other approaches\n    are also possible, such as the Message Passing Interface (MPI) used on\n    clusters [6]_ [7]_. The overhead from these approaches (creating new\n    Processes, etc) may be significant, meaning that computational speed\n    doesn't necessarily scale with the number of processors used.\n    Parallelization is best suited to computationally expensive objective\n    functions. If the objective function is less expensive, then\n    ``'vectorized'`` may aid by only calling the objective function once per\n    iteration, rather than multiple times for all the population members; the\n    interpreter overhead is reduced.\n\n    .. versionadded:: 0.15.0\n\n    References\n    ----------\n    .. [1] Differential evolution, Wikipedia,\n           http://en.wikipedia.org/wiki/Differential_evolution\n    .. [2] Storn, R and Price, K, Differential Evolution - a Simple and\n           Efficient Heuristic for Global Optimization over Continuous Spaces,\n           Journal of Global Optimization, 1997, 11, 341 - 359.\n    .. [3] Qiang, J., Mitchell, C., A Unified Differential Evolution Algorithm\n            for Global Optimization, 2014, https://www.osti.gov/servlets/purl/1163659\n    .. [4] Wormington, M., Panaccione, C., Matney, K. M., Bowen, D. K., -\n           Characterization of structures from X-ray scattering data using\n           genetic algorithms, Phil. Trans. R. Soc. Lond. A, 1999, 357,\n           2827-2848\n    .. [5] Lampinen, J., A constraint handling approach for the differential\n           evolution algorithm. Proceedings of the 2002 Congress on\n           Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). Vol. 2. IEEE,\n           2002.\n    .. [6] https://mpi4py.readthedocs.io/en/stable/\n    .. [7] https://schwimmbad.readthedocs.io/en/latest/\n \n\n    Examples\n    --------\n    Let us consider the problem of minimizing the Rosenbrock function. This\n    function is implemented in `rosen` in `scipy.optimize`.\n\n    >>> import numpy as np\n    >>> from scipy.optimize import rosen, differential_evolution\n    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]\n    >>> result = differential_evolution(rosen, bounds)\n    >>> result.x, result.fun\n    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)\n\n    Now repeat, but with parallelization.\n\n    >>> result = differential_evolution(rosen, bounds, updating='deferred',\n    ...                                 workers=2)\n    >>> result.x, result.fun\n    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)\n\n    Let's do a constrained minimization.\n\n    >>> from scipy.optimize import LinearConstraint, Bounds\n\n    We add the constraint that the sum of ``x[0]`` and ``x[1]`` must be less\n    than or equal to 1.9.  This is a linear constraint, which may be written\n    ``A @ x <= 1.9``, where ``A = array([[1, 1]])``.  This can be encoded as\n    a `LinearConstraint` instance:\n\n    >>> lc = LinearConstraint([[1, 1]], -np.inf, 1.9)\n\n    Specify limits using a `Bounds` object.\n\n    >>> bounds = Bounds([0., 0.], [2., 2.])\n    >>> result = differential_evolution(rosen, bounds, constraints=lc,\n    ...                                 seed=1)\n    >>> result.x, result.fun\n    (array([0.96632622, 0.93367155]), 0.0011352416852625719)\n\n    Next find the minimum of the Ackley function\n    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).\n\n    >>> def ackley(x):\n    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))\n    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))\n    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e\n    >>> bounds = [(-5, 5), (-5, 5)]\n    >>> result = differential_evolution(ackley, bounds, seed=1)\n    >>> result.x, result.fun\n    (array([0., 0.]), 4.440892098500626e-16)\n\n    The Ackley function is written in a vectorized manner, so the\n    ``'vectorized'`` keyword can be employed. Note the reduced number of\n    function evaluations.\n\n    >>> result = differential_evolution(\n    ...     ackley, bounds, vectorized=True, updating='deferred', seed=1\n    ... )\n    >>> result.x, result.fun\n    (array([0., 0.]), 4.440892098500626e-16)\n\n    The following custom strategy function mimics 'best1bin':\n\n    >>> def custom_strategy_fn(candidate, population, rng=None):\n    ...     parameter_count = population.shape(-1)\n    ...     mutation, recombination = 0.7, 0.9\n    ...     trial = np.copy(population[candidate])\n    ...     fill_point = rng.choice(parameter_count)\n    ...\n    ...     pool = np.arange(len(population))\n    ...     rng.shuffle(pool)\n    ...\n    ...     # two unique random numbers that aren't the same, and\n    ...     # aren't equal to candidate.\n    ...     idxs = []\n    ...     while len(idxs) < 2 and len(pool) > 0:\n    ...         idx = pool[0]\n    ...         pool = pool[1:]\n    ...         if idx != candidate:\n    ...             idxs.append(idx)\n    ...\n    ...     r0, r1 = idxs[:2]\n    ...\n    ...     bprime = (population[0] + mutation *\n    ...               (population[r0] - population[r1]))\n    ...\n    ...     crossovers = rng.uniform(size=parameter_count)\n    ...     crossovers = crossovers < recombination\n    ...     crossovers[fill_point] = True\n    ...     trial = np.where(crossovers, bprime, trial)\n    ...     return trial\n\n    "
    with DifferentialEvolutionSolver(func, bounds, args=args, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, recombination=recombination, seed=seed, polish=polish, callback=callback, disp=disp, init=init, atol=atol, updating=updating, workers=workers, constraints=constraints, x0=x0, integrality=integrality, vectorized=vectorized) as solver:
        ret = solver.solve()
    return ret

class DifferentialEvolutionSolver:
    """This class implements the differential evolution solver

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'

        The default is 'best1bin'. Strategies that may be
        implemented are outlined in 'Notes'.

        Alternatively the differential evolution strategy can be customized
        by providing a callable that constructs a trial vector. The callable
        must have the form
        ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape `(N,)`. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    popsize : int, optional
        A multiplier for setting the total population size. The population has
        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
        an initial population is supplied via the `init` keyword. When using
        ``init='sobol'`` the population size is calculated as the next power
        of 2 after ``popsize * (N - N_equal)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, optional
        A callable called after each iteration. Has the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
        found so far and the objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`.

        The callback also supports a signature like:

            ``callback(x, convergence: float=val)``

        ``val`` represents the fractional value of the population convergence.
         When ``val`` is greater than ``1.0``, the function halts.

        Introspection is used to determine which of the signatures is invoked.

        Global minimization will halt if the callback raises ``StopIteration``
        or returns ``True``; any polishing is still carried out.

        .. versionchanged:: 1.12.0
            callback accepts the ``intermediate_result`` keyword.

    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead. For large
        problems with many constraints, polishing can take a long time due to
        the Jacobian computations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(S, N)``, where S is the total population size and
              N is the number of parameters.
              `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial population
        size which is calculated as the next power of 2 after
        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
        less efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the population randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a population could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation [4]_. This can lead to faster convergence as
        trial vectors can take advantage of continuous improvements in the best
        solution.
        With ``'deferred'``, the best solution vector is updated once per
        generation. Only ``'deferred'`` is compatible with parallelization or
        vectorization, and the `workers` and `vectorized` keywords can
        over-ride this option.
    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        `updating='deferred'` if `workers != 1`.
        Requires that `func` be pickleable.
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen.
    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the population has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial population.
        ``x0.shape == (N,)``.
    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values. The array is
        broadcast to ``(N,)``.
        If any decision variables are constrained to be integral, they will not
        be changed during polishing.
        Only integer values lying between the lower and upper bounds are used.
        If there are no integer values lying between the bounds then a
        `ValueError` is raised.
    vectorized : bool, optional
        If ``vectorized is True``, `func` is sent an `x` array with
        ``x.shape == (N, S)``, and is expected to return an array of shape
        ``(S,)``, where `S` is the number of solution vectors to be calculated.
        If constraints are applied, each of the functions used to construct
        a `Constraint` object should accept an `x` array with
        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
        `M` is the number of constraint components.
        This option is an alternative to the parallelization offered by
        `workers`, and may help in optimization speed. This keyword is
        ignored if ``workers != 1``.
        This option will override the `updating` keyword to
        ``updating='deferred'``.
    """
    _binomial = {'best1bin': '_best1', 'randtobest1bin': '_randtobest1', 'currenttobest1bin': '_currenttobest1', 'best2bin': '_best2', 'rand2bin': '_rand2', 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1', 'rand1exp': '_rand1', 'randtobest1exp': '_randtobest1', 'currenttobest1exp': '_currenttobest1', 'best2exp': '_best2', 'rand2exp': '_rand2'}
    __init_error_msg = "The population initialization method must be one of 'latinhypercube' or 'random', or an array of shape (S, N) where N is the number of parameters and S>5"

    def __init__(self, func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, maxfun=np.inf, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, *, integrality=None, vectorized=False):
        if False:
            print('Hello World!')
        if callable(strategy):
            pass
        elif strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError('Please select a valid mutation strategy')
        self.strategy = strategy
        self.callback = _wrap_callback(callback, 'differential_evolution')
        self.polish = polish
        if updating in ['immediate', 'deferred']:
            self._updating = updating
        self.vectorized = vectorized
        if workers != 1 and updating == 'immediate':
            warnings.warn("differential_evolution: the 'workers' keyword has overridden updating='immediate' to updating='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'
        if vectorized and workers != 1:
            warnings.warn("differential_evolution: the 'workers' keyword overrides the 'vectorized' keyword", stacklevel=2)
            self.vectorized = vectorized = False
        if vectorized and updating == 'immediate':
            warnings.warn("differential_evolution: the 'vectorized' keyword has overridden updating='immediate' to updating='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'
        if vectorized:

            def maplike_for_vectorized_func(func, x):
                if False:
                    for i in range(10):
                        print('nop')
                return np.atleast_1d(func(x.T))
            workers = maplike_for_vectorized_func
        self._mapwrapper = MapWrapper(workers)
        (self.tol, self.atol) = (tol, atol)
        self.scale = mutation
        if not np.all(np.isfinite(mutation)) or np.any(np.array(mutation) >= 2) or np.any(np.array(mutation) < 0):
            raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()
        self.cross_over_probability = recombination
        self.func = _FunctionWrapper(func, args)
        self.args = args
        if isinstance(bounds, Bounds):
            self.limits = np.array(new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb)), dtype=float).T
        else:
            self.limits = np.array(bounds, dtype='float').T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError('bounds should be a sequence containing finite real valued (min, max) pairs for each value in x')
        if maxiter is None:
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        with np.errstate(divide='ignore'):
            self.__recip_scale_arg2 = 1 / self.__scale_arg2
            self.__recip_scale_arg2[~np.isfinite(self.__recip_scale_arg2)] = 0
        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)
        if np.any(integrality):
            integrality = np.broadcast_to(integrality, self.parameter_count)
            integrality = np.asarray(integrality, bool)
            (lb, ub) = np.copy(self.limits)
            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                raise ValueError('One of the integrality constraints does not have any possible integer values between the lower/upper bounds.')
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)
            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False
        eb = self.limits[0] == self.limits[1]
        eb_count = np.count_nonzero(eb)
        self.num_population_members = max(5, popsize * max(1, self.parameter_count - eb_count))
        self.population_shape = (self.num_population_members, self.parameter_count)
        self._nfev = 0
        if isinstance(init, str):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'sobol':
                n_s = int(2 ** np.ceil(np.log2(self.num_population_members)))
                self.num_population_members = n_s
                self.population_shape = (self.num_population_members, self.parameter_count)
                self.init_population_qmc(qmc_engine='sobol')
            elif init == 'halton':
                self.init_population_qmc(qmc_engine='halton')
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)
        if x0 is not None:
            x0_scaled = self._unscale_parameters(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError('Some entries in x0 lay outside the specified bounds')
            self.population[0] = x0_scaled
        self.constraints = constraints
        self._wrapped_constraints = []
        if hasattr(constraints, '__len__'):
            for c in constraints:
                self._wrapped_constraints.append(_ConstraintWrapper(c, self.x))
        else:
            self._wrapped_constraints = [_ConstraintWrapper(constraints, self.x)]
        self.total_constraints = np.sum([c.num_constr for c in self._wrapped_constraints])
        self.constraint_violation = np.zeros((self.num_population_members, 1))
        self.feasible = np.ones(self.num_population_members, bool)
        self.disp = disp

    def init_population_lhs(self):
        if False:
            while True:
                i = 10
        '\n        Initializes the population with Latin Hypercube Sampling.\n        Latin Hypercube Sampling ensures that each parameter is uniformly\n        sampled over its range.\n        '
        rng = self.random_number_generator
        segsize = 1.0 / self.num_population_members
        samples = segsize * rng.uniform(size=self.population_shape) + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[:, np.newaxis]
        self.population = np.zeros_like(samples)
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]
        self.population_energies = np.full(self.num_population_members, np.inf)
        self._nfev = 0

    def init_population_qmc(self, qmc_engine):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the population with a QMC method.\n\n        QMC methods ensures that each parameter is uniformly\n        sampled over its range.\n\n        Parameters\n        ----------\n        qmc_engine : str\n            The QMC method to use for initialization. Can be one of\n            ``latinhypercube``, ``sobol`` or ``halton``.\n\n        '
        from scipy.stats import qmc
        rng = self.random_number_generator
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.parameter_count, seed=rng)
        else:
            raise ValueError(self.__init_error_msg)
        self.population = sampler.random(n=self.num_population_members)
        self.population_energies = np.full(self.num_population_members, np.inf)
        self._nfev = 0

    def init_population_random(self):
        if False:
            while True:
                i = 10
        '\n        Initializes the population at random. This type of initialization\n        can possess clustering, Latin Hypercube sampling is generally better.\n        '
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)
        self.population_energies = np.full(self.num_population_members, np.inf)
        self._nfev = 0

    def init_population_array(self, init):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the population with a user specified population.\n\n        Parameters\n        ----------\n        init : np.ndarray\n            Array specifying subset of the initial population. The array should\n            have shape (S, N), where N is the number of parameters.\n            The population is clipped to the lower and upper bounds.\n        '
        popn = np.asarray(init, dtype=np.float64)
        if np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2:
            raise ValueError('The population supplied needs to have shape (S, len(x)), where S > 4.')
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = np.size(self.population, 0)
        self.population_shape = (self.num_population_members, self.parameter_count)
        self.population_energies = np.full(self.num_population_members, np.inf)
        self._nfev = 0

    @property
    def x(self):
        if False:
            while True:
                i = 10
        '\n        The best solution from the solver\n        '
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The standard deviation of the population energies divided by their\n        mean.\n        '
        if np.any(np.isinf(self.population_energies)):
            return np.inf
        return np.std(self.population_energies) / (np.abs(np.mean(self.population_energies)) + _MACHEPS)

    def converged(self):
        if False:
            return 10
        '\n        Return True if the solver has converged.\n        '
        if np.any(np.isinf(self.population_energies)):
            return False
        return np.std(self.population_energies) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))

    def solve(self):
        if False:
            return 10
        '\n        Runs the DifferentialEvolutionSolver.\n\n        Returns\n        -------\n        res : OptimizeResult\n            The optimization result represented as a `OptimizeResult` object.\n            Important attributes are: ``x`` the solution array, ``success`` a\n            Boolean flag indicating if the optimizer exited successfully,\n            ``message`` which describes the cause of the termination,\n            ``population`` the solution vectors present in the population, and\n            ``population_energies`` the value of the objective function for\n            each entry in ``population``.\n            See `OptimizeResult` for a description of other attributes. If\n            `polish` was employed, and a lower minimum was obtained by the\n            polishing, then OptimizeResult also contains the ``jac`` attribute.\n            If the eventual solution does not satisfy the applied constraints\n            ``success`` will be `False`.\n        '
        (nit, warning_flag) = (0, False)
        status_message = _status_message['success']
        if np.all(np.isinf(self.population_energies)):
            (self.feasible, self.constraint_violation) = self._calculate_population_feasibilities(self.population)
            self.population_energies[self.feasible] = self._calculate_population_energies(self.population[self.feasible])
            self._promote_lowest_energy()
        for nit in range(1, self.maxiter + 1):
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = _status_message['maxfev']
                elif self._nfev == self.maxfun:
                    status_message = 'Maximum number of function evaluations has been reached.'
                break
            if self.disp:
                print(f'differential_evolution step {nit}: f(x)= {self.population_energies[0]}')
            if self.callback:
                c = self.tol / (self.convergence + _MACHEPS)
                res = self._result(nit=nit, message='in progress')
                res.convergence = c
                try:
                    warning_flag = bool(self.callback(res))
                except StopIteration:
                    warning_flag = True
                if warning_flag:
                    status_message = 'callback function requested stop early'
            if warning_flag or self.converged():
                break
        else:
            status_message = _status_message['maxiter']
            warning_flag = True
        DE_result = self._result(nit=nit, message=status_message, warning_flag=warning_flag)
        if self.polish and (not np.all(self.integrality)):
            if np.any(self.integrality):
                (limits, integrality) = (self.limits, self.integrality)
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]
            polish_method = 'L-BFGS-B'
            if self._wrapped_constraints:
                polish_method = 'trust-constr'
                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.0):
                    warnings.warn("differential evolution didn't find a solution satisfying the constraints, attempting to polish from the least infeasible solution", UserWarning)
            if self.disp:
                print(f"Polishing solution with '{polish_method}'")
            result = minimize(self.func, np.copy(DE_result.x), method=polish_method, bounds=self.limits.T, constraints=self.constraints)
            self._nfev += result.nfev
            DE_result.nfev = self._nfev
            if result.fun < DE_result.fun and result.success and np.all(result.x <= self.limits[1]) and np.all(self.limits[0] <= result.x):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)
        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                DE_result.success = False
                DE_result.message = f'The solution does not satisfy the constraints, MAXCV = {DE_result.maxcv}'
        return DE_result

    def _result(self, **kwds):
        if False:
            while True:
                i = 10
        nit = kwds.get('nit', None)
        message = kwds.get('message', None)
        warning_flag = kwds.get('warning_flag', False)
        result = OptimizeResult(x=self.x, fun=self.population_energies[0], nfev=self._nfev, nit=nit, message=message, success=warning_flag is not True, population=self._scale_parameters(self.population), population_energies=self.population_energies)
        if self._wrapped_constraints:
            result.constr = [c.violation(result.x) for c in self._wrapped_constraints]
            result.constr_violation = np.max(np.concatenate(result.constr))
            result.maxcv = result.constr_violation
            if result.maxcv > 0:
                result.success = False
        return result

    def _calculate_population_energies(self, population):
        if False:
            print('Hello World!')
        '\n        Calculate the energies of a population.\n\n        Parameters\n        ----------\n        population : ndarray\n            An array of parameter vectors normalised to [0, 1] using lower\n            and upper limits. Has shape ``(np.size(population, 0), N)``.\n\n        Returns\n        -------\n        energies : ndarray\n            An array of energies corresponding to each population member. If\n            maxfun will be exceeded during this call, then the number of\n            function evaluations will be reduced and energies will be\n            right-padded with np.inf. Has shape ``(np.size(population, 0),)``\n        '
        num_members = np.size(population, 0)
        S = min(num_members, self.maxfun - self._nfev)
        energies = np.full(num_members, np.inf)
        parameters_pop = self._scale_parameters(population)
        try:
            calc_energies = list(self._mapwrapper(self.func, parameters_pop[0:S]))
            calc_energies = np.squeeze(calc_energies)
        except (TypeError, ValueError) as e:
            raise RuntimeError("The map-like callable must be of the form f(func, iterable), returning a sequence of numbers the same length as 'iterable'") from e
        if calc_energies.size != S:
            if self.vectorized:
                raise RuntimeError('The vectorized function must return an array of shape (S,) when given an array of shape (len(x), S)')
            raise RuntimeError('func(x, *args) must return a scalar value')
        energies[0:S] = calc_energies
        if self.vectorized:
            self._nfev += 1
        else:
            self._nfev += S
        return energies

    def _promote_lowest_energy(self):
        if False:
            return 10
        idx = np.arange(self.num_population_members)
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            l = feasible_solutions[idx_t]
        else:
            l = np.argmin(np.sum(self.constraint_violation, axis=1))
        self.population_energies[[0, l]] = self.population_energies[[l, 0]]
        self.population[[0, l], :] = self.population[[l, 0], :]
        self.feasible[[0, l]] = self.feasible[[l, 0]]
        self.constraint_violation[[0, l], :] = self.constraint_violation[[l, 0], :]

    def _constraint_violation_fn(self, x):
        if False:
            print('Hello World!')
        '\n        Calculates total constraint violation for all the constraints, for a\n        set of solutions.\n\n        Parameters\n        ----------\n        x : ndarray\n            Solution vector(s). Has shape (S, N), or (N,), where S is the\n            number of solutions to investigate and N is the number of\n            parameters.\n\n        Returns\n        -------\n        cv : ndarray\n            Total violation of constraints. Has shape ``(S, M)``, where M is\n            the total number of constraint components (which is not necessarily\n            equal to len(self._wrapped_constraints)).\n        '
        S = np.size(x) // self.parameter_count
        _out = np.zeros((S, self.total_constraints))
        offset = 0
        for con in self._wrapped_constraints:
            c = con.violation(x.T).T
            if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
                raise RuntimeError('An array returned from a Constraint has the wrong shape. If `vectorized is False` the Constraint should return an array of shape (M,). If `vectorized is True` then the Constraint must return an array of shape (M, S), where S is the number of solution vectors and M is the number of constraint components in a given Constraint object.')
            c = np.reshape(c, (S, con.num_constr))
            _out[:, offset:offset + con.num_constr] = c
            offset += con.num_constr
        return _out

    def _calculate_population_feasibilities(self, population):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the feasibilities of a population.\n\n        Parameters\n        ----------\n        population : ndarray\n            An array of parameter vectors normalised to [0, 1] using lower\n            and upper limits. Has shape ``(np.size(population, 0), N)``.\n\n        Returns\n        -------\n        feasible, constraint_violation : ndarray, ndarray\n            Boolean array of feasibility for each population member, and an\n            array of the constraint violation for each population member.\n            constraint_violation has shape ``(np.size(population, 0), M)``,\n            where M is the number of constraints.\n        '
        num_members = np.size(population, 0)
        if not self._wrapped_constraints:
            return (np.ones(num_members, bool), np.zeros((num_members, 1)))
        parameters_pop = self._scale_parameters(population)
        if self.vectorized:
            constraint_violation = np.array(self._constraint_violation_fn(parameters_pop))
        else:
            constraint_violation = np.array([self._constraint_violation_fn(x) for x in parameters_pop])
            constraint_violation = constraint_violation[:, 0]
        feasible = ~(np.sum(constraint_violation, axis=1) > 0)
        return (feasible, constraint_violation)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        return self._mapwrapper.__exit__(*args)

    def _accept_trial(self, energy_trial, feasible_trial, cv_trial, energy_orig, feasible_orig, cv_orig):
        if False:
            for i in range(10):
                print('nop')
        '\n        Trial is accepted if:\n        * it satisfies all constraints and provides a lower or equal objective\n          function value, while both the compared solutions are feasible\n        - or -\n        * it is feasible while the original solution is infeasible,\n        - or -\n        * it is infeasible, but provides a lower or equal constraint violation\n          for all constraint functions.\n\n        This test corresponds to section III of Lampinen [1]_.\n\n        Parameters\n        ----------\n        energy_trial : float\n            Energy of the trial solution\n        feasible_trial : float\n            Feasibility of trial solution\n        cv_trial : array-like\n            Excess constraint violation for the trial solution\n        energy_orig : float\n            Energy of the original solution\n        feasible_orig : float\n            Feasibility of original solution\n        cv_orig : array-like\n            Excess constraint violation for the original solution\n\n        Returns\n        -------\n        accepted : bool\n\n        '
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and (not feasible_orig):
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            return True
        return False

    def __next__(self):
        if False:
            i = 10
            return i + 15
        '\n        Evolve the population by a single generation\n\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        fun : float\n            Value of objective function obtained from the best solution.\n        '
        if np.all(np.isinf(self.population_energies)):
            (self.feasible, self.constraint_violation) = self._calculate_population_feasibilities(self.population)
            self.population_energies[self.feasible] = self._calculate_population_energies(self.population[self.feasible])
            self._promote_lowest_energy()
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0], self.dither[1])
        if self._updating == 'immediate':
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration
                trial = self._mutate(candidate)
                self._ensure_constraint(trial)
                parameters = self._scale_parameters(trial)
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.0])
                    energy = self.func(parameters)
                    self._nfev += 1
                if self._accept_trial(energy, feasible, cv, self.population_energies[candidate], self.feasible[candidate], self.constraint_violation[candidate]):
                    self.population[candidate] = trial
                    self.population_energies[candidate] = np.squeeze(energy)
                    self.feasible[candidate] = feasible
                    self.constraint_violation[candidate] = cv
                    if self._accept_trial(energy, feasible, cv, self.population_energies[0], self.feasible[0], self.constraint_violation[0]):
                        self._promote_lowest_energy()
        elif self._updating == 'deferred':
            if self._nfev >= self.maxfun:
                raise StopIteration
            trial_pop = np.array([self._mutate(i) for i in range(self.num_population_members)])
            self._ensure_constraint(trial_pop)
            (feasible, cv) = self._calculate_population_feasibilities(trial_pop)
            trial_energies = np.full(self.num_population_members, np.inf)
            trial_energies[feasible] = self._calculate_population_energies(trial_pop[feasible])
            loc = [self._accept_trial(*val) for val in zip(trial_energies, feasible, cv, self.population_energies, self.feasible, self.constraint_violation)]
            loc = np.array(loc)
            self.population = np.where(loc[:, np.newaxis], trial_pop, self.population)
            self.population_energies = np.where(loc, trial_energies, self.population_energies)
            self.feasible = np.where(loc, feasible, self.feasible)
            self.constraint_violation = np.where(loc[:, np.newaxis], cv, self.constraint_violation)
            self._promote_lowest_energy()
        return (self.x, self.population_energies[0])

    def _scale_parameters(self, trial):
        if False:
            while True:
                i = 10
        'Scale from a number between 0 and 1 to parameters.'
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.any(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _unscale_parameters(self, parameters):
        if False:
            i = 10
            return i + 15
        'Scale from parameters to a number between 0 and 1.'
        return (parameters - self.__scale_arg1) * self.__recip_scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        if False:
            while True:
                i = 10
        'Make sure the parameters lie between the limits.'
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)

    def _mutate(self, candidate):
        if False:
            for i in range(10):
                print('nop')
        'Create a trial vector based on a mutation strategy.'
        rng = self.random_number_generator
        if callable(self.strategy):
            _population = self._scale_parameters(self.population)
            trial = np.array(self.strategy(candidate, _population, rng=rng), dtype=float)
            if trial.shape != (self.parameter_count,):
                raise RuntimeError('strategy must have signature f(candidate: int, population: np.ndarray, rng=None) returning an array of shape (N,)')
            return self._unscale_parameters(trial)
        trial = np.copy(self.population[candidate])
        fill_point = rng.choice(self.parameter_count)
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))
        if self.strategy in self._binomial:
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial
        elif self.strategy in self._exponential:
            i = 0
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[0] = True
            while i < self.parameter_count and crossovers[i]:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1
            return trial

    def _best1(self, samples):
        if False:
            return 10
        'best1bin, best1exp'
        (r0, r1) = samples[:2]
        return self.population[0] + self.scale * (self.population[r0] - self.population[r1])

    def _rand1(self, samples):
        if False:
            for i in range(10):
                print('nop')
        'rand1bin, rand1exp'
        (r0, r1, r2) = samples[:3]
        return self.population[r0] + self.scale * (self.population[r1] - self.population[r2])

    def _randtobest1(self, samples):
        if False:
            print('Hello World!')
        'randtobest1bin, randtobest1exp'
        (r0, r1, r2) = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        if False:
            i = 10
            return i + 15
        'currenttobest1bin, currenttobest1exp'
        (r0, r1) = samples[:2]
        bprime = self.population[candidate] + self.scale * (self.population[0] - self.population[candidate] + self.population[r0] - self.population[r1])
        return bprime

    def _best2(self, samples):
        if False:
            i = 10
            return i + 15
        'best2bin, best2exp'
        (r0, r1, r2, r3) = samples[:4]
        bprime = self.population[0] + self.scale * (self.population[r0] + self.population[r1] - self.population[r2] - self.population[r3])
        return bprime

    def _rand2(self, samples):
        if False:
            while True:
                i = 10
        'rand2bin, rand2exp'
        (r0, r1, r2, r3, r4) = samples
        bprime = self.population[r0] + self.scale * (self.population[r1] + self.population[r2] - self.population[r3] - self.population[r4])
        return bprime

    def _select_samples(self, candidate, number_samples):
        if False:
            while True:
                i = 10
        "\n        obtain random integers from range(self.num_population_members),\n        without replacement. You can't have the original candidate either.\n        "
        pool = np.arange(self.num_population_members)
        self.random_number_generator.shuffle(pool)
        idxs = []
        while len(idxs) < number_samples and len(pool) > 0:
            idx = pool[0]
            pool = pool[1:]
            if idx != candidate:
                idxs.append(idx)
        return idxs

class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    """

    def __init__(self, constraint, x0):
        if False:
            print('Hello World!')
        self.constraint = constraint
        if isinstance(constraint, NonlinearConstraint):

            def fun(x):
                if False:
                    while True:
                        i = 10
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))
        elif isinstance(constraint, LinearConstraint):

            def fun(x):
                if False:
                    while True:
                        i = 10
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)
                return A.dot(x)
        elif isinstance(constraint, Bounds):

            def fun(x):
                if False:
                    while True:
                        i = 10
                return np.asarray(x)
        else:
            raise ValueError('`constraint` of an unknown type is passed.')
        self.fun = fun
        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        x0 = np.asarray(x0)
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size
        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)
        self.bounds = (lb, ub)

    def __call__(self, x):
        if False:
            print('Hello World!')
        return np.atleast_1d(self.fun(x))

    def violation(self, x):
        if False:
            return 10
        'How much the constraint is exceeded by.\n\n        Parameters\n        ----------\n        x : array-like\n            Vector of independent variables, (N, S), where N is number of\n            parameters and S is the number of solutions to be investigated.\n\n        Returns\n        -------\n        excess : array-like\n            How much the constraint is exceeded by, for each of the\n            constraints specified by `_ConstraintWrapper.fun`.\n            Has shape (M, S) where M is the number of constraint components.\n        '
        ev = self.fun(np.asarray(x))
        try:
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            raise RuntimeError('An array returned from a Constraint has the wrong shape. If `vectorized is False` the Constraint should return an array of shape (M,). If `vectorized is True` then the Constraint must return an array of shape (M, S), where S is the number of solution vectors and M is the number of constraint components in a given Constraint object.') from e
        v = (excess_lb + excess_ub).T
        return v