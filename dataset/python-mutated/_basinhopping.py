"""
basinhopping: The basinhopping global optimization algorithm
"""
import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
__all__ = ['basinhopping']
_params = (inspect.Parameter('res_new', kind=inspect.Parameter.KEYWORD_ONLY), inspect.Parameter('res_old', kind=inspect.Parameter.KEYWORD_ONLY))
_new_accept_test_signature = inspect.Signature(parameters=_params)

class Storage:
    """
    Class used to store the lowest energy structure
    """

    def __init__(self, minres):
        if False:
            while True:
                i = 10
        self._add(minres)

    def _add(self, minres):
        if False:
            i = 10
            return i + 15
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        if False:
            while True:
                i = 10
        if minres.success and (minres.fun < self.minres.fun or not self.minres.success):
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        if False:
            print('Hello World!')
        return self.minres

class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether or not to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """

    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        if False:
            print('Hello World!')
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp
        self.nstep = 0
        self.res = scipy.optimize.OptimizeResult()
        self.res.minimization_failures = 0
        minres = minimizer(self.x)
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print('warning: basinhopping: local minimization failure')
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        self.incumbent_minres = minres
        if self.disp:
            print('basinhopping step %d: f %g' % (self.nstep, self.energy))
        self.storage = Storage(minres)
        if hasattr(minres, 'nfev'):
            self.res.nfev = minres.nfev
        if hasattr(minres, 'njev'):
            self.res.njev = minres.njev
        if hasattr(minres, 'nhev'):
            self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        if False:
            print('Hello World!')
        'Do one Monte Carlo iteration\n\n        Randomly displace the coordinates, minimize, and decide whether\n        or not to accept the new coordinates.\n        '
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print('warning: basinhopping: local minimization failure')
        if hasattr(minres, 'nfev'):
            self.res.nfev += minres.nfev
        if hasattr(minres, 'njev'):
            self.res.njev += minres.njev
        if hasattr(minres, 'nhev'):
            self.res.nhev += minres.nhev
        accept = True
        for test in self.accept_tests:
            if inspect.signature(test) == _new_accept_test_signature:
                testres = test(res_new=minres, res_old=self.incumbent_minres)
            else:
                testres = test(f_new=energy_after_quench, x_new=x_after_quench, f_old=self.energy, x_old=self.x)
            if testres == 'force accept':
                accept = True
                break
            elif testres is None:
                raise ValueError("accept_tests must return True, False, or 'force accept'")
            elif not testres:
                accept = False
        if hasattr(self.step_taking, 'report'):
            self.step_taking.report(accept, f_new=energy_after_quench, x_new=x_after_quench, f_old=self.energy, x_old=self.x)
        return (accept, minres)

    def one_cycle(self):
        if False:
            return 10
        'Do one cycle of the basinhopping algorithm\n        '
        self.nstep += 1
        new_global_min = False
        (accept, minres) = self._monte_carlo_step()
        if accept:
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            self.incumbent_minres = minres
            new_global_min = self.storage.update(minres)
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                print('found new global minimum on step %d with function value %g' % (self.nstep, self.energy))
        self.xtrial = minres.x
        self.energy_trial = minres.fun
        self.accept = accept
        return new_global_min

    def print_report(self, energy_trial, accept):
        if False:
            for i in range(10):
                print('nop')
        'print a status update'
        minres = self.storage.get_lowest()
        print('basinhopping step %d: f %g trial_f %g accepted %d  lowest_f %g' % (self.nstep, self.energy, energy_trial, accept, minres.fun))

class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """

    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9, verbose=True):
        if False:
            print('Hello World!')
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose
        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.take_step(x)

    def _adjust_step_size(self):
        if False:
            for i in range(10):
                print('nop')
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            self.takestep.stepsize /= self.factor
        else:
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print('adaptive stepsize: acceptance rate {:f} target {:f} new stepsize {:g} old stepsize {:g}'.format(accept_rate, self.target_accept_rate, self.takestep.stepsize, old_stepsize))

    def take_step(self, x):
        if False:
            return 10
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        if False:
            return 10
        'called by basinhopping to report the result of the step'
        if accept:
            self.naccept += 1

class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """

    def __init__(self, stepsize=0.5, random_gen=None):
        if False:
            return 10
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        x += self.random_gen.uniform(-self.stepsize, self.stepsize, np.shape(x))
        return x

class MinimizerWrapper:
    """
    wrap a minimizer function as a minimizer class
    """

    def __init__(self, minimizer, func=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        if False:
            return 10
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)

class Metropolis:
    """Metropolis acceptance criterion.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Random number generator used for acceptance test.

    """

    def __init__(self, T, random_gen=None):
        if False:
            print('Hello World!')
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.random_gen = check_random_state(random_gen)

    def accept_reject(self, res_new, res_old):
        if False:
            return 10
        '\n        Assuming the local search underlying res_new was successful:\n        If new energy is lower than old, it will always be accepted.\n        If new is higher than old, there is a chance it will be accepted,\n        less likely for larger differences.\n        '
        with np.errstate(invalid='ignore'):
            prod = -(res_new.fun - res_old.fun) * self.beta
            w = math.exp(min(0, prod))
        rand = self.random_gen.uniform()
        return w >= rand and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        if False:
            return 10
        '\n        f_new and f_old are mandatory in kwargs\n        '
        return bool(self.accept_reject(res_new, res_old))

def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=None, seed=None, *, target_accept_rate=0.5, stepwise_factor=0.9):
    if False:
        i = 10
        return i + 15
    'Find the global minimum of a function using the basin-hopping algorithm.\n\n    Basin-hopping is a two-phase method that combines a global stepping\n    algorithm with local minimization at each step. Designed to mimic\n    the natural process of energy minimization of clusters of atoms, it works\n    well for similar problems with "funnel-like, but rugged" energy landscapes\n    [5]_.\n\n    As the step-taking, step acceptance, and minimization methods are all\n    customizable, this function can also be used to implement other two-phase\n    methods.\n\n    Parameters\n    ----------\n    func : callable ``f(x, *args)``\n        Function to be optimized.  ``args`` can be passed as an optional item\n        in the dict `minimizer_kwargs`\n    x0 : array_like\n        Initial guess.\n    niter : integer, optional\n        The number of basin-hopping iterations. There will be a total of\n        ``niter + 1`` runs of the local minimizer.\n    T : float, optional\n        The "temperature" parameter for the acceptance or rejection criterion.\n        Higher "temperatures" mean that larger jumps in function value will be\n        accepted.  For best results `T` should be comparable to the\n        separation (in function value) between local minima.\n    stepsize : float, optional\n        Maximum step size for use in the random displacement.\n    minimizer_kwargs : dict, optional\n        Extra keyword arguments to be passed to the local minimizer\n        `scipy.optimize.minimize` Some important options could be:\n\n            method : str\n                The minimization method (e.g. ``"L-BFGS-B"``)\n            args : tuple\n                Extra arguments passed to the objective function (`func`) and\n                its derivatives (Jacobian, Hessian).\n\n    take_step : callable ``take_step(x)``, optional\n        Replace the default step-taking routine with this routine. The default\n        step-taking routine is a random displacement of the coordinates, but\n        other step-taking algorithms may be better for some systems.\n        `take_step` can optionally have the attribute ``take_step.stepsize``.\n        If this attribute exists, then `basinhopping` will adjust\n        ``take_step.stepsize`` in order to try to optimize the global minimum\n        search.\n    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional\n        Define a test which will be used to judge whether to accept the\n        step. This will be used in addition to the Metropolis test based on\n        "temperature" `T`. The acceptable return values are True,\n        False, or ``"force accept"``. If any of the tests return False\n        then the step is rejected. If the latter, then this will override any\n        other tests in order to accept the step. This can be used, for example,\n        to forcefully escape from a local minimum that `basinhopping` is\n        trapped in.\n    callback : callable, ``callback(x, f, accept)``, optional\n        A callback function which will be called for all minima found. ``x``\n        and ``f`` are the coordinates and function value of the trial minimum,\n        and ``accept`` is whether that minimum was accepted. This can\n        be used, for example, to save the lowest N minima found. Also,\n        `callback` can be used to specify a user defined stop criterion by\n        optionally returning True to stop the `basinhopping` routine.\n    interval : integer, optional\n        interval for how often to update the `stepsize`\n    disp : bool, optional\n        Set to True to print status messages\n    niter_success : integer, optional\n        Stop the run if the global minimum candidate remains the same for this\n        number of iterations.\n    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional\n\n        If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n        singleton is used.\n        If `seed` is an int, a new ``RandomState`` instance is used,\n        seeded with `seed`.\n        If `seed` is already a ``Generator`` or ``RandomState`` instance then\n        that instance is used.\n        Specify `seed` for repeatable minimizations. The random numbers\n        generated with this seed only affect the default Metropolis\n        `accept_test` and the default `take_step`. If you supply your own\n        `take_step` and `accept_test`, and these functions use random\n        number generation, then those functions are responsible for the state\n        of their random number generator.\n    target_accept_rate : float, optional\n        The target acceptance rate that is used to adjust the `stepsize`.\n        If the current acceptance rate is greater than the target,\n        then the `stepsize` is increased. Otherwise, it is decreased.\n        Range is (0, 1). Default is 0.5.\n\n        .. versionadded:: 1.8.0\n\n    stepwise_factor : float, optional\n        The `stepsize` is multiplied or divided by this stepwise factor upon\n        each update. Range is (0, 1). Default is 0.9.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a `OptimizeResult` object.\n        Important attributes are: ``x`` the solution array, ``fun`` the value\n        of the function at the solution, and ``message`` which describes the\n        cause of the termination. The ``OptimizeResult`` object returned by the\n        selected minimizer at the lowest minimum is also contained within this\n        object and can be accessed through the ``lowest_optimization_result``\n        attribute.  See `OptimizeResult` for a description of other attributes.\n\n    See Also\n    --------\n    minimize :\n        The local minimization function called once for each basinhopping step.\n        `minimizer_kwargs` is passed to this routine.\n\n    Notes\n    -----\n    Basin-hopping is a stochastic algorithm which attempts to find the global\n    minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_\n    [4]_. The algorithm in its current form was described by David Wales and\n    Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.\n\n    The algorithm is iterative with each cycle composed of the following\n    features\n\n    1) random perturbation of the coordinates\n\n    2) local minimization\n\n    3) accept or reject the new coordinates based on the minimized function\n       value\n\n    The acceptance test used here is the Metropolis criterion of standard Monte\n    Carlo algorithms, although there are many other possibilities [3]_.\n\n    This global minimization method has been shown to be extremely efficient\n    for a wide variety of problems in physics and chemistry. It is\n    particularly useful when the function has many minima separated by large\n    barriers. See the `Cambridge Cluster Database\n    <https://www-wales.ch.cam.ac.uk/CCD.html>`_ for databases of molecular\n    systems that have been optimized primarily using basin-hopping. This\n    database includes minimization problems exceeding 300 degrees of freedom.\n\n    See the free software program `GMIN <https://www-wales.ch.cam.ac.uk/GMIN>`_\n    for a Fortran implementation of basin-hopping. This implementation has many\n    variations of the procedure described above, including more\n    advanced step taking algorithms and alternate acceptance criterion.\n\n    For stochastic global optimization there is no way to determine if the true\n    global minimum has actually been found. Instead, as a consistency check,\n    the algorithm can be run from a number of different random starting points\n    to ensure the lowest minimum found in each example has converged to the\n    global minimum. For this reason, `basinhopping` will by default simply\n    run for the number of iterations `niter` and return the lowest minimum\n    found. It is left to the user to ensure that this is in fact the global\n    minimum.\n\n    Choosing `stepsize`:  This is a crucial parameter in `basinhopping` and\n    depends on the problem being solved. The step is chosen uniformly in the\n    region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it\n    should be comparable to the typical separation (in argument values) between\n    local minima of the function being optimized. `basinhopping` will, by\n    default, adjust `stepsize` to find an optimal value, but this may take\n    many iterations. You will get quicker results if you set a sensible\n    initial value for ``stepsize``.\n\n    Choosing `T`: The parameter `T` is the "temperature" used in the\n    Metropolis criterion. Basinhopping steps are always accepted if\n    ``func(xnew) < func(xold)``. Otherwise, they are accepted with\n    probability::\n\n        exp( -(func(xnew) - func(xold)) / T )\n\n    So, for best results, `T` should to be comparable to the typical\n    difference (in function values) between local minima. (The height of\n    "walls" between local minima is irrelevant.)\n\n    If `T` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all\n    steps that increase energy are rejected.\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,\n        Cambridge, UK.\n    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and\n        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to\n        110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.\n    .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the\n        multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,\n        1987, 84, 6611.\n    .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,\n        crystals, and biomolecules, Science, 1999, 285, 1368.\n    .. [5] Olson, B., Hashmi, I., Molloy, K., and Shehu1, A., Basin Hopping as\n        a General and Versatile Optimization Framework for the Characterization\n        of Biological Macromolecules, Advances in Artificial Intelligence,\n        Volume 2012 (2012), Article ID 674832, :doi:`10.1155/2012/674832`\n\n    Examples\n    --------\n    The following example is a 1-D minimization problem, with many\n    local minima superimposed on a parabola.\n\n    >>> import numpy as np\n    >>> from scipy.optimize import basinhopping\n    >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x\n    >>> x0 = [1.]\n\n    Basinhopping, internally, uses a local minimization algorithm. We will use\n    the parameter `minimizer_kwargs` to tell basinhopping which algorithm to\n    use and how to set up that minimizer. This parameter will be passed to\n    `scipy.optimize.minimize`.\n\n    >>> minimizer_kwargs = {"method": "BFGS"}\n    >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200)\n    >>> print("global minimum: x = %.4f, f(x) = %.4f" % (ret.x, ret.fun))\n    global minimum: x = -0.1951, f(x) = -1.0009\n\n    Next consider a 2-D minimization problem. Also, this time, we\n    will use gradient information to significantly speed up the search.\n\n    >>> def func2d(x):\n    ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +\n    ...                                                            0.2) * x[0]\n    ...     df = np.zeros(2)\n    ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2\n    ...     df[1] = 2. * x[1] + 0.2\n    ...     return f, df\n\n    We\'ll also use a different local minimization algorithm. Also, we must tell\n    the minimizer that our function returns both energy and gradient (Jacobian).\n\n    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}\n    >>> x0 = [1.0, 1.0]\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200)\n    >>> print("global minimum: x = [%.4f, %.4f], f(x) = %.4f" % (ret.x[0],\n    ...                                                           ret.x[1],\n    ...                                                           ret.fun))\n    global minimum: x = [-0.1951, -0.1000], f(x) = -1.0109\n\n    Here is an example using a custom step-taking routine. Imagine you want\n    the first coordinate to take larger steps than the rest of the coordinates.\n    This can be implemented like so:\n\n    >>> class MyTakeStep:\n    ...    def __init__(self, stepsize=0.5):\n    ...        self.stepsize = stepsize\n    ...        self.rng = np.random.default_rng()\n    ...    def __call__(self, x):\n    ...        s = self.stepsize\n    ...        x[0] += self.rng.uniform(-2.*s, 2.*s)\n    ...        x[1:] += self.rng.uniform(-s, s, x[1:].shape)\n    ...        return x\n\n    Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude\n    of `stepsize` to optimize the search. We\'ll use the same 2-D function as\n    before\n\n    >>> mytakestep = MyTakeStep()\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=200, take_step=mytakestep)\n    >>> print("global minimum: x = [%.4f, %.4f], f(x) = %.4f" % (ret.x[0],\n    ...                                                           ret.x[1],\n    ...                                                           ret.fun))\n    global minimum: x = [-0.1951, -0.1000], f(x) = -1.0109\n\n    Now, let\'s do an example using a custom callback function which prints the\n    value of every minimum found\n\n    >>> def print_fun(x, f, accepted):\n    ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))\n\n    We\'ll run it for only 10 basinhopping steps this time.\n\n    >>> rng = np.random.default_rng()\n    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,\n    ...                    niter=10, callback=print_fun, seed=rng)\n    at minimum 0.4159 accepted 1\n    at minimum -0.4317 accepted 1\n    at minimum -1.0109 accepted 1\n    at minimum -0.9073 accepted 1\n    at minimum -0.4317 accepted 0\n    at minimum -0.1021 accepted 1\n    at minimum -0.7425 accepted 1\n    at minimum -0.9073 accepted 1\n    at minimum -0.4317 accepted 0\n    at minimum -0.7425 accepted 1\n    at minimum -0.9073 accepted 1\n\n    The minimum at -1.0109 is actually the global minimum, found already on the\n    8th iteration.\n\n    '
    if target_accept_rate <= 0.0 or target_accept_rate >= 1.0:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0.0 or stepwise_factor >= 1.0:
        raise ValueError('stepwise_factor has to be in range (0, 1)')
    x0 = np.array(x0)
    rng = check_random_state(seed)
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func, **minimizer_kwargs)
    if take_step is not None:
        if not callable(take_step):
            raise TypeError('take_step must be callable')
        if hasattr(take_step, 'stepsize'):
            take_step_wrapped = AdaptiveStepsize(take_step, interval=interval, accept_rate=target_accept_rate, factor=stepwise_factor, verbose=disp)
        else:
            take_step_wrapped = take_step
    else:
        displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval, accept_rate=target_accept_rate, factor=stepwise_factor, verbose=disp)
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError('accept_test must be callable')
        accept_tests = [accept_test]
    metropolis = Metropolis(T, random_gen=rng)
    accept_tests.append(metropolis)
    if niter_success is None:
        niter_success = niter + 2
    bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped, accept_tests, disp=disp)
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)
    (count, i) = (0, 0)
    message = ['requested number of basinhopping iterations completed successfully']
    for i in range(niter):
        new_global_min = bh.one_cycle()
        if callable(callback):
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ['callback function requested stop early byreturning True']
                    break
        count += 1
        if new_global_min:
            count = 0
        elif count > niter_success:
            message = ['success condition satisfied']
            break
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success
    return res