from typing import Callable, Dict, Optional, Union
import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import NullTypeGradError
from scipy import optimize
import pymc as pm
__all__ = ['find_constrained_prior']

def find_constrained_prior(distribution: pm.Distribution, lower: float, upper: float, init_guess: Dict[str, float], mass: float=0.95, fixed_params: Optional[Dict[str, float]]=None, mass_below_lower: Optional[float]=None, **kwargs) -> Dict[str, float]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Find optimal parameters to get `mass` % of probability\n    of a :ref:`distribution <api_distributions>` between `lower` and `upper`.\n\n    Note: only works for one- and two-parameter distributions, as there\n    are exactly two constraints. Fix some combination of parameters\n    if you want to use it on >=3-parameter distributions.\n\n    Parameters\n    ----------\n    distribution : Distribution\n        PyMC distribution you want to set a prior on.\n        Needs to have a ``logcdf`` method implemented in PyMC.\n    lower : float\n        Lower bound to get `mass` % of probability of `pm_dist`.\n    upper : float\n        Upper bound to get `mass` % of probability of `pm_dist`.\n    init_guess : dict of {str : float}\n        Initial guess for ``scipy.optimize.least_squares`` to find the\n        optimal parameters of `pm_dist` fitting the interval constraint.\n        Must be a dictionary with the name of the PyMC distribution\'s\n        parameter as keys and the initial guess as values.\n    mass : float, default 0.95\n        Share of the probability mass we want between ``lower`` and ``upper``.\n        Defaults to 95%.\n    fixed_params : str or float, optional, default None\n        Only used when `pm_dist` has at least three parameters.\n        Dictionary of fixed parameters, so that there are only 2 to optimize.\n        For instance, for a StudentT, you fix nu to a constant and get the optimized\n        mu and sigma.\n    mass_below_lower : float, optional, default None\n        The probability mass below the ``lower`` bound. If ``None``,\n        defaults to ``(1 - mass) / 2``, which implies that the probability\n        mass below the ``lower`` value will be equal to the probability\n        mass above the ``upper`` value.\n\n    Returns\n    -------\n    opt_params : dict\n        The optimized distribution parameters as a dictionary.\n        Dictionary keys are the parameter names and\n        dictionary values are the optimized parameter values.\n\n    Notes\n    -----\n    Optional keyword arguments can be passed to ``find_constrained_prior``. These will be\n    delivered to the underlying call to :external:py:func:`scipy.optimize.minimize`.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        # get parameters obeying constraints\n        opt_params = pm.find_constrained_prior(\n            pm.Gamma, lower=0.1, upper=0.4, mass=0.75, init_guess={"alpha": 1, "beta": 10}\n        )\n\n        # use these parameters to draw random samples\n        samples = pm.Gamma.dist(**opt_params, size=100).eval()\n\n        # use these parameters in a model\n        with pm.Model():\n            x = pm.Gamma(\'x\', **opt_params)\n\n        # specify fixed values before optimization\n        opt_params = pm.find_constrained_prior(\n            pm.StudentT,\n            lower=0,\n            upper=1,\n            init_guess={"mu": 5, "sigma": 2},\n            fixed_params={"nu": 7},\n        )\n\n    Under some circumstances, you might not want to have the same cumulative\n    probability below the ``lower`` threshold and above the ``upper`` threshold.\n    For example, you might want to constrain an Exponential distribution to\n    find the parameter that yields 90% of the mass below the ``upper`` bound,\n    and have zero mass below ``lower``. You can do that with the following call\n    to ``find_constrained_prior``\n\n    .. code-block:: python\n\n        opt_params = pm.find_constrained_prior(\n            pm.Exponential,\n            lower=0,\n            upper=3.,\n            mass=0.9,\n            init_guess={"lam": 1},\n            mass_below_lower=0,\n        )\n    '
    assert 0.01 <= mass <= 0.99, f'This function optimizes the mass of the given distribution +/- 1%, so `mass` has to be between 0.01 and 0.99. You provided {mass}.'
    if mass_below_lower is None:
        mass_below_lower = (1 - mass) / 2
    if np.any(np.asarray(distribution.rv_op.ndims_params) != 0):
        raise NotImplementedError('`pm.find_constrained_prior` does not work with non-scalar parameters yet.\nFeel free to open a pull request on PyMC repo if you really need this feature.')
    dist_params = pt.vector('dist_params')
    params_to_optim = {arg_name: dist_params[i] for (arg_name, i) in zip(init_guess.keys(), range(len(init_guess)))}
    if fixed_params is not None:
        params_to_optim.update(fixed_params)
    dist = distribution.dist(**params_to_optim)
    try:
        logcdf_lower = pm.logcdf(dist, pm.floatX(lower))
        logcdf_upper = pm.logcdf(dist, pm.floatX(upper))
    except AttributeError:
        raise AttributeError(f"You cannot use `find_constrained_prior` with {distribution} -- it doesn't have a logcdf method yet.\nOpen an issue or, even better, a pull request on PyMC repo if you really need it.")
    target = (pt.exp(logcdf_lower) - mass_below_lower) ** 2
    target_fn = pm.pytensorf.compile_pymc([dist_params], target, allow_input_downcast=True)
    constraint = pt.exp(logcdf_upper) - pt.exp(logcdf_lower)
    constraint_fn = pm.pytensorf.compile_pymc([dist_params], constraint, allow_input_downcast=True)
    jac: Union[str, Callable]
    constraint_jac: Union[str, Callable]
    try:
        pytensor_jac = pm.gradient(target, [dist_params])
        jac = pm.pytensorf.compile_pymc([dist_params], pytensor_jac, allow_input_downcast=True)
        pytensor_constraint_jac = pm.gradient(constraint, [dist_params])
        constraint_jac = pm.pytensorf.compile_pymc([dist_params], pytensor_constraint_jac, allow_input_downcast=True)
    except (NotImplementedError, NullTypeGradError):
        jac = '2-point'
        constraint_jac = '2-point'
    cons = optimize.NonlinearConstraint(constraint_fn, lb=mass, ub=mass, jac=constraint_jac)
    opt = optimize.minimize(target_fn, x0=list(init_guess.values()), jac=jac, constraints=cons, **kwargs)
    if not opt.success:
        raise ValueError(f'Optimization of parameters failed.\nOptimization termination details:\n{opt}')
    opt_params = {param_name: param_value for (param_name, param_value) in zip(init_guess.keys(), opt.x)}
    if fixed_params is not None:
        opt_params.update(fixed_params)
    return opt_params