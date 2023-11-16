from abc import abstractmethod
from typing import Callable, List, Tuple, Union, cast
import numpy as np
from numpy.random import uniform
from pymc.blocking import DictToArrayBijection, PointType, RaveledVars, StatsType
from pymc.model import modelcontext
from pymc.step_methods.compound import BlockedStep
from pymc.util import get_var_name
__all__ = ['ArrayStep', 'ArrayStepShared', 'metrop_select']

class ArrayStep(BlockedStep):
    """
    Blocked step method that is generalized to accept vectors of variables.

    Parameters
    ----------
    vars: list
        List of value variables for sampler.
    fs: list of logp PyTensor functions
    allvars: Boolean (default False)
    blocked: Boolean (default True)
    """

    def __init__(self, vars, fs, allvars=False, blocked=True):
        if False:
            while True:
                i = 10
        self.vars = vars
        self.fs = fs
        self.allvars = allvars
        self.blocked = blocked

    def step(self, point: PointType) -> Tuple[PointType, StatsType]:
        if False:
            for i in range(10):
                print('nop')
        partial_funcs_and_point: List[Union[Callable, PointType]] = [DictToArrayBijection.mapf(x, start_point=point) for x in self.fs]
        if self.allvars:
            partial_funcs_and_point.append(point)
        var_dict = {cast(str, v.name): point[cast(str, v.name)] for v in self.vars}
        apoint = DictToArrayBijection.map(var_dict)
        (apoint_new, stats) = self.astep(apoint, *partial_funcs_and_point)
        if not isinstance(apoint_new, RaveledVars):
            apoint_new = RaveledVars(apoint_new, apoint.point_map_info)
        point_new = DictToArrayBijection.rmap(apoint_new, start_point=point)
        return (point_new, stats)

    @abstractmethod
    def astep(self, apoint: RaveledVars, *args) -> Tuple[RaveledVars, StatsType]:
        if False:
            print('Hello World!')
        'Perform a single sample step in a raveled and concatenated parameter space.'

class ArrayStepShared(BlockedStep):
    """Faster version of ArrayStep that requires the substep method that does not wrap
       the functions the step method uses.

    Works by setting shared variables before using the step. This eliminates the mapping
    and unmapping overhead as well as moving fewer variables around.
    """

    def __init__(self, vars, shared, blocked=True):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        vars: list of sampling value variables\n        shared: dict of PyTensor variable -> shared variable\n        blocked: Boolean (default True)\n        '
        self.vars = vars
        self.shared = {get_var_name(var): shared for (var, shared) in shared.items()}
        self.blocked = blocked

    def step(self, point: PointType) -> Tuple[PointType, StatsType]:
        if False:
            print('Hello World!')
        for (name, shared_var) in self.shared.items():
            shared_var.set_value(point[name])
        var_dict = {cast(str, v.name): point[cast(str, v.name)] for v in self.vars}
        q = DictToArrayBijection.map(var_dict)
        (apoint, stats) = self.astep(q)
        if not isinstance(apoint, RaveledVars):
            apoint = RaveledVars(apoint, q.point_map_info)
        new_point = DictToArrayBijection.rmap(apoint, start_point=point)
        return (new_point, stats)

    @abstractmethod
    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, StatsType]:
        if False:
            i = 10
            return i + 15
        'Perform a single sample step in a raveled and concatenated parameter space.'

class PopulationArrayStepShared(ArrayStepShared):
    """Version of ArrayStepShared that allows samplers to access the states
    of other chains in the population.

    Works by linking a list of Points that is updated as the chains are iterated.
    """

    def __init__(self, vars, shared, blocked=True):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        vars: list of sampling value variables\n        shared: dict of PyTensor variable -> shared variable\n        blocked: Boolean (default True)\n        '
        self.population = None
        self.this_chain = None
        self.other_chains = None
        return super().__init__(vars, shared, blocked)

    def link_population(self, population, chain_index):
        if False:
            i = 10
            return i + 15
        'Links the sampler to the population.\n\n        Parameters\n        ----------\n        population: list of Points. (The elements of this list must be\n            replaced with current chain states in every iteration.)\n        chain_index: int of the index of this sampler in the population\n        '
        self.population = population
        self.this_chain = chain_index
        self.other_chains = [c for c in range(len(population)) if c != chain_index]
        if not len(self.other_chains) > 1:
            raise ValueError('Population is just {} + {}. This is too small and the error should have been raised earlier.'.format(self.this_chain, self.other_chains))
        return

class GradientSharedStep(ArrayStepShared):

    def __init__(self, vars, model=None, blocked=True, dtype=None, logp_dlogp_func=None, **pytensor_kwargs):
        if False:
            for i in range(10):
                print('nop')
        model = modelcontext(model)
        if logp_dlogp_func is None:
            func = model.logp_dlogp_function(vars, dtype=dtype, **pytensor_kwargs)
        else:
            func = logp_dlogp_func
        self._logp_dlogp_func = func
        super().__init__(vars, func._extra_vars_shared, blocked)

    def step(self, point) -> Tuple[PointType, StatsType]:
        if False:
            print('Hello World!')
        self._logp_dlogp_func._extra_are_set = True
        return super().step(point)

def metrop_select(mr: np.ndarray, q: np.ndarray, q0: np.ndarray) -> Tuple[np.ndarray, bool]:
    if False:
        for i in range(10):
            print('nop')
    'Perform rejection/acceptance step for Metropolis class samplers.\n\n    Returns the new sample q if a uniform random number is less than the\n    metropolis acceptance rate (`mr`), and the old sample otherwise, along\n    with a boolean indicating whether the sample was accepted.\n\n    Parameters\n    ----------\n    mr: float, Metropolis acceptance rate\n    q: proposed sample\n    q0: current sample\n\n    Returns\n    -------\n    q or q0\n    '
    if np.isfinite(mr) and np.log(uniform()) < mr:
        return (q, True)
    else:
        return (q0, False)