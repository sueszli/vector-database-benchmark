"""
State Space Representation, Kalman Filter, Smoother, and Simulation Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
SIMULATION_STATE = 1
SIMULATION_DISTURBANCE = 4
SIMULATION_ALL = SIMULATION_STATE | SIMULATION_DISTURBANCE

def check_random_state(seed=None):
    if False:
        return 10
    'Turn `seed` into a `numpy.random.Generator` instance.\n    Parameters\n    ----------\n    seed : {None, int, Generator, RandomState}, optional\n        If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n        singleton is used.\n        If `seed` is an int, a new ``numpy.random.RandomState`` instance\n        is used, seeded with `seed`.\n        If `seed` is already a ``numpy.random.Generator`` or\n        ``numpy.random.RandomState`` instance then that instance is used.\n    Returns\n    -------\n    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}\n        Random number generator.\n    '
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a numpy.random.Generator instance')

class SimulationSmoother(KalmanSmoother):
    """
    State space representation of a time series process, with Kalman filter
    and smoother, and with simulation smoother.

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    simulation_smooth_results_class : class, optional
        Default results class to use to save output of simulation smoothing.
        Default is `SimulationSmoothResults`. If specified, class must extend
        from `SimulationSmoothResults`.
    simulation_smoother_classes : dict, optional
        Dictionary with BLAS prefixes as keys and classes as values.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, for Kalman smoothing
        options, or for Simulation smoothing options.
        See `Representation`, `KalmanFilter`, and `KalmanSmoother` for more
        details.
    """
    simulation_outputs = ['simulate_state', 'simulate_disturbance', 'simulate_all']

    def __init__(self, k_endog, k_states, k_posdef=None, simulation_smooth_results_class=None, simulation_smoother_classes=None, **kwargs):
        if False:
            while True:
                i = 10
        super(SimulationSmoother, self).__init__(k_endog, k_states, k_posdef, **kwargs)
        if simulation_smooth_results_class is None:
            simulation_smooth_results_class = SimulationSmoothResults
        self.simulation_smooth_results_class = simulation_smooth_results_class
        self.prefix_simulation_smoother_map = simulation_smoother_classes if simulation_smoother_classes is not None else tools.prefix_simulation_smoother_map.copy()
        self._simulators = {}

    def get_simulation_output(self, simulation_output=None, simulate_state=None, simulate_disturbance=None, simulate_all=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Get simulation output bitmask\n\n        Helper method to get final simulation output bitmask from a set of\n        optional arguments including the bitmask itself and possibly boolean\n        flags.\n\n        Parameters\n        ----------\n        simulation_output : int, optional\n            Simulation output bitmask. If this is specified, it is simply\n            returned and the other arguments are ignored.\n        simulate_state : bool, optional\n            Whether or not to include the state in the simulation output.\n        simulate_disturbance : bool, optional\n            Whether or not to include the state and observation disturbances\n            in the simulation output.\n        simulate_all : bool, optional\n            Whether or not to include all simulation output.\n        \\*\\*kwargs\n            Additional keyword arguments. Present so that calls to this method\n            can use \\*\\*kwargs without clearing out additional arguments.\n        '
        if simulation_output is None:
            simulation_output = 0
            if simulate_state:
                simulation_output |= SIMULATION_STATE
            if simulate_disturbance:
                simulation_output |= SIMULATION_DISTURBANCE
            if simulate_all:
                simulation_output |= SIMULATION_ALL
            if simulation_output == 0:
                argument_set = not all([simulate_state is None, simulate_disturbance is None, simulate_all is None])
                if argument_set:
                    raise ValueError('Invalid simulation output options: given options would result in no output.')
                simulation_output = self.smoother_output
        return simulation_output

    def _simulate(self, nsimulations, simulator=None, random_state=None, return_simulator=False, **kwargs):
        if False:
            i = 10
            return i + 15
        if simulator is None:
            simulator = self.simulator(nsimulations, random_state=random_state)
        simulator.simulate(**kwargs)
        simulated_obs = np.array(simulator.generated_obs, copy=True)
        simulated_state = np.array(simulator.generated_state, copy=True)
        out = (simulated_obs.T[:nsimulations], simulated_state.T[:nsimulations])
        if return_simulator:
            out = out + (simulator,)
        return out

    def simulator(self, nsimulations, random_state=None):
        if False:
            for i in range(10):
                print('nop')
        return self.simulation_smoother(simulation_output=0, method='kfs', nobs=nsimulations, random_state=random_state)

    def simulation_smoother(self, simulation_output=None, method='kfs', results_class=None, prefix=None, nobs=-1, random_state=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Retrieve a simulation smoother for the statespace model.\n\n        Parameters\n        ----------\n        simulation_output : int, optional\n            Determines which simulation smoother output is calculated.\n            Default is all (including state and disturbances).\n        method : {'kfs', 'cfa'}, optional\n            Method for simulation smoothing. If `method='kfs'`, then the\n            simulation smoother is based on Kalman filtering and smoothing\n            recursions. If `method='cfa'`, then the simulation smoother is\n            based on the Cholesky Factor Algorithm (CFA) approach. The CFA\n            approach is not applicable to all state space models, but can be\n            faster for the cases in which it is supported.\n        results_class : class, optional\n            Default results class to use to save output of simulation\n            smoothing. Default is `SimulationSmoothResults`. If specified,\n            class must extend from `SimulationSmoothResults`.\n        prefix : str\n            The prefix of the datatype. Usually only used internally.\n        nobs : int\n            The number of observations to simulate. If set to anything other\n            than -1, only simulation will be performed (i.e. simulation\n            smoothing will not be performed), so that only the `generated_obs`\n            and `generated_state` attributes will be available.\n        random_state : {None, int, Generator, RandomState}, optional\n            If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n            singleton is used.\n            If `seed` is an int, a new ``numpy.random.RandomState`` instance\n            is used, seeded with `seed`.\n            If `seed` is already a ``numpy.random.Generator`` or\n            ``numpy.random.RandomState`` instance then that instance is used.\n        **kwargs\n            Additional keyword arguments, used to set the simulation output.\n            See `set_simulation_output` for more details.\n\n        Returns\n        -------\n        SimulationSmoothResults\n        "
        method = method.lower()
        if method == 'cfa':
            if simulation_output not in [None, 1, -1]:
                raise ValueError('Can only retrieve simulations of the state vector using the CFA simulation smoother.')
            return CFASimulationSmoother(self)
        elif method != 'kfs':
            raise ValueError('Invalid simulation smoother method "%s". Valid methods are "kfs" or "cfa".' % method)
        if results_class is None:
            results_class = self.simulation_smooth_results_class
        if not issubclass(results_class, SimulationSmoothResults):
            raise ValueError('Invalid results class provided.')
        (prefix, dtype, create_smoother, create_filter, create_statespace) = self._initialize_smoother()
        simulation_output = self.get_simulation_output(simulation_output, **kwargs)
        smoother_output = kwargs.get('smoother_output', simulation_output)
        filter_method = kwargs.get('filter_method', self.filter_method)
        inversion_method = kwargs.get('inversion_method', self.inversion_method)
        stability_method = kwargs.get('stability_method', self.stability_method)
        conserve_memory = kwargs.get('conserve_memory', self.conserve_memory)
        filter_timing = kwargs.get('filter_timing', self.filter_timing)
        loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
        tolerance = kwargs.get('tolerance', self.tolerance)
        cls = self.prefix_simulation_smoother_map[prefix]
        simulation_smoother = cls(self._statespaces[prefix], filter_method, inversion_method, stability_method, conserve_memory, filter_timing, tolerance, loglikelihood_burn, smoother_output, simulation_output, nobs)
        results = results_class(self, simulation_smoother, random_state=random_state)
        return results

class SimulationSmoothResults:
    """
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation
    simulation_smoother : {{prefix}}SimulationSmoother object
        The Cython simulation smoother object with which to simulation smooth.
    random_state : {None, int, Generator, RandomState}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``numpy.random.RandomState`` instance
        is used, seeded with `seed`.
        If `seed` is already a ``numpy.random.Generator`` or
        ``numpy.random.RandomState`` instance then that instance is used.

    Attributes
    ----------
    model : Representation
        A Statespace representation
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    simulation_output : int
        Bitmask controlling simulation output.
    simulate_state : bool
        Flag for if the state is included in simulation output.
    simulate_disturbance : bool
        Flag for if the state and observation disturbances are included in
        simulation output.
    simulate_all : bool
        Flag for if simulation output should include everything.
    generated_measurement_disturbance : ndarray
        Measurement disturbance variates used to genereate the observation
        vector.
    generated_state_disturbance : ndarray
        State disturbance variates used to genereate the state and
        observation vectors.
    generated_obs : ndarray
        Generated observation vector produced as a byproduct of simulation
        smoothing.
    generated_state : ndarray
        Generated state vector produced as a byproduct of simulation smoothing.
    simulated_state : ndarray
        Simulated state.
    simulated_measurement_disturbance : ndarray
        Simulated measurement disturbance.
    simulated_state_disturbance : ndarray
        Simulated state disturbance.
    """

    def __init__(self, model, simulation_smoother, random_state=None):
        if False:
            return 10
        self.model = model
        self.prefix = model.prefix
        self.dtype = model.dtype
        self._simulation_smoother = simulation_smoother
        self.random_state = check_random_state(random_state)
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

    @property
    def simulation_output(self):
        if False:
            i = 10
            return i + 15
        return self._simulation_smoother.simulation_output

    @simulation_output.setter
    def simulation_output(self, value):
        if False:
            while True:
                i = 10
        self._simulation_smoother.simulation_output = value

    @property
    def simulate_state(self):
        if False:
            i = 10
            return i + 15
        return bool(self.simulation_output & SIMULATION_STATE)

    @simulate_state.setter
    def simulate_state(self, value):
        if False:
            i = 10
            return i + 15
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_STATE
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_STATE

    @property
    def simulate_disturbance(self):
        if False:
            while True:
                i = 10
        return bool(self.simulation_output & SIMULATION_DISTURBANCE)

    @simulate_disturbance.setter
    def simulate_disturbance(self, value):
        if False:
            i = 10
            return i + 15
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_DISTURBANCE
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_DISTURBANCE

    @property
    def simulate_all(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.simulation_output & SIMULATION_ALL)

    @simulate_all.setter
    def simulate_all(self, value):
        if False:
            return 10
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_ALL
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_ALL

    @property
    def generated_measurement_disturbance(self):
        if False:
            print('Hello World!')
        '\n        Randomly drawn measurement disturbance variates\n\n        Used to construct `generated_obs`.\n\n        Notes\n        -----\n\n        .. math::\n\n           \\varepsilon_t^+ ~ N(0, H_t)\n\n        If `disturbance_variates` were provided to the `simulate()` method,\n        then this returns those variates (which were N(0,1)) transformed to the\n        distribution above.\n        '
        if self._generated_measurement_disturbance is None:
            self._generated_measurement_disturbance = np.array(self._simulation_smoother.measurement_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_endog)
        return self._generated_measurement_disturbance

    @property
    def generated_state_disturbance(self):
        if False:
            return 10
        '\n        Randomly drawn state disturbance variates, used to construct\n        `generated_state` and `generated_obs`.\n\n        Notes\n        -----\n\n        .. math::\n\n            \\eta_t^+ ~ N(0, Q_t)\n\n        If `disturbance_variates` were provided to the `simulate()` method,\n        then this returns those variates (which were N(0,1)) transformed to the\n        distribution above.\n        '
        if self._generated_state_disturbance is None:
            self._generated_state_disturbance = np.array(self._simulation_smoother.state_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_posdef)
        return self._generated_state_disturbance

    @property
    def generated_obs(self):
        if False:
            i = 10
            return i + 15
        '\n        Generated vector of observations by iterating on the observation and\n        transition equations, given a random initial state draw and random\n        disturbance draws.\n\n        Notes\n        -----\n\n        .. math::\n\n            y_t^+ = d_t + Z_t \\alpha_t^+ + \\varepsilon_t^+\n        '
        if self._generated_obs is None:
            self._generated_obs = np.array(self._simulation_smoother.generated_obs, copy=True)
        return self._generated_obs

    @property
    def generated_state(self):
        if False:
            return 10
        '\n        Generated vector of states by iterating on the transition equation,\n        given a random initial state draw and random disturbance draws.\n\n        Notes\n        -----\n\n        .. math::\n\n            \\alpha_{t+1}^+ = c_t + T_t \\alpha_t^+ + \\eta_t^+\n        '
        if self._generated_state is None:
            self._generated_state = np.array(self._simulation_smoother.generated_state, copy=True)
        return self._generated_state

    @property
    def simulated_state(self):
        if False:
            i = 10
            return i + 15
        '\n        Random draw of the state vector from its conditional distribution.\n\n        Notes\n        -----\n\n        .. math::\n\n            \\alpha ~ p(\\alpha \\mid Y_n)\n        '
        if self._simulated_state is None:
            self._simulated_state = np.array(self._simulation_smoother.simulated_state, copy=True)
        return self._simulated_state

    @property
    def simulated_measurement_disturbance(self):
        if False:
            print('Hello World!')
        '\n        Random draw of the measurement disturbance vector from its conditional\n        distribution.\n\n        Notes\n        -----\n\n        .. math::\n\n            \\varepsilon ~ N(\\hat \\varepsilon, Var(\\hat \\varepsilon \\mid Y_n))\n        '
        if self._simulated_measurement_disturbance is None:
            self._simulated_measurement_disturbance = np.array(self._simulation_smoother.simulated_measurement_disturbance, copy=True)
        return self._simulated_measurement_disturbance

    @property
    def simulated_state_disturbance(self):
        if False:
            while True:
                i = 10
        '\n        Random draw of the state disturbanc e vector from its conditional\n        distribution.\n\n        Notes\n        -----\n\n        .. math::\n\n            \\eta ~ N(\\hat \\eta, Var(\\hat \\eta \\mid Y_n))\n        '
        if self._simulated_state_disturbance is None:
            self._simulated_state_disturbance = np.array(self._simulation_smoother.simulated_state_disturbance, copy=True)
        return self._simulated_state_disturbance

    def simulate(self, simulation_output=-1, disturbance_variates=None, measurement_disturbance_variates=None, state_disturbance_variates=None, initial_state_variates=None, pretransformed=None, pretransformed_measurement_disturbance_variates=None, pretransformed_state_disturbance_variates=None, pretransformed_initial_state_variates=False, random_state=None):
        if False:
            while True:
                i = 10
        "\n        Perform simulation smoothing\n\n        Does not return anything, but populates the object's `simulated_*`\n        attributes, as specified by simulation output.\n\n        Parameters\n        ----------\n        simulation_output : int, optional\n            Bitmask controlling simulation output. Default is to use the\n            simulation output defined in object initialization.\n        measurement_disturbance_variates : array_like, optional\n            If specified, these are the shocks to the measurement equation,\n            :math:`\\varepsilon_t`. If unspecified, these are automatically\n            generated using a pseudo-random number generator. If specified,\n            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the\n            same as in the state space model.\n        state_disturbance_variates : array_like, optional\n            If specified, these are the shocks to the state equation,\n            :math:`\\eta_t`. If unspecified, these are automatically\n            generated using a pseudo-random number generator. If specified,\n            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the\n            same as in the state space model.\n        initial_state_variates : array_like, optional\n            If specified, this is the state vector at time zero, which should\n            be shaped (`k_states` x 1), where `k_states` is the same as in the\n            state space model. If unspecified, but the model has been\n            initialized, then that initialization is used.\n        initial_state_variates : array_likes, optional\n            Random values to use as initial state variates. Usually only\n            specified if results are to be replicated (e.g. to enforce a seed)\n            or for testing. If not specified, random variates are drawn.\n        pretransformed_measurement_disturbance_variates : bool, optional\n            If `measurement_disturbance_variates` is provided, this flag\n            indicates whether it should be directly used as the shocks. If\n            False, then it is assumed to contain draws from the standard Normal\n            distribution that must be transformed using the `obs_cov`\n            covariance matrix. Default is False.\n        pretransformed_state_disturbance_variates : bool, optional\n            If `state_disturbance_variates` is provided, this flag indicates\n            whether it should be directly used as the shocks. If False, then it\n            is assumed to contain draws from the standard Normal distribution\n            that must be transformed using the `state_cov` covariance matrix.\n            Default is False.\n        pretransformed_initial_state_variates : bool, optional\n            If `initial_state_variates` is provided, this flag indicates\n            whether it should be directly used as the initial_state. If False,\n            then it is assumed to contain draws from the standard Normal\n            distribution that must be transformed using the `initial_state_cov`\n            covariance matrix. Default is False.\n        random_state : {None, int, Generator, RandomState}, optional\n            If `seed` is None (or `np.random`), the `numpy.random.RandomState`\n            singleton is used.\n            If `seed` is an int, a new ``numpy.random.RandomState`` instance\n            is used, seeded with `seed`.\n            If `seed` is already a ``numpy.random.Generator`` or\n            ``numpy.random.RandomState`` instance then that instance is used.\n        disturbance_variates : bool, optional\n            Deprecated, please use pretransformed_measurement_shocks and\n            pretransformed_state_shocks instead.\n\n            .. deprecated:: 0.14.0\n\n               Use ``measurement_disturbance_variates`` and\n               ``state_disturbance_variates`` as replacements.\n\n        pretransformed : bool, optional\n            Deprecated, please use pretransformed_measurement_shocks and\n            pretransformed_state_shocks instead.\n\n            .. deprecated:: 0.14.0\n\n               Use ``pretransformed_measurement_disturbance_variates`` and\n               ``pretransformed_state_disturbance_variates`` as replacements.\n        "
        if disturbance_variates is not None:
            msg = '`disturbance_variates` keyword is deprecated, use `measurement_disturbance_variates` and `state_disturbance_variates` instead.'
            warnings.warn(msg, FutureWarning)
            if measurement_disturbance_variates is not None or state_disturbance_variates is not None:
                raise ValueError('Cannot use `disturbance_variates` in combination with  `measurement_disturbance_variates` or `state_disturbance_variates`.')
            if disturbance_variates is not None:
                disturbance_variates = disturbance_variates.ravel()
                n_mds = self.model.nobs * self.model.k_endog
                measurement_disturbance_variates = disturbance_variates[:n_mds]
                state_disturbance_variates = disturbance_variates[n_mds:]
        if pretransformed is not None:
            msg = '`pretransformed` keyword is deprecated, use `pretransformed_measurement_disturbance_variates` and `pretransformed_state_disturbance_variates` instead.'
            warnings.warn(msg, FutureWarning)
            if pretransformed_measurement_disturbance_variates is not None or pretransformed_state_disturbance_variates is not None:
                raise ValueError('Cannot use `pretransformed` in combination with  `pretransformed_measurement_disturbance_variates` or `pretransformed_state_disturbance_variates`.')
            if pretransformed is not None:
                pretransformed_measurement_disturbance_variates = pretransformed
                pretransformed_state_disturbance_variates = pretransformed
        if pretransformed_measurement_disturbance_variates is None:
            pretransformed_measurement_disturbance_variates = False
        if pretransformed_state_disturbance_variates is None:
            pretransformed_state_disturbance_variates = False
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_state = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None
        if random_state is None:
            random_state = self.random_state
        else:
            random_state = check_random_state(random_state)
        (prefix, dtype, create_smoother, create_filter, create_statespace) = self.model._initialize_smoother()
        if create_statespace:
            raise ValueError('The simulation smoother currently cannot replace the underlying _{{prefix}}Representation model object if it changes (which happens e.g. if the dimensions of some system matrices change.')
        self.model._initialize_state(prefix=prefix)
        if measurement_disturbance_variates is not None:
            self._simulation_smoother.set_measurement_disturbance_variates(np.array(measurement_disturbance_variates, dtype=self.dtype).ravel(), pretransformed=pretransformed_measurement_disturbance_variates)
        else:
            self._simulation_smoother.draw_measurement_disturbance_variates(random_state)
        if state_disturbance_variates is not None:
            self._simulation_smoother.set_state_disturbance_variates(np.array(state_disturbance_variates, dtype=self.dtype).ravel(), pretransformed=pretransformed_state_disturbance_variates)
        else:
            self._simulation_smoother.draw_state_disturbance_variates(random_state)
        if initial_state_variates is not None:
            if pretransformed_initial_state_variates:
                self._simulation_smoother.set_initial_state(np.array(initial_state_variates, dtype=self.dtype))
            else:
                self._simulation_smoother.set_initial_state_variates(np.array(initial_state_variates, dtype=self.dtype), pretransformed=False)
        else:
            self._simulation_smoother.draw_initial_state_variates(random_state)
        self._simulation_smoother.simulate(simulation_output)