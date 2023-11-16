"""Continuous-time population dynamics."""
import numpy as np

def replicator(state, fitness):
    if False:
        i = 10
        return i + 15
    'Continuous-time replicator dynamics.\n\n  This is the standard form of the continuous-time replicator dynamics also\n  known as selection dynamics.\n\n  For more details, see equation (5) page 9 in\n  https://jair.org/index.php/jair/article/view/10952\n\n  Args:\n    state: Probability distribution as an `np.array(shape=num_strategies)`.\n    fitness: Fitness vector as an `np.array(shape=num_strategies)`.\n\n  Returns:\n    Time derivative of the population state.\n  '
    avg_fitness = state.dot(fitness)
    return state * (fitness - avg_fitness)

def boltzmannq(state, fitness, temperature=1.0):
    if False:
        i = 10
        return i + 15
    'Selection-mutation dynamics modeling Q-learning with Boltzmann exploration.\n\n  For more details, see equation (10) page 15 in\n  https://jair.org/index.php/jair/article/view/10952\n\n  Args:\n    state: Probability distribution as an `np.array(shape=num_strategies)`.\n    fitness: Fitness vector as an `np.array(shape=num_strategies)`.\n    temperature: A scalar parameter determining the rate of exploration.\n\n  Returns:\n    Time derivative of the population state.\n  '
    exploitation = 1.0 / temperature * replicator(state, fitness)
    exploration = np.log(state) - state.dot(np.log(state).transpose())
    return exploitation - state * exploration

def qpg(state, fitness):
    if False:
        i = 10
        return i + 15
    'Q-based policy gradient dynamics (QPG).\n\n  For more details, see equation (12) on page 18 in\n  https://arxiv.org/pdf/1810.09026.pdf\n\n  Args:\n    state: Probability distribution as an `np.array(shape=num_strategies)`.\n    fitness: Fitness vector as an `np.array(shape=num_strategies)`.\n\n  Returns:\n    Time derivative of the population state.\n  '
    regret = fitness - state.dot(fitness)
    return state * (state * regret - np.sum(state ** 2 * regret))

class SinglePopulationDynamics(object):
    """Continuous-time single population dynamics.

  Attributes:
    payoff_matrix: The payoff matrix as an `numpy.ndarray` of shape `[2, k_1,
      k_2]`, where `k_1` is the number of strategies of the first player and
      `k_2` for the second player. The game is assumed to be symmetric.
    dynamics: A callback function that returns the time-derivative of the
      population state.
  """

    def __init__(self, payoff_matrix, dynamics):
        if False:
            i = 10
            return i + 15
        'Initializes the single-population dynamics.'
        assert payoff_matrix.ndim == 3
        assert payoff_matrix.shape[0] == 2
        assert np.allclose(payoff_matrix[0], payoff_matrix[1].T)
        self.payoff_matrix = payoff_matrix[0]
        self.dynamics = dynamics

    def __call__(self, state=None, time=None):
        if False:
            return 10
        'Time derivative of the population state.\n\n    Args:\n      state: Probability distribution as list or\n        `numpy.ndarray(shape=num_strategies)`.\n      time: Time is ignored (time-invariant dynamics). Including the argument in\n        the function signature supports numerical integration via e.g.\n        `scipy.integrate.odeint` which requires that the callback function has\n        at least two arguments (state and time).\n\n    Returns:\n      Time derivative of the population state as\n      `numpy.ndarray(shape=num_strategies)`.\n    '
        state = np.array(state)
        assert state.ndim == 1
        assert state.shape[0] == self.payoff_matrix.shape[0]
        fitness = np.matmul(state, self.payoff_matrix.T)
        return self.dynamics(state, fitness)

class MultiPopulationDynamics(object):
    """Continuous-time multi-population dynamics.

  Attributes:
    payoff_tensor: The payoff tensor as an numpy.ndarray of size `[n, k0, k1,
      k2, ...]`, where n is the number of players and `k0` is the number of
      strategies of the first player, `k1` of the second player and so forth.
    dynamics: List of callback functions for the time-derivative of the
      population states, where `dynamics[i]` computes the time-derivative of the
      i-th player's population state. If at construction, only a single callback
      function is provided, the same function is used for all populations.
  """

    def __init__(self, payoff_tensor, dynamics):
        if False:
            i = 10
            return i + 15
        'Initializes the multi-population dynamics.'
        if isinstance(dynamics, list) or isinstance(dynamics, tuple):
            assert payoff_tensor.shape[0] == len(dynamics)
        else:
            dynamics = [dynamics] * payoff_tensor.shape[0]
        self.payoff_tensor = payoff_tensor
        self.dynamics = dynamics

    def __call__(self, state, time=None):
        if False:
            while True:
                i = 10
        'Time derivative of the population states.\n\n    Args:\n      state: Combined population state for all populations as a list or flat\n        `numpy.ndarray` (ndim=1). Probability distributions are concatenated in\n        order of the players.\n      time: Time is ignored (time-invariant dynamics). Including the argument in\n        the function signature supports numerical integration via e.g.\n        `scipy.integrate.odeint` which requires that the callback function has\n        at least two arguments (state and time).\n\n    Returns:\n      Time derivative of the combined population state as `numpy.ndarray`.\n    '
        state = np.array(state)
        n = self.payoff_tensor.shape[0]
        ks = self.payoff_tensor.shape[1:]
        assert state.shape[0] == sum(ks)
        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * n
        for i in range(n):
            fitness = np.moveaxis(self.payoff_tensor[i], i, 0)
            for i_ in set(range(n)) - {i}:
                fitness = np.tensordot(states[i_], fitness, axes=[0, 1])
            dstates[i] = self.dynamics[i](states[i], fitness)
        return np.concatenate(dstates)

def time_average(traj):
    if False:
        i = 10
        return i + 15
    'Time-averaged population state trajectory.\n\n  Args:\n    traj: Trajectory as `numpy.ndarray`. Time is along the first dimension,\n      types/strategies along the second.\n\n  Returns:\n    Time-averaged trajectory.\n  '
    n = traj.shape[0]
    sum_traj = np.cumsum(traj, axis=0)
    norm = 1.0 / np.arange(1, n + 1)
    return sum_traj * norm[:, np.newaxis]