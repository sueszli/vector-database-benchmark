"""Boltzmann DQN agent implemented in JAX.

This algorithm is a variation of DQN that uses a softmax policy directly with
the unregularized action-value function. See https://arxiv.org/abs/2102.01585.
"""
import jax
import jax.numpy as jnp
import numpy as np
from open_spiel.python.jax import dqn

class BoltzmannDQN(dqn.DQN):
    """Boltzmann DQN implementation in JAX."""

    def __init__(self, *args, eta: float=1.0, seed: int=42, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the Boltzmann DQN agent.\n\n    Args:\n      *args: args passed to the underlying DQN agent.\n      eta: Temperature parameter used in the softmax function.\n      seed: Random seed used for action selection.\n      **kwargs: kwargs passed to the underlying DQN agent.\n    '
        self._eta = eta
        self._rs = np.random.RandomState(seed)
        super().__init__(*args, seed=seed, **kwargs)

    def _create_networks(self, rng, state_representation_size):
        if False:
            i = 10
            return i + 15
        'Called to create the networks.'
        super()._create_networks(rng, state_representation_size)
        self.params_prev_q_network = self.hk_network.init(rng, jnp.ones([1, state_representation_size]))

    def _softmax_action_probs(self, params, info_state, legal_actions, coeff=None):
        if False:
            i = 10
            return i + 15
        'Returns a valid soft-max action and action probabilities.\n\n    Args:\n      params: Parameters of the Q-network.\n      info_state: Observations from the environment.\n      legal_actions: List of legal actions.\n      coeff: If not None, then the terms in softmax function will be\n        element-wise multiplied with these coefficients.\n\n    Returns:\n      a valid soft-max action and action probabilities.\n    '
        info_state = np.reshape(info_state, [1, -1])
        q_values = self.hk_network_apply(params, info_state)[0]
        legal_one_hot = self._to_one_hot(legal_actions)
        legal_q_values = q_values + (1 - legal_one_hot) * dqn.ILLEGAL_ACTION_LOGITS_PENALTY
        temp = legal_q_values / self._eta
        unnormalized = np.exp(temp - np.amax(temp))
        if coeff is not None:
            unnormalized = np.multiply(coeff, unnormalized)
        probs = unnormalized / unnormalized.sum()
        action = self._rs.choice(legal_actions, p=probs[legal_actions])
        return (action, probs)

    def _get_action_probs(self, info_state, legal_actions, is_evaluation=False):
        if False:
            i = 10
            return i + 15
        'Returns a selected action and the probabilities of legal actions.'
        if is_evaluation:
            (_, prev_probs) = self._softmax_action_probs(self.params_prev_q_network, info_state, legal_actions)
            return self._softmax_action_probs(self.params_q_network, info_state, legal_actions, prev_probs)
        return super()._get_action_probs(info_state, legal_actions, is_evaluation=False)

    def update_prev_q_network(self):
        if False:
            while True:
                i = 10
        'Updates the parameters of the previous Q-network.'
        self.params_prev_q_network = jax.tree_map(lambda x: x.copy(), self.params_q_network)