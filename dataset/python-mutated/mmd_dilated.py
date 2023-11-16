"""Python implementation of the magnetic mirror descent (MMD) algorithm.

The algorithm operated over the sequence-from with dilated entropy.

See https://arxiv.org/abs/2206.05825.

One iteration of MMD consists of:
1) Compute gradients of dilated entropy
   and payoffs for current sequence form policies.
2) Compute behavioural form policy starting from the bottom
    of the tree and updating gradients of parent nodes along the way.
3) Convert behavioural form policy to equivalent sequence form policy.

The last sequence form policy converges linearly (exponentially fast)
to a \\alpha-reduced normal-form  QRE.
"""
import copy
import warnings
import numpy as np
from scipy import stats as scipy_stats
from open_spiel.python import policy
from open_spiel.python.algorithms.sequence_form_utils import _EMPTY_INFOSET_ACTION_KEYS
from open_spiel.python.algorithms.sequence_form_utils import _EMPTY_INFOSET_KEYS
from open_spiel.python.algorithms.sequence_form_utils import _get_action_from_key
from open_spiel.python.algorithms.sequence_form_utils import construct_vars
from open_spiel.python.algorithms.sequence_form_utils import is_root
from open_spiel.python.algorithms.sequence_form_utils import policy_to_sequence
from open_spiel.python.algorithms.sequence_form_utils import sequence_to_policy
from open_spiel.python.algorithms.sequence_form_utils import uniform_random_seq
import pyspiel

def neg_entropy(probs):
    if False:
        while True:
            i = 10
    return -scipy_stats.entropy(probs)

def softmax(x):
    if False:
        while True:
            i = 10
    unnormalized = np.exp(x - np.max(x))
    return unnormalized / np.sum(unnormalized)

def divergence(x, y, psi_x, psi_y, grad_psi_y):
    if False:
        for i in range(10):
            print('nop')
    'Compute Bregman divergence between x and y, B_psi(x;y).\n\n  Args:\n      x: Numpy array.\n      y: Numpy array.\n      psi_x: Value of psi evaluated at x.\n      psi_y: Value of psi evaluated at y.\n      grad_psi_y: Gradient of psi evaluated at y.\n\n  Returns:\n      Scalar.\n  '
    return psi_x - psi_y - np.dot(grad_psi_y, x - y)

def dilated_dgf_divergence(mmd_1, mmd_2):
    if False:
        i = 10
        return i + 15
    'Bregman divergence between two MMDDilatedEnt objects.\n\n      The value is equivalent to a sum of two Bregman divergences\n      over the sequence form, one for each player.\n\n  Args:\n      mmd_1: MMDDilatedEnt Object\n      mmd_2: MMDDilatedEnt Object\n\n  Returns:\n      Scalar.\n  '
    dgf_values = [mmd_1.dgf_eval(), mmd_2.dgf_eval()]
    dgf_grads = mmd_2.dgf_grads()
    div = 0
    for player in range(2):
        div += divergence(mmd_1.sequences[player], mmd_2.sequences[player], dgf_values[0][player], dgf_values[1][player], dgf_grads[player])
    return div

class MMDDilatedEnt(object):
    """Implements Magnetic Mirror Descent (MMD) with Dilated Entropy.

  The implementation uses the sequence form representation.

  The policies converge to a \\alpha-reduced normal form QRE of a
  two-player zero-sum extensive-form game. If \\alpha is set
  to zero then the method is equivalent to mirror descent ascent
  over the sequence form with dilated entropy and the policies
  will converge on average to a nash equilibrium with
  the appropriate stepsize schedule (or approximate equilirbrium
  for fixed stepsize).

  The main iteration loop is implemented in `update_sequences`:

  ```python
    game = pyspiel.load_game("game_name")
    mmd = MMDDilatedEnt(game, alpha=0.1)
    for i in range(num_iterations):
      mmd.update_sequences()
  ```
  The gap in the regularized game (i.e. 2x exploitability) converges
  to zero and can be computed:

  ```python
      gap = mmd.get_gap()
  ```
  The policy (i.e. behavioural form policy) can be retrieved:
  ```python
      policies = mmd.get_policies()
  ```

  The average sequences and policies can be retrieved:

  ```python
      avg_sequences = mmd.get_avg_sequences()
      avg_policies = mmd.get_avg_policies()
  ```

  """
    empy_state_action_keys = _EMPTY_INFOSET_ACTION_KEYS[:]
    empty_infoset_keys = _EMPTY_INFOSET_KEYS[:]

    def __init__(self, game, alpha, stepsize=None):
        if False:
            while True:
                i = 10
        'Initialize the solver object.\n\n    Args:\n        game: a zeros-um spiel game with two players.\n        alpha: weight of dilated entropy regularization. If alpha > 0 MMD\n          will converge to an alpha-QRE. If alpha = 0 mmd will converge to\n          Nash on average.\n        stepsize: MMD stepsize. Will be set automatically if None.\n    '
        assert game.num_players() == 2
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
        assert game.get_type().chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC or game.get_type().chance_mode == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
        assert alpha >= 0
        self.game = game
        self.alpha = float(alpha)
        (self.infosets, self.infoset_actions_to_seq, self.infoset_action_maps, self.infoset_parent_map, self.payoff_mat, self.infoset_actions_children) = construct_vars(game)
        if stepsize is not None:
            self.stepsize = stepsize
        else:
            self.stepsize = self.alpha / np.max(np.abs(self.payoff_mat)) ** 2
        if self.stepsize == 0.0:
            warnings.warn('MMD stepsize is 0, probably because alpha = 0.')
        self.sequences = uniform_random_seq(game, self.infoset_actions_to_seq)
        self.avg_sequences = copy.deepcopy(self.sequences)
        self.iteration_count = 1

    def get_parent_seq(self, player, infostate):
        if False:
            i = 10
            return i + 15
        'Looks up the parent sequence value for a given infostate.\n\n    Args:\n        player: player number, either 0 or 1.\n        infostate: infostate id string.\n\n    Returns:\n        Scalar.\n    '
        parent_isa_key = self.infoset_parent_map[player][infostate]
        seq_id = self.infoset_actions_to_seq[player][parent_isa_key]
        parent_seq = self.sequences[player][seq_id]
        return parent_seq

    def get_infostate_seq(self, player, infostate):
        if False:
            print('Hello World!')
        'Gets vector of sequence form values corresponding to a given infostate.\n\n    Args:\n        player: player number, either 0 or 1.\n        infostate: infostate id string.\n\n    Returns:\n        Numpy array.\n    '
        seq_idx = [self.infoset_actions_to_seq[player][isa_key] for isa_key in self.infoset_action_maps[player][infostate]]
        seqs = np.array([self.sequences[player][idx] for idx in seq_idx])
        return seqs

    def dgf_eval(self):
        if False:
            while True:
                i = 10
        'Computes the value of dilated entropy for current sequences.\n\n    Returns:\n        List of values, one for each player.\n    '
        dgf_value = [0.0, 0.0]
        for player in range(2):
            for infostate in self.infosets[player]:
                if is_root(infostate):
                    continue
                parent_seq = self.get_parent_seq(player, infostate)
                if parent_seq > 0:
                    children_seq = self.get_infostate_seq(player, infostate)
                    dgf_value[player] += parent_seq * neg_entropy(children_seq / parent_seq)
        return dgf_value

    def dgf_grads(self):
        if False:
            for i in range(10):
                print('nop')
        'Computes gradients of dilated entropy for each player and current seqs.\n\n    Returns:\n        A list of numpy arrays.\n    '
        grads = [np.zeros(len(self.sequences[0])), np.zeros(len(self.sequences[1]))]
        for player in range(2):
            for infostate in self.infosets[player]:
                if is_root(infostate):
                    continue
                parent_seq = self.get_parent_seq(player, infostate)
                if parent_seq > 0:
                    for isa_key in self.infoset_action_maps[player][infostate]:
                        seq_idx = self.infoset_actions_to_seq[player][isa_key]
                        seq = self.sequences[player][seq_idx]
                        grads[player][seq_idx] += np.log(seq / parent_seq) + 1
                        num_children = len(self.infoset_actions_children[player].get(isa_key, []))
                        grads[player][seq_idx] -= num_children
        return grads

    def update_sequences(self):
        if False:
            while True:
                i = 10
        'Performs one step of MMD.'
        self.iteration_count += 1
        psi_grads = self.dgf_grads()
        grads = [(self.stepsize * self.payoff_mat @ self.sequences[1] - psi_grads[0]) / (1 + self.stepsize * self.alpha), (-self.stepsize * self.payoff_mat.T @ self.sequences[0] - psi_grads[1]) / (1 + self.stepsize * self.alpha)]
        new_policy = policy.TabularPolicy(self.game)
        for player in range(2):
            self._update_state_sequences(self.empty_infoset_keys[player], grads[player], player, new_policy)
        self.sequences = policy_to_sequence(self.game, new_policy, self.infoset_actions_to_seq)
        self.update_avg_sequences()

    def _update_state_sequences(self, infostate, g, player, pol):
        if False:
            for i in range(10):
                print('nop')
        'Update the state sequences.'
        isa_keys = self.infoset_action_maps[player][infostate]
        seq_idx = [self.infoset_actions_to_seq[player][isa_key] for isa_key in isa_keys]
        for (isa_key, isa_idx) in zip(isa_keys, seq_idx):
            children = self.infoset_actions_children[player].get(isa_key, [])
            for child in children:
                self._update_state_sequences(child, g, player, pol)
                child_isa_keys = self.infoset_action_maps[player][child]
                child_seq_idx = [self.infoset_actions_to_seq[player][child_isa_key] for child_isa_key in child_isa_keys]
                g_child = np.array([g[idx] for idx in child_seq_idx])
                actions_child = [_get_action_from_key(child_isa_key) for child_isa_key in child_isa_keys]
                policy_child = pol.policy_for_key(child)[:]
                policy_child = np.array([policy_child[a] for a in actions_child])
                g[isa_idx] += np.dot(g_child, policy_child)
                g[isa_idx] += neg_entropy(policy_child)
        if is_root(infostate):
            return
        state_policy = pol.policy_for_key(infostate)
        g_infostate = np.array([g[idx] for idx in seq_idx])
        actions = [_get_action_from_key(isa_key) for isa_key in isa_keys]
        new_state_policy = softmax(-g_infostate)
        for (action, pr) in zip(actions, new_state_policy):
            state_policy[action] = pr

    def get_gap(self):
        if False:
            return 10
        'Computes saddle point gap of the regularized game.\n\n    The gap measures convergence to the alpha-QRE.\n\n    Returns:\n        Scalar.\n    '
        assert self.alpha > 0, 'gap cannot be computed for alpha = 0'
        grads = [self.payoff_mat @ self.sequences[1] / self.alpha, -self.payoff_mat.T @ self.sequences[0] / self.alpha]
        dgf_values = self.dgf_eval()
        br_policy = policy.TabularPolicy(self.game)
        for player in range(2):
            self._update_state_sequences(self.empty_infoset_keys[player], grads[player], player, br_policy)
        br_sequences = policy_to_sequence(self.game, br_policy, self.infoset_actions_to_seq)
        curr_sequences = copy.deepcopy(self.sequences)
        self.sequences = br_sequences
        br_dgf_values = self.dgf_eval()
        self.sequences = curr_sequences
        gap = 0
        gap += curr_sequences[0].T @ self.payoff_mat @ br_sequences[1]
        gap += self.alpha * (dgf_values[1] - br_dgf_values[1])
        gap += self.alpha * (dgf_values[0] - br_dgf_values[0])
        gap += -br_sequences[0].T @ self.payoff_mat @ curr_sequences[1]
        return gap

    def update_avg_sequences(self):
        if False:
            while True:
                i = 10
        for player in range(2):
            self.avg_sequences[player] = self.avg_sequences[player] * (self.iteration_count - 1) + self.sequences[player]
            self.avg_sequences[player] = self.avg_sequences[player] / self.iteration_count

    def current_sequences(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the current sequences.\n\n    Returns:\n      the current sequences for each player as list of numpy arrays.\n    '
        return self.sequences

    def get_avg_sequences(self):
        if False:
            return 10
        'Retrieves the average sequences.\n\n    Returns:\n      the average sequences for each player as list of numpy arrays.\n    '
        return self.avg_sequences

    def get_policies(self):
        if False:
            while True:
                i = 10
        'Convert current sequences to equivalent behavioural form policies.\n\n    Returns:\n        spiel TabularPolicy Object.\n    '
        return sequence_to_policy(self.sequences, self.game, self.infoset_actions_to_seq, self.infoset_action_maps)

    def get_avg_policies(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert average sequences to equivalent behavioural form policies.\n\n    Returns:\n        spiel TabularPolicy Object.\n    '
        return sequence_to_policy(self.avg_sequences, self.game, self.infoset_actions_to_seq, self.infoset_action_maps)