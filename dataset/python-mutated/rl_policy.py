"""DQN as a policy.

Treating RL Oracles as policies allows us to streamline their use with tabular
policies and other policies in OpenSpiel, and freely mix populations using
different types of oracles.
"""
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import policy_gradient

def rl_policy_factory(rl_class):
    if False:
        i = 10
        return i + 15
    "Transforms an RL Agent into an OpenSpiel policy.\n\n  Args:\n    rl_class: An OpenSpiel class inheriting from 'rl_agent.AbstractAgent' such\n      as policy_gradient.PolicyGradient or dqn.DQN.\n\n  Returns:\n    An RLPolicy class that wraps around an instance of rl_class to transform it\n    into a policy.\n  "

    class RLPolicy(policy.Policy):
        """A 'policy.Policy' wrapper around an 'rl_agent.AbstractAgent' instance."""

        def __init__(self, env, player_id, **kwargs):
            if False:
                i = 10
                return i + 15
            "Constructs an RL Policy.\n\n      Args:\n        env: An OpenSpiel RL Environment instance.\n        player_id: The ID of the DQN policy's player.\n        **kwargs: Various kwargs used to initialize rl_class.\n      "
            game = env.game
            super(RLPolicy, self).__init__(game, player_id)
            self._policy = rl_class(**{'player_id': player_id, **kwargs})
            self._frozen = False
            self._rl_class = rl_class
            self._env = env
            self._obs = {'info_state': [None] * self.game.num_players(), 'legal_actions': [None] * self.game.num_players()}

        def get_time_step(self):
            if False:
                print('Hello World!')
            time_step = self._env.get_time_step()
            return time_step

        def action_probabilities(self, state, player_id=None):
            if False:
                print('Hello World!')
            cur_player = state.current_player()
            legal_actions = state.legal_actions(cur_player)
            step_type = rl_environment.StepType.LAST if state.is_terminal() else rl_environment.StepType.MID
            self._obs['current_player'] = cur_player
            self._obs['info_state'][cur_player] = state.information_state_tensor(cur_player)
            self._obs['legal_actions'][cur_player] = legal_actions
            rewards = state.rewards()
            if rewards:
                time_step = rl_environment.TimeStep(observations=self._obs, rewards=rewards, discounts=self._env._discounts, step_type=step_type)
            else:
                rewards = [0] * self._num_players
                time_step = rl_environment.TimeStep(observations=self._obs, rewards=rewards, discounts=self._env._discounts, step_type=rl_environment.StepType.FIRST)
            p = self._policy.step(time_step, is_evaluation=True).probs
            prob_dict = {action: p[action] for action in legal_actions}
            return prob_dict

        def step(self, time_step, is_evaluation=False):
            if False:
                print('Hello World!')
            is_evaluation = is_evaluation or self._frozen
            return self._policy.step(time_step, is_evaluation)

        def freeze(self):
            if False:
                return 10
            "This method freezes the policy's weights.\n\n      The weight freezing effect is implemented by preventing any training to\n      take place through calls to the step function. The weights are therefore\n      not effectively frozen, and unconventional calls may trigger weights\n      training.\n\n      The weight-freezing effect is especially needed in PSRO, where all\n      policies that aren't being trained by the oracle must be static. Freezing\n      trained policies permitted us not to change how 'step' was called when\n      introducing self-play (By not changing 'is_evaluation' depending on the\n      current player).\n      "
            self._frozen = True

        def unfreeze(self):
            if False:
                while True:
                    i = 10
            self._frozen = False

        def is_frozen(self):
            if False:
                while True:
                    i = 10
            return self._frozen

        def get_weights(self):
            if False:
                return 10
            return self._policy.get_weights()

        def copy_with_noise(self, sigma=0.0):
            if False:
                while True:
                    i = 10
            copied_object = RLPolicy.__new__(RLPolicy)
            super(RLPolicy, copied_object).__init__(self.game, self.player_ids)
            setattr(copied_object, '_rl_class', self._rl_class)
            setattr(copied_object, '_obs', self._obs)
            setattr(copied_object, '_policy', self._policy.copy_with_noise(sigma=sigma))
            setattr(copied_object, '_env', self._env)
            copied_object.unfreeze()
            return copied_object
    return RLPolicy
PGPolicy = rl_policy_factory(policy_gradient.PolicyGradient)
DQNPolicy = rl_policy_factory(dqn.DQN)