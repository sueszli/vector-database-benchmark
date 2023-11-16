"""Tests for open_spiel.python.policy."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python import games
from open_spiel.python import policy
from open_spiel.python.algorithms import get_all_states
import pyspiel
SEED = 187461917
_TIC_TAC_TOE_STATES = [{'state': '3, 4, 8, 5', 'legal_actions': (0, 1, 2, 6, 7)}, {'state': '4, 1, 0, 3, 5, 6', 'legal_actions': (2, 7, 8)}, {'state': '', 'legal_actions': (0, 1, 2, 3, 4, 5, 6, 7, 8)}]

class DerivedPolicyTest(absltest.TestCase):

    def test_derive_from_policy(self):
        if False:
            while True:
                i = 10

        class DerivedPolicy(pyspiel.Policy):

            def action_probabilities(self, state):
                if False:
                    print('Hello World!')
                return {0: 0.1, 1: 0.9}

            def get_state_policy(self, infostate):
                if False:
                    while True:
                        i = 10
                return {10: 0.9, 11: 0.1}
        policy_obj = DerivedPolicy()
        self.assertEqual(DerivedPolicy.__bases__, (pyspiel.Policy,))
        self.assertIsInstance(policy_obj, pyspiel.Policy)
        self.assertEqual({0: 0.1, 1: 0.9}, policy_obj.action_probabilities(pyspiel.load_game('kuhn_poker').new_initial_state()))
        self.assertEqual({0: 0.1, 1: 0.9}, policy_obj.action_probabilities('some infostate'))
        self.assertEqual({10: 0.9, 11: 0.1}, policy_obj.get_state_policy('some infostate'))
        with self.assertRaises(RuntimeError):
            policy_obj.serialize()

    def test_cpp_policy_from_py(self):
        if False:
            for i in range(10):
                print('nop')

        class DerivedPolicy(pyspiel.Policy):

            def action_probabilities(self, state):
                if False:
                    while True:
                        i = 10
                return {0: 0.0, 1: 0.0}

            def get_state_policy(self, infostate):
                if False:
                    for i in range(10):
                        print('nop')
                return [(2, 0.0), (3, 0.0)]

            def get_state_policy_as_parallel_vectors(self, state):
                if False:
                    while True:
                        i = 10
                if isinstance(state, str):
                    return ([4, 5], [0, 0])
                else:
                    return ([6, 7], [0, 0])

            def serialize(self, precision, delim):
                if False:
                    print('Hello World!')
                return f'Serialized string, precision={precision!r}, delim={delim!r}'
        policy_obj = DerivedPolicy()
        self.assertEqual({0: 0.0, 1: 0.0}, pyspiel._policy_trampoline_testing.call_action_probabilities(policy_obj, pyspiel.load_game('kuhn_poker').new_initial_state()))
        self.assertEqual({0: 0.0, 1: 0.0}, pyspiel._policy_trampoline_testing.call_action_probabilities(policy_obj, 'some infostate'))
        self.assertEqual([(2, 0.0), (3, 0.0)], pyspiel._policy_trampoline_testing.call_get_state_policy(policy_obj, pyspiel.load_game('kuhn_poker').new_initial_state()))
        self.assertEqual([(2, 0.0), (3, 0.0)], pyspiel._policy_trampoline_testing.call_get_state_policy(policy_obj, 'some infostate'))
        self.assertEqual(([4, 5], [0, 0]), pyspiel._policy_trampoline_testing.call_get_state_policy_as_parallel_vectors(policy_obj, 'some infostate'))
        self.assertEqual(([6, 7], [0, 0]), pyspiel._policy_trampoline_testing.call_get_state_policy_as_parallel_vectors(policy_obj, pyspiel.load_game('kuhn_poker').new_initial_state()))
        self.assertEqual(pyspiel._policy_trampoline_testing.call_serialize(policy_obj, 3, '!?'), "Serialized string, precision=3, delim='!?'")

def test_policy_on_game(self, game, policy_object, player=-1):
    if False:
        print('Hello World!')
    'Checks the policy conforms to the conventions.\n\n  Checks the Policy.action_probabilities contains only legal actions (but not\n  necessarily all).\n  Checks that the probabilities are positive and sum to 1.\n\n  Args:\n    self: The Test class. This methid targets as being used as a utility\n      function to test policies.\n    game: A `pyspiel.Game`, same as the one used in the policy.\n    policy_object: A `policy.Policy` object on `game`. to test.\n    player: Restrict testing policy to a player.\n  '
    all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, to_string=lambda s: s.information_state_string())
    for state in all_states.values():
        legal_actions = set(state.legal_actions())
        action_probabilities = policy_object.action_probabilities(state)
        for action in action_probabilities.keys():
            actions_missing = set(legal_actions) - set(action_probabilities.keys())
            illegal_actions = set(action_probabilities.keys()) - set(legal_actions)
            self.assertIn(action, legal_actions, msg='The action {} is present in the policy but is not a legal actions (these are {})\nLegal actions missing from policy: {}\nIllegal actions present in policy: {}'.format(action, legal_actions, actions_missing, illegal_actions))
        sum_ = 0
        for prob in action_probabilities.values():
            sum_ += prob
            self.assertGreaterEqual(prob, 0)
        if player < 0 or state.current_player() == player:
            self.assertAlmostEqual(1, sum_)
        else:
            self.assertAlmostEqual(0, sum_)
_LEDUC_POKER = pyspiel.load_game('leduc_poker')

class CommonTest(parameterized.TestCase):

    @parameterized.parameters([policy.TabularPolicy(_LEDUC_POKER), policy.UniformRandomPolicy(_LEDUC_POKER), policy.FirstActionPolicy(_LEDUC_POKER)])
    def test_policy_on_leduc(self, policy_object):
        if False:
            i = 10
            return i + 15
        test_policy_on_game(self, _LEDUC_POKER, policy_object)

    @parameterized.named_parameters([('pyspiel.UniformRandomPolicy', pyspiel.UniformRandomPolicy(_LEDUC_POKER)), ('pyspiel.GetRandomPolicy', pyspiel.GetRandomPolicy(_LEDUC_POKER, 1)), ('pyspiel.GetFlatDirichletPolicy', pyspiel.GetFlatDirichletPolicy(_LEDUC_POKER, 1)), ('pyspiel.GetRandomDeterministicPolicy', pyspiel.GetRandomDeterministicPolicy(_LEDUC_POKER, 1))])
    def test_cpp_policies_on_leduc(self, policy_object):
        if False:
            print('Hello World!')
        test_policy_on_game(self, _LEDUC_POKER, policy_object)

    @parameterized.named_parameters([('pyspiel.GetRandomPolicy0', pyspiel.GetRandomPolicy(_LEDUC_POKER, 1, 0), 0), ('pyspiel.GetFlatDirichletPolicy1', pyspiel.GetFlatDirichletPolicy(_LEDUC_POKER, 1, 1), 1), ('pyspiel.GetRandomDeterministicPolicym1', pyspiel.GetRandomDeterministicPolicy(_LEDUC_POKER, 1, -1), -1)])
    def test_cpp_player_policies_on_leduc(self, policy_object, player):
        if False:
            i = 10
            return i + 15
        test_policy_on_game(self, _LEDUC_POKER, policy_object, player)

class TabularTicTacToePolicyTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(TabularTicTacToePolicyTest, cls).setUpClass()
        cls.game = pyspiel.load_game('tic_tac_toe')
        cls.tabular_policy = policy.TabularPolicy(cls.game)

    def test_policy_shape(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.tabular_policy.action_probability_array.shape, (294778, 9))

    def test_policy_attributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.tabular_policy.player_ids, [0, 1])

    @parameterized.parameters(*_TIC_TAC_TOE_STATES)
    def test_policy_at_state(self, state, legal_actions):
        if False:
            print('Hello World!')
        index = self.tabular_policy.state_lookup[state]
        prob = 1 / len(legal_actions)
        np.testing.assert_array_equal(self.tabular_policy.action_probability_array[index], [prob if action in legal_actions else 0 for action in range(9)])

    @parameterized.parameters(*_TIC_TAC_TOE_STATES)
    def test_legal_actions_at_state(self, state, legal_actions):
        if False:
            for i in range(10):
                print('nop')
        index = self.tabular_policy.state_lookup[state]
        np.testing.assert_array_equal(self.tabular_policy.legal_actions_mask[index], [1 if action in legal_actions else 0 for action in range(9)])

    def test_call_for_state(self):
        if False:
            i = 10
            return i + 15
        state = self.game.new_initial_state()
        state.apply_action(3)
        state.apply_action(4)
        state.apply_action(5)
        state.apply_action(6)
        state.apply_action(7)
        self.assertEqual(self.tabular_policy.action_probabilities(state), {0: 0.25, 1: 0.25, 2: 0.25, 8: 0.25})

    def test_states_ordered_by_player(self):
        if False:
            print('Hello World!')
        max_player0_index = max((self.tabular_policy.state_lookup[state] for state in self.tabular_policy.states_per_player[0]))
        min_player1_index = min((self.tabular_policy.state_lookup[state] for state in self.tabular_policy.states_per_player[1]))
        self.assertEqual(max_player0_index + 1, min_player1_index)

    def test_state_in(self):
        if False:
            return 10
        self.assertEqual(self.tabular_policy.state_in.shape, (294778, 27))

    @parameterized.parameters(*_TIC_TAC_TOE_STATES)
    def test_policy_for_state_string(self, state, legal_actions):
        if False:
            return 10
        prob = 1 / len(legal_actions)
        np.testing.assert_array_equal(self.tabular_policy.policy_for_key(state), [prob if action in legal_actions else 0 for action in range(9)])

class TabularPolicyTest(parameterized.TestCase):

    def test_update_elementwise(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('kuhn_poker')
        tabular_policy = policy.TabularPolicy(game)
        state = '0pb'
        np.testing.assert_array_equal(tabular_policy.policy_for_key(state), [0.5, 0.5])
        tabular_policy.policy_for_key(state)[0] = 0.9
        tabular_policy.policy_for_key(state)[1] = 0.1
        np.testing.assert_array_equal(tabular_policy.policy_for_key(state), [0.9, 0.1])

    def test_update_slice(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('kuhn_poker')
        tabular_policy = policy.TabularPolicy(game)
        state = '2b'
        np.testing.assert_array_equal(tabular_policy.policy_for_key(state), [0.5, 0.5])
        tabular_policy.policy_for_key(state)[:] = [0.8, 0.2]
        np.testing.assert_array_equal(tabular_policy.policy_for_key(state), [0.8, 0.2])

    def test_state_ordering_is_deterministic(self):
        if False:
            return 10
        game = pyspiel.load_game('kuhn_poker')
        tabular_policy = policy.TabularPolicy(game)
        expected = {'0': 0, '0pb': 1, '1': 2, '1pb': 3, '2': 4, '2pb': 5, '1p': 6, '1b': 7, '2p': 8, '2b': 9, '0p': 10, '0b': 11}
        self.assertEqual(expected, tabular_policy.state_lookup)

    def test_partial_tabular_policy_empty_uniform(self):
        if False:
            return 10
        'Tests that a partial tabular policy works for an empty policy.'
        game = pyspiel.load_game('kuhn_poker')
        python_tabular_policy = policy.TabularPolicy(game)
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy()
        self.assertNotEmpty(python_tabular_policy.state_lookup)
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, include_mean_field_states=False)
        self.assertNotEmpty(all_states)
        for (_, state) in all_states.items():
            tabular_probs = python_tabular_policy.action_probabilities(state)
            state_policy = partial_pyspiel_policy.get_state_policy(state)
            self.assertLen(state_policy, 2)
            for (a, p) in state_policy:
                self.assertAlmostEqual(p, tabular_probs[a])

    def test_partial_tabular_policy_set_full(self):
        if False:
            i = 10
            return i + 15
        'Tests the partial tabular policy works for a complete policy.'
        game = pyspiel.load_game('kuhn_poker')
        python_tabular_policy = policy.TabularPolicy(game)
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy()
        self.assertNotEmpty(python_tabular_policy.state_lookup)
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, include_mean_field_states=False)
        self.assertNotEmpty(all_states)
        policy_dict = python_tabular_policy.to_dict()
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy(policy_dict)
        for (_, state) in all_states.items():
            tabular_probs = python_tabular_policy.action_probabilities(state)
            state_policy = partial_pyspiel_policy.get_state_policy(state)
            self.assertLen(state_policy, 2)
            for (a, p) in state_policy:
                self.assertAlmostEqual(p, tabular_probs[a])

    def test_partial_tabular_policy_override_fallback(self):
        if False:
            i = 10
            return i + 15
        'Tests the partial tabular policy for a truly partial policy.\n\n    Specifically: assigns a full policy, overrides some entries, and\n    removes others. Checks that the overridden ones return correctly and that\n    the missing ones return the fallback.\n    '
        game = pyspiel.load_game('kuhn_poker')
        python_tabular_policy = policy.TabularPolicy(game)
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy()
        self.assertNotEmpty(python_tabular_policy.state_lookup)
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, include_mean_field_states=False)
        self.assertNotEmpty(all_states)
        policy_dict = python_tabular_policy.to_dict()
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy(policy_dict)
        perturbed_policy_dict = {}
        for key in policy_dict:
            if np.random.uniform() < 0.5:
                perturbed_policy_dict[key] = [(0, 1.0)]
        partial_pyspiel_policy = pyspiel.PartialTabularPolicy(perturbed_policy_dict)
        for (_, state) in all_states.items():
            infostate_key = state.information_state_string()
            state_policy = partial_pyspiel_policy.get_state_policy(state)
            if infostate_key in perturbed_policy_dict:
                self.assertLen(state_policy, 1)
                self.assertAlmostEqual(state_policy[0][1], 1.0)
            else:
                tabular_probs = python_tabular_policy.action_probabilities(state)
                for (a, p) in state_policy:
                    self.assertAlmostEqual(p, tabular_probs[a])

    def test_states(self):
        if False:
            return 10
        game = pyspiel.load_game('leduc_poker')
        tabular_policy = policy.TabularPolicy(game)
        i = 0
        for state in tabular_policy.states:
            self.assertEqual(i, tabular_policy.state_index(state))
            i += 1
        self.assertEqual(936, i)

    @parameterized.parameters((policy.FirstActionPolicy, 'kuhn_poker'), (policy.UniformRandomPolicy, 'kuhn_poker'), (policy.FirstActionPolicy, 'leduc_poker'), (policy.UniformRandomPolicy, 'leduc_poker'))
    def test_can_turn_policy_into_tabular_policy(self, policy_class, game_name):
        if False:
            return 10
        game = pyspiel.load_game(game_name)
        realized_policy = policy_class(game)
        tabular_policy = realized_policy.to_tabular()
        for state in tabular_policy.states:
            self.assertEqual(realized_policy.action_probabilities(state), tabular_policy.action_probabilities(state))

class TabularRockPaperScissorsPolicyTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(TabularRockPaperScissorsPolicyTest, cls).setUpClass()
        game = pyspiel.load_game_as_turn_based('matrix_rps')
        cls.tabular_policy = policy.TabularPolicy(game)

    def test_policy_attributes(self):
        if False:
            return 10
        self.assertEqual(self.tabular_policy.player_ids, [0, 1])

    def test_tabular_policy(self):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_array_equal(self.tabular_policy.action_probability_array, [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])

    def test_states_lookup(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game_as_turn_based('matrix_rps')
        state = game.new_initial_state()
        first_info_state = state.information_state_string()
        state.apply_action(state.legal_actions()[0])
        second_info_state = state.information_state_string()
        self.assertCountEqual(self.tabular_policy.state_lookup, [first_info_state, second_info_state])
        self.assertCountEqual(self.tabular_policy.state_lookup.values(), [0, 1])

    def test_legal_actions_mask(self):
        if False:
            i = 10
            return i + 15
        np.testing.assert_array_equal(self.tabular_policy.legal_actions_mask, [[1, 1, 1], [1, 1, 1]])

class UniformRandomPolicyTest(absltest.TestCase):

    def test_policy_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('tiny_bridge_4p')
        uniform_random_policy = policy.UniformRandomPolicy(game)
        self.assertEqual(uniform_random_policy.player_ids, [0, 1, 2, 3])

    def test_policy_at_state(self):
        if False:
            return 10
        game = pyspiel.load_game('tic_tac_toe')
        uniform_random_policy = policy.UniformRandomPolicy(game)
        state = game.new_initial_state()
        state.apply_action(2)
        state.apply_action(4)
        state.apply_action(6)
        state.apply_action(8)
        self.assertEqual(uniform_random_policy.action_probabilities(state), {0: 0.2, 1: 0.2, 3: 0.2, 5: 0.2, 7: 0.2})

    def test_players_have_different_legal_actions(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('oshi_zumo')
        uniform_random_policy = policy.UniformRandomPolicy(game)
        state = game.new_initial_state()
        state.apply_actions([46, 49])
        self.assertEqual(uniform_random_policy.action_probabilities(state, player_id=0), {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
        self.assertEqual(uniform_random_policy.action_probabilities(state, player_id=1), {0: 0.5, 1: 0.5})

class MergeTabularPoliciesTest(absltest.TestCase):

    def test_identity(self):
        if False:
            print('Hello World!')
        num_players = 2
        game = pyspiel.load_game('kuhn_poker', {'players': num_players})
        tabular_policies = [policy.TabularPolicy(game, players=(player,)) for player in range(num_players)]
        for (player, tabular_policy) in enumerate(tabular_policies):
            tabular_policy.action_probability_array[:] = 0
            tabular_policy.action_probability_array[:, player] = 1.0
        merged_tabular_policy = policy.merge_tabular_policies(tabular_policies, game)
        self.assertIdentityPoliciesEqual(tabular_policies, merged_tabular_policy, game)

    def test_identity_redundant(self):
        if False:
            i = 10
            return i + 15
        num_players = 2
        game = pyspiel.load_game('kuhn_poker', {'players': num_players})
        tabular_policies = [policy.TabularPolicy(game, players=None) for player in range(num_players)]
        for (player, tabular_policy) in enumerate(tabular_policies):
            tabular_policy.action_probability_array[:] = 0
            tabular_policy.action_probability_array[:, player] = 1.0
        merged_tabular_policy = policy.merge_tabular_policies(tabular_policies, game)
        self.assertIdentityPoliciesEqual(tabular_policies, merged_tabular_policy, game)

    def test_identity_missing(self):
        if False:
            for i in range(10):
                print('nop')
        num_players = 2
        game = pyspiel.load_game('kuhn_poker', {'players': num_players})
        tabular_policies = [policy.TabularPolicy(game, players=(0,)) for player in range(num_players)]
        for (player, tabular_policy) in enumerate(tabular_policies):
            tabular_policy.action_probability_array[:] = 0
            tabular_policy.action_probability_array[:, player] = 1.0
        merged_tabular_policy = policy.merge_tabular_policies(tabular_policies, game)
        for player in range(game.num_players()):
            if player == 0:
                self.assertListEqual(tabular_policies[player].states_per_player[player], merged_tabular_policy.states_per_player[player])
                for p_state in merged_tabular_policy.states_per_player[player]:
                    to_index = merged_tabular_policy.state_lookup[p_state]
                    from_index = tabular_policies[player].state_lookup[p_state]
                    self.assertTrue(np.allclose(merged_tabular_policy.action_probability_array[to_index], tabular_policies[player].action_probability_array[from_index]))
                    self.assertTrue(np.allclose(merged_tabular_policy.action_probability_array[to_index, player], 1))
            else:
                self.assertEmpty(tabular_policies[player].states_per_player[player])
                for p_state in merged_tabular_policy.states_per_player[player]:
                    to_index = merged_tabular_policy.state_lookup[p_state]
                    self.assertTrue(np.allclose(merged_tabular_policy.action_probability_array[to_index, player], 0.5))

    def assertIdentityPoliciesEqual(self, tabular_policies, merged_tabular_policy, game):
        if False:
            i = 10
            return i + 15
        for player in range(game.num_players()):
            self.assertListEqual(tabular_policies[player].states_per_player[player], merged_tabular_policy.states_per_player[player])
            for p_state in merged_tabular_policy.states_per_player[player]:
                to_index = merged_tabular_policy.state_lookup[p_state]
                from_index = tabular_policies[player].state_lookup[p_state]
                self.assertTrue(np.allclose(merged_tabular_policy.action_probability_array[to_index], tabular_policies[player].action_probability_array[from_index]))
                self.assertTrue(np.allclose(merged_tabular_policy.action_probability_array[to_index, player], 1))

class JointActionProbTest(absltest.TestCase):

    def test_joint_action_probabilities(self):
        if False:
            return 10
        'Test expected behavior of joint_action_probabilities.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        uniform_policy = policy.UniformRandomPolicy(game)
        joint_action_probs = policy.joint_action_probabilities(game.new_initial_state(), uniform_policy)
        self.assertCountEqual(list(joint_action_probs), [((0, 0), 0.25), ((1, 1), 0.25), ((1, 0), 0.25), ((0, 1), 0.25)])

    def test_joint_action_probabilities_failure_on_seq_game(self):
        if False:
            while True:
                i = 10
        'Test failure of child on sequential games.'
        game = pyspiel.load_game('kuhn_poker')
        with self.assertRaises(AssertionError):
            list(policy.joint_action_probabilities(game.new_initial_state(), policy.UniformRandomPolicy(game)))

class ChildTest(absltest.TestCase):

    def test_child_function_expected_behavior_for_seq_game(self):
        if False:
            while True:
                i = 10
        'Test expected behavior of child on sequential games.'
        game = pyspiel.load_game('tic_tac_toe')
        initial_state = game.new_initial_state()
        action = 3
        new_state = policy.child(initial_state, action)
        self.assertNotEqual(new_state.history(), initial_state.history())
        expected_new_state = initial_state.child(action)
        self.assertNotEqual(new_state, expected_new_state)
        self.assertEqual(new_state.history(), expected_new_state.history())

    def test_child_function_expected_behavior_for_sim_game(self):
        if False:
            while True:
                i = 10
        'Test expected behavior of child on simultaneous games.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        parameter_state = game.new_initial_state()
        actions = [1, 1]
        new_state = policy.child(parameter_state, actions)
        self.assertEqual(str(new_state), 'p0:D p1:D')

    def test_child_function_failure_behavior_for_sim_game(self):
        if False:
            for i in range(10):
                print('nop')
        'Test failure behavior of child on simultaneous games.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        parameter_state = game.new_initial_state()
        with self.assertRaises(AssertionError):
            policy.child(parameter_state, 0)
if __name__ == '__main__':
    np.random.seed(SEED)
    absltest.main()