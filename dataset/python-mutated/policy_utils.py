"""Adds useful functions for working with dictionaries representing policies."""
from open_spiel.python.algorithms import get_all_states

def policy_to_dict(player_policy, game, all_states=None, state_to_information_state=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts a Policy instance into a tabular policy represented as a dict.\n\n  This is compatible with the C++ TabularExploitability code (i.e.\n  pyspiel.exploitability, pyspiel.TabularBestResponse, etc.).\n\n  While you do not have to pass the all_states and state_to_information_state\n  arguments, creating them outside of this funciton will speed your code up\n  dramatically.\n\n  Args:\n    player_policy: The policy you want to convert to a dict.\n    game: The game the policy is for.\n    all_states: The result of calling get_all_states.get_all_states. Can be\n      cached for improved performance.\n    state_to_information_state: A dict mapping str(state) to\n      state.information_state for every state in the game. Can be cached for\n      improved performance.\n\n  Returns:\n    A dictionary version of player_policy that can be passed to the C++\n    TabularBestResponse, Exploitability, and BestResponse functions/classes.\n  '
    if all_states is None:
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False)
        state_to_information_state = {state: all_states[state].information_state_string() for state in all_states}
    tabular_policy = dict()
    for state in all_states:
        information_state = state_to_information_state[state]
        tabular_policy[information_state] = list(player_policy.action_probabilities(all_states[state]).items())
    return tabular_policy

def get_best_response_actions_as_string(best_response_actions):
    if False:
        for i in range(10):
            print('nop')
    'Turns a dict<bytes, int> into a bytestring compatible with C++.\n\n  i.e. the bytestring can be copy-pasted as the brace initialization for a\n  {std::unordered_,std::,absl::flat_hash_}map<std::string, int>.\n\n  Args:\n    best_response_actions: A dict mapping bytes to ints.\n\n  Returns:\n    A bytestring that can be copy-pasted to brace-initialize a C++\n    std::map<std::string, T>.\n  '
    best_response_keys = sorted(best_response_actions.keys())
    best_response_strings = ['%s: %i' % (k, best_response_actions[k]) for k in best_response_keys]
    return '{%s}' % ', '.join(best_response_strings)

def tabular_policy_to_cpp_map(policy):
    if False:
        for i in range(10):
            print('nop')
    'Turns a policy into a C++ compatible bytestring for brace-initializing.\n\n  Args:\n    policy: A dict representing a tabular policy. The keys are infostate\n      bytestrings.\n\n  Returns:\n    A bytestring that can be copy-pasted to brace-initialize a C++\n    std::map<std::string, open_spiel::ActionsAndProbs>.\n  '
    cpp_entries = []
    policy_keys = sorted(policy.keys())
    for key in policy_keys:
        tuple_strs = ['{%i, %s}' % (p[0], p[1].astype(str)) for p in policy[key]]
        value = '{' + ', '.join(tuple_strs) + '}'
        cpp_entries.append('{"%s", %s}' % (key, value))
    return '{%s}' % ',\n'.join(cpp_entries)