"""An implementation of sequence-form linear programming.

This is a classic algorithm for solving two-player zero-sum games with imperfect
information. For a general introduction to the concepts, see Sec 5.2.3 of
Shoham & Leyton-Brown '09, Multiagent Systems: Algorithmic, Game-Theoretic, and
Logical Foundations http://www.masfoundations.org/mas.pdf.

In this implementation, we follow closely the construction in Koller, Megiddo,
and von Stengel, Fast Algorithms for Finding Randomized Strategies in Game Trees
http://theory.stanford.edu/~megiddo/pdf/stoc94.pdf. Specifically, we construct
and solve equations (8) and (9) from this paper.
"""
from open_spiel.python import policy
from open_spiel.python.algorithms import lp_solver
import pyspiel
_DELIMITER = ' -=- '
_EMPTY_INFOSET_KEYS = ['***EMPTY_INFOSET_P0***', '***EMPTY_INFOSET_P1***']
_EMPTY_INFOSET_ACTION_KEYS = ['***EMPTY_INFOSET_ACTION_P0***', '***EMPTY_INFOSET_ACTION_P1***']

def _construct_lps(state, infosets, infoset_actions, infoset_action_maps, chance_reach, lps, parent_is_keys, parent_isa_keys):
    if False:
        i = 10
        return i + 15
    "Build the linear programs recursively from this state.\n\n  Args:\n    state: an open spiel state (root of the game tree)\n    infosets: a list of dicts, one per player, that maps infostate to an id. The\n      dicts are filled by this function and should initially only contain root\n      values.\n    infoset_actions: a list of dicts, one per player, that maps a string of\n      (infostate, action) pair to an id. The dicts are filled by this function\n      and should inirially only contain the root values\n    infoset_action_maps: a list of dicts, one per player, that maps each\n      info_state to a list of (infostate, action) string\n    chance_reach: the contribution of chance's reach probability (should start\n      at 1).\n    lps: a list of linear programs, one per player. The first one will be\n      constructred as in Eq (8) of Koller, Megiddo and von Stengel. The second\n      lp is Eq (9). Initially these should contain only the root-level\n      constraints and variables.\n    parent_is_keys: a list of parent information state keys for this state\n    parent_isa_keys: a list of parent (infostate, action) keys\n  "
    if state.is_terminal():
        returns = state.returns()
        lps[0].add_or_reuse_constraint(parent_isa_keys[0], lp_solver.CONS_TYPE_GEQ)
        lps[0].add_to_cons_coeff(parent_isa_keys[0], parent_isa_keys[1], -1.0 * returns[0] * chance_reach)
        lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], 1.0)
        lps[1].add_or_reuse_constraint(parent_isa_keys[1], lp_solver.CONS_TYPE_LEQ)
        lps[1].add_to_cons_coeff(parent_isa_keys[1], parent_isa_keys[0], -1.0 * returns[0] * chance_reach)
        lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], -1.0)
        return
    if state.is_chance_node():
        for (action, prob) in state.chance_outcomes():
            new_state = state.child(action)
            _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps, prob * chance_reach, lps, parent_is_keys, parent_isa_keys)
        return
    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions(player)
    if player == 0:
        lps[0].add_or_reuse_variable(info_state)
        lps[0].add_or_reuse_constraint(parent_isa_keys[0], lp_solver.CONS_TYPE_GEQ)
        lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], 1.0)
        lps[0].set_cons_coeff(parent_isa_keys[0], info_state, -1.0)
        lps[1].add_or_reuse_constraint(info_state, lp_solver.CONS_TYPE_EQ)
        lps[1].set_cons_coeff(info_state, parent_isa_keys[0], -1.0)
    else:
        lps[1].add_or_reuse_variable(info_state)
        lps[1].add_or_reuse_constraint(parent_isa_keys[1], lp_solver.CONS_TYPE_LEQ)
        lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], -1.0)
        lps[1].set_cons_coeff(parent_isa_keys[1], info_state, 1.0)
        lps[0].add_or_reuse_constraint(info_state, lp_solver.CONS_TYPE_EQ)
        lps[0].set_cons_coeff(info_state, parent_isa_keys[1], -1.0)
    if info_state not in infosets[player]:
        infosets[player][info_state] = len(infosets[player])
    if info_state not in infoset_action_maps[player]:
        infoset_action_maps[player][info_state] = []
    new_parent_is_keys = parent_is_keys[:]
    new_parent_is_keys[player] = info_state
    for action in legal_actions:
        isa_key = info_state + _DELIMITER + str(action)
        if isa_key not in infoset_actions[player]:
            infoset_actions[player][isa_key] = len(infoset_actions[player])
        if isa_key not in infoset_action_maps[player][info_state]:
            infoset_action_maps[player][info_state].append(isa_key)
        if player == 0:
            lps[1].add_or_reuse_variable(isa_key, lb=0)
            lps[1].set_cons_coeff(info_state, isa_key, 1.0)
        else:
            lps[0].add_or_reuse_variable(isa_key, lb=0)
            lps[0].set_cons_coeff(info_state, isa_key, 1.0)
        new_parent_isa_keys = parent_isa_keys[:]
        new_parent_isa_keys[player] = isa_key
        new_state = state.child(action)
        _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps, chance_reach, lps, new_parent_is_keys, new_parent_isa_keys)

def solve_zero_sum_game(game, solver=None):
    if False:
        return 10
    "Solve the two-player zero-sum game using sequence-form LPs.\n\n  Args:\n    game: the spiel game tp solve (must be zero-sum, sequential, and have chance\n      mode of deterministic or explicit stochastic).\n    solver: a specific solver to use, sent to cvxopt (i.e. 'lapack', 'blas',\n      'glpk'). A value of None uses cvxopt's default solver.\n\n  Returns:\n    A 4-tuple containing:\n      - player 0 value\n      - player 1 value\n      - player 0 policy: a policy.TabularPolicy for player 0\n      - player 1 policy: a policy.TabularPolicy for player 1\n  "
    assert game.num_players() == 2
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert game.get_type().chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC or game.get_type().chance_mode == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
    infosets = [{_EMPTY_INFOSET_KEYS[0]: 0}, {_EMPTY_INFOSET_KEYS[1]: 0}]
    infoset_actions = [{_EMPTY_INFOSET_ACTION_KEYS[0]: 0}, {_EMPTY_INFOSET_ACTION_KEYS[1]: 0}]
    infoset_action_maps = [{}, {}]
    lps = [lp_solver.LinearProgram(lp_solver.OBJ_MIN), lp_solver.LinearProgram(lp_solver.OBJ_MAX)]
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[1], lb=0)
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[0])
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[0], lb=0)
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[1])
    lps[0].set_obj_coeff(_EMPTY_INFOSET_KEYS[0], 1.0)
    lps[1].set_obj_coeff(_EMPTY_INFOSET_KEYS[1], -1.0)
    lps[0].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[1], lp_solver.CONS_TYPE_EQ)
    lps[0].set_cons_coeff(_EMPTY_INFOSET_KEYS[1], _EMPTY_INFOSET_ACTION_KEYS[1], -1.0)
    lps[0].set_cons_rhs(_EMPTY_INFOSET_KEYS[1], -1.0)
    lps[1].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[0], lp_solver.CONS_TYPE_EQ)
    lps[1].set_cons_coeff(_EMPTY_INFOSET_KEYS[0], _EMPTY_INFOSET_ACTION_KEYS[0], 1.0)
    lps[1].set_cons_rhs(_EMPTY_INFOSET_KEYS[0], 1.0)
    _construct_lps(game.new_initial_state(), infosets, infoset_actions, infoset_action_maps, 1.0, lps, _EMPTY_INFOSET_KEYS[:], _EMPTY_INFOSET_ACTION_KEYS[:])
    solutions = [lps[0].solve(solver=solver), lps[1].solve(solver=solver)]
    policies = [policy.TabularPolicy(game), policy.TabularPolicy(game)]
    for i in range(2):
        for info_state in infoset_action_maps[i]:
            total_weight = 0
            num_actions = 0
            for isa_key in infoset_action_maps[i][info_state]:
                total_weight += solutions[1 - i][lps[1 - i].get_var_id(isa_key)]
                num_actions += 1
            unif_pr = 1.0 / num_actions
            state_policy = policies[i].policy_for_key(info_state)
            for isa_key in infoset_action_maps[i][info_state]:
                rel_weight = solutions[1 - i][lps[1 - i].get_var_id(isa_key)]
                (_, action_str) = isa_key.split(_DELIMITER)
                action = int(action_str)
                pr_action = rel_weight / total_weight if total_weight > 0 else unif_pr
                state_policy[action] = pr_action
    return (solutions[0][lps[0].get_var_id(_EMPTY_INFOSET_KEYS[0])], solutions[1][lps[1].get_var_id(_EMPTY_INFOSET_KEYS[1])], policies[0], policies[1])