"""Strategy selectors repository."""
import numpy as np
DEFAULT_STRATEGY_SELECTION_METHOD = 'probabilistic'
EPSILON_MIN_POSITIVE_PROBA = 1e-08

def exhaustive(solver, number_policies_selected=1):
    if False:
        i = 10
        return i + 15
    "Returns every player's policies.\n\n  Args:\n    solver: A GenPSROSolver instance.\n    number_policies_selected: Number of policies to return for each player.\n      (Compatibility argument)\n\n  Returns:\n    used_policies : List of size 'num_players' of lists of size\n      min('number_policies_selected', num_policies') containing selected\n      policies.\n    used_policies_indexes: List of lists of the same shape as used_policies,\n      containing the list indexes of selected policies.\n  "
    del number_policies_selected
    policies = solver.get_policies()
    indexes = [list(range(len(pol))) for pol in policies]
    return (policies, indexes)

def filter_function_factory(filter_function):
    if False:
        while True:
            i = 10
    "Returns a function filtering players' strategies wrt.\n\n  'filter_function'.\n\n  This function is used to select which strategy to start training from. As\n  such, and in the Rectified Nash Response logic, filter_function expects a\n  certain set of arguments:\n    - player_policies: The list of policies for the current player.\n    - player: The current player id.\n    - effective_number_selected: The effective number of policies to select.\n    - solver: In case the above arguments weren't enough, the solver instance so\n    the filter_function can have more complex behavior.\n  And returns the selected policies and policy indexes for the current player.\n\n  Args:\n    filter_function: A filter function following the specifications above, used\n      to filter which strategy to start training from for each player.\n\n  Returns:\n    A filter function on all players.\n  "

    def filter_policies(solver, number_policies_selected=1):
        if False:
            while True:
                i = 10
        "Filters each player's policies according to 'filter_function'.\n\n    Args:\n      solver: The PSRO solver.\n      number_policies_selected: The expected number of policies to select. If\n        there are fewer policies than 'number_policies_selected', behavior will\n        saturate at num_policies.\n\n    Returns:\n      used_policies : List of length 'num_players' of lists of length\n        min('number_policies_selected', num_policies') containing selected\n        policies.\n      used_policies_indexes: List of lists of the same shape as used_policies,\n        containing the list indexes of selected policies.\n\n    "
        policies = solver.get_policies()
        num_players = len(policies)
        meta_strategy_probabilities = solver.get_meta_strategies()
        used_policies = []
        used_policy_indexes = []
        for player in range(num_players):
            player_policies = policies[player]
            current_selection_probabilities = meta_strategy_probabilities[player]
            effective_number = min(number_policies_selected, len(player_policies))
            (used_policy, used_policy_index) = filter_function(player_policies, current_selection_probabilities, player, effective_number, solver)
            used_policies.append(used_policy)
            used_policy_indexes.append(used_policy_index)
        return (used_policies, used_policy_indexes)
    return filter_policies

def rectified_filter(player_policies, selection_probabilities, player, effective_number_to_select, solver):
    if False:
        for i in range(10):
            print('nop')
    "Returns every strategy with nonzero selection probability.\n\n  Args:\n    player_policies: A list of policies for the current player.\n    selection_probabilities: Selection probabilities for 'player_policies'.\n    player: Player id.\n    effective_number_to_select: Effective number of policies to select.\n    solver: PSRO solver instance if kwargs needed.\n\n  Returns:\n    selected_policies : List of size 'effective_number_to_select'\n      containing selected policies.\n    selected_indexes: List of the same shape as selected_policies,\n      containing the list indexes of selected policies.\n  "
    del effective_number_to_select, solver, player
    selected_indexes = [i for i in range(len(player_policies)) if selection_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA]
    selected_policies = [player_policies[i] for i in selected_indexes]
    return (selected_policies, selected_indexes)

def probabilistic_filter(player_policies, selection_probabilities, player, effective_number_to_select, solver):
    if False:
        return 10
    "Returns every strategy with nonzero selection probability.\n\n  Args:\n    player_policies: A list of policies for the current player.\n    selection_probabilities: Selection probabilities for 'player_policies'.\n    player: Player id.\n    effective_number_to_select: Effective number of policies to select.\n    solver: PSRO solver instance if kwargs needed.\n\n  Returns:\n    selected_policies : List of size 'effective_number_to_select'\n      containing selected policies.\n    selected_indexes: List of the same shape as selected_policies,\n      containing the list indexes of selected policies.\n  "
    del solver, player
    selected_indexes = list(np.random.choice(list(range(len(player_policies))), effective_number_to_select, replace=False, p=selection_probabilities))
    selected_policies = [player_policies[i] for i in selected_indexes]
    return (selected_policies, selected_indexes)

def top_k_probabilities_filter(player_policies, selection_probabilities, player, effective_number_to_select, solver):
    if False:
        while True:
            i = 10
    "Returns top 'effective_number_to_select' highest probability policies.\n\n  Args:\n    player_policies: A list of policies for the current player.\n    selection_probabilities: Selection probabilities for 'player_policies'.\n    player: Player id.\n    effective_number_to_select: Effective number of policies to select.\n    solver: PSRO solver instance if kwargs needed.\n\n  Returns:\n    selected_policies : List of size 'effective_number_to_select'\n      containing selected policies.\n    selected_indexes: List of the same shape as selected_policies,\n      containing the list indexes of selected policies.\n  "
    del player, solver
    selected_indexes = [index for (_, index) in sorted(zip(selection_probabilities, list(range(len(player_policies)))), key=lambda pair: pair[0])][:effective_number_to_select]
    selected_policies = [player_policies[i] for i in selected_indexes]
    return (selected_policies, selected_indexes)

def uniform_filter(player_policies, selection_probabilities, player, effective_number_to_select, solver):
    if False:
        i = 10
        return i + 15
    "Returns 'effective_number_to_select' uniform-randomly selected policies.\n\n  Args:\n    player_policies: A list of policies for the current player.\n    selection_probabilities: Selection probabilities for 'player_policies'.\n    player: Player id.\n    effective_number_to_select: Effective number of policies to select.\n    solver: PSRO solver instance if kwargs needed.\n\n  Returns:\n    selected_policies : List of size 'effective_number_to_select'\n      containing selected policies.\n    selected_indexes: List of the same shape as selected_policies,\n      containing the list indexes of selected policies.\n  "
    del solver, selection_probabilities, player
    selected_indexes = list(np.random.choice(list(range(len(player_policies))), effective_number_to_select, replace=False, p=np.ones(len(player_policies)) / len(player_policies)))
    selected_policies = [player_policies[i] for i in selected_indexes]
    return (selected_policies, selected_indexes)

def functional_probabilistic_filter(player_policies, selection_probabilities, player, effective_number_to_select, solver):
    if False:
        while True:
            i = 10
    "Returns effective_number_to_select randomly selected policies by function.\n\n  Args:\n    player_policies: A list of policies for the current player.\n    selection_probabilities: Selection probabilities for 'player_policies'.\n    player: Player id.\n    effective_number_to_select: Effective number of policies to select.\n    solver: PSRO solver instance if kwargs needed.\n\n  Returns:\n    selected_policies : List of size 'effective_number_to_select'\n      containing selected policies.\n    selected_indexes: List of the same shape as selected_policies,\n      containing the list indexes of selected policies.\n  "
    kwargs = solver.get_kwargs()
    probability_computation_function = kwargs.get('selection_probability_function') or (lambda x: x.get_meta_strategies())
    selection_probabilities = probability_computation_function(solver)[player]
    selected_indexes = list(np.random.choice(list(range(len(player_policies))), effective_number_to_select, replace=False, p=selection_probabilities))
    selected_policies = [player_policies[i] for i in selected_indexes]
    return (selected_policies, selected_indexes)
uniform = filter_function_factory(uniform_filter)
rectified = filter_function_factory(rectified_filter)
probabilistic = filter_function_factory(probabilistic_filter)
top_k_probabilities = filter_function_factory(top_k_probabilities_filter)
functional_probabilistic = filter_function_factory(functional_probabilistic_filter)
'Selectors below are used to rectify probabilities.\n'

def get_current_and_average_payoffs(ps2ro_trainer, current_player, current_strategy):
    if False:
        return 10
    "Returns the current player's and average players' payoffs.\n\n  These payoffs are returned when current_player's strategy's index is\n  'current_strategy'.\n\n  Args:\n    ps2ro_trainer: A ps2ro object.\n    current_player: Integer, current player index.\n    current_strategy: Integer, current player's strategy index.\n\n  Returns:\n    Payoff tensor for current player, Average payoff tensor over all players.\n  "
    meta_games = ps2ro_trainer.meta_games
    current_payoff = meta_games[current_player]
    current_payoff = np.take(current_payoff, current_strategy, axis=current_player)
    average_payoffs = np.mean(meta_games, axis=0)
    average_payoffs = np.take(average_payoffs, current_strategy, axis=current_player)
    return (current_payoff, average_payoffs)

def rectified_selector(ps2ro_trainer, current_player, current_strategy):
    if False:
        print('Hello World!')
    (current_payoff, average_payoffs) = get_current_and_average_payoffs(ps2ro_trainer, current_player, current_strategy)
    res = current_payoff >= average_payoffs
    return np.expand_dims(res, axis=current_player)
'When using joint strategies, use the selectors below.\n'

def empty_list_generator(number_dimensions):
    if False:
        while True:
            i = 10
    result = []
    for _ in range(number_dimensions - 1):
        result = [result]
    return result

def get_indices_from_non_marginalized(policies):
    if False:
        i = 10
        return i + 15
    'Get a list of lists of indices from joint policies.\n\n  These are the ones used for training strategy selector.\n\n  Args:\n    policies: a list of joint policies.\n\n  Returns:\n    A list of lists of indices.\n  '
    num_players = len(policies[0])
    num_strategies = len(policies)
    return [list(range(num_strategies)) for _ in range(num_players)]

def rectified_non_marginalized(solver):
    if False:
        for i in range(10):
            print('nop')
    'Returns every strategy with nonzero selection probability.\n\n  Args:\n    solver: A GenPSROSolver instance.\n  '
    used_policies = []
    policies = solver.get_policies()
    num_players = len(policies)
    meta_strategy_probabilities = solver.get_and_update_non_marginalized_meta_strategies(update=False)
    for k in range(num_players):
        current_policies = policies[k]
        current_probabilities = meta_strategy_probabilities[k]
        current_policies = [current_policies[i] for i in range(len(current_policies)) if current_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA]
        used_policies.append(current_policies)
    return (used_policies, get_indices_from_non_marginalized(used_policies))

def exhaustive_non_marginalized(solver):
    if False:
        print('Hello World!')
    "Returns every player's policies.\n\n  Args:\n    solver: A GenPSROSolver instance.\n  "
    used_policies = solver.get_policies()
    return (used_policies, get_indices_from_non_marginalized(used_policies))

def probabilistic_non_marginalized(solver):
    if False:
        while True:
            i = 10
    'Returns [kwargs] policies randomly, proportionally with selection probas.\n\n  Args:\n    solver: A GenPSROSolver instance.\n  '
    kwargs = solver.get_kwargs()
    number_policies_to_select = kwargs.get('number_policies_selected') or 1
    ids = solver.get_joint_policy_ids()
    joint_strategy_probabilities = solver.get_and_update_non_marginalized_meta_strategies(update=False)
    effective_number = min(number_policies_to_select, len(ids))
    selected_policy_ids = list(np.random.choice(ids, effective_number, replace=False, p=joint_strategy_probabilities))
    used_policies = solver.get_joint_policies_from_id_list(selected_policy_ids)
    return (used_policies, get_indices_from_non_marginalized(used_policies))

def top_k_probabilites_non_marginalized(solver):
    if False:
        for i in range(10):
            print('nop')
    'Returns [kwargs] policies with highest selection probabilities.\n\n  Args:\n    solver: A GenPSROSolver instance.\n  '
    kwargs = solver.get_kwargs()
    number_policies_to_select = kwargs.get('number_policies_selected') or 1
    ids = solver.get_joint_policy_ids()
    effective_number = min(number_policies_to_select, len(ids))
    joint_strategy_probabilities = solver.get_and_update_non_marginalized_meta_strategies(update=False)
    sorted_list = sorted(zip(joint_strategy_probabilities, ids), reverse=True, key=lambda pair: pair[0])
    selected_policy_ids = [id_selected for (_, id_selected) in sorted_list][:effective_number]
    used_policies = solver.get_joint_policies_from_id_list(selected_policy_ids)
    return (used_policies, get_indices_from_non_marginalized(used_policies))

def uniform_non_marginalized(solver):
    if False:
        while True:
            i = 10
    'Returns [kwargs] randomly selected policies (Uniform probability).\n\n  Args:\n    solver: A GenPSROSolver instance.\n  '
    kwargs = solver.get_kwargs()
    number_policies_to_select = kwargs.get('number_policies_selected') or 1
    ids = solver.get_joint_policy_ids()
    effective_number = min(number_policies_to_select, len(ids))
    selected_policy_ids = list(np.random.choice(ids, effective_number, replace=False, p=np.ones(len(ids)) / len(ids)))
    used_policies = solver.get_joint_policies_from_id_list(selected_policy_ids)
    return (used_policies, get_indices_from_non_marginalized(used_policies))

def compressed_lambda(x):
    if False:
        print('Hello World!')
    return x.get_and_update_non_marginalized_meta_strategies(update=False)

def functional_probabilistic_non_marginalized(solver):
    if False:
        for i in range(10):
            print('nop')
    'Returns [kwargs] randomly selected policies with generated probabilities.\n\n  Args:\n    solver: A GenPSROSolver instance.\n  '
    kwargs = solver.get_kwargs()
    number_policies_to_select = kwargs.get('number_policies_selected') or 1
    probability_computation_function = kwargs.get('selection_probability_function') or compressed_lambda
    ids = solver.get_joint_policy_ids()
    joint_strategy_probabilities = probability_computation_function(solver)
    effective_number = min(number_policies_to_select, len(ids))
    selected_policies = list(np.random.choice(ids, effective_number, replace=False, p=joint_strategy_probabilities))
    used_policies = solver.get_joint_policies_from_id_list(selected_policies)
    return (used_policies, get_indices_from_non_marginalized(used_policies))
TRAINING_STRATEGY_SELECTORS = {'functional_probabilistic': functional_probabilistic, 'top_k_probabilities': top_k_probabilities, 'probabilistic': probabilistic, 'exhaustive': exhaustive, 'rectified': rectified, 'uniform': uniform, 'functional_probabilistic_non_marginalized': functional_probabilistic_non_marginalized, 'top_k_probabilites_non_marginalized': top_k_probabilites_non_marginalized, 'probabilistic_non_marginalized': probabilistic_non_marginalized, 'exhaustive_non_marginalized': exhaustive_non_marginalized, 'rectified_non_marginalized': rectified_non_marginalized, 'uniform_non_marginalized': uniform_non_marginalized}