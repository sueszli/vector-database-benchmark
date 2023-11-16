"""Implementation of Alpha-Rank for general games.

Namely, computes fixation probabilities, Markov chain, and associated
stationary distribution given a population size and payoff matrix involving
n-strategy interactions.

All equations and variable names correspond to the following paper:
  https://arxiv.org/abs/1903.01373

"""
import numpy as np
import scipy.linalg as la
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils

def _get_payoff(payoff_table_k, payoffs_are_hpt_format, strat_profile, k=None):
    if False:
        print('Hello World!')
    "Gets the payoff of the k-th agent in a single or multi-population game.\n\n  Namely, accepts the payoff table of the k-th agent (which can be matrix or\n  HPT format), the index k of the agent of interest (so its payoff can be looked\n  up in case of an HPT format payoff table), and the pure strategy profile.\n\n  For multipopulation games, we currently only support games where the k-th\n  agent's payoff is a function of the HPT distribution (a vector\n  indicating the number of players playing each strategy), as opposed to the\n  strategy profile (a vector indicating the strategy of each player). This is\n  due to the nature of the PayoffTable class, which currently only tracks\n  distributions in the first k columns (rather than profiles).\n\n  Args:\n    payoff_table_k: The k-th agent's payoff table, in matrix or HPT format.\n    payoffs_are_hpt_format: Boolean indicating whether payoff_table_k is a\n      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT) or a\n      numpy array. True indicates HPT format, False indicates numpy array.\n    strat_profile: The pure strategy profile.\n    k: The index of the agent of interest. Only used for HPT case, and only >0\n      for a multi-population game.\n\n  Returns:\n    The k-th agent's payoff.\n  "
    if payoffs_are_hpt_format:
        assert k is not None
        distribution = payoff_table_k.get_distribution_from_profile(strat_profile)
        payoff_profile = payoff_table_k[tuple(distribution)]
        return payoff_profile[strat_profile[k]]
    else:
        return payoff_table_k[tuple(strat_profile)]

def _get_singlepop_2player_fitness(payoff_table, payoffs_are_hpt_format, m, my_popsize, my_strat, opponent_strat, use_local_selection_model):
    if False:
        i = 10
        return i + 15
    'Gets a target agent fitness given a finite population of competitors.\n\n  Note that this is only applicable to 2-player symmetric games.\n  Namely, gets fitness of an agent i playing my_strat in underlying population\n  of (my_popsize agents playing my_strat) and (m-my_popsize agents playing\n  opponent_strat).\n\n  Args:\n    payoff_table: A payoff table.\n    payoffs_are_hpt_format: Boolean indicating whether payoff_table is a\n      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or a\n      numpy array. True indicates HPT format, False indicates numpy array.\n    m: The total number of agents in the population.\n    my_popsize: The number of agents in the population playing my strategy.\n    my_strat: Index of my strategy.\n    opponent_strat: Index of the opposing strategy.\n    use_local_selection_model: Enable local evolutionary selection model, which\n      considers fitness against the current opponent only, rather than the\n      global population state.\n\n  Returns:\n    The fitness of agent i.\n  '
    if use_local_selection_model:
        fitness = payoff_table[tuple([my_strat, opponent_strat])]
    else:
        fitness = (my_popsize - 1) / (m - 1) * _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[my_strat, my_strat], k=0) + (m - my_popsize) / (m - 1) * _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[my_strat, opponent_strat], k=0)
    return fitness

def _get_rho_sr(payoff_table, payoffs_are_hpt_format, m, r, s, alpha, game_is_constant_sum, use_local_selection_model, payoff_sum=None):
    if False:
        i = 10
        return i + 15
    'Gets fixation probability of rogue strategy r in population playing s.\n\n  Args:\n    payoff_table: A payoff table.\n    payoffs_are_hpt_format: Boolean indicating whether payoff_table is a\n      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or a\n      numpy array. True indicates HPT format, False indicates numpy array.\n    m: The total number of agents in the population.\n    r: Rogue strategy r.\n    s: Population strategy s.\n    alpha: Fermi distribution temperature parameter.\n    game_is_constant_sum: Boolean indicating if the game is constant sum.\n    use_local_selection_model: Enable local evolutionary selection model, which\n      considers fitness against the current opponent only, rather than the\n      global population state.\n    payoff_sum: The payoff sum if the game is constant sum, or None otherwise.\n\n  Returns:\n    The fixation probability.\n  '
    if use_local_selection_model or game_is_constant_sum:
        payoff_rs = _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[r, s], k=0)
        if use_local_selection_model:
            payoff_sr = _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[s, r], k=0)
            u = alpha * (payoff_rs - payoff_sr)
        else:
            assert payoff_sum is not None
            u = alpha * m / (m - 1) * (payoff_rs - payoff_sum / 2)
        if np.isclose(u, 0, atol=1e-14):
            result = 1 / m
        else:
            result = (1 - np.exp(-u)) / (1 - np.exp(-m * u))
    else:
        assert payoff_sum is None
        summed = 0
        for l in range(1, m):
            t_mult = 1.0
            for p_r in range(1, l + 1):
                p_s = m - p_r
                f_ri = _get_singlepop_2player_fitness(payoff_table, payoffs_are_hpt_format, m, my_popsize=p_r, my_strat=r, opponent_strat=s, use_local_selection_model=use_local_selection_model)
                f_sj = _get_singlepop_2player_fitness(payoff_table, payoffs_are_hpt_format, m, my_popsize=p_s, my_strat=s, opponent_strat=r, use_local_selection_model=use_local_selection_model)
                t_mult *= np.exp(-alpha * (f_ri - f_sj))
            summed += t_mult
        result = (1 + summed) ** (-1)
    return result

def _get_rho_sr_multipop(payoff_table_k, payoffs_are_hpt_format, k, m, r, s, alpha, use_fast_compute=True):
    if False:
        return 10
    "Gets fixation probability for multi-population games.\n\n  Specifically, considers the fitnesses of two strategy profiles r and s given\n  the payoff table of the k-th population. Profile s is the current profile and\n  r is a mutant profile. Profiles r and s are identical except for the k-th\n  element, which corresponds to the deviation of the k-th population's\n  monomorphic strategy from s[k] to r[k].\n\n  Args:\n    payoff_table_k: The k-th population's payoff table.\n    payoffs_are_hpt_format: Boolean indicating whether payoff_table_k is a\n      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or numpy\n      array. True indicates HPT format, False indicates numpy array.\n    k: Index of the k-th population.\n    m: Total number of agents in the k-th population.\n    r: Strategy profile containing mutant strategy r for population k.\n    s: Current strategy profile.\n    alpha: Fermi distribution temperature parameter.\n    use_fast_compute: Boolean indicating whether closed-form computation should\n      be used.\n\n  Returns:\n    Probability of strategy r fixating in population k.\n  "
    f_r = _get_payoff(payoff_table_k, payoffs_are_hpt_format, r, k)
    f_s = _get_payoff(payoff_table_k, payoffs_are_hpt_format, s, k)
    if use_fast_compute:
        u = alpha * (f_r - f_s)
        if np.isclose(u, 0, atol=1e-14):
            result = 1 / m
        else:
            result = (1 - np.exp(-u)) / (1 - np.exp(-m * u))
    else:
        summed = 0
        for l in range(1, m):
            t_mult = 1.0
            for p_r in range(1, l + 1):
                t_mult *= np.exp(-alpha * (f_r - f_s))
            summed += t_mult
        result = (1 + summed) ** (-1)
    return result

def _get_singlepop_transition_matrix(payoff_table, payoffs_are_hpt_format, m, alpha, game_is_constant_sum, use_local_selection_model, payoff_sum, use_inf_alpha=False, inf_alpha_eps=0.1):
    if False:
        while True:
            i = 10
    'Gets the Markov transition matrix for a single-population game.\n\n  Args:\n    payoff_table: A payoff table.\n    payoffs_are_hpt_format: Boolean indicating whether payoff_table is a\n      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or a\n      numpy array. True indicates HPT format, False indicates numpy array.\n    m: Total number of agents in the k-th population.\n    alpha: Fermi distribution temperature parameter.\n    game_is_constant_sum: Boolean indicating if the game is constant sum.\n    use_local_selection_model: Enable local evolutionary selection model, which\n      considers fitness against the current opponent only, rather than the\n      global population state.\n    payoff_sum: The payoff sum if the game is constant sum, or None otherwise.\n    use_inf_alpha: Use infinite-alpha alpharank model.\n    inf_alpha_eps: Noise term (epsilon) used in infinite-alpha alpharank model.\n\n  Returns:\n    Markov transition matrix.\n  '
    num_strats_per_population = utils.get_num_strats_per_population([payoff_table], payoffs_are_hpt_format)
    num_strats = num_strats_per_population[0]
    c = np.zeros((num_strats, num_strats))
    rhos = np.zeros((num_strats, num_strats))
    for s in range(num_strats):
        for r in range(num_strats):
            if s != r:
                if use_inf_alpha:
                    eta = 1.0 / (num_strats - 1)
                    payoff_rs = _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[r, s], k=0)
                    payoff_sr = _get_payoff(payoff_table, payoffs_are_hpt_format, strat_profile=[s, r], k=0)
                    if np.isclose(payoff_rs, payoff_sr, atol=1e-14):
                        c[s, r] = eta * 0.5
                    elif payoff_rs > payoff_sr:
                        c[s, r] = eta * (1 - inf_alpha_eps)
                    else:
                        c[s, r] = eta * inf_alpha_eps
                else:
                    rhos[s, r] = _get_rho_sr(payoff_table, payoffs_are_hpt_format, m, r, s, alpha, game_is_constant_sum, use_local_selection_model, payoff_sum)
                    eta = 1.0 / (num_strats - 1)
                    c[s, r] = eta * rhos[s, r]
        c[s, s] = 1 - sum(c[s, :])
    return (c, rhos)

def _get_multipop_transition_matrix(payoff_tables, payoffs_are_hpt_format, m, alpha, use_inf_alpha=False, inf_alpha_eps=0.1):
    if False:
        while True:
            i = 10
    'Gets Markov transition matrix for multipopulation games.'
    num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
    num_profiles = utils.get_num_profiles(num_strats_per_population)
    eta = 1.0 / np.sum(num_strats_per_population - 1)
    c = np.zeros((num_profiles, num_profiles))
    rhos = np.zeros((num_profiles, num_profiles))
    for id_row_profile in range(num_profiles):
        row_profile = utils.get_strat_profile_from_id(num_strats_per_population, id_row_profile)
        next_profile_gen = utils.get_valid_next_profiles(num_strats_per_population, row_profile)
        for (index_population_that_changed, col_profile) in next_profile_gen:
            id_col_profile = utils.get_id_from_strat_profile(num_strats_per_population, col_profile)
            if use_inf_alpha:
                payoff_col = _get_payoff(payoff_tables[index_population_that_changed], payoffs_are_hpt_format, col_profile, k=index_population_that_changed)
                payoff_row = _get_payoff(payoff_tables[index_population_that_changed], payoffs_are_hpt_format, row_profile, k=index_population_that_changed)
                if np.isclose(payoff_col, payoff_row, atol=1e-14):
                    c[id_row_profile, id_col_profile] = eta * 0.5
                elif payoff_col > payoff_row:
                    c[id_row_profile, id_col_profile] = eta * (1 - inf_alpha_eps)
                else:
                    c[id_row_profile, id_col_profile] = eta * inf_alpha_eps
            else:
                rhos[id_row_profile, id_col_profile] = _get_rho_sr_multipop(payoff_table_k=payoff_tables[index_population_that_changed], payoffs_are_hpt_format=payoffs_are_hpt_format, k=index_population_that_changed, m=m, r=col_profile, s=row_profile, alpha=alpha)
                c[id_row_profile, id_col_profile] = eta * rhos[id_row_profile, id_col_profile]
        c[id_row_profile, id_row_profile] = 1 - sum(c[id_row_profile, :])
    return (c, rhos)

def _get_stationary_distr(c):
    if False:
        i = 10
        return i + 15
    'Gets stationary distribution of transition matrix c.'
    (eigenvals, left_eigenvecs, _) = la.eig(c, left=True, right=True)
    mask = abs(eigenvals - 1.0) < 1e-10
    left_eigenvecs = left_eigenvecs[:, mask]
    num_stationary_eigenvecs = np.shape(left_eigenvecs)[1]
    if num_stationary_eigenvecs != 1:
        raise ValueError('Expected 1 stationary distribution, but found %d' % num_stationary_eigenvecs)
    left_eigenvecs *= 1.0 / sum(left_eigenvecs)
    return left_eigenvecs.real.flatten()

def print_results(payoff_tables, payoffs_are_hpt_format, rhos=None, rho_m=None, c=None, pi=None):
    if False:
        i = 10
        return i + 15
    'Prints the finite-population analysis results.'
    print('Payoff tables:\n')
    if payoffs_are_hpt_format:
        for payoff_table in payoff_tables:
            print(payoff_table())
    else:
        print(payoff_tables)
    if rho_m is not None:
        print('\nNeutral fixation probability (rho_m):\n', rho_m)
    if rhos is not None and rho_m is not None:
        print('\nFixation probability matrix (rho_{r,s}/rho_m):\n', np.around(rhos / rho_m, decimals=2))
    if c is not None:
        print('\nMarkov transition matrix (c):\n', np.around(c, decimals=2))
    if pi is not None:
        print('\nStationary distribution (pi):\n', pi)

def sweep_pi_vs_epsilon(payoff_tables, strat_labels=None, warm_start_epsilon=None, visualize=False, return_epsilon=False, min_iters=10, max_iters=100, min_epsilon=1e-14, num_strats_to_label=10, legend_sort_clusters=False):
    if False:
        print('Hello World!')
    'Computes infinite-alpha distribution for a range of perturbations.\n\n  The range of response graph perturbations is defined in epsilon_list.\n\n  Note that min_iters and max_iters is necessary as it may sometimes appear the\n  stationary distribution has converged for a game in the first few iterations,\n  where in reality a sufficiently smaller epsilon is needed for the distribution\n  to first diverge, then reconverge. This behavior is dependent on both the\n  payoff structure and bounds, so the parameters min_iters and max_iters can be\n  used to fine-tune this.\n\n  Args:\n    payoff_tables: List of game payoff tables, one for each agent identity.\n      Each payoff_table may be either a numpy array, or a\n      _PayoffTableInterface object.\n    strat_labels: Human-readable strategy labels. See get_strat_profile_labels()\n      in utils.py for formatting details.\n    warm_start_epsilon: Initial value of epsilon to use.\n    visualize: Plot the sweep results.\n    return_epsilon: Whether to return the final epsilon used.\n    min_iters: the minimum number of sweep iterations.\n    max_iters: the maximum number of sweep iterations.\n    min_epsilon: the minimum value of epsilon to be tested, at which point the\n      sweep terminates (if not converged already).\n    num_strats_to_label: Number of strats to label in legend\n    legend_sort_clusters: If true, strategies in the same cluster are sorted in\n      the legend according to orderings for earlier alpha values. Primarily for\n      visualization purposes! Rankings for lower alpha values should be\n      interpreted carefully.\n\n  Returns:\n   pi: AlphaRank stationary distribution.\n   epsilon: The AlphaRank transition matrix noise level resulting from sweep.\n  '
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    num_populations = len(payoff_tables)
    num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
    if num_populations == 1:
        num_profiles = num_strats_per_population[0]
    else:
        num_profiles = utils.get_num_profiles(num_strats_per_population)
    assert strat_labels is None or isinstance(strat_labels, dict) or len(strat_labels) == num_profiles
    pi_list = np.empty((num_profiles, 0))
    (pi, alpha, m) = (None, None, None)
    epsilon_list = []
    epsilon_pi_hist = {}
    num_iters = 0
    epsilon_mult_factor = 0.5
    alpharank_succeeded_once = False
    if warm_start_epsilon is not None:
        epsilon = warm_start_epsilon
    else:
        epsilon = 0.5
    while True:
        try:
            pi_prev = pi
            (_, _, pi, _, _) = compute(payoff_tables, m=m, alpha=alpha, use_inf_alpha=True, inf_alpha_eps=epsilon)
            epsilon_pi_hist[epsilon] = pi
            if num_iters > min_iters and np.allclose(pi, pi_prev):
                break
            epsilon *= epsilon_mult_factor
            num_iters += 1
            alpharank_succeeded_once = True
            assert num_iters < max_iters, 'Alpharank stationary distr. not foundafter {} iterations of pi_vs_epsilonsweep'.format(num_iters)
        except ValueError as _:
            print('Error: ', _, epsilon, min_epsilon)
            assert epsilon >= min_epsilon, 'AlphaRank stationary distr. not found &epsilon < min_epsilon.'
            epsilon /= epsilon_mult_factor
            if alpharank_succeeded_once:
                epsilon_mult_factor = (epsilon_mult_factor + 1.0) / 2.0
                epsilon *= epsilon_mult_factor
    (epsilon_list, pi_list) = zip(*[(epsilon, epsilon_pi_hist[epsilon]) for epsilon in sorted(epsilon_pi_hist.keys(), reverse=True)])
    pi_list = np.asarray(pi_list)
    if visualize:
        if strat_labels is None:
            strat_labels = utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
        alpharank_visualizer.plot_pi_vs_alpha(pi_list.T, epsilon_list, num_populations, num_strats_per_population, strat_labels, num_strats_to_label=num_strats_to_label, legend_sort_clusters=legend_sort_clusters, xlabel='Infinite-AlphaRank Noise $\\epsilon$')
    if return_epsilon:
        return (pi_list[-1], epsilon_list[-1])
    else:
        return pi_list[-1]

def sweep_pi_vs_alpha(payoff_tables, strat_labels=None, warm_start_alpha=None, visualize=False, return_alpha=False, m=50, rtol=1e-05, atol=1e-08, num_strats_to_label=10, legend_sort_clusters=False):
    if False:
        i = 10
        return i + 15
    'Computes stationary distribution, pi, for range of selection intensities.\n\n  The range of selection intensities is defined in alpha_list and corresponds\n  to the temperature of the Fermi selection function.\n\n  Args:\n    payoff_tables: List of game payoff tables, one for each agent identity. Each\n      payoff_table may be either a numpy array, or a _PayoffTableInterface\n      object.\n    strat_labels: Human-readable strategy labels. See get_strat_profile_labels()\n      in utils.py for formatting details.\n    warm_start_alpha: Initial value of alpha to use.\n    visualize: Plot the sweep results.\n    return_alpha: Whether to return the final alpha used.\n    m: AlphaRank population size.\n    rtol: The relative tolerance parameter for np.allclose calls.\n    atol: The absolute tolerance parameter for np.allclose calls.\n    num_strats_to_label: Number of strats to label in legend\n    legend_sort_clusters: If true, strategies in the same cluster are sorted in\n      the legend according to orderings for earlier alpha values. Primarily for\n      visualization purposes! Rankings for lower alpha values should be\n      interpreted carefully.\n\n  Returns:\n   pi: AlphaRank stationary distribution.\n   alpha: The AlphaRank selection-intensity level resulting from sweep.\n  '
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    num_populations = len(payoff_tables)
    num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
    if num_populations == 1:
        num_profiles = num_strats_per_population[0]
    else:
        num_profiles = utils.get_num_profiles(num_strats_per_population)
    assert strat_labels is None or isinstance(strat_labels, dict) or len(strat_labels) == num_profiles
    pi_list = np.empty((num_profiles, 0))
    alpha_list = []
    num_iters = 0
    alpha_mult_factor = 2.0
    if warm_start_alpha is not None:
        alpha = warm_start_alpha
        alpharank_succeeded_once = False
    else:
        alpha = 0.0001
    while 1:
        try:
            (_, _, pi, _, _) = compute(payoff_tables, alpha=alpha, m=m)
            pi_list = np.append(pi_list, np.reshape(pi, (-1, 1)), axis=1)
            alpha_list.append(alpha)
            if num_iters > 0 and np.allclose(pi, pi_list[:, num_iters - 1], rtol, atol):
                break
            alpha *= alpha_mult_factor
            num_iters += 1
            alpharank_succeeded_once = True
        except ValueError as _:
            if warm_start_alpha is not None and (not alpharank_succeeded_once):
                alpha /= 2
            elif not np.allclose(pi_list[:, -1], pi_list[:, -2], rtol, atol):
                alpha /= alpha_mult_factor
                alpha_mult_factor = (alpha_mult_factor + 1.0) / 2.0
                alpha *= alpha_mult_factor
            else:
                break
    if visualize:
        if strat_labels is None:
            strat_labels = utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
        alpharank_visualizer.plot_pi_vs_alpha(pi_list.T, alpha_list, num_populations, num_strats_per_population, strat_labels, num_strats_to_label=num_strats_to_label, legend_sort_clusters=legend_sort_clusters)
    if return_alpha:
        return (pi, alpha)
    else:
        return pi

def compute_and_report_alpharank(payoff_tables, m=50, alpha=100, verbose=False, num_top_strats_to_print=8):
    if False:
        while True:
            i = 10
    'Computes and visualizes Alpha-Rank outputs.\n\n  Args:\n    payoff_tables: List of game payoff tables, one for each agent identity. Each\n      payoff_table may be either a numpy array, or a _PayoffTableInterface\n      object.\n    m: Finite population size.\n    alpha: Fermi distribution temperature parameter.\n    verbose: Set to True to print intermediate results.\n    num_top_strats_to_print: Number of top strategies to print.\n\n  Returns:\n    pi: AlphaRank stationary distribution/rankings.\n  '
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    (rhos, rho_m, pi, _, _) = compute(payoff_tables, m=m, alpha=alpha)
    strat_labels = utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
    if verbose:
        print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)
    utils.print_rankings_table(payoff_tables, pi, strat_labels, num_top_strats_to_print=num_top_strats_to_print)
    m_network_plotter = alpharank_visualizer.NetworkPlot(payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
    m_network_plotter.compute_and_draw_network()
    return pi

def compute(payoff_tables, m=50, alpha=100, use_local_selection_model=True, verbose=False, use_inf_alpha=False, inf_alpha_eps=0.01):
    if False:
        i = 10
        return i + 15
    'Computes the finite population stationary statistics.\n\n  Args:\n    payoff_tables: List of game payoff tables, one for each agent identity. Each\n      payoff_table may be either a numpy array, or a _PayoffTableInterface\n      object.\n    m: Finite population size.\n    alpha: Fermi distribution temperature parameter.\n    use_local_selection_model: Enable local evolutionary selection model, which\n      considers fitness against the current opponent only, rather than the\n      global population state.\n    verbose: Set to True to print intermediate results.\n    use_inf_alpha: Use infinite-alpha alpharank model.\n    inf_alpha_eps: Noise term to use in infinite-alpha alpharank model.\n\n  Returns:\n    rhos: Matrix of strategy-to-strategy fixation probabilities.\n    rho_m: Neutral fixation probability.\n    pi: Finite population stationary distribution.\n    num_strats: Number of available strategies.\n  '
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    num_populations = len(payoff_tables)
    num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
    if np.array_equal(num_strats_per_population, np.ones(len(num_strats_per_population))):
        rhos = np.asarray([[1]])
        rho_m = 1.0 / m if not use_inf_alpha else 1
        num_profiles = 1
        pi = np.asarray([1.0])
        return (rhos, rho_m, pi, num_profiles, num_strats_per_population)
    if verbose:
        print('Constructing c matrix')
        print('num_strats_per_population:', num_strats_per_population)
    if num_populations == 1:
        (game_is_constant_sum, payoff_sum) = utils.check_is_constant_sum(payoff_tables[0], payoffs_are_hpt_format)
        if verbose:
            print('game_is_constant_sum:', game_is_constant_sum, 'payoff sum: ', payoff_sum)
        (c, rhos) = _get_singlepop_transition_matrix(payoff_tables[0], payoffs_are_hpt_format, m, alpha, game_is_constant_sum, use_local_selection_model, payoff_sum, use_inf_alpha=use_inf_alpha, inf_alpha_eps=inf_alpha_eps)
        num_profiles = num_strats_per_population[0]
    else:
        (c, rhos) = _get_multipop_transition_matrix(payoff_tables, payoffs_are_hpt_format, m, alpha, use_inf_alpha=use_inf_alpha, inf_alpha_eps=inf_alpha_eps)
        num_profiles = utils.get_num_profiles(num_strats_per_population)
    pi = _get_stationary_distr(c)
    rho_m = 1.0 / m if not use_inf_alpha else 1
    if verbose:
        print_results(payoff_tables, payoffs_are_hpt_format, rhos, rho_m, c, pi)
    return (rhos, rho_m, pi, num_profiles, num_strats_per_population)

def suggest_alpha(payoff_tables, tol=0.1):
    if False:
        return 10
    "Suggests an alpha for use in alpha-rank.\n\n  The suggested alpha is approximately the smallest possible alpha such that\n  the ranking has 'settled out'. It is calculated as\n  -ln(tol)/min_gap_between_payoffs.\n\n  The logic behind this settling out is that the fixation probabilities can be\n  expanded as a series, and the relative size of each term in this series\n  changes with alpha. As alpha gets larger and larger, one of the terms in\n  this series comes to dominate, and this causes the ranking to settle\n  down. Just how fast this domination happens is easy to calculate, and this\n  function uses it to estimate the alpha by which the ranking has settled.\n\n  You can find further discussion at the PR:\n\n  https://github.com/deepmind/open_spiel/pull/403\n\n  Args:\n    payoff_tables: List of game payoff tables, one for each agent identity. Each\n      payoff_table may be either a numpy array, or a _PayoffTableInterface\n      object.\n    tol: the desired gap between the first and second terms in the fixation\n      probability expansion. A smaller tolerance leads to a larger alpha, and\n      a 'more settled out' ranking.\n\n  Returns:\n    A suggested alpha.\n  "
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
    num_profiles = utils.get_num_profiles(num_strats_per_population)
    gap = np.inf
    for id_row_profile in range(num_profiles):
        row_profile = utils.get_strat_profile_from_id(num_strats_per_population, id_row_profile)
        next_profile_gen = utils.get_valid_next_profiles(num_strats_per_population, row_profile)
        for (index_population_that_changed, col_profile) in next_profile_gen:
            payoff_table_k = payoff_tables[index_population_that_changed]
            f_r = _get_payoff(payoff_table_k, payoffs_are_hpt_format, col_profile, index_population_that_changed)
            f_s = _get_payoff(payoff_table_k, payoffs_are_hpt_format, row_profile, index_population_that_changed)
            if f_r > f_s:
                gap = min(gap, f_r - f_s)
    return -np.log(tol) / gap