"""Example running AlphaRank on OpenSpiel games.

  AlphaRank output variable names corresponds to the following paper:
    https://arxiv.org/abs/1903.01373
"""
from absl import app
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel

def get_kuhn_poker_data(num_players=3):
    if False:
        print('Hello World!')
    'Returns the kuhn poker data for the number of players specified.'
    game = pyspiel.load_game('kuhn_poker', {'players': num_players})
    xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
    for _ in range(3):
        xfp_solver.iteration()
    if num_players == 2:
        meta_games = xfp_solver.get_empirical_metagame(100, seed=1)
    elif num_players == 3:
        meta_games = xfp_solver.get_empirical_metagame(100, seed=5)
    elif num_players == 4:
        meta_games = xfp_solver.get_empirical_metagame(100, seed=2)
    payoff_tables = []
    for i in range(num_players):
        payoff_tables.append(meta_games[i])
    return payoff_tables

def main(unused_arg):
    if False:
        i = 10
        return i + 15
    payoff_tables = get_kuhn_poker_data()
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    strat_labels = utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
    (rhos, rho_m, pi, _, _) = alpharank.compute(payoff_tables, alpha=100.0)
    alpharank.print_results(payoff_tables, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)
    utils.print_rankings_table(payoff_tables, pi, strat_labels)
    m_network_plotter = alpharank_visualizer.NetworkPlot(payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
    m_network_plotter.compute_and_draw_network()
if __name__ == '__main__':
    app.run(main)