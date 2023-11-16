"""Tests for adidas."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import adidas
from open_spiel.python.algorithms.adidas_utils.games.big import ElFarol
from open_spiel.python.algorithms.adidas_utils.games.small import MatrixGame
from open_spiel.python.algorithms.adidas_utils.solvers.symmetric import qre_anneal as qre_anneal_sym

class AdidasTest(absltest.TestCase):

    def test_adidas_on_prisoners_dilemma(self):
        if False:
            while True:
                i = 10
        "Tests ADIDAS on a 2-player prisoner's dilemma game."
        pt_r = np.array([[-1, -3], [0, -2]])
        pt_r -= pt_r.min()
        pt_c = pt_r.T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        pt /= pt.max()
        game = MatrixGame(pt, seed=0)
        solver = qre_anneal_sym.Solver(temperature=100, proj_grad=False, euclidean=True, lrs=(0.0001, 0.0001), exp_thresh=0.01, rnd_init=True, seed=0)
        lle = adidas.ADIDAS(seed=0)
        lle.approximate_nash(game, solver, sym=True, num_iterations=1, num_samples=1, num_eval_samples=int(100000.0), approx_eval=True, exact_eval=True, avg_trajectory=False)
        self.assertLess(lle.results['exps_exact'][-1], 0.2)

    def test_adidas_on_elfarol(self):
        if False:
            return 10
        'Test ADIDAS on a 10-player, symmetric El Farol bar game.'
        game = ElFarol(n=10, c=0.7)
        solver = qre_anneal_sym.Solver(temperature=100, proj_grad=False, euclidean=False, lrs=(0.0001, 0.01), exp_thresh=0.01, seed=0)
        lle = adidas.ADIDAS(seed=0)
        lle.approximate_nash(game, solver, sym=True, num_iterations=1, num_samples=np.inf, num_eval_samples=int(100000.0), approx_eval=True, exact_eval=True, avg_trajectory=False)
        self.assertLess(lle.results['exps_exact'][-1], 0.5)
if __name__ == '__main__':
    absltest.main()