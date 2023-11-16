from absl.testing import absltest
import torch
import torch.nn.functional as F
import pyspiel
from open_spiel.python.pytorch import neurd
_GAME = pyspiel.load_game('kuhn_poker')

def _new_model():
    if False:
        i = 10
        return i + 15
    return neurd.DeepNeurdModel(_GAME, num_hidden_layers=1, num_hidden_units=13, num_hidden_factors=1, use_skip_connections=True, autoencode=True)

class NeurdTest(absltest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(NeurdTest, self).setUp()
        torch.manual_seed(42)

    def test_neurd(self):
        if False:
            for i in range(10):
                print('nop')
        num_iterations = 2
        models = [_new_model() for _ in range(_GAME.num_players())]
        solver = neurd.CounterfactualNeurdSolver(_GAME, models)
        average_policy = solver.average_policy()
        self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

        def _train(model, data):
            if False:
                print('Hello World!')
            neurd.train(model=model, data=data, batch_size=12, step_size=10.0, autoencoder_loss=F.huber_loss)
        for _ in range(num_iterations):
            solver.evaluate_and_update_policy(_train)
        average_policy = solver.average_policy()
        self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)
if __name__ == '__main__':
    absltest.main()