from absl.testing import absltest
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import neurd
import pyspiel
tf.disable_v2_behavior()
tf.enable_eager_execution()
_GAME = pyspiel.load_game('kuhn_poker')

def _new_model():
    if False:
        i = 10
        return i + 15
    return neurd.DeepNeurdModel(_GAME, num_hidden_layers=1, num_hidden_units=13, num_hidden_factors=1, use_skip_connections=True, autoencode=True)

class NeurdTest(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(NeurdTest, self).setUp()
        tf.set_random_seed(42)

    def test_neurd(self):
        if False:
            return 10
        num_iterations = 2
        models = [_new_model() for _ in range(_GAME.num_players())]
        solver = neurd.CounterfactualNeurdSolver(_GAME, models)
        average_policy = solver.average_policy()
        self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

        @tf.function
        def _train(model, data):
            if False:
                i = 10
                return i + 15
            neurd.train(model=model, data=data, batch_size=12, step_size=10.0, autoencoder_loss=tf.losses.huber_loss)
        for _ in range(num_iterations):
            solver.evaluate_and_update_policy(_train)
        average_policy = solver.average_policy()
        self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)
if __name__ == '__main__':
    absltest.main()