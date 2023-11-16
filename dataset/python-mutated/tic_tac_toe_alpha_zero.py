"""Simple AlphaZero tic tac toe example.

Take a look at the log-learner.txt in the output directory.

If you want more control, check out `alpha_zero.py`.
"""
from absl import app
from absl import flags
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.utils import spawn
flags.DEFINE_string('path', None, 'Where to save checkpoints.')
FLAGS = flags.FLAGS

def main(unused_argv):
    if False:
        for i in range(10):
            print('nop')
    config = alpha_zero.Config(game='tic_tac_toe', path=FLAGS.path, learning_rate=0.01, weight_decay=0.0001, train_batch_size=128, replay_buffer_size=2 ** 14, replay_buffer_reuse=4, max_steps=25, checkpoint_freq=25, actors=4, evaluators=4, uct_c=1, max_simulations=20, policy_alpha=0.25, policy_epsilon=1, temperature=1, temperature_drop=4, evaluation_window=50, eval_levels=7, nn_model='resnet', nn_width=128, nn_depth=2, observation_shape=None, output_size=None, quiet=True)
    alpha_zero.alpha_zero(config)
if __name__ == '__main__':
    with spawn.main_handler():
        app.run(main)