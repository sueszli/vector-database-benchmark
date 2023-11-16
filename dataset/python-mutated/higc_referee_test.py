"""Tests for open_spiel.python.referee."""
import os
from absl import flags
from absl.testing import absltest
import pyspiel
flags.DEFINE_string('bot_dir', os.path.dirname(__file__) + '/../bots', 'Path to python implementation of bots.')
FLAGS = flags.FLAGS

class RefereeTest(absltest.TestCase):

    def test_playing_tournament(self):
        if False:
            i = 10
            return i + 15
        ref = pyspiel.Referee('kuhn_poker', [f'python {FLAGS.bot_dir}/higc_random_bot_test.py'] * 2, settings=pyspiel.TournamentSettings(timeout_ready=2000, timeout_start=500))
        results = ref.play_tournament(num_matches=1)
        self.assertLen(results.matches, 1)
if __name__ == '__main__':
    absltest.main()