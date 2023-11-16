"""Tests for crowd_modelling_2d."""
from absl.testing import absltest
from open_spiel.python.mfg.games import crowd_modelling_2d

class CrowdModelling2DTest(absltest.TestCase):

    def test_grid_to_forbidden_states(self):
        if False:
            for i in range(10):
                print('nop')
        forbidden_states = crowd_modelling_2d.grid_to_forbidden_states(['#####', '# # #', '#   #', '#####'])
        self.assertEqual(forbidden_states, '[0|0;1|0;2|0;3|0;4|0;0|1;2|1;4|1;0|2;4|2;0|3;1|3;2|3;3|3;4|3]')
if __name__ == '__main__':
    absltest.main()