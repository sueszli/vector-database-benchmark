"""Tests for factory."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg.games import factory
import pyspiel

class FactoryTest(parameterized.TestCase):

    @parameterized.parameters(('mfg_crowd_modelling_2d', None), ('mfg_crowd_modelling_2d', 'crowd_modelling_2d_10x10'), ('mfg_crowd_modelling_2d', 'crowd_modelling_2d_four_rooms'), ('mfg_dynamic_routing', None), ('mfg_dynamic_routing', 'dynamic_routing_line'), ('mfg_dynamic_routing', 'dynamic_routing_braess'), ('python_mfg_dynamic_routing', None), ('python_mfg_dynamic_routing', 'dynamic_routing_line'), ('python_mfg_dynamic_routing', 'dynamic_routing_braess'), ('python_mfg_dynamic_routing', 'dynamic_routing_sioux_falls_dummy_demand'), ('python_mfg_dynamic_routing', 'dynamic_routing_sioux_falls'), ('python_mfg_periodic_aversion', None), ('python_mfg_predator_prey', None), ('python_mfg_predator_prey', 'predator_prey_5x5x3'))
    def test_smoke(self, game_name, setting):
        if False:
            for i in range(10):
                print('nop')
        game = factory.create_game_with_setting(game_name, setting)
        self.assertIsInstance(game, pyspiel.Game)
if __name__ == '__main__':
    absltest.main()