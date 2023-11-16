import unittest, pickle, os
from mock import patch
from pokemongo_bot.cell_workers.follow_cluster import FollowCluster

class FollowClusterTestCase(unittest.TestCase):

    @patch('pokemongo_bot.PokemonGoBot')
    def testWorkAway(self, mock_pokemongo_bot):
        if False:
            while True:
                i = 10
        forts_path = os.path.join(os.path.dirname(__file__), 'resources', 'example_forts.pickle')
        with open(forts_path, 'rb') as forts:
            ex_forts = pickle.load(forts)
        config = {'radius': 50, 'lured': False}
        mock_pokemongo_bot.position = (37.396787, -5.994587, 0)
        mock_pokemongo_bot.config.walk_max = 4.16
        mock_pokemongo_bot.config.walk_min = 2.16
        mock_pokemongo_bot.get_forts.return_value = ex_forts
        follow_cluster = FollowCluster(mock_pokemongo_bot, config)
        expected = (37.397183750142624, -5.993291250000001)
        result = follow_cluster.work()
        self.assertAlmostEqual(expected[0], result[0], delta=1e-11)
        self.assertAlmostEqual(expected[1], result[1], delta=1e-11)
        assert follow_cluster.is_at_destination == False
        assert follow_cluster.announced == False

    @patch('pokemongo_bot.PokemonGoBot')
    def testWorkArrived(self, mock_pokemongo_bot):
        if False:
            for i in range(10):
                print('nop')
        forts_path = os.path.join(os.path.dirname(__file__), 'resources', 'example_forts.pickle')
        with open(forts_path, 'rb') as forts:
            ex_forts = pickle.load(forts)
        config = {'radius': 50, 'lured': False}
        mock_pokemongo_bot.position = (37.39718375014263, -5.993291250000001, 0)
        mock_pokemongo_bot.config.walk_max = 4.16
        mock_pokemongo_bot.config.walk_min = 2.16
        mock_pokemongo_bot.get_forts.return_value = ex_forts
        follow_cluster = FollowCluster(mock_pokemongo_bot, config)
        expected = (37.397183750142624, -5.993291250000001)
        result = follow_cluster.work()
        self.assertAlmostEqual(expected[0], result[0], delta=1e-11)
        self.assertAlmostEqual(expected[1], result[1], delta=1e-11)
        assert follow_cluster.is_at_destination == True
        assert follow_cluster.announced == False