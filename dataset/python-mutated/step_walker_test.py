import unittest
from geographiclib.geodesic import Geodesic
from mock import MagicMock, patch, mock
from pokemongo_bot.walkers.step_walker import StepWalker
NORMALIZED_LAT_LNG_DISTANCE = (6.3948578954430175e-06, 6.35204828670955e-06)

class TestStepWalker(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patcherSleep = patch('pokemongo_bot.walkers.step_walker.sleep')
        self.patcherSleep.start()
        self.bot = MagicMock()
        self.bot.position = [0, 0, 0]
        self.bot.api = MagicMock()

        def api_set_position(lat, lng, alt):
            if False:
                while True:
                    i = 10
            self.bot.position = [lat, lng, alt]
        self.bot.api.set_position = api_set_position

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.bot.position = [0, 0, 0]
        self.patcherSleep.stop()

    def test_normalized_distance(self):
        if False:
            return 10
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 1
        self.bot.config.walk_min = 1
        sw = StepWalker(self.bot, 0.1, 0.1, precision=0.0)
        self.assertGreater(sw.dest_lat, 0)
        self.assertGreater(sw.dest_lng, 0)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                while True:
                    i = 10
            mock_random.return_value = 0.0
            return sw.step()
        stayInPlace = run_step()
        self.assertFalse(stayInPlace)
        self.assertAlmostEqual(self.bot.position[0], NORMALIZED_LAT_LNG_DISTANCE[0], places=6)
        self.assertAlmostEqual(self.bot.position[1], NORMALIZED_LAT_LNG_DISTANCE[1], places=6)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_normalized_distance_times_2(self):
        if False:
            i = 10
            return i + 15
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 2
        self.bot.config.walk_min = 2
        sw = StepWalker(self.bot, 0.1, 0.1, precision=0.0)
        self.assertTrue(sw.dest_lat > 0)
        self.assertTrue(sw.dest_lng > 0)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                while True:
                    i = 10
            mock_random.return_value = 0.0
            return sw.step()
        stayInPlace = run_step()
        self.assertFalse(stayInPlace)
        self.assertAlmostEqual(self.bot.position[0], NORMALIZED_LAT_LNG_DISTANCE[0] * 2, places=6)
        self.assertAlmostEqual(self.bot.position[1], NORMALIZED_LAT_LNG_DISTANCE[1] * 2, places=6)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_small_distance_same_spot(self):
        if False:
            print('Hello World!')
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 1
        self.bot.config.walk_min = 1
        sw = StepWalker(self.bot, 0, 0, precision=0.0)
        self.assertEqual(sw.dest_lat, 0, 'dest_lat should be 0')
        self.assertEqual(sw.dest_lng, 0, 'dest_lng should be 0')

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                i = 10
                return i + 15
            mock_random.return_value = 0.0
            return sw.step()
        moveInprecision = run_step()
        self.assertTrue(moveInprecision, 'step should return True')
        distance = Geodesic.WGS84.Inverse(0.0, 0.0, self.bot.position[0], self.bot.position[1])['s12']
        self.assertTrue(0.0 <= distance <= sw.precision + sw.epsilon)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_small_distance_small_step(self):
        if False:
            i = 10
            return i + 15
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 1
        self.bot.config.walk_min = 1
        total_distance = Geodesic.WGS84.Inverse(0.0, 0.0, 1e-06, 1e-06)['s12']
        sw = StepWalker(self.bot, 1e-06, 1e-06, precision=0.2)
        self.assertEqual(sw.dest_lat, 1e-06)
        self.assertEqual(sw.dest_lng, 1e-06)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                while True:
                    i = 10
            mock_random.return_value = 0.0
            return sw.step()
        moveInprecistion = run_step()
        self.assertTrue(moveInprecistion, 'step should return True')
        distance = Geodesic.WGS84.Inverse(0.0, 0.0, self.bot.position[0], self.bot.position[1])['s12']
        self.assertTrue(0.0 <= abs(total_distance - distance) <= sw.precision + sw.epsilon)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_big_distances(self):
        if False:
            i = 10
            return i + 15
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 1
        self.bot.config.walk_min = 1
        sw = StepWalker(self.bot, 10, 10, precision=0.0)
        self.assertEqual(sw.dest_lat, 10)
        self.assertEqual(sw.dest_lng, 10)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                return 10
            mock_random.return_value = 0.0
            return sw.step()
        finishedWalking = run_step()
        self.assertFalse(finishedWalking, 'step should return False')
        self.assertAlmostEqual(self.bot.position[0], NORMALIZED_LAT_LNG_DISTANCE[0], places=6)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_stay_put(self):
        if False:
            return 10
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 4
        self.bot.config.walk_min = 2
        sw = StepWalker(self.bot, 10, 10, precision=0.0)
        self.assertEqual(sw.dest_lat, 10)
        self.assertEqual(sw.dest_lng, 10)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                return 10
            mock_random.return_value = 0.0
            return sw.step(speed=0)
        finishedWalking = run_step()
        self.assertFalse(finishedWalking, 'step should return False')
        distance = Geodesic.WGS84.Inverse(0.0, 0.0, self.bot.position[0], self.bot.position[1])['s12']
        self.assertTrue(0.0 <= distance <= sw.precision + sw.epsilon)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min

    def test_teleport(self):
        if False:
            i = 10
            return i + 15
        walk_max = self.bot.config.walk_max
        walk_min = self.bot.config.walk_min
        self.bot.config.walk_max = 4
        self.bot.config.walk_min = 2
        sw = StepWalker(self.bot, 10, 10, precision=0.0)
        self.assertEqual(sw.dest_lat, 10)
        self.assertEqual(sw.dest_lng, 10)

        @mock.patch('random.uniform')
        def run_step(mock_random):
            if False:
                while True:
                    i = 10
            mock_random.return_value = 0.0
            return sw.step(speed=float('inf'))
        finishedWalking = run_step()
        self.assertTrue(finishedWalking, 'step should return True')
        total_distance = Geodesic.WGS84.Inverse(0.0, 0.0, 10, 10)['s12']
        distance = Geodesic.WGS84.Inverse(0.0, 0.0, self.bot.position[0], self.bot.position[1])['s12']
        self.assertTrue(0.0 <= abs(total_distance - distance) <= sw.precision + sw.epsilon)
        self.bot.config.walk_max = walk_max
        self.bot.config.walk_min = walk_min