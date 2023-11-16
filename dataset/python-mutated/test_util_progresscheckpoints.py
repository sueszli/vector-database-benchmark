from test.picardtestcase import PicardTestCase
from picard.util.progresscheckpoints import ProgressCheckpoints

class ProgressCheckpointsTest(PicardTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()

    def test_empty_jobs(self):
        if False:
            print('Hello World!')
        checkpoints = ProgressCheckpoints(0, 1)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [])
        checkpoints = ProgressCheckpoints(0, 0)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [])
        checkpoints = ProgressCheckpoints(1, 0)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [])

    def test_uniformly_spaced_integer_distance(self):
        if False:
            while True:
                i = 10
        checkpoints = ProgressCheckpoints(100, 10)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    def test_uniformly_spaced_fractional_distance(self):
        if False:
            print('Hello World!')
        checkpoints = ProgressCheckpoints(100, 7)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [14, 28, 42, 57, 71, 85, 99])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [14, 28, 42, 57, 71, 85, 100])
        checkpoints = ProgressCheckpoints(10, 20)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [5, 15, 25, 35, 45, 55, 65, 75, 85, 100])
        checkpoints = ProgressCheckpoints(5, 10)
        self.assertEqual(list(sorted(checkpoints._checkpoints.keys())), [0, 1, 2, 3, 4])
        self.assertEqual(list(sorted(checkpoints._checkpoints.values())), [10, 30, 50, 70, 100])