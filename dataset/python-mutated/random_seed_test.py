"""Tests for tensorflow.python.framework.random_seed."""
from tensorflow.python.eager import context
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class RandomSeedTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testRandomSeed(self):
        if False:
            print('Hello World!')
        test_cases = [((None, None), (None, None)), ((None, 1), (random_seed.DEFAULT_GRAPH_SEED, 1)), ((1, 1), (1, 1)), ((0, 0), (0, 2 ** 31 - 1)), ((2 ** 31 - 1, 0), (0, 2 ** 31 - 1)), ((0, 2 ** 31 - 1), (0, 2 ** 31 - 1))]
        if context.executing_eagerly():
            pass
        else:
            test_cases.append(((1, None), (1, 0)))
        for tc in test_cases:
            (tinput, toutput) = (tc[0], tc[1])
            random_seed.set_random_seed(tinput[0])
            (g_seed, op_seed) = random_seed.get_seed(tinput[1])
            msg = 'test_case = {0}, got {1}, want {2}'.format(tinput, (g_seed, op_seed), toutput)
            self.assertEqual((g_seed, op_seed), toutput, msg=msg)
            random_seed.set_random_seed(None)
if __name__ == '__main__':
    test.main()