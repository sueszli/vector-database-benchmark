import unittest
import numpy as np
import numba

class NumpyGeneratorUsageTest(unittest.TestCase):

    def test_numpy_gen_usage(self):
        if False:
            return 10
        x = np.random.default_rng(1)
        y = np.random.default_rng(1)
        size = 10

        @numba.njit
        def do_stuff(gen):
            if False:
                return 10
            return gen.random(size=int(size / 2))
        original = x.random(size=size)
        numba_func_res = do_stuff(y)
        after_numba = y.random(size=int(size / 2))
        numba_res = np.concatenate((numba_func_res, after_numba))
        for (_np_res, _nb_res) in zip(original, numba_res):
            self.assertEqual(_np_res, _nb_res)
if __name__ == '__main__':
    unittest.main()