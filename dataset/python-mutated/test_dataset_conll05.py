import os
import unittest
import numpy as np
from paddle.text.datasets import Conll05st

class TestConll05st(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        conll05st = Conll05st()
        self.assertTrue(len(conll05st) == 5267)
        idx = np.random.randint(0, 5267)
        sample = conll05st[idx]
        self.assertTrue(len(sample) == 9)
        for s in sample:
            self.assertTrue(len(s.shape) == 1)
        assert os.path.exists(conll05st.get_embedding())
if __name__ == '__main__':
    unittest.main()