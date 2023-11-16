import unittest
import numpy as np
import pandas as pd
from qlib.contrib.data.utils.sepdf import SepDataFrame

class SepDF(unittest.TestCase):

    def to_str(self, obj):
        if False:
            return 10
        return ''.join(str(obj).split())

    def test_index_data(self):
        if False:
            while True:
                i = 10
        np.random.seed(42)
        index = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']), np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
        cols = [np.repeat(np.array(['g1', 'g2']), 2), np.arange(4)]
        df = pd.DataFrame(np.random.randn(8, 4), index=index, columns=cols)
        sdf = SepDataFrame(df_dict={'g2': df['g2']}, join=None)
        sdf['g2', 4] = 3
        sdf['g1'] = df['g1']
        exp = "\n        {'g2':                 2         3  4\n        bar one  0.647689  1.523030  3\n            two  1.579213  0.767435  3\n        baz one -0.463418 -0.465730  3\n            two -1.724918 -0.562288  3\n        foo one -0.908024 -1.412304  3\n            two  0.067528 -1.424748  3\n        qux one -1.150994  0.375698  3\n            two -0.601707  1.852278  3, 'g1':                 0         1\n        bar one  0.496714 -0.138264\n            two -0.234153 -0.234137\n        baz one -0.469474  0.542560\n            two  0.241962 -1.913280\n        foo one -1.012831  0.314247\n            two  1.465649 -0.225776\n        qux one -0.544383  0.110923\n            two -0.600639 -0.291694}\n        "
        self.assertEqual(self.to_str(sdf._df_dict), self.to_str(exp))
        del df['g1']
        del df['g2']
        del sdf['g1']
        del sdf['g2']
if __name__ == '__main__':
    unittest.main()