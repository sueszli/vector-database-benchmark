import unittest
import pandas as pd
from bigdl.ppml.fl.data_utils import *

class TestDataUtils(unittest.TestCase):

    def test_pandas_api(self):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        (array, _) = convert_to_jtensor(df, feature_columns=['f1'])
        self.assert_(isinstance(array, JTensor))
        self.assertEqual(array.storage.shape, (2, 1))

    def test_numpy_api(self):
        if False:
            print('Hello World!')
        array = np.array([[1, 2], [3, 4]])
        (array, _) = convert_to_jtensor(array)
        self.assert_(isinstance(array, JTensor))
        self.assertEqual(array.storage.shape, (2, 2))
if __name__ == '__main__':
    unittest.main()