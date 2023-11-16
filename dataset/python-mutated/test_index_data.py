import numpy as np
import pandas as pd
import qlib.utils.index_data as idd
import unittest

class IndexDataTest(unittest.TestCase):

    def test_index_single_data(self):
        if False:
            for i in range(10):
                print('nop')
        sd = idd.SingleData(0, index=['foo', 'bar'])
        print(sd)
        sd = idd.SingleData()
        print(sd)
        with self.assertRaises(ValueError):
            idd.SingleData(range(10), index=['foo', 'bar'])
        sd = idd.SingleData([1, 2, 3, 4], index=['foo', 'bar', 'f', 'g'])
        print(sd)
        print(sd.iloc[1])
        with self.assertRaises(KeyError):
            print(sd.loc[1])
        print(sd.loc['foo'])
        print(sd.loc[:'bar'])
        print(sd.iloc[:3])

    def test_index_multi_data(self):
        if False:
            for i in range(10):
                print('nop')
        sd = idd.MultiData(0, index=['foo', 'bar'], columns=['f', 'g'])
        print(sd)
        with self.assertRaises(ValueError):
            idd.MultiData(range(10), index=['foo', 'bar'], columns=['f', 'g'])
        sd = idd.MultiData(np.arange(4).reshape(2, 2), index=['foo', 'bar'], columns=['f', 'g'])
        print(sd)
        print(sd.iloc[1])
        with self.assertRaises(KeyError):
            print(sd.loc[1])
        print(sd.loc['foo'])
        print(sd.loc[:'foo'])
        print(sd.loc[:, 'g':])

    def test_sorting(self):
        if False:
            return 10
        sd = idd.MultiData(np.arange(4).reshape(2, 2), index=['foo', 'bar'], columns=['f', 'g'])
        print(sd)
        sd.sort_index()
        print(sd)
        print(sd.loc[:'c'])

    def test_corner_cases(self):
        if False:
            while True:
                i = 10
        sd = idd.MultiData([[1, 2], [3, np.NaN]], index=['foo', 'bar'], columns=['f', 'g'])
        print(sd)
        self.assertTrue(np.isnan(sd.loc['bar', 'g']))
        print(sd.loc[~sd.loc[:, 'g'].isna().data.astype(bool)])
        print(self.assertTrue(idd.SingleData().index == idd.SingleData().index))
        print(idd.SingleData({}))
        print(idd.SingleData(pd.Series()))
        sd = idd.SingleData()
        with self.assertRaises(KeyError):
            sd.loc['foo']
        sd = idd.SingleData([1, 2, 3, 4], index=['foo', 'bar', 'f', 'g'])
        sd = sd.replace(dict(zip(range(1, 5), range(2, 6))))
        print(sd)
        self.assertTrue(sd.iloc[0] == 2)

    def test_ops(self):
        if False:
            print('Hello World!')
        sd1 = idd.SingleData([1, 2, 3, 4], index=['foo', 'bar', 'f', 'g'])
        sd2 = idd.SingleData([1, 2, 3, 4], index=['foo', 'bar', 'f', 'g'])
        print(sd1 + sd2)
        new_sd = sd2 * 2
        self.assertTrue(new_sd.index == sd2.index)
        sd1 = idd.SingleData([1, 2, None, 4], index=['foo', 'bar', 'f', 'g'])
        sd2 = idd.SingleData([1, 2, 3, None], index=['foo', 'bar', 'f', 'g'])
        self.assertTrue(np.isnan((sd1 + sd2).iloc[3]))
        self.assertTrue(sd1.add(sd2).sum() == 13)
        self.assertTrue(idd.sum_by_index([sd1, sd2], sd1.index, fill_value=0.0).sum() == 13)

    def test_todo(self):
        if False:
            print('Hello World!')
        pass

    def test_squeeze(self):
        if False:
            i = 10
            return i + 15
        sd1 = idd.SingleData([1, 2, 3, 4], index=['foo', 'bar', 'f', 'g'])
        self.assertTrue(not isinstance(np.nansum(sd1), idd.IndexData))
        self.assertTrue(not isinstance(np.sum(sd1), idd.IndexData))
        self.assertTrue(not isinstance(sd1.sum(), idd.IndexData))
        self.assertEqual(np.nansum(sd1), 10)
        self.assertEqual(np.sum(sd1), 10)
        self.assertEqual(sd1.sum(), 10)
        self.assertEqual(np.nanmean(sd1), 2.5)
        self.assertEqual(np.mean(sd1), 2.5)
        self.assertEqual(sd1.mean(), 2.5)
if __name__ == '__main__':
    unittest.main()