import os
import shutil
import test
import unittest
from tensorflowonspark import dfutil

class DFUtilTest(test.SparkTest):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(DFUtilTest, cls).setUpClass()
        cls.tfrecord_dir = os.getcwd() + os.sep + 'test_tfr'

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        super(DFUtilTest, cls).tearDownClass()

    def setUp(self):
        if False:
            print('Hello World!')
        super(DFUtilTest, self).setUp()
        shutil.rmtree(self.tfrecord_dir, ignore_errors=True)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_dfutils(self):
        if False:
            return 10
        row1 = ('text string', 1, [2, 3, 4, 5], -1.1, [-2.2, -3.3, -4.4, -5.5], bytearray(b'\xff\xfe\xfd\xfc'))
        rdd = self.sc.parallelize([row1])
        df1 = self.spark.createDataFrame(rdd, ['a', 'b', 'c', 'd', 'e', 'f'])
        print('schema: {}'.format(df1.schema))
        dfutil.saveAsTFRecords(df1, self.tfrecord_dir)
        self.assertTrue(os.path.isdir(self.tfrecord_dir))
        df2 = dfutil.loadTFRecords(self.sc, self.tfrecord_dir, binary_features=['f'])
        row2 = df2.take(1)[0]
        print('row_saved: {}'.format(row1))
        print('row_loaded: {}'.format(row2))
        self.assertEqual(row1[0], row2['a'])
        self.assertEqual(row1[1], row2['b'])
        self.assertEqual(row1[2], row2['c'])
        self.assertAlmostEqual(row1[3], row2['d'], 6)
        for i in range(len(row1[4])):
            self.assertAlmostEqual(row1[4][i], row2['e'][i], 6)
        print('type(f): {}'.format(type(row2['f'])))
        for i in range(len(row1[5])):
            self.assertEqual(row1[5][i], row2['f'][i])
        self.assertFalse(dfutil.isLoadedDF(df1))
        self.assertTrue(dfutil.isLoadedDF(df2))
        df_ref = df2
        self.assertTrue(dfutil.isLoadedDF(df_ref))
        df3 = df2.filter(df2.a == 'string_label')
        self.assertFalse(dfutil.isLoadedDF(df3))
        df2 = df3
        self.assertFalse(dfutil.isLoadedDF(df2))
if __name__ == '__main__':
    unittest.main()