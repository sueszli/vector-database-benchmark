import unittest
from pyspark import SparkContext
from pyspark.sql import SparkSession

class MLlibTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.sc = SparkContext('local[4]', 'MLlib tests')
        self.spark = SparkSession(self.sc)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.spark.stop()