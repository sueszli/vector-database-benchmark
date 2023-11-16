import tempfile
import unittest
import os
import sys
from io import StringIO
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.profiler import UDFBasicProfiler

class UDFProfilerTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._old_sys_path = list(sys.path)
        class_name = self.__class__.__name__
        conf = SparkConf().set('spark.python.profile', 'true')
        self.spark = SparkSession.builder.master('local[4]').config(conf=conf).appName(class_name).getOrCreate()
        self.sc = self.spark.sparkContext

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.spark.stop()
        sys.path = self._old_sys_path

    def test_udf_profiler(self):
        if False:
            return 10
        self.do_computation()
        profilers = self.sc.profiler_collector.profilers
        self.assertEqual(3, len(profilers))
        old_stdout = sys.stdout
        try:
            sys.stdout = io = StringIO()
            self.sc.show_profiles()
        finally:
            sys.stdout = old_stdout
        d = tempfile.gettempdir()
        self.sc.dump_profiles(d)
        for (i, udf_name) in enumerate(['add1', 'add2', 'add1']):
            (id, profiler, _) = profilers[i]
            with self.subTest(id=id, udf_name=udf_name):
                stats = profiler.stats()
                self.assertTrue(stats is not None)
                (width, stat_list) = stats.get_print_list([])
                func_names = [func_name for (fname, n, func_name) in stat_list]
                self.assertTrue(udf_name in func_names)
                self.assertTrue(udf_name in io.getvalue())
                self.assertTrue('udf_%d.pstats' % id in os.listdir(d))

    def test_custom_udf_profiler(self):
        if False:
            for i in range(10):
                print('nop')

        class TestCustomProfiler(UDFBasicProfiler):

            def show(self, id):
                if False:
                    print('Hello World!')
                self.result = 'Custom formatting'
        self.sc.profiler_collector.udf_profiler_cls = TestCustomProfiler
        self.do_computation()
        profilers = self.sc.profiler_collector.profilers
        self.assertEqual(3, len(profilers))
        (_, profiler, _) = profilers[0]
        self.assertTrue(isinstance(profiler, TestCustomProfiler))
        self.sc.show_profiles()
        self.assertEqual('Custom formatting', profiler.result)

    def do_computation(self):
        if False:
            print('Hello World!')

        @udf
        def add1(x):
            if False:
                print('Hello World!')
            return x + 1

        @udf
        def add2(x):
            if False:
                return 10
            return x + 2
        df = self.spark.range(10)
        df.select(add1('id'), add2('id'), add1('id')).collect()
if __name__ == '__main__':
    from pyspark.sql.tests.test_udf_profiler import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)