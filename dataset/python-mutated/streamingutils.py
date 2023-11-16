import os
import tempfile
import time
import unittest
from pyspark import SparkConf, SparkContext, RDD
from pyspark.streaming import StreamingContext
from pyspark.testing.utils import search_jar
kinesis_test_environ_var = 'ENABLE_KINESIS_TESTS'
should_skip_kinesis_tests = not os.environ.get(kinesis_test_environ_var) == '1'
if should_skip_kinesis_tests:
    kinesis_requirement_message = "Skipping all Kinesis Python tests as environmental variable 'ENABLE_KINESIS_TESTS' was not set."
else:
    kinesis_asl_assembly_jar = search_jar('connector/kinesis-asl-assembly', 'spark-streaming-kinesis-asl-assembly-', 'spark-streaming-kinesis-asl-assembly_')
    if kinesis_asl_assembly_jar is None:
        kinesis_requirement_message = "Skipping all Kinesis Python tests as the optional Kinesis project was not compiled into a JAR. To run these tests, you need to build Spark with 'build/sbt -Pkinesis-asl assembly/package streaming-kinesis-asl-assembly/assembly' or 'build/mvn -Pkinesis-asl package' before running this test."
    else:
        existing_args = os.environ.get('PYSPARK_SUBMIT_ARGS', 'pyspark-shell')
        jars_args = '--jars %s' % kinesis_asl_assembly_jar
        os.environ['PYSPARK_SUBMIT_ARGS'] = ' '.join([jars_args, existing_args])
        kinesis_requirement_message = None
should_test_kinesis = kinesis_requirement_message is None

class PySparkStreamingTestCase(unittest.TestCase):
    timeout = 30
    duration = 0.5

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        class_name = cls.__name__
        conf = SparkConf().set('spark.default.parallelism', 1)
        cls.sc = SparkContext(appName=class_name, conf=conf)
        cls.sc.setCheckpointDir(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.sc.stop()
        try:
            jSparkContextOption = SparkContext._jvm.SparkContext.get()
            if jSparkContextOption.nonEmpty():
                jSparkContextOption.get().stop()
        except BaseException:
            pass

    def setUp(self):
        if False:
            while True:
                i = 10
        self.ssc = StreamingContext(self.sc, self.duration)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ssc is not None:
            self.ssc.stop(False)
        try:
            jStreamingContextOption = StreamingContext._jvm.SparkContext.getActive()
            if jStreamingContextOption.nonEmpty():
                jStreamingContextOption.get().stop(False)
        except BaseException:
            pass

    def wait_for(self, result, n):
        if False:
            while True:
                i = 10
        start_time = time.time()
        while len(result) < n and time.time() - start_time < self.timeout:
            time.sleep(0.01)
        if len(result) < n:
            print('timeout after', self.timeout)

    def _take(self, dstream, n):
        if False:
            i = 10
            return i + 15
        '\n        Return the first `n` elements in the stream (will start and stop).\n        '
        results = []

        def take(_, rdd):
            if False:
                i = 10
                return i + 15
            if rdd and len(results) < n:
                results.extend(rdd.take(n - len(results)))
        dstream.foreachRDD(take)
        self.ssc.start()
        self.wait_for(results, n)
        return results

    def _collect(self, dstream, n, block=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Collect each RDDs into the returned list.\n\n        Returns\n        -------\n        list\n            which will have the collected items.\n        '
        result = []

        def get_output(_, rdd):
            if False:
                print('Hello World!')
            if rdd and len(result) < n:
                r = rdd.collect()
                if r:
                    result.append(r)
        dstream.foreachRDD(get_output)
        if not block:
            return result
        self.ssc.start()
        self.wait_for(result, n)
        return result

    def _test_func(self, input, func, expected, sort=False, input2=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        input : list\n            dataset for the test. This should be list of lists.\n        func : function\n            wrapped function. This function should return PythonDStream object.\n        expected\n            expected output for this testcase.\n        '
        if not isinstance(input[0], RDD):
            input = [self.sc.parallelize(d, 1) for d in input]
        input_stream = self.ssc.queueStream(input)
        if input2 and (not isinstance(input2[0], RDD)):
            input2 = [self.sc.parallelize(d, 1) for d in input2]
        input_stream2 = self.ssc.queueStream(input2) if input2 is not None else None
        if input2:
            stream = func(input_stream, input_stream2)
        else:
            stream = func(input_stream)
        result = self._collect(stream, len(expected))
        if sort:
            self._sort_result_based_on_key(result)
            self._sort_result_based_on_key(expected)
        self.assertEqual(expected, result)

    def _sort_result_based_on_key(self, outputs):
        if False:
            i = 10
            return i + 15
        'Sort the list based on first value.'
        for output in outputs:
            output.sort(key=lambda x: x[0])