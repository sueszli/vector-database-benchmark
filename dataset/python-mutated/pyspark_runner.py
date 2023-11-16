"""
The pyspark program.

This module will be run by spark-submit for PySparkTask jobs.

The first argument is a path to the pickled instance of the PySparkTask,
other arguments are the ones returned by PySparkTask.app_options()

"""
import abc
import logging
import os
import pickle
import sys
from luigi import configuration
sys.path.append(sys.path.pop(0))

class _SparkEntryPoint(metaclass=abc.ABCMeta):

    def __init__(self, conf):
        if False:
            for i in range(10):
                print('nop')
        self.conf = conf

    @abc.abstractmethod
    def __enter__(self):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        pass

class SparkContextEntryPoint(_SparkEntryPoint):
    sc = None

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        from pyspark import SparkContext
        self.sc = SparkContext(conf=self.conf)
        return (self.sc, self.sc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.sc.stop()

class SparkSessionEntryPoint(_SparkEntryPoint):
    spark = None

    def _check_major_spark_version(self):
        if False:
            return 10
        from pyspark import __version__ as spark_version
        major_version = int(spark_version.split('.')[0])
        if major_version < 2:
            raise RuntimeError("\n                Apache Spark {} does not support SparkSession entrypoint.\n                Try to set 'pyspark_runner.use_spark_session' to 'False' and switch to old-style syntax\n                ".format(spark_version))

    def __enter__(self):
        if False:
            return 10
        self._check_major_spark_version()
        from pyspark.sql import SparkSession
        self.spark = SparkSession.builder.config(conf=self.conf).enableHiveSupport().getOrCreate()
        return (self.spark, self.spark.sparkContext)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.spark.stop()

class AbstractPySparkRunner(object):
    _entry_point_class = None

    def __init__(self, job, *args):
        if False:
            i = 10
            return i + 15
        sys.path.append(os.path.dirname(job))
        with open(job, 'rb') as fd:
            self.job = pickle.load(fd)
        self.args = args

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        from pyspark import SparkConf
        conf = SparkConf()
        self.job.setup(conf)
        with self._entry_point_class(conf=conf) as (entry_point, sc):
            self.job.setup_remote(sc)
            self.job.main(entry_point, *self.args)

def _pyspark_runner_with(name, entry_point_class):
    if False:
        print('Hello World!')
    return type(name, (AbstractPySparkRunner,), {'_entry_point_class': entry_point_class})
PySparkRunner = _pyspark_runner_with('PySparkRunner', SparkContextEntryPoint)
PySparkSessionRunner = _pyspark_runner_with('PySparkSessionRunner', SparkSessionEntryPoint)

def _use_spark_session():
    if False:
        for i in range(10):
            print('nop')
    return bool(configuration.get_config().get('pyspark_runner', 'use_spark_session', False))

def _get_runner_class():
    if False:
        while True:
            i = 10
    if _use_spark_session():
        return PySparkSessionRunner
    return PySparkRunner
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    _get_runner_class()(*sys.argv[1:]).run()