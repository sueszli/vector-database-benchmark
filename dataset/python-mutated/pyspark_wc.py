import luigi
from luigi.contrib.s3 import S3Target
from luigi.contrib.spark import SparkSubmitTask, PySparkTask

class InlinePySparkWordCount(PySparkTask):
    """
    This task runs a :py:class:`luigi.contrib.spark.PySparkTask` task
    over the target data in :py:meth:`wordcount.input` (a file in S3) and
    writes the result into its :py:meth:`wordcount.output` target (a file in S3).

    This class uses :py:meth:`luigi.contrib.spark.PySparkTask.main`.

    Example luigi configuration::

        [spark]
        spark-submit: /usr/local/spark/bin/spark-submit
        master: spark://spark.example.org:7077
        # py-packages: numpy, pandas

    """
    driver_memory = '2g'
    executor_memory = '3g'

    def input(self):
        if False:
            print('Hello World!')
        return S3Target('s3n://bucket.example.org/wordcount.input')

    def output(self):
        if False:
            print('Hello World!')
        return S3Target('s3n://bucket.example.org/wordcount.output')

    def main(self, sc, *args):
        if False:
            i = 10
            return i + 15
        sc.textFile(self.input().path).flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).saveAsTextFile(self.output().path)

class PySparkWordCount(SparkSubmitTask):
    """
    This task is the same as :py:class:`InlinePySparkWordCount` above but uses
    an external python driver file specified in :py:meth:`app`

    It runs a :py:class:`luigi.contrib.spark.SparkSubmitTask` task
    over the target data in :py:meth:`wordcount.input` (a file in S3) and
    writes the result into its :py:meth:`wordcount.output` target (a file in S3).

    This class uses :py:meth:`luigi.contrib.spark.SparkSubmitTask.run`.

    Example luigi configuration::

        [spark]
        spark-submit: /usr/local/spark/bin/spark-submit
        master: spark://spark.example.org:7077
        deploy-mode: client

    """
    driver_memory = '2g'
    executor_memory = '3g'
    total_executor_cores = luigi.IntParameter(default=100, significant=False)
    name = 'PySpark Word Count'
    app = 'wordcount.py'

    def app_options(self):
        if False:
            while True:
                i = 10
        return [self.input().path, self.output().path]

    def input(self):
        if False:
            while True:
                i = 10
        return S3Target('s3n://bucket.example.org/wordcount.input')

    def output(self):
        if False:
            print('Hello World!')
        return S3Target('s3n://bucket.example.org/wordcount.output')
'\n// Corresponding example Spark Job, running Word count with Spark\'s Python API\n// This file would have to be saved into wordcount.py\n\nimport sys\nfrom pyspark import SparkContext\n\nif __name__ == "__main__":\n\n    sc = SparkContext()\n    sc.textFile(sys.argv[1])       .flatMap(lambda line: line.split())       .map(lambda word: (word, 1))       .reduceByKey(lambda a, b: a + b)       .saveAsTextFile(sys.argv[2])\n'