import random
import luigi
import luigi.format
import luigi.contrib.hdfs
from luigi.contrib.spark import SparkSubmitTask

class UserItemMatrix(luigi.Task):
    data_size = luigi.IntParameter()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates :py:attr:`~.UserItemMatrix.data_size` elements.\n        Writes this data in \\ separated value format into the target :py:func:`~/.UserItemMatrix.output`.\n\n        The data has the following elements:\n\n        * `user` is the default Elasticsearch id field,\n        * `track`: the text,\n        * `rating`: the day when the data was created.\n\n        '
        w = self.output().open('w')
        for user in range(self.data_size):
            track = int(random.random() * self.data_size)
            w.write('%d\\%d\\%f' % (user, track, 1.0))
        w.close()

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget('data-matrix', format=luigi.format.Gzip)

class SparkALS(SparkSubmitTask):
    """
    This task runs a :py:class:`luigi.contrib.spark.SparkSubmitTask` task
    over the target data returned by :py:meth:`~/.UserItemMatrix.output` and
    writes the result into its :py:meth:`~.SparkALS.output` target (a file in HDFS).

    This class uses :py:meth:`luigi.contrib.spark.SparkSubmitTask.run`.

    Example luigi configuration::

        [spark]
        spark-submit: /usr/local/spark/bin/spark-submit
        master: yarn-client

    """
    data_size = luigi.IntParameter(default=1000)
    driver_memory = '2g'
    executor_memory = '3g'
    num_executors = luigi.IntParameter(default=100)
    app = 'my-spark-assembly.jar'
    entry_class = 'com.spotify.spark.ImplicitALS'

    def app_options(self):
        if False:
            return 10
        return [self.input().path, self.output().path]

    def requires(self):
        if False:
            print('Hello World!')
        "\n        This task's dependencies:\n\n        * :py:class:`~.UserItemMatrix`\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        return UserItemMatrix(self.data_size)

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget('als-output/', format=luigi.format.Gzip)
'\n// Corresponding example Spark Job, a wrapper around the MLLib ALS job.\n// This class would have to be jarred into my-spark-assembly.jar\n// using sbt assembly (or package) and made available to the Luigi job\n// above.\n\npackage com.spotify.spark\n\nimport org.apache.spark._\nimport org.apache.spark.mllib.recommendation.{Rating, ALS}\nimport org.apache.hadoop.io.compress.GzipCodec\n\nobject ImplicitALS {\n\n  def main(args: Array[String]) {\n    val sc = new SparkContext(args(0), "ImplicitALS")\n    val input = args(1)\n    val output = args(2)\n\n    val ratings = sc.textFile(input)\n      .map { l: String =>\n        val t = l.split(\'\t\')\n        Rating(t(0).toInt, t(1).toInt, t(2).toFloat)\n      }\n\n    val model = ALS.trainImplicit(ratings, 40, 20, 0.8, 150)\n    model\n      .productFeatures\n      .map { case (id, vec) =>\n        id + "\t" + vec.map(d => "%.6f".format(d)).mkString(" ")\n      }\n      .saveAsTextFile(output, classOf[GzipCodec])\n\n    sc.stop()\n  }\n}\n'