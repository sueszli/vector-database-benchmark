import logging
import os
import luigi
import luigi.contrib.hadoop_jar
import luigi.contrib.hdfs
logger = logging.getLogger('luigi-interface')

def hadoop_examples_jar():
    if False:
        i = 10
        return i + 15
    config = luigi.configuration.get_config()
    examples_jar = config.get('hadoop', 'examples-jar')
    if not examples_jar:
        logger.error('You must specify hadoop:examples-jar in luigi.cfg')
        raise
    if not os.path.exists(examples_jar):
        logger.error("Can't find example jar: " + examples_jar)
        raise
    return examples_jar
DEFAULT_TERASORT_IN = '/tmp/terasort-in'
DEFAULT_TERASORT_OUT = '/tmp/terasort-out'

class TeraGen(luigi.contrib.hadoop_jar.HadoopJarJobTask):
    """
    Runs TeraGen, by default with 1TB of data (10B records)
    """
    records = luigi.Parameter(default='10000000000', description='Number of records, each record is 100 Bytes')
    terasort_in = luigi.Parameter(default=DEFAULT_TERASORT_IN, description='directory to store terasort input into.')

    def output(self):
        if False:
            print('Hello World!')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget(self.terasort_in)

    def jar(self):
        if False:
            print('Hello World!')
        return hadoop_examples_jar()

    def main(self):
        if False:
            print('Hello World!')
        return 'teragen'

    def args(self):
        if False:
            while True:
                i = 10
        return [self.records, self.output()]

class TeraSort(luigi.contrib.hadoop_jar.HadoopJarJobTask):
    """
    Runs TeraGent, by default using
    """
    terasort_in = luigi.Parameter(default=DEFAULT_TERASORT_IN, description='directory to store terasort input into.')
    terasort_out = luigi.Parameter(default=DEFAULT_TERASORT_OUT, description='directory to store terasort output into.')

    def requires(self):
        if False:
            i = 10
            return i + 15
        "\n        This task's dependencies:\n\n        * :py:class:`~.TeraGen`\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        return TeraGen(terasort_in=self.terasort_in)

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget(self.terasort_out)

    def jar(self):
        if False:
            while True:
                i = 10
        return hadoop_examples_jar()

    def main(self):
        if False:
            print('Hello World!')
        return 'terasort'

    def args(self):
        if False:
            print('Hello World!')
        return [self.input(), self.output()]