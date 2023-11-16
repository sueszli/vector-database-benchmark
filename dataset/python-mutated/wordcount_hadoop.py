import luigi
import luigi.contrib.hadoop
import luigi.contrib.hdfs

class InputText(luigi.ExternalTask):
    """
    This task is a :py:class:`luigi.task.ExternalTask` which means it doesn't generate the
    :py:meth:`~.InputText.output` target on its own instead relying on the execution something outside of Luigi
    to produce it.
    """
    date = luigi.DateParameter()

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, it expects a file to be present in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget(self.date.strftime('/tmp/text/%Y-%m-%d.txt'))

class WordCount(luigi.contrib.hadoop.JobTask):
    """
    This task runs a :py:class:`luigi.contrib.hadoop.JobTask`
    over the target data returned by :py:meth:`~/.InputText.output` and
    writes the result into its :py:meth:`~.WordCount.output` target.

    This class uses :py:meth:`luigi.contrib.hadoop.JobTask.run`.
    """
    date_interval = luigi.DateIntervalParameter()

    def requires(self):
        if False:
            print('Hello World!')
        "\n        This task's dependencies:\n\n        * :py:class:`~.InputText`\n\n        :return: list of object (:py:class:`luigi.task.Task`)\n        "
        return [InputText(date) for date in self.date_interval.dates()]

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget('/tmp/text-count/%s' % self.date_interval)

    def mapper(self, line):
        if False:
            print('Hello World!')
        for word in line.strip().split():
            yield (word, 1)

    def reducer(self, key, values):
        if False:
            i = 10
            return i + 15
        yield (key, sum(values))
if __name__ == '__main__':
    luigi.run()