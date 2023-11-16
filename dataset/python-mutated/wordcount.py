import luigi

class InputText(luigi.ExternalTask):
    """
    This class represents something that was created elsewhere by an external process,
    so all we want to do is to implement the output method.
    """
    date = luigi.DateParameter()

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, it expects a file to be present in the local file system.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget(self.date.strftime('/var/tmp/text/%Y-%m-%d.txt'))

class WordCount(luigi.Task):
    date_interval = luigi.DateIntervalParameter()

    def requires(self):
        if False:
            return 10
        "\n        This task's dependencies:\n\n        * :py:class:`~.InputText`\n\n        :return: list of object (:py:class:`luigi.task.Task`)\n        "
        return [InputText(date) for date in self.date_interval.dates()]

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('/var/tmp/text-count/%s' % self.date_interval)

    def run(self):
        if False:
            print('Hello World!')
        '\n        1. count the words for each of the :py:meth:`~.InputText.output` targets created by :py:class:`~.InputText`\n        2. write the count into the :py:meth:`~.WordCount.output` target\n        '
        count = {}
        for f in self.input():
            for line in f.open('r'):
                for word in line.strip().split():
                    count[word] = count.get(word, 0) + 1
        f = self.output().open('w')
        for (word, count) in count.items():
            f.write('%s\t%d\n' % (word, count))
        f.close()