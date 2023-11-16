import luigi
from luigi.contrib.ftp import RemoteTarget
HOST = 'some_host'
USER = 'user'
PWD = 'some_password'

class ExperimentTask(luigi.ExternalTask):
    """
    This class represents something that was created elsewhere by an external process,
    so all we want to do is to implement the output method.
    """

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file that will be created in a FTP server.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return RemoteTarget('/experiment/output1.txt', HOST, username=USER, password=PWD)

    def run(self):
        if False:
            i = 10
            return i + 15
        "\n        The execution of this task will write 4 lines of data on this task's target output.\n        "
        with self.output().open('w') as outfile:
            print('data 0 200 10 50 60', file=outfile)
            print('data 1 190 9 52 60', file=outfile)
            print('data 2 200 10 52 60', file=outfile)
            print('data 3 195 1 52 60', file=outfile)

class ProcessingTask(luigi.Task):
    """
    This class represents something that was created elsewhere by an external process,
    so all we want to do is to implement the output method.
    """

    def requires(self):
        if False:
            return 10
        "\n        This task's dependencies:\n\n        * :py:class:`~.ExperimentTask`\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        return ExperimentTask()

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return luigi.LocalTarget('/tmp/processeddata.txt')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        avg = 0.0
        elements = 0
        sumval = 0.0
        for line in self.input().open('r'):
            values = line.split(' ')
            avg += float(values[2])
            sumval += float(values[3])
            elements = elements + 1
        avg = avg / elements
        with self.output().open('w') as outfile:
            print(avg, sumval, file=outfile)
if __name__ == '__main__':
    luigi.run()