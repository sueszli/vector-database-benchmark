"""
You can run this example like this:

    .. code:: console

            $ rm -rf '/tmp/bar'
            $ luigi --module examples.foo_complex examples.Foo --workers 2 --local-scheduler

"""
import time
import random
import luigi
max_depth = 10
max_total_nodes = 50
current_nodes = 0

class Foo(luigi.Task):
    task_namespace = 'examples'

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print('Running Foo')

    def requires(self):
        if False:
            while True:
                i = 10
        global current_nodes
        for i in range(30 // max_depth):
            current_nodes += 1
            yield Bar(i)

class Bar(luigi.Task):
    task_namespace = 'examples'
    num = luigi.IntParameter()

    def run(self):
        if False:
            return 10
        time.sleep(1)
        self.output().open('w').close()

    def requires(self):
        if False:
            i = 10
            return i + 15
        global current_nodes
        if max_total_nodes > current_nodes:
            valor = int(random.uniform(1, 30))
            for i in range(valor // max_depth):
                current_nodes += 1
                yield Bar(current_nodes)

    def output(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the target output for this task.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        time.sleep(1)
        return luigi.LocalTarget('/tmp/bar/%d' % self.num)