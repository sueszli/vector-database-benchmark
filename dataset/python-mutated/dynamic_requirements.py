import os
import random as rnd
import time
import luigi

class Configuration(luigi.Task):
    seed = luigi.IntParameter()

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('/tmp/Config_%d.txt' % self.seed)

    def run(self):
        if False:
            i = 10
            return i + 15
        time.sleep(5)
        rnd.seed(self.seed)
        result = ','.join([str(x) for x in rnd.sample(list(range(300)), rnd.randint(7, 25))])
        with self.output().open('w') as f:
            f.write(result)

class Data(luigi.Task):
    magic_number = luigi.IntParameter()

    def output(self):
        if False:
            print('Hello World!')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('/tmp/Data_%d.txt' % self.magic_number)

    def run(self):
        if False:
            return 10
        time.sleep(1)
        with self.output().open('w') as f:
            f.write('%s' % self.magic_number)

class Dynamic(luigi.Task):
    seed = luigi.IntParameter(default=1)

    def output(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('/tmp/Dynamic_%d.txt' % self.seed)

    def run(self):
        if False:
            while True:
                i = 10
        config = self.clone(Configuration)
        yield config
        with config.output().open() as f:
            data = [int(x) for x in f.read().split(',')]
        data_dependent_deps = [Data(magic_number=x) for x in data]
        yield data_dependent_deps
        with self.output().open('w') as f:
            f.write('Tada!')

        def custom_complete(complete_fn):
            if False:
                return 10
            if not complete_fn(data_dependent_deps[0]):
                return False
            paths = [task.output().path for task in data_dependent_deps]
            basenames = os.listdir(os.path.dirname(paths[0]))
            return all((os.path.basename(path) in basenames for path in paths))
        yield luigi.DynamicRequirements(data_dependent_deps, custom_complete)
if __name__ == '__main__':
    luigi.run()