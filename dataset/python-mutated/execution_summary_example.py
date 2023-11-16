"""
You can run this example like this:

    .. code:: console

            $ luigi --module examples.execution_summary_example examples.EntryPoint --local-scheduler
            ...
            ... lots of spammy output
            ...
            INFO: There are 11 pending tasks unique to this worker
            INFO: Worker Worker(salt=843361665, workers=1, host=arash-spotify-T440s, username=arash, pid=18534) was stopped. Shutting down Keep-Alive thread
            INFO:
            ===== Luigi Execution Summary =====

            Scheduled 218 tasks of which:
            * 195 complete ones were encountered:
                - 195 examples.Bar(num=5...199)
            * 1 ran successfully:
                - 1 examples.Boom(...)
            * 22 were left pending, among these:
                * 1 were missing external dependencies:
                    - 1 MyExternal()
                * 21 had missing dependencies:
                    - 1 examples.EntryPoint()
                    - examples.Foo(num=100, num2=16) and 9 other examples.Foo
                    - 10 examples.DateTask(date=1998-03-23...1998-04-01, num=5)

            This progress looks :| because there were missing external dependencies

            ===== Luigi Execution Summary =====
"""
import datetime
import luigi

class MyExternal(luigi.ExternalTask):

    def complete(self):
        if False:
            while True:
                i = 10
        return False

class Boom(luigi.Task):
    task_namespace = 'examples'
    this_is_a_really_long_I_mean_way_too_long_and_annoying_parameter = luigi.IntParameter()

    def run(self):
        if False:
            i = 10
            return i + 15
        print('Running Boom')

    def requires(self):
        if False:
            while True:
                i = 10
        for i in range(5, 200):
            yield Bar(i)

class Foo(luigi.Task):
    task_namespace = 'examples'
    num = luigi.IntParameter()
    num2 = luigi.IntParameter()

    def run(self):
        if False:
            print('Hello World!')
        print('Running Foo')

    def requires(self):
        if False:
            print('Hello World!')
        yield MyExternal()
        yield Boom(0)

class Bar(luigi.Task):
    task_namespace = 'examples'
    num = luigi.IntParameter()

    def run(self):
        if False:
            while True:
                i = 10
        self.output().open('w').close()

    def output(self):
        if False:
            for i in range(10):
                print('nop')
        return luigi.LocalTarget('/tmp/bar/%d' % self.num)

class DateTask(luigi.Task):
    task_namespace = 'examples'
    date = luigi.DateParameter()
    num = luigi.IntParameter()

    def run(self):
        if False:
            while True:
                i = 10
        print('Running DateTask')

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        yield MyExternal()
        yield Boom(0)

class EntryPoint(luigi.Task):
    task_namespace = 'examples'

    def run(self):
        if False:
            while True:
                i = 10
        print('Running EntryPoint')

    def requires(self):
        if False:
            i = 10
            return i + 15
        for i in range(10):
            yield Foo(100, 2 * i)
        for i in range(10):
            yield DateTask(datetime.date(1998, 3, 23) + datetime.timedelta(days=i), 5)