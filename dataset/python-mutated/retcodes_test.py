from helpers import LuigiTestCase, with_config
import mock
import luigi
import luigi.scheduler
from luigi.cmdline import luigi_run

class RetcodesTest(LuigiTestCase):

    def run_and_expect(self, joined_params, retcode, extra_args=['--local-scheduler', '--no-lock']):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(SystemExit) as cm:
            luigi_run(joined_params.split(' ') + extra_args)
        self.assertEqual(cm.exception.code, retcode)

    def run_with_config(self, retcode_config, *args, **kwargs):
        if False:
            return 10
        with_config(dict(retcode=retcode_config))(self.run_and_expect)(*args, **kwargs)

    def test_task_failed(self):
        if False:
            return 10

        class FailingTask(luigi.Task):

            def run(self):
                if False:
                    return 10
                raise ValueError()
        self.run_and_expect('FailingTask', 0)
        self.run_and_expect('FailingTask --retcode-task-failed 5', 5)
        self.run_with_config(dict(task_failed='3'), 'FailingTask', 3)

    def test_missing_data(self):
        if False:
            i = 10
            return i + 15

        class MissingDataTask(luigi.ExternalTask):

            def complete(self):
                if False:
                    print('Hello World!')
                return False
        self.run_and_expect('MissingDataTask', 0)
        self.run_and_expect('MissingDataTask --retcode-missing-data 5', 5)
        self.run_with_config(dict(missing_data='3'), 'MissingDataTask', 3)

    def test_already_running(self):
        if False:
            for i in range(10):
                print('nop')

        class AlreadyRunningTask(luigi.Task):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                pass
        old_func = luigi.scheduler.Scheduler.get_work

        def new_func(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            kwargs['current_tasks'] = None
            old_func(*args, **kwargs)
            res = old_func(*args, **kwargs)
            res['running_tasks'][0]['worker'] = 'not me :)'
            return res
        with mock.patch('luigi.scheduler.Scheduler.get_work', new_func):
            self.run_and_expect('AlreadyRunningTask', 0)
            self.run_and_expect('AlreadyRunningTask --retcode-already-running 5', 5)
            self.run_with_config(dict(already_running='3'), 'AlreadyRunningTask', 3)

    def test_when_locked(self):
        if False:
            while True:
                i = 10

        def new_func(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return False
        with mock.patch('luigi.lock.acquire_for', new_func):
            self.run_and_expect('Task', 0, extra_args=['--local-scheduler'])
            self.run_and_expect('Task --retcode-already-running 5', 5, extra_args=['--local-scheduler'])
            self.run_with_config(dict(already_running='3'), 'Task', 3, extra_args=['--local-scheduler'])

    def test_failure_in_complete(self):
        if False:
            return 10

        class FailingComplete(luigi.Task):

            def complete(self):
                if False:
                    print('Hello World!')
                raise Exception

        class RequiringTask(luigi.Task):

            def requires(self):
                if False:
                    print('Hello World!')
                yield FailingComplete()
        self.run_and_expect('RequiringTask', 0)

    def test_failure_in_requires(self):
        if False:
            return 10

        class FailingRequires(luigi.Task):

            def requires(self):
                if False:
                    print('Hello World!')
                raise Exception
        self.run_and_expect('FailingRequires', 0)

    def test_validate_dependency_error(self):
        if False:
            while True:
                i = 10

        class DependencyTask:
            pass

        class RequiringTask(luigi.Task):

            def requires(self):
                if False:
                    while True:
                        i = 10
                yield DependencyTask()
        self.run_and_expect('RequiringTask', 4)

    def test_task_limit(self):
        if False:
            print('Hello World!')

        class TaskB(luigi.Task):

            def complete(self):
                if False:
                    print('Hello World!')
                return False

        class TaskA(luigi.Task):

            def requires(sefl):
                if False:
                    i = 10
                    return i + 15
                yield TaskB()

        class TaskLimitTest(luigi.Task):

            def requires(self):
                if False:
                    return 10
                yield TaskA()
        self.run_and_expect('TaskLimitTest --worker-task-limit 2', 0)
        self.run_and_expect('TaskLimitTest --worker-task-limit 2 --retcode-scheduling-error 3', 3)

    def test_unhandled_exception(self):
        if False:
            return 10

        def new_func(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise Exception()
        with mock.patch('luigi.worker.Worker.add', new_func):
            self.run_and_expect('Task', 4)
            self.run_and_expect('Task --retcode-unhandled-exception 2', 2)

        class TaskWithRequiredParam(luigi.Task):
            param = luigi.Parameter()
        self.run_and_expect('TaskWithRequiredParam --param hello', 0)
        self.run_and_expect('TaskWithRequiredParam', 4)

    def test_when_mixed_errors(self):
        if False:
            while True:
                i = 10

        class FailingTask(luigi.Task):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                raise ValueError()

        class MissingDataTask(luigi.ExternalTask):

            def complete(self):
                if False:
                    print('Hello World!')
                return False

        class RequiringTask(luigi.Task):

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                yield FailingTask()
                yield MissingDataTask()
        self.run_and_expect('RequiringTask --retcode-task-failed 4 --retcode-missing-data 5', 5)
        self.run_and_expect('RequiringTask --retcode-task-failed 7 --retcode-missing-data 6', 7)

    def test_unknown_reason(self):
        if False:
            print('Hello World!')

        class TaskA(luigi.Task):

            def complete(self):
                if False:
                    while True:
                        i = 10
                return True

        class RequiringTask(luigi.Task):

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                yield TaskA()

        def new_func(*args, **kwargs):
            if False:
                return 10
            return None
        with mock.patch('luigi.scheduler.Scheduler.add_task', new_func):
            self.run_and_expect('RequiringTask', 0)
            self.run_and_expect('RequiringTask --retcode-not-run 5', 5)
    '\n    Test that a task once crashing and then succeeding should be counted as no failure.\n    '

    def test_retry_sucess_task(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            run_count = 0

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.run_count += 1
                if self.run_count == 1:
                    raise ValueError()

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.run_count > 0
        self.run_and_expect('Foo --scheduler-retry-delay=0', 0)
        self.run_and_expect('Foo --scheduler-retry-delay=0 --retcode-task-failed=5', 0)
        self.run_with_config(dict(task_failed='3'), 'Foo', 0)