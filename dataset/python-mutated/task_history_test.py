from helpers import LuigiTestCase
import luigi
import luigi.scheduler
import luigi.task_history
import luigi.worker
luigi.notifications.DEBUG = True

class SimpleTaskHistory(luigi.task_history.TaskHistory):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.actions = []

    def task_scheduled(self, task):
        if False:
            while True:
                i = 10
        self.actions.append(('scheduled', task.id))

    def task_finished(self, task, successful):
        if False:
            return 10
        self.actions.append(('finished', task.id))

    def task_started(self, task, worker_host):
        if False:
            i = 10
            return i + 15
        self.actions.append(('started', task.id))

class TaskHistoryTest(LuigiTestCase):

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        th = SimpleTaskHistory()
        sch = luigi.scheduler.Scheduler(task_history_impl=th)
        with luigi.worker.Worker(scheduler=sch) as w:

            class MyTask(luigi.Task):
                pass
            task = MyTask()
            w.add(task)
            w.run()
            self.assertEqual(th.actions, [('scheduled', task.task_id), ('started', task.task_id), ('finished', task.task_id)])