from helpers import LuigiTestCase
import luigi
import luigi.scheduler
import luigi.worker
luigi.notifications.DEBUG = True

class TaskStatusMessageTest(LuigiTestCase):

    def test_run(self):
        if False:
            return 10
        message = 'test message'
        sch = luigi.scheduler.Scheduler()
        with luigi.worker.Worker(scheduler=sch) as w:

            class MyTask(luigi.Task):

                def run(self):
                    if False:
                        print('Hello World!')
                    self.set_status_message(message)
            task = MyTask()
            w.add(task)
            w.run()
            self.assertEqual(sch.get_task_status_message(task.task_id)['statusMessage'], message)