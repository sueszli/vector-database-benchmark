from helpers import LuigiTestCase
import luigi
import luigi.scheduler
import luigi.worker

class TaskProgressPercentageTest(LuigiTestCase):

    def test_run(self):
        if False:
            i = 10
            return i + 15
        sch = luigi.scheduler.Scheduler()
        with luigi.worker.Worker(scheduler=sch) as w:

            class MyTask(luigi.Task):

                def run(self):
                    if False:
                        while True:
                            i = 10
                    self.set_progress_percentage(30)
            task = MyTask()
            w.add(task)
            w.run()
            self.assertEqual(sch.get_task_progress_percentage(task.task_id)['progressPercentage'], 30)