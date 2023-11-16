from helpers import LuigiTestCase
import luigi
from luigi.task_register import Register, TaskClassNotFoundException, TaskClassAmbigiousException

class TaskRegisterTest(LuigiTestCase):

    def test_externalize_taskclass(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TaskClassNotFoundException):
            Register.get_task_cls('scooby.Doo')

        class Task1(luigi.Task):

            @classmethod
            def get_task_family(cls):
                if False:
                    return 10
                return 'scooby.Doo'
        self.assertEqual(Task1, Register.get_task_cls('scooby.Doo'))

        class Task2(luigi.Task):

            @classmethod
            def get_task_family(cls):
                if False:
                    i = 10
                    return i + 15
                return 'scooby.Doo'
        with self.assertRaises(TaskClassAmbigiousException):
            Register.get_task_cls('scooby.Doo')

        class Task3(luigi.Task):

            @classmethod
            def get_task_family(cls):
                if False:
                    while True:
                        i = 10
                return 'scooby.Doo'
        with self.assertRaises(TaskClassAmbigiousException):
            Register.get_task_cls('scooby.Doo')