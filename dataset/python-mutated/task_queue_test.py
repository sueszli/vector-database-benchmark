import operator
import os
import unittest
from google.appengine.api import taskqueue
from google.appengine.ext import deferred
from google.appengine.ext import testbed

class TaskQueueTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.testbed = testbed.Testbed()
        self.testbed.activate()
        self.testbed.init_taskqueue_stub(root_path=os.path.join(os.path.dirname(__file__), 'resources'))
        self.taskqueue_stub = self.testbed.get_stub(testbed.TASKQUEUE_SERVICE_NAME)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.testbed.deactivate()

    def testTaskAddedToQueue(self):
        if False:
            i = 10
            return i + 15
        taskqueue.Task(name='my_task', url='/url/of/my/task/').add()
        tasks = self.taskqueue_stub.get_filtered_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].name, 'my_task')

    def testFiltering(self):
        if False:
            for i in range(10):
                print('nop')
        taskqueue.Task(name='task_one', url='/url/of/task/1/').add('queue-1')
        taskqueue.Task(name='task_two', url='/url/of/task/2/').add('queue-2')
        tasks = self.taskqueue_stub.get_filtered_tasks()
        self.assertEqual(len(tasks), 2)
        tasks = self.taskqueue_stub.get_filtered_tasks(name='task_one')
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].name, 'task_one')
        tasks = self.taskqueue_stub.get_filtered_tasks(url='/url/of/task/1/')
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].name, 'task_one')
        tasks = self.taskqueue_stub.get_filtered_tasks(queue_names='queue-1')
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].name, 'task_one')
        tasks = self.taskqueue_stub.get_filtered_tasks(queue_names=['queue-1', 'queue-2'])
        self.assertEqual(len(tasks), 2)

    def testTaskAddedByDeferred(self):
        if False:
            i = 10
            return i + 15
        deferred.defer(operator.add, 1, 2)
        tasks = self.taskqueue_stub.get_filtered_tasks()
        self.assertEqual(len(tasks), 1)
        result = deferred.run(tasks[0].payload)
        self.assertEqual(result, 3)
if __name__ == '__main__':
    unittest.main()