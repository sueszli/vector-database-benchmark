import unittest
import paddle
from paddle.base import core
from paddle.distributed.fleet.fleet_executor_utils import TaskNode
paddle.enable_static()

class TestFleetExecutorTaskNode(unittest.TestCase):

    def test_task_node(self):
        if False:
            while True:
                i = 10
        program = paddle.static.Program()
        task_node_0 = core.TaskNode(program.desc, 0, 0, 1)
        task_node_1 = core.TaskNode(program.desc, 0, 1, 1)
        task_node_2 = core.TaskNode(program.desc, 0, 2, 1)
        self.assertEqual(task_node_0.task_id(), 0)
        self.assertEqual(task_node_1.task_id(), 1)
        self.assertEqual(task_node_2.task_id(), 2)
        self.assertTrue(task_node_0.add_downstream_task(task_node_1.task_id(), 1, core.DependType.NORMAL))
        self.assertTrue(task_node_1.add_upstream_task(task_node_0.task_id(), 1, core.DependType.NORMAL))

    def test_lazy_task_node(self):
        if False:
            print('Hello World!')
        program = paddle.static.Program()
        task = TaskNode(program=program, rank=0, max_run_times=1, lazy_initialize=True)
        task_node = task.task_node()
if __name__ == '__main__':
    unittest.main()