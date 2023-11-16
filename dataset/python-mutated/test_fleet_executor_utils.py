import unittest
import paddle
from paddle.distributed.fleet.fleet_executor_utils import FleetExecutorUtils
paddle.enable_static()

class TestFleetExecutorUtils(unittest.TestCase):

    def test_construct_program(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding_configs = {'dp_degree': 2, 'mp_degree': 2, 'pp_degree': 2}
        fleet_executor_utils = FleetExecutorUtils(dist_strategy=strategy.sharding_configs, rank=0, nrank=1, max_run_times=1)
        op_list = {'lr': [], 'fwd': [], 'bwd': [], 'opt': []}
        program_map = fleet_executor_utils.convert_op_list_to_program(op_list, paddle.static.Program())
        task_node_map = fleet_executor_utils.construct_task_nodes_1f1b(program_map)
if __name__ == '__main__':
    unittest.main()