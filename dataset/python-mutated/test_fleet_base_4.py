import os
import unittest
import paddle
from paddle.distributed import fleet
paddle.enable_static()

class TestFleetBase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36001'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'

    def test_fleet_init(self):
        if False:
            print('Hello World!')
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        role = fleet.PaddleCloudRoleMaker(is_collective=False)
        fleet.init(role)
        fleet.init()
        fleet.init(is_collective=False)
        self.assertRaises(Exception, fleet.init, is_collective='F')
        self.assertRaises(Exception, fleet.init, role_maker='F')
if __name__ == '__main__':
    unittest.main()