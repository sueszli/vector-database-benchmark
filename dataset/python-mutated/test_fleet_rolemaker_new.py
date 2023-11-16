"""Test cloud role maker."""
import os
import unittest
from paddle.distributed.fleet.base import role_maker

class TestRoleMakerBase(unittest.TestCase):
    """
    Test cases for RoleMakerBase
    """

    def test_rolemaker_base(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.RoleMakerBase()
        self.assertRaises(Exception, role._is_worker)
        self.assertRaises(Exception, role._is_server)
        self.assertRaises(Exception, role._is_first_worker)
        self.assertRaises(Exception, role._worker_num)
        self.assertRaises(Exception, role._server_num)
        self.assertRaises(Exception, role._worker_index)
        self.assertRaises(Exception, role._server_index)
        self.assertRaises(Exception, role._role_id)
        self.assertRaises(Exception, role._node_num)
        trainer_endpoints = role._get_trainer_endpoints()
        self.assertTrue(len(trainer_endpoints) == 0)
        pserver_endpoints = role._get_pserver_endpoints()
        self.assertTrue(len(pserver_endpoints) == 0)
        print(role.to_string())
        self.assertIsNone(role._all_gather(1, 'worker'))
        self.assertIsNone(role._all_reduce(1, 'sum', 'worker'))
        role._barrier('worker')

class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up, set envs.'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36001,127.0.0.2:36001'
        os.environ['POD_IP'] = '127.0.0.1'

    def test_tr_rolemaker(self):
        if False:
            return 10
        'Test tr rolenamer.'
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertTrue(ro._is_worker())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_server())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._worker_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertTrue(ro._is_first_worker())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        worker_endpoints = ro._get_trainer_endpoints()
        self.assertEqual(worker_endpoints[0], '127.0.0.1:36001')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._role_id(), 0)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._node_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_non_distributed())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._heter_worker_num(), 0)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_heter_worker())

    def test_tr_rolemaker_collective(self):
        if False:
            for i in range(10):
                print('nop')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=True)
        self.assertEqual(ro._worker_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=True)
        self.assertEqual(ro._node_num(), 2)

    def test_ps_rolemaker(self):
        if False:
            print('Hello World!')
        'Test ps rolemaker.'
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False, init_gloo=False)
        self.assertEqual(ro._server_index(), 0)
        self.assertFalse(ro._is_worker())
        self.assertTrue(ro._is_server())
        self.assertEqual(ro._server_num(), 2)
        pserver_endpoints = ro._get_pserver_endpoints()
        self.assertEqual(pserver_endpoints[0], '127.0.0.1:36001')
        self.assertEqual(ro._all_gather(1, 'worker'), 1)
        self.assertEqual(ro._all_reduce(1, 'sum', 'worker'), 1)

    def test_training_role(self):
        if False:
            while True:
                i = 10
        'Test training role.'
        os.environ['TRAINING_ROLE'] = 'TEST'
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestUserDefinedRoleMaker(unittest.TestCase):
    """
    Test cases for UserDefinedRoleMaker.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_ps_rolemaker(self):
        if False:
            i = 10
            return i + 15
        ro = role_maker.UserDefinedRoleMaker(is_collective=False, init_gloo=False, server_endpoints=['127.0.0.1:36001', '127.0.0.1:36001'], role=role_maker.Role.SERVER, current_id=0, worker_num=2)
        self.assertEqual(ro._server_num(), 2)
        ro._generate_role()
        self.assertTrue(ro._is_server())
        self.assertEqual(ro._role_id(), 0)

    def test_tr_rolemaker(self):
        if False:
            for i in range(10):
                print('nop')
        ro = role_maker.UserDefinedRoleMaker(is_collective=False, init_gloo=False, server_endpoints=['127.0.0.1:36001', '127.0.0.1:36001'], role=role_maker.Role.WORKER, current_id=0, worker_num=2)
        self.assertIn('127.0.0.1:36001', ro._get_pserver_endpoints())
        self.assertTrue(ro._is_worker())
        self.assertEqual(ro._role_id(), 0)
'\nclass TestGlooWithCloudRoleMaker(unittest.TestCase):\n    def setUp(self):\n        os.environ["PADDLE_TRAINERS_NUM"] = "1"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_TRAINER_ID"] = "0"\n\n    def case(self, role, comm_world):\n        role._barrier(comm_world)\n\n        gather = role._all_gather(1, comm_world)\n        self.assertEqual(gather[0], 1)\n\n        all_reduce = role._all_reduce(1, "sum", comm_world)\n        self.assertEqual(1, all_reduce)\n\n    def mkdir(self):\n        tmp = tempfile.mkdtemp()\n        return tmp\n\n    def clean(self, tmp):\n        shutil.rmtree(tmp)\n\n    def test_hdfs_gloo(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n        os.environ["TRAINING_ROLE"] = "TRAINER"\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"\n        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "worker")\n        self.clean(tmp)\n\n    def test_fs_gloo(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n        os.environ["TRAINING_ROLE"] = "TRAINER"\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "worker")\n        self.clean(tmp)\n\n    def test_fs_gloo2(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "server")\n        self.clean(tmp)\n\n    def test_fs_gloo3(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"\n        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "server")\n        self.clean(tmp)\n\n    def test_fs_gloo4(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        os.environ["TRAINING_ROLE"] = "WORKER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "3"\n        os.environ["PADDLE_GLOO_HTTP_ENDPOINT"] = "127.0.0.1:30019"\n\n        role = role_maker.PaddleCloudRoleMaker(is_collective=True)\n        role._generate_role()\n        import time\n\n        time.sleep(3)\n\n    def test_fs_gloo5(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n        os.environ["PADDLE_TRAINERS_NUM"] = "0"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "2"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "server")\n        self.case(role, "all")\n        self.clean(tmp)\n\n    def test_fs_gloo6(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n        os.environ["PADDLE_TRAINERS_NUM"] = "0"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n\n        os.environ["PADDLE_WITH_GLOO"] = "2"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"\n        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        role = role_maker.PaddleCloudRoleMaker()\n        role._generate_role()\n        self.case(role, "server")\n        self.case(role, "all")\n        self.clean(tmp)\n\n    def test_fs_gloo7(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n        os.environ["PADDLE_TRAINERS_NUM"] = "0"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "5"\n\n        role = role_maker.PaddleCloudRoleMaker()\n        self.assertRaises(ValueError, role._generate_role)\n\n    def test_hdfs_gloo_v2(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        os.environ["TRAINING_ROLE"] = "TRAINER"\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"\n        os.environ["PADDLE_GLOO_FS_NAME"] = ""\n        os.environ["PADDLE_GLOO_FS_UGI"] = ""\n        os.environ["PADDLE_GLOO_FS_PATH"] = ""\n\n        role = role_maker.PaddleCloudRoleMaker()\n        self.assertRaises(ValueError, role._generate_role)\n\n    def test_fs_gloo_v2(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n        os.environ["PADDLE_TRAINERS_NUM"] = "0"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"\n        os.environ["PADDLE_GLOO_FS_PATH"] = ""\n\n        role = role_maker.PaddleCloudRoleMaker()\n        self.assertRaises(ValueError, role._generate_role)\n\n    def test_http_gloo_v2(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n        os.environ["PADDLE_WITH_GLOO"] = "1"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "3"\n        os.environ["PADDLE_GLOO_HTTP_ENDPOINT"] = ""\n\n        role = role_maker.PaddleCloudRoleMaker()\n\n    def test_fs_gloo8(self):\n        plats = platform.platform()\n        if \'Linux\' not in plats:\n            print("skip gloo UT on MacOS/Win")\n            return\n\n        tmp = self.mkdir()\n\n        os.environ["TRAINING_ROLE"] = "PSERVER"\n        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"\n        os.environ["POD_IP"] = "127.0.0.1"\n        os.environ["PADDLE_PORT"] = "36001"\n        os.environ["PADDLE_TRAINERS_NUM"] = "0"\n\n        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"\n\n        os.environ["PADDLE_WITH_GLOO"] = "2"\n        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"\n        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"\n        os.environ["PADDLE_GLOO_FS_PATH"] = tmp\n\n        def net():\n            x = paddle.static.data(name=\'x\', shape=[-1, 13], dtype=\'float32\')\n            y_predict = paddle.static.nn.fc(x, size=1, activation=None)\n            y = paddle.static.data(name=\'y\', shape=[-1, 1], dtype=\'float32\')\n            cost = paddle.nn.functional.square_error_cost(\n                input=y_predict, label=y\n            )\n            avg_cost = paddle.mean(cost)\n            return avg_cost\n\n        from paddle.distributed import fleet\n\n        role = role_maker.PaddleCloudRoleMaker()\n        fleet.init(role)\n        avg_cost = net()\n\n        strategy = paddle.distributed.fleet.DistributedStrategy()\n        strategy.a_sync = False\n\n        optimizer = paddle.optimizer.SGD(0.01)\n        optimizer = fleet.distributed_optimizer(optimizer, strategy)\n        optimizer.minimize(avg_cost)\n\n        comm_world = "server"\n        fleet.util.barrier(comm_world)\n\n        gather = fleet.util.all_gather(1, comm_world)\n        self.assertEqual(gather[0], 1)\n\n        all_reduce = fleet.util.all_reduce(1, "sum", comm_world)\n        self.assertEqual(1, all_reduce)\n\n        self.clean(tmp)\n'
if __name__ == '__main__':
    unittest.main()