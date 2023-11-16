import os
import unittest
from paddle.distributed.fleet.elastic.manager import ELASTIC_AUTO_PARALLEL_EXIT_CODE, ElasticManager, LauncherInterface

class MockLease:

    def refresh(self):
        if False:
            print('Hello World!')
        pass

class MockKVMetadata:

    def __init__(self, key):
        if False:
            i = 10
            return i + 15
        self.key = key
        self.create_revision = 2
        self.mod_revision = 3
        self.version = 2
        self.lease_id = 0
        self.response_header = None

class MockEtcdClient:

    def __init__(self, lease=None):
        if False:
            print('Hello World!')
        self._lease = lease

    def put(self, key, value, lease=None):
        if False:
            while True:
                i = 10
        pass

    def get(self, key):
        if False:
            return 10
        return (b'0', MockKVMetadata(b'/prefix'))

    def delete_prefix(self, key):
        if False:
            return 10
        pass

    def get_prefix(self, key_prefix):
        if False:
            print('Hello World!')
        hosts = [(b'/prefix/host1', b'10.10.10.1:6001'), (b'/prefix/host2', b'10.10.10.2:6001')]
        return ((v, MockKVMetadata(k)) for (k, v) in hosts)

    def add_watch_callback(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return 0

    def add_watch_prefix_callback(self, key_prefix, callback, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        callback(None)
        return 0

    def cancel_watch(self, watch_id):
        if False:
            i = 10
            return i + 15
        pass

    def delete(self, key):
        if False:
            for i in range(10):
                print('nop')
        return True

    def lease(self, ttl):
        if False:
            return 10
        if self._lease:
            return self._lease
        else:
            return MockLease()

class TestElasticManager(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.etcd_client = MockEtcdClient()

    def test_elastic_manager_init(self):
        if False:
            i = 10
            return i + 15

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        args = Argument()

        class _MockLease:

            def refresh(self):
                if False:
                    return 10
                raise ValueError('valid error, this only for unittest')
        etcd_client = MockEtcdClient(lease=_MockLease())
        elastic = ElasticManager(args, etcd_client=etcd_client)

    def test_match_faulttolerance(self):
        if False:
            for i in range(10):
                print('nop')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        args = Argument()
        args.ips = '10.10.10.1,10.10.10.2'
        elastic = ElasticManager(args, self.etcd_client)
        os.environ['FLAGS_START_PORT'] = '6001'
        hosts = ['10.10.10.1:6001', '10.10.10.2:6001']
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        self.assertEqual(elastic._match(hosts), True)
        hosts = ['10.10.10.1:6001']
        args.ips = '10.10.10.1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001'
        self.assertEqual(elastic._match(hosts), False)

    def test_match_elastic(self):
        if False:
            return 10

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2:4'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        os.environ['PADDLE_ELASTIC_TIMEOUT'] = '60'
        args = Argument()
        args.ips = '10.10.10.1,10.10.10.2,10.10.10.3,10.10.10.4'
        os.environ['FLAGS_START_PORT'] = '6001'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001'
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ['10.10.10.1:6001', '10.10.10.2:6001']
        self.assertEqual(elastic._match(hosts), False)
        hosts = ['10.10.10.1:6001', '10.10.10.2:6001', '10.10.10.3:6001', '10.10.10.4:6001']
        self.assertEqual(elastic._match(hosts), True)
        hosts = ['10.10.10.1:6001', '10.10.10.2:6001', '10.10.10.3:6001']
        self.assertEqual(elastic._match(hosts), False)
        hosts = ['10.10.10.1:6001']
        self.assertEqual(elastic._match(hosts), False)
        args.ips = '10.10.10.1,10.10.10.2'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ['10.10.10.1:6001', '10.10.10.2:6001']
        self.assertEqual(elastic._match(hosts), True)

    def test_update_hosts_for_faulttolerance(self):
        if False:
            return 10

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '0'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        args = Argument()
        os.environ['FLAGS_START_PORT'] = '6001'
        os.environ['PADDLE_ELASTIC_NP'] = '2'
        os.environ['PADDLE_TRAINERS'] = '10.10.10.1,10.10.10.2'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        elastic = ElasticManager(args, self.etcd_client)
        os.environ['PADDLE_TRAINER_ID'] = '0'
        elastic.curr_host = '10.10.10.1:6001'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.2:6001']
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.1,10.10.10.2')
        elastic.curr_host = '10.10.10.3:6001'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.3:6001']
        os.environ['PADDLE_TRAINER_ID'] = '1'
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.1,10.10.10.3')
        elastic.curr_host = '10.10.10.3:6001'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.3:6001']
        os.environ['PADDLE_TRAINER_ID'] = '-1'
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.1,10.10.10.3')

    def test_update_hosts_for_elastic(self):
        if False:
            for i in range(10):
                print('nop')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2:4'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        args = Argument()
        os.environ['FLAGS_START_PORT'] = '6001'
        os.environ['PADDLE_TRAINERS'] = '10.10.10.1,10.10.10.2'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.2:6001'
        elastic = ElasticManager(args, self.etcd_client)
        elastic.curr_host = '10.10.10.1:6001'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.2:6001', '10.10.10.3:6001']
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.1,10.10.10.2,10.10.10.3')
        os.environ['PADDLE_TRAINERS'] = '10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001'
        os.environ['POD_IP'] = '10.10.10.1'
        os.environ['TRAINER_PORTS_NUM'] = '4'
        os.environ['PADDLE_TRAINER_ID'] = '1'
        os.environ['PADDLE_PORT'] = '6001'
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.curr_host = '10.10.10.1:6001'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.2:6001', '10.10.10.3:6001']
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.3,10.10.10.1,10.10.10.2')
        self.assertEqual(os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'), '10.10.10.3:6001,10.10.10.1:6001,10.10.10.2:6001')
        os.environ['PADDLE_TRAINERS'] = '10.10.10.1,10.10.10.1,10.10.10.1,10.10.10.1'
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004'
        os.environ['POD_IP'] = '10.10.10.1'
        os.environ['TRAINER_PORTS_NUM'] = '4'
        os.environ['PADDLE_PORT'] = '6001'
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.curr_host = '10.10.10.1:6001'
        os.environ['PADDLE_TRAINER_ID'] = '-1'
        elastic.hosts = ['10.10.10.1:6001', '10.10.10.1:6003']
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), '10.10.10.1,10.10.10.1')
        self.assertEqual(os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'), '10.10.10.1:6001,10.10.10.1:6003')

    def test_exit(self):
        if False:
            for i in range(10):
                print('nop')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.exit()

    def test_pre_hook(self):
        if False:
            print('Hello World!')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
            elastic_pre_hook = None
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.pre_hook()
        args.elastic_pre_hook = 'hostname'
        elastic.pre_hook()

    def test_watch(self):
        if False:
            print('Hello World!')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
            elastic_pre_hook = None

        class ElasticLauncher:

            def watch(self):
                if False:
                    return 10
                return ELASTIC_AUTO_PARALLEL_EXIT_CODE

            def stop(self):
                if False:
                    while True:
                        i = 10
                pass
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.stopped = False
        elastic.launcher = ElasticLauncher()
        elastic.watch()

    def test_launcher_interface_check_procs(self):
        if False:
            i = 10
            return i + 15

        class Proc:

            def poll(self):
                if False:
                    while True:
                        i = 10
                return ELASTIC_AUTO_PARALLEL_EXIT_CODE

        class ProcList:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.proc = Proc()
        launch = LauncherInterface(None)
        launch.procs = [ProcList()]
        launch._check_procs()
if __name__ == '__main__':
    unittest.main()