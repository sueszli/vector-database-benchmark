from unittest import TestCase
import pytest
import ray
from bigdl.orca import stop_orca_context
from bigdl.orca.ray import OrcaRayContext

@ray.remote
class TestRay:

    def hostname(self):
        if False:
            return 10
        import socket
        return socket.gethostname()

class TestUtil(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.ray_ctx = OrcaRayContext('ray', cores=2, num_nodes=1)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        stop_orca_context()

    def test_init(self):
        if False:
            print('Hello World!')
        node_num = 4
        address_info = self.ray_ctx.init()
        assert OrcaRayContext._active_ray_context, 'OrcaRayContext has not been initialized'
        assert 'object_store_address' in address_info
        actors = [TestRay.remote() for i in range(0, node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))

    def test_stop(self):
        if False:
            return 10
        self.ray_ctx.stop()
        assert not self.ray_ctx.initialized, 'The Ray cluster has been stopped.'

    def test_get(self):
        if False:
            while True:
                i = 10
        self.ray_ctx.get()
        assert self.ray_ctx.initialized, 'The Ray cluster has been launched.'
if __name__ == '__main__':
    pytest.main([__file__])