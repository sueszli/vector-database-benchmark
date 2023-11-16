import time
from unittest import TestCase
import numpy as np
import pytest
import ray
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.common import stop_orca_context
np.random.seed(1337)

@ray.remote
class TestRay:

    def hostname(self):
        if False:
            while True:
                i = 10
        import socket
        return socket.gethostname()

class TestUtil(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.ray_ctx = OrcaRayContext('ray', cores=2, num_nodes=1)
        self.node_num = 4

    def tearDown(self):
        if False:
            print('Hello World!')
        stop_orca_context()

    def test_init_and_stop(self):
        if False:
            i = 10
            return i + 15
        self.ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, self.node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        self.ray_ctx.stop()
        assert not self.ray_ctx.initialized, 'The Ray cluster has been stopped.'
        time.sleep(3)

    def test_reinit(self):
        if False:
            print('Hello World!')
        print('-------------------first repeat begin!------------------')
        self.ray_ctx = OrcaRayContext('ray', cores=2, num_nodes=1)
        assert OrcaRayContext._active_ray_context, 'Please create an OrcaRayContext First.'
        assert not self.ray_ctx.initialized, 'The Ray cluster has not been launched.'
        self.ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, self.node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        self.ray_ctx.stop()
        time.sleep(3)
if __name__ == '__main__':
    pytest.main([__file__])