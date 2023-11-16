import time
from unittest import TestCase
import numpy as np
import psutil
import pytest
import ray
from bigdl.dllib.nncontext import init_spark_on_local
from bigdl.orca.ray import OrcaRayContext
np.random.seed(1337)

@ray.remote
class TestRay:

    def hostname(self):
        if False:
            i = 10
            return i + 15
        import socket
        return socket.gethostname()

class TestUtil(TestCase):

    def test_local(self):
        if False:
            i = 10
            return i + 15
        node_num = 4
        sc = init_spark_on_local(cores=node_num)
        ray_ctx = OrcaRayContext(sc=sc, object_store_memory='1g')
        ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        ray_ctx.stop()
        time.sleep(3)
        print('-------------------first repeat begin!------------------')
        ray_ctx = OrcaRayContext(sc=sc, object_store_memory='1g')
        ray_ctx.init()
        actors = [TestRay.remote() for i in range(0, node_num)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        ray_ctx.stop()
        sc.stop()
        time.sleep(3)
if __name__ == '__main__':
    pytest.main([__file__])