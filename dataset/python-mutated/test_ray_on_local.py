from unittest import TestCase
import pytest
import ray
from bigdl.dllib.nncontext import init_spark_on_local
from bigdl.orca.ray import OrcaRayContext

class TestRayLocal(TestCase):

    def test_local(self):
        if False:
            while True:
                i = 10

        @ray.remote
        class TestRay:

            def hostname(self):
                if False:
                    print('Hello World!')
                import socket
                return socket.gethostname()
        sc = init_spark_on_local(cores=8)
        config = {'object_spilling_config': '{"type":"filesystem","params":{"directory_path":"/tmp/spill"}}'}
        ray_ctx = OrcaRayContext(sc=sc, object_store_memory='1g', ray_node_cpu_cores=4, system_config=config)
        address_info = ray_ctx.init()
        assert 'object_store_address' in address_info
        actors = [TestRay.remote() for i in range(0, 4)]
        print(ray.get([actor.hostname.remote() for actor in actors]))
        ray_ctx.stop()
        sc.stop()
if __name__ == '__main__':
    pytest.main([__file__])