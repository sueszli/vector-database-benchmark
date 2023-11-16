import ray
from bigdl.dllib.nncontext import init_spark_on_yarn
from bigdl.orca.ray import OrcaRayContext
slave_num = 2
sc = init_spark_on_yarn(hadoop_conf='/opt/work/almaren-yarn-config/', conda_name='ray_train', num_executors=slave_num, executor_cores=28, executor_memory='10g', driver_memory='2g', driver_cores=4, extra_executor_memory_for_ray='30g', conf={'hello': 'world'})
ray_ctx = OrcaRayContext(sc=sc, object_store_memory='25g', extra_params={'temp-dir': '/tmp/hello/'}, env={'http_proxy': 'http://child-prc.intel.com:913', 'http_proxys': 'http://child-prc.intel.com:913'})
ray_ctx.init()

@ray.remote
class TestRay:

    def hostname(self):
        if False:
            while True:
                i = 10
        import socket
        return socket.gethostname()

    def check_cv2(self):
        if False:
            for i in range(10):
                print('nop')
        import cv2
        return cv2.__version__

    def ip(self):
        if False:
            for i in range(10):
                print('nop')
        return ray._private.services.get_node_ip_address()

    def network(self):
        if False:
            print('Hello World!')
        from urllib.request import urlopen
        try:
            urlopen('http://www.baidu.com', timeout=3)
            return True
        except Exception as err:
            return False
actors = [TestRay.remote() for i in range(0, slave_num)]
print(ray.get([actor.hostname.remote() for actor in actors]))
print(ray.get([actor.ip.remote() for actor in actors]))
ray_ctx.stop()