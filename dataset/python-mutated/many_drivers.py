import time
import argparse
import ray
from ray.cluster_utils import Cluster
from ray._private.test_utils import run_string_as_driver, safe_write_to_results_json

def update_progress(result):
    if False:
        print('Hello World!')
    result['last_update'] = time.time()
    safe_write_to_results_json(result)
num_redis_shards = 5
redis_max_memory = 10 ** 8
object_store_memory = 10 ** 8
num_nodes = 4
message = 'Make sure there is enough memory on this machine to run this workload. We divide the system memory by 2 to provide a buffer.'
assert num_nodes * object_store_memory + num_redis_shards * redis_max_memory < ray._private.utils.get_system_memory() / 2, message
cluster = Cluster()
for i in range(num_nodes):
    cluster.add_node(redis_port=6379 if i == 0 else None, num_redis_shards=num_redis_shards if i == 0 else None, num_cpus=4, num_gpus=0, resources={str(i): 5}, object_store_memory=object_store_memory, redis_max_memory=redis_max_memory, dashboard_host='0.0.0.0')
ray.init(address=cluster.address)
driver_script = '\nimport ray\n\nray.init(address="{}")\n\nnum_nodes = {}\n\n\n@ray.remote\ndef f():\n    return 1\n\n\n@ray.remote\nclass Actor(object):\n    def method(self):\n        return 1\n\n\nfor _ in range(5):\n    for i in range(num_nodes):\n        assert (ray.get(\n            f._remote(args=[],\n            kwargs={{}},\n            resources={{str(i): 1}})) == 1)\n        actor = Actor._remote(\n            args=[], kwargs={{}}, resources={{str(i): 1}})\n        assert ray.get(actor.method.remote()) == 1\n\n# Tests datasets doesn\'t leak workers.\nray.data.range(100).map(lambda x: x).take()\n\nprint("success")\n'.format(cluster.address, num_nodes)

@ray.remote
def run_driver():
    if False:
        i = 10
        return i + 15
    output = run_string_as_driver(driver_script, encode='utf-8')
    assert 'success' in output
iteration = 0
running_ids = [run_driver._remote(args=[], kwargs={}, num_cpus=0, resources={str(i): 0.01}) for i in range(num_nodes)]
start_time = time.time()
previous_time = start_time
parser = argparse.ArgumentParser(prog='Many Drivers long running tests')
parser.add_argument('--iteration-num', type=int, help='How many iterations to run', required=False)
parser.add_argument('--smoke-test', action='store_true', help='Whether or not the test is smoke test.', default=False)
args = parser.parse_args()
iteration_num = args.iteration_num
if args.smoke_test:
    iteration_num = 400
while True:
    if iteration_num is not None and iteration_num < iteration:
        break
    ([ready_id], running_ids) = ray.wait(running_ids, num_returns=1)
    ray.get(ready_id)
    running_ids.append(run_driver._remote(args=[], kwargs={}, num_cpus=0, resources={str(iteration % num_nodes): 0.01}))
    new_time = time.time()
    print('Iteration {}:\n  - Iteration time: {}.\n  - Absolute time: {}.\n  - Total elapsed time: {}.'.format(iteration, new_time - previous_time, new_time, new_time - start_time))
    update_progress({'iteration': iteration, 'iteration_time': new_time - previous_time, 'absolute_time': new_time, 'elapsed_time': new_time - start_time})
    previous_time = new_time
    iteration += 1