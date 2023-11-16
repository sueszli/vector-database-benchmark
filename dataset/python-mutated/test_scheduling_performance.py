import logging
import sys
import time
import pytest
import ray
import ray._private.test_utils
import ray.cluster_utils
logger = logging.getLogger(__name__)

@pytest.mark.parametrize('args', [[16, 16, 4, False], [16, 16, 4, True]])
def test_actor_scheduling_latency(ray_start_cluster, args):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    node_count = args[0]
    actor_count = args[1]
    upper_count = args[2]
    gcs_sched = args[3]
    for i in range(node_count):
        cluster.add_node(memory=1024 ** 2, _system_config={'gcs_actor_scheduling_enabled': gcs_sched} if i == 0 else {})
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(memory=1024 ** 2)
    class UpperActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.start = time.time()

        def info(self):
            if False:
                i = 10
                return i + 15
            return [ray._private.worker.global_worker.node.unique_id, self.start]

        def create(self, num):
            if False:
                print('Hello World!')
            ret_list = []
            for _ in range(num):
                start_time = time.time()
                ret_list.append([start_time, BottomActor.remote()])
            return ret_list

    @ray.remote(memory=1024 ** 2)
    class BottomActor:

        def __init__(self):
            if False:
                print('Hello World!')
            self.start = time.time()

        def info(self):
            if False:
                for i in range(10):
                    print('nop')
            return [ray._private.worker.global_worker.node.unique_id, self.start]
    actor_distribution = {}
    actor_list = []
    start_list = []
    end_list = []
    ref_list = []
    for _ in range(upper_count):
        start_list.append(time.time())
        actor_list.append(UpperActor.remote())
    for actor in actor_list:
        ref_list.append(actor.create.remote(int((actor_count - upper_count) / upper_count)))
    for ref in ref_list:
        ret_list = ray.get(ref)
        for ret in ret_list:
            start_list.append(ret[0])
            actor_list.append(ret[1])
    for actor in actor_list:
        ret_list = ray.get(actor.info.remote())
        if ret_list[0] not in actor_distribution.keys():
            actor_distribution[ret_list[0]] = []
        actor_distribution[ret_list[0]].append(actor)
        end_list.append(ret_list[1])
    assert len(actor_distribution) == node_count
    for (node_id, actors) in actor_distribution.items():
        assert len(actors) <= int(actor_count / node_count)
    latency_list = []
    for i in range(actor_count):
        latency_list.append(end_list[i] - start_list[i])
        print(latency_list[i])
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))