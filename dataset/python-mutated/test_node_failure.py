import time
import unittest
import ray
from ray._private.test_utils import get_other_nodes
from ray.cluster_utils import Cluster
from ray.util.state import list_actors
from ray.rllib.algorithms.ppo import PPO, PPOConfig
num_redis_shards = 5
redis_max_memory = 10 ** 8
object_store_memory = 10 ** 8
num_nodes = 3
assert num_nodes * object_store_memory + num_redis_shards * redis_max_memory < ray._private.utils.get_system_memory() / 2, 'Make sure there is enough memory on this machine to run this workload. We divide the system memory by 2 to provide a buffer.'

class NodeFailureTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.cluster = Cluster()
        for i in range(num_nodes):
            self.cluster.add_node(redis_port=6379 if i == 0 else None, num_redis_shards=num_redis_shards if i == 0 else None, num_cpus=2, num_gpus=0, object_store_memory=object_store_memory, redis_max_memory=redis_max_memory, dashboard_host='0.0.0.0')
        self.cluster.wait_for_nodes()
        ray.init(address=self.cluster.address)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ray.shutdown()
        self.cluster.shutdown()

    def test_continue_training_on_failure(self):
        if False:
            i = 10
            return i + 15
        config = PPOConfig().environment('CartPole-v1').rollouts(num_rollout_workers=6, recreate_failed_workers=True, validate_workers_after_construction=True).training(train_batch_size=300)
        ppo = PPO(config=config)
        ppo.train()
        self.assertEqual(ppo.workers.num_healthy_remote_workers(), 6)
        self.assertEqual(ppo.workers.num_remote_workers(), 6)
        node_to_kill = get_other_nodes(self.cluster, exclude_head=True)[0]
        self.cluster.remove_node(node_to_kill)
        ppo.train()
        self.assertEqual(ppo.workers.num_healthy_remote_workers(), 4)
        self.assertEqual(ppo.workers.num_remote_workers(), 6)
        self.cluster.add_node(redis_port=None, num_redis_shards=None, num_cpus=2, num_gpus=0, object_store_memory=object_store_memory, redis_max_memory=redis_max_memory, dashboard_host='0.0.0.0')
        while True:
            states = [a['state'] == 'ALIVE' for a in list_actors() if a['class_name'] == 'RolloutWorker']
            if all(states):
                break
            time.sleep(1)
        ppo.train()
        self.assertEqual(ppo.workers.num_healthy_remote_workers(), 6)
        self.assertEqual(ppo.workers.num_remote_workers(), 6)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))