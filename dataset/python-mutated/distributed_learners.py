import math
from typing import Any, Dict, List, Optional, Type
import ray
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import AlgorithmConfigDict, PolicyID

class DistributedLearners:
    """Container class for n learning @ray.remote-turned policies.

    The container contains n "learner shards", each one consisting of one
    multi-agent replay buffer and m policy actors that share this replay
    buffer.
    """

    def __init__(self, *, config, max_num_policies_to_train: int, replay_actor_class: Type[ActorHandle], replay_actor_args: List[Any], num_learner_shards: Optional[int]=None):
        if False:
            return 10
        'Initializes a DistributedLearners instance.\n\n        Args:\n            config: The Algorithm\'s config dict.\n            max_num_policies_to_train: Maximum number of policies that will ever be\n                trainable. For these policies, we\'ll have to create remote\n                policy actors, distributed across n "learner shards".\n            num_learner_shards: Optional number of "learner shards" to reserve.\n                Each one consists of one multi-agent replay actor and\n                m policy actors that share this replay buffer. If None,\n                will infer this number automatically from the number of GPUs\n                and the max. number of learning policies.\n            replay_actor_class: The class to use to produce one multi-agent\n                replay buffer on each learner shard (shared by all policy actors\n                on that shard).\n            replay_actor_args: The args to pass to the remote replay buffer\n                actor\'s constructor.\n        '
        self.config = config
        self.num_gpus = self.config.num_gpus
        self.max_num_policies_to_train = max_num_policies_to_train
        self.replay_actor_class = replay_actor_class
        self.replay_actor_args = replay_actor_args
        if num_learner_shards is None:
            self.num_learner_shards = min(self.num_gpus or self.max_num_policies_to_train, self.max_num_policies_to_train)
        else:
            self.num_learner_shards = num_learner_shards
        self.num_gpus_per_shard = self.num_gpus // self.num_learner_shards
        if self.num_gpus_per_shard == 0:
            self.num_gpus_per_shard = self.num_gpus / self.num_learner_shards
        num_policies_per_shard = self.max_num_policies_to_train / self.num_learner_shards
        self.num_gpus_per_policy = self.num_gpus_per_shard / num_policies_per_shard
        self.num_policies_per_shard = math.ceil(num_policies_per_shard)
        self.shards = [_Shard(config=self.config, max_num_policies=self.num_policies_per_shard, num_gpus_per_policy=self.num_gpus_per_policy, replay_actor_class=self.replay_actor_class, replay_actor_args=self.replay_actor_args) for _ in range(self.num_learner_shards)]

    def add_policy(self, policy_id, policy_spec):
        if False:
            return 10
        for shard in self.shards:
            if shard.max_num_policies > len(shard.policy_actors):
                pol_actor = shard.add_policy(policy_id, policy_spec)
                return pol_actor
        raise RuntimeError('All shards are full!')

    def remove_policy(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_policy_actor(self, policy_id):
        if False:
            while True:
                i = 10
        for shard in self.shards:
            if policy_id in shard.policy_actors:
                return shard.policy_actors[policy_id]
        raise None

    def get_replay_and_policy_actors(self, policy_id):
        if False:
            while True:
                i = 10
        for shard in self.shards:
            if policy_id in shard.policy_actors:
                return (shard.replay_actor, shard.policy_actors[policy_id])
        return (None, None)

    def get_policy_id(self, policy_actor):
        if False:
            return 10
        for shard in self.shards:
            for (pid, act) in shard.policy_actors.items():
                if act == policy_actor:
                    return pid
        raise None

    def get_replay_actors(self):
        if False:
            i = 10
            return i + 15
        return [shard.replay_actor for shard in self.shards]

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        'Terminates all ray actors.'
        for shard in self.shards:
            shard.stop()

    def __len__(self):
        if False:
            while True:
                i = 10
        'Returns the number of all Policy actors in all our shards.'
        return sum((len(s) for s in self.shards))

    def __iter__(self):
        if False:
            i = 10
            return i + 15

        def _gen():
            if False:
                print('Hello World!')
            for shard in self.shards:
                for (pid, policy_actor) in shard.policy_actors.items():
                    yield (pid, policy_actor, shard.replay_actor)
        return _gen()

class _Shard:

    def __init__(self, config, max_num_policies, num_gpus_per_policy, replay_actor_class, replay_actor_args):
        if False:
            print('Hello World!')
        if isinstance(config, AlgorithmConfig):
            config = config.to_dict()
        self.config = config
        self.has_replay_buffer = False
        self.max_num_policies = max_num_policies
        self.num_gpus_per_policy = num_gpus_per_policy
        self.replay_actor_class = replay_actor_class
        self.replay_actor_args = replay_actor_args
        self.replay_actor: Optional[ActorHandle] = None
        self.policy_actors: Dict[str, ActorHandle] = {}

    def add_policy(self, policy_id: PolicyID, policy_spec: PolicySpec):
        if False:
            print('Hello World!')
        cfg = Algorithm.merge_trainer_configs(self.config, dict(policy_spec.config, **{'num_gpus': self.num_gpus_per_policy}))
        if self.replay_actor is None:
            return self._add_replay_buffer_and_policy(policy_id, policy_spec, cfg)
        assert len(self.policy_actors) < self.max_num_policies
        actual_policy_class = get_tf_eager_cls_if_necessary(policy_spec.policy_class, cfg)
        colocated = create_colocated_actors(actor_specs=[(ray.remote(num_cpus=1, num_gpus=self.num_gpus_per_policy if not cfg['_fake_gpus'] else 0)(actual_policy_class), (policy_spec.observation_space, policy_spec.action_space, cfg), {}, 1)], node=ray.get(self.replay_actor.get_host.remote()))
        self.policy_actors[policy_id] = colocated[0][0]
        return self.policy_actors[policy_id]

    def _add_replay_buffer_and_policy(self, policy_id: PolicyID, policy_spec: PolicySpec, config: AlgorithmConfigDict):
        if False:
            while True:
                i = 10
        assert self.replay_actor is None
        assert len(self.policy_actors) == 0
        actual_policy_class = get_tf_eager_cls_if_necessary(policy_spec.policy_class, config)
        if isinstance(config, AlgorithmConfig):
            config = config.to_dict()
        colocated = create_colocated_actors(actor_specs=[(self.replay_actor_class, self.replay_actor_args, {}, 1)] + [(ray.remote(num_cpus=1, num_gpus=self.num_gpus_per_policy if not config['_fake_gpus'] else 0)(actual_policy_class), (policy_spec.observation_space, policy_spec.action_space, config), {}, 1)], node=None)
        self.replay_actor = colocated[0][0]
        self.policy_actors[policy_id] = colocated[1][0]
        self.has_replay_buffer = True
        return self.policy_actors[policy_id]

    def stop(self):
        if False:
            i = 10
            return i + 15
        'Terminates all ray actors (replay and n policy actors).'
        self.replay_actor.__ray_terminate__.remote()
        for (pid, policy_actor) in self.policy_actors.items():
            policy_actor.__ray_terminate__.remote()

    def __len__(self):
        if False:
            return 10
        'Returns the number of Policy actors in this shard.'
        return len(self.policy_actors)