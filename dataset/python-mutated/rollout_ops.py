import logging
from typing import List, Optional, Union
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import _check_sample_batch_type
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, concat_samples
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.typing import SampleBatchType
logger = logging.getLogger(__name__)

@ExperimentalAPI
def synchronous_parallel_sample(*, worker_set: WorkerSet, max_agent_steps: Optional[int]=None, max_env_steps: Optional[int]=None, concat: bool=True) -> Union[List[SampleBatchType], SampleBatchType]:
    if False:
        i = 10
        return i + 15
    'Runs parallel and synchronous rollouts on all remote workers.\n\n    Waits for all workers to return from the remote calls.\n\n    If no remote workers exist (num_workers == 0), use the local worker\n    for sampling.\n\n    Alternatively to calling `worker.sample.remote()`, the user can provide a\n    `remote_fn()`, which will be applied to the worker(s) instead.\n\n    Args:\n        worker_set: The WorkerSet to use for sampling.\n        remote_fn: If provided, use `worker.apply.remote(remote_fn)` instead\n            of `worker.sample.remote()` to generate the requests.\n        max_agent_steps: Optional number of agent steps to be included in the\n            final batch.\n        max_env_steps: Optional number of environment steps to be included in the\n            final batch.\n        concat: Whether to concat all resulting batches at the end and return the\n            concat\'d batch.\n\n    Returns:\n        The list of collected sample batch types (one for each parallel\n        rollout worker in the given `worker_set`).\n\n    .. testcode::\n\n        # Define an RLlib Algorithm.\n        from ray.rllib.algorithms.ppo import PPO, PPOConfig\n        config = PPOConfig().environment("CartPole-v1")\n        algorithm = PPO(config=config)\n        # 2 remote workers (num_workers=2):\n        batches = synchronous_parallel_sample(worker_set=algorithm.workers,\n            concat=False)\n        print(len(batches))\n\n    .. testoutput::\n\n        2\n    '
    assert not (max_agent_steps is not None and max_env_steps is not None)
    agent_or_env_steps = 0
    max_agent_or_env_steps = max_agent_steps or max_env_steps or None
    all_sample_batches = []
    while max_agent_or_env_steps is None and agent_or_env_steps == 0 or (max_agent_or_env_steps is not None and agent_or_env_steps < max_agent_or_env_steps):
        if worker_set.num_remote_workers() <= 0:
            sample_batches = [worker_set.local_worker().sample()]
        else:
            sample_batches = worker_set.foreach_worker(lambda w: w.sample(), local_worker=False, healthy_only=True)
            if worker_set.num_healthy_remote_workers() <= 0:
                break
        for b in sample_batches:
            if max_agent_steps:
                agent_or_env_steps += b.agent_steps()
            else:
                agent_or_env_steps += b.env_steps()
        all_sample_batches.extend(sample_batches)
    if concat is True:
        full_batch = concat_samples(all_sample_batches)
        return full_batch
    else:
        return all_sample_batches

def standardize_fields(samples: SampleBatchType, fields: List[str]) -> SampleBatchType:
    if False:
        while True:
            i = 10
    'Standardize fields of the given SampleBatch'
    _check_sample_batch_type(samples)
    wrapped = False
    if isinstance(samples, SampleBatch):
        samples = samples.as_multi_agent()
        wrapped = True
    for policy_id in samples.policy_batches:
        batch = samples.policy_batches[policy_id]
        for field in fields:
            if field in batch:
                batch[field] = standardized(batch[field])
    if wrapped:
        samples = samples.policy_batches[DEFAULT_POLICY_ID]
    return samples