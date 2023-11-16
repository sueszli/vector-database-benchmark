from collections import defaultdict
import logging
import time
import tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import ActionConnectorDataType, AgentConnectorDataType, AgentID, EnvActionType, EnvID, EnvInfoDict, EnvObsType, MultiAgentDict, MultiEnvDict, PolicyID, PolicyOutputType, SampleBatchType, StateBatches, TensorStructType
from ray.util.debug import log_once
if TYPE_CHECKING:
    from gymnasium.envs.classic_control.rendering import SimpleImageViewer
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
logger = logging.getLogger(__name__)
MIN_LARGE_BATCH_THRESHOLD = 1000
DEFAULT_LARGE_BATCH_THRESHOLD = 5000
MS_TO_SEC = 1000.0

class _PerfStats:
    """Sampler perf stats that will be included in rollout metrics."""

    def __init__(self, ema_coef: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        self.ema_coef = ema_coef
        self.iters = 0
        self.raw_obs_processing_time = 0.0
        self.inference_time = 0.0
        self.action_processing_time = 0.0
        self.env_wait_time = 0.0
        self.env_render_time = 0.0

    def incr(self, field: str, value: Union[int, float]):
        if False:
            for i in range(10):
                print('nop')
        if field == 'iters':
            self.iters += value
            return
        if self.ema_coef is None:
            self.__dict__[field] += value
        else:
            self.__dict__[field] = (1.0 - self.ema_coef) * self.__dict__[field] + self.ema_coef * value

    def _get_avg(self):
        if False:
            return 10
        factor = MS_TO_SEC / self.iters
        return {'mean_raw_obs_processing_ms': self.raw_obs_processing_time * factor, 'mean_inference_ms': self.inference_time * factor, 'mean_action_processing_ms': self.action_processing_time * factor, 'mean_env_wait_ms': self.env_wait_time * factor, 'mean_env_render_ms': self.env_render_time * factor}

    def _get_ema(self):
        if False:
            return 10
        return {'mean_raw_obs_processing_ms': self.raw_obs_processing_time * MS_TO_SEC, 'mean_inference_ms': self.inference_time * MS_TO_SEC, 'mean_action_processing_ms': self.action_processing_time * MS_TO_SEC, 'mean_env_wait_ms': self.env_wait_time * MS_TO_SEC, 'mean_env_render_ms': self.env_render_time * MS_TO_SEC}

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ema_coef is None:
            return self._get_avg()
        else:
            return self._get_ema()

class _NewDefaultDict(defaultdict):

    def __missing__(self, env_id):
        if False:
            while True:
                i = 10
        ret = self[env_id] = self.default_factory(env_id)
        return ret

def _build_multi_agent_batch(episode_id: int, batch_builder: _PolicyCollectorGroup, large_batch_threshold: int, multiple_episodes_in_batch: bool) -> MultiAgentBatch:
    if False:
        while True:
            i = 10
    'Build MultiAgentBatch from a dict of _PolicyCollectors.\n\n    Args:\n        env_steps: total env steps.\n        policy_collectors: collected training SampleBatchs by policy.\n\n    Returns:\n        Always returns a sample batch in MultiAgentBatch format.\n    '
    ma_batch = {}
    for (pid, collector) in batch_builder.policy_collectors.items():
        if collector.agent_steps <= 0:
            continue
        if batch_builder.agent_steps > large_batch_threshold and log_once('large_batch_warning'):
            logger.warning('More than {} observations in {} env steps for episode {} '.format(batch_builder.agent_steps, batch_builder.env_steps, episode_id) + 'are buffered in the sampler. If this is more than you expected, check that that you set a horizon on your environment correctly and that it terminates at some point. Note: In multi-agent environments, `rollout_fragment_length` sets the batch size based on (across-agents) environment steps, not the steps of individual agents, which can result in unexpectedly large batches.' + ('Also, you may be waiting for your Env to terminate (batch_mode=`complete_episodes`). Make sure it does at some point.' if not multiple_episodes_in_batch else ''))
        batch = collector.build()
        policy = collector.policy
        if policy.config.get('_enable_new_api_stack', False):
            seq_lens = batch.get(SampleBatch.SEQ_LENS)
            pad_batch_to_sequences_of_same_size(batch=batch, max_seq_len=policy.config['model']['max_seq_len'], shuffle=False, batch_divisibility_req=getattr(policy, 'batch_divisibility_req', 1), view_requirements=getattr(policy, 'view_requirements', None), _enable_new_api_stack=True)
            batch = policy.maybe_add_time_dimension(batch, seq_lens=seq_lens, framework='np')
        ma_batch[pid] = batch
    return MultiAgentBatch(policy_batches=ma_batch, env_steps=batch_builder.env_steps)

def _batch_inference_sample_batches(eval_data: List[SampleBatch]) -> SampleBatch:
    if False:
        i = 10
        return i + 15
    'Batch a list of input SampleBatches into a single SampleBatch.\n\n    Args:\n        eval_data: list of SampleBatches.\n\n    Returns:\n        single batched SampleBatch.\n    '
    inference_batch = concat_samples(eval_data)
    if 'state_in_0' in inference_batch:
        batch_size = len(eval_data)
        inference_batch[SampleBatch.SEQ_LENS] = np.ones(batch_size, dtype=np.int32)
    return inference_batch

@DeveloperAPI
class EnvRunnerV2:
    """Collect experiences from user environment using Connectors."""

    def __init__(self, worker: 'RolloutWorker', base_env: BaseEnv, multiple_episodes_in_batch: bool, callbacks: 'DefaultCallbacks', perf_stats: _PerfStats, rollout_fragment_length: int=200, count_steps_by: str='env_steps', render: bool=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            worker: Reference to the current rollout worker.\n            base_env: Env implementing BaseEnv.\n            multiple_episodes_in_batch: Whether to pack multiple\n                episodes into each batch. This guarantees batches will be exactly\n                `rollout_fragment_length` in size.\n            callbacks: User callbacks to run on episode events.\n            perf_stats: Record perf stats into this object.\n            rollout_fragment_length: The length of a fragment to collect\n                before building a SampleBatch from the data and resetting\n                the SampleBatchBuilder object.\n            count_steps_by: One of "env_steps" (default) or "agent_steps".\n                Use "agent_steps", if you want rollout lengths to be counted\n                by individual agent steps. In a multi-agent env,\n                a single env_step contains one or more agent_steps, depending\n                on how many agents are present at any given time in the\n                ongoing episode.\n            render: Whether to try to render the environment after each\n                step.\n        '
        self._worker = worker
        if isinstance(base_env, ExternalEnvWrapper):
            raise ValueError('Policies using the new Connector API do not support ExternalEnv.')
        self._base_env = base_env
        self._multiple_episodes_in_batch = multiple_episodes_in_batch
        self._callbacks = callbacks
        self._perf_stats = perf_stats
        self._rollout_fragment_length = rollout_fragment_length
        self._count_steps_by = count_steps_by
        self._render = render
        self._simple_image_viewer: Optional['SimpleImageViewer'] = self._get_simple_image_viewer()
        self._active_episodes: Dict[EnvID, EpisodeV2] = {}
        self._batch_builders: Dict[EnvID, _PolicyCollectorGroup] = _NewDefaultDict(self._new_batch_builder)
        self._large_batch_threshold: int = max(MIN_LARGE_BATCH_THRESHOLD, self._rollout_fragment_length * 10) if self._rollout_fragment_length != float('inf') else DEFAULT_LARGE_BATCH_THRESHOLD

    def _get_simple_image_viewer(self):
        if False:
            while True:
                i = 10
        'Maybe construct a SimpleImageViewer instance for episode rendering.'
        if not self._render:
            return None
        try:
            from gymnasium.envs.classic_control.rendering import SimpleImageViewer
            return SimpleImageViewer()
        except (ImportError, ModuleNotFoundError):
            self._render = False
            logger.warning('Could not import gymnasium.envs.classic_control.rendering! Try `pip install gymnasium[all]`.')
        return None

    def _call_on_episode_start(self, episode, env_id):
        if False:
            return 10
        for p in self._worker.policy_map.cache.values():
            if getattr(p, 'exploration', None) is not None:
                p.exploration.on_episode_start(policy=p, environment=self._base_env, episode=episode, tf_sess=p.get_session())
        self._callbacks.on_episode_start(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, env_index=env_id, episode=episode)

    def _new_batch_builder(self, _) -> _PolicyCollectorGroup:
        if False:
            i = 10
            return i + 15
        'Create a new batch builder.\n\n        We create a _PolicyCollectorGroup based on the full policy_map\n        as the batch builder.\n        '
        return _PolicyCollectorGroup(self._worker.policy_map)

    def run(self) -> Iterator[SampleBatchType]:
        if False:
            for i in range(10):
                print('nop')
        'Samples and yields training episodes continuously.\n\n        Yields:\n            Object containing state, action, reward, terminal condition,\n            and other fields as dictated by `policy`.\n        '
        while True:
            outputs = self.step()
            for o in outputs:
                yield o

    def step(self) -> List[SampleBatchType]:
        if False:
            for i in range(10):
                print('nop')
        'Samples training episodes by stepping through environments.'
        self._perf_stats.incr('iters', 1)
        t0 = time.time()
        (unfiltered_obs, rewards, terminateds, truncateds, infos, off_policy_actions) = self._base_env.poll()
        env_poll_time = time.time() - t0
        t1 = time.time()
        (active_envs, to_eval, outputs) = self._process_observations(unfiltered_obs=unfiltered_obs, rewards=rewards, terminateds=terminateds, truncateds=truncateds, infos=infos)
        self._perf_stats.incr('raw_obs_processing_time', time.time() - t1)
        t2 = time.time()
        eval_results = self._do_policy_eval(to_eval=to_eval)
        self._perf_stats.incr('inference_time', time.time() - t2)
        t3 = time.time()
        actions_to_send: Dict[EnvID, Dict[AgentID, EnvActionType]] = self._process_policy_eval_results(active_envs=active_envs, to_eval=to_eval, eval_results=eval_results, off_policy_actions=off_policy_actions)
        self._perf_stats.incr('action_processing_time', time.time() - t3)
        t4 = time.time()
        self._base_env.send_actions(actions_to_send)
        self._perf_stats.incr('env_wait_time', env_poll_time + time.time() - t4)
        self._maybe_render()
        return outputs

    def _get_rollout_metrics(self, episode: EpisodeV2, policy_map: Dict[str, Policy]) -> List[RolloutMetrics]:
        if False:
            for i in range(10):
                print('nop')
        'Get rollout metrics from completed episode.'
        atari_metrics: List[RolloutMetrics] = _fetch_atari_metrics(self._base_env)
        if atari_metrics is not None:
            for m in atari_metrics:
                m._replace(custom_metrics=episode.custom_metrics)
            return atari_metrics
        connector_metrics = {}
        active_agents = episode.get_agents()
        for agent in active_agents:
            policy_id = episode.policy_for(agent)
            policy = episode.policy_map[policy_id]
            connector_metrics[policy_id] = policy.get_connector_metrics()
        return [RolloutMetrics(episode_length=episode.length, episode_reward=episode.total_reward, agent_rewards=dict(episode.agent_rewards), custom_metrics=episode.custom_metrics, perf_stats={}, hist_data=episode.hist_data, media=episode.media, connector_metrics=connector_metrics)]

    def _process_observations(self, unfiltered_obs: MultiEnvDict, rewards: MultiEnvDict, terminateds: MultiEnvDict, truncateds: MultiEnvDict, infos: MultiEnvDict) -> Tuple[Set[EnvID], Dict[PolicyID, List[AgentConnectorDataType]], List[Union[RolloutMetrics, SampleBatchType]]]:
        if False:
            return 10
        'Process raw obs from env.\n\n        Group data for active agents by policy. Reset environments that are done.\n\n        Args:\n            unfiltered_obs: The unfiltered, raw observations from the BaseEnv\n                (vectorized, possibly multi-agent). Dict of dict: By env index,\n                then agent ID, then mapped to actual obs.\n            rewards: The rewards MultiEnvDict of the BaseEnv.\n            terminateds: The `terminated` flags MultiEnvDict of the BaseEnv.\n            truncateds: The `truncated` flags MultiEnvDict of the BaseEnv.\n            infos: The MultiEnvDict of infos dicts of the BaseEnv.\n\n        Returns:\n            A tuple of:\n                A list of envs that were active during this step.\n                AgentConnectorDataType for active agents for policy evaluation.\n                SampleBatches and RolloutMetrics for completed agents for output.\n        '
        active_envs: Set[EnvID] = set()
        to_eval: Dict[PolicyID, List[AgentConnectorDataType]] = defaultdict(list)
        outputs: List[Union[RolloutMetrics, SampleBatchType]] = []
        for (env_id, env_obs) in unfiltered_obs.items():
            if isinstance(env_obs, Exception):
                assert terminateds[env_id]['__all__'] is True, f'ERROR: When a sub-environment (env-id {env_id}) returns an error as observation, the terminateds[__all__] flag must also be set to True!'
                self._handle_done_episode(env_id=env_id, env_obs_or_exception=env_obs, is_done=True, active_envs=active_envs, to_eval=to_eval, outputs=outputs)
                continue
            if env_id not in self._active_episodes:
                episode: EpisodeV2 = self.create_episode(env_id)
                self._active_episodes[env_id] = episode
            else:
                episode: EpisodeV2 = self._active_episodes[env_id]
            if not episode.has_init_obs():
                self._call_on_episode_start(episode, env_id)
            if terminateds[env_id]['__all__'] or truncateds[env_id]['__all__']:
                all_agents_done = True
            else:
                all_agents_done = False
                active_envs.add(env_id)
            episode.set_last_info('__common__', infos[env_id].get('__common__', {}))
            sample_batches_by_policy = defaultdict(list)
            agent_terminateds = {}
            agent_truncateds = {}
            for (agent_id, obs) in env_obs.items():
                assert agent_id != '__all__'
                policy_id: PolicyID = episode.policy_for(agent_id)
                agent_terminated = bool(terminateds[env_id]['__all__'] or terminateds[env_id].get(agent_id))
                agent_terminateds[agent_id] = agent_terminated
                agent_truncated = bool(truncateds[env_id]['__all__'] or truncateds[env_id].get(agent_id, False))
                agent_truncateds[agent_id] = agent_truncated
                if not episode.has_init_obs(agent_id) and (agent_terminated or agent_truncated):
                    continue
                values_dict = {SampleBatch.T: episode.length, SampleBatch.ENV_ID: env_id, SampleBatch.AGENT_INDEX: episode.agent_index(agent_id), SampleBatch.REWARDS: rewards[env_id].get(agent_id, 0.0), SampleBatch.TERMINATEDS: agent_terminated, SampleBatch.TRUNCATEDS: agent_truncated, SampleBatch.INFOS: infos[env_id].get(agent_id, {}), SampleBatch.NEXT_OBS: obs}
                sample_batches_by_policy[policy_id].append((agent_id, values_dict))
            if all_agents_done:
                for agent_id in episode.get_agents():
                    if agent_terminateds.get(agent_id, False) or agent_truncateds.get(agent_id, False) or episode.is_done(agent_id):
                        continue
                    policy_id: PolicyID = episode.policy_for(agent_id)
                    policy = self._worker.policy_map[policy_id]
                    obs_space = get_original_space(policy.observation_space)
                    reward = rewards[env_id].get(agent_id, 0.0)
                    info = infos[env_id].get(agent_id, {})
                    values_dict = {SampleBatch.T: episode.length, SampleBatch.ENV_ID: env_id, SampleBatch.AGENT_INDEX: episode.agent_index(agent_id), SampleBatch.REWARDS: reward, SampleBatch.TERMINATEDS: True, SampleBatch.TRUNCATEDS: truncateds[env_id].get(agent_id, False), SampleBatch.INFOS: info, SampleBatch.NEXT_OBS: obs_space.sample()}
                    sample_batches_by_policy[policy_id].append((agent_id, values_dict))
            for (policy_id, batches) in sample_batches_by_policy.items():
                policy: Policy = self._worker.policy_map[policy_id]
                assert policy.agent_connectors, 'EnvRunnerV2 requires agent connectors to work.'
                acd_list: List[AgentConnectorDataType] = [AgentConnectorDataType(env_id, agent_id, data) for (agent_id, data) in batches]
                processed = policy.agent_connectors(acd_list)
                for d in processed:
                    if not episode.has_init_obs(d.agent_id):
                        episode.add_init_obs(agent_id=d.agent_id, init_obs=d.data.raw_dict[SampleBatch.NEXT_OBS], init_infos=d.data.raw_dict[SampleBatch.INFOS], t=d.data.raw_dict[SampleBatch.T])
                    else:
                        episode.add_action_reward_done_next_obs(d.agent_id, d.data.raw_dict)
                    if not (all_agents_done or agent_terminateds.get(d.agent_id, False) or agent_truncateds.get(d.agent_id, False) or episode.is_done(d.agent_id)):
                        item = AgentConnectorDataType(d.env_id, d.agent_id, d.data)
                        to_eval[policy_id].append(item)
            episode.step()
            if episode.length > 0:
                self._callbacks.on_episode_step(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, episode=episode, env_index=env_id)
            if all_agents_done:
                self._handle_done_episode(env_id, env_obs, terminateds[env_id]['__all__'] or truncateds[env_id]['__all__'], active_envs, to_eval, outputs)
            if self._multiple_episodes_in_batch:
                sample_batch = self._try_build_truncated_episode_multi_agent_batch(self._batch_builders[env_id], episode)
                if sample_batch:
                    outputs.append(sample_batch)
                    del self._batch_builders[env_id]
        return (active_envs, to_eval, outputs)

    def _build_done_episode(self, env_id: EnvID, is_done: bool, outputs: List[SampleBatchType]):
        if False:
            print('Hello World!')
        'Builds a MultiAgentSampleBatch from the episode and adds it to outputs.\n\n        Args:\n            env_id: The env id.\n            is_done: Whether the env is done.\n            outputs: The list of outputs to add the\n        '
        episode: EpisodeV2 = self._active_episodes[env_id]
        batch_builder = self._batch_builders[env_id]
        episode.postprocess_episode(batch_builder=batch_builder, is_done=is_done, check_dones=is_done)
        if not self._multiple_episodes_in_batch:
            ma_sample_batch = _build_multi_agent_batch(episode.episode_id, batch_builder, self._large_batch_threshold, self._multiple_episodes_in_batch)
            if ma_sample_batch:
                outputs.append(ma_sample_batch)
            del self._batch_builders[env_id]

    def __process_resetted_obs_for_eval(self, env_id: EnvID, obs: Dict[EnvID, Dict[AgentID, EnvObsType]], infos: Dict[EnvID, Dict[AgentID, EnvInfoDict]], episode: EpisodeV2, to_eval: Dict[PolicyID, List[AgentConnectorDataType]]):
        if False:
            return 10
        'Process resetted obs through agent connectors for policy eval.\n\n        Args:\n            env_id: The env id.\n            obs: The Resetted obs.\n            episode: New episode.\n            to_eval: List of agent connector data for policy eval.\n        '
        per_policy_resetted_obs: Dict[PolicyID, List] = defaultdict(list)
        for (agent_id, raw_obs) in obs[env_id].items():
            policy_id: PolicyID = episode.policy_for(agent_id)
            per_policy_resetted_obs[policy_id].append((agent_id, raw_obs))
        for (policy_id, agents_obs) in per_policy_resetted_obs.items():
            policy = self._worker.policy_map[policy_id]
            acd_list: List[AgentConnectorDataType] = [AgentConnectorDataType(env_id, agent_id, {SampleBatch.NEXT_OBS: obs, SampleBatch.INFOS: infos, SampleBatch.T: episode.length, SampleBatch.AGENT_INDEX: episode.agent_index(agent_id)}) for (agent_id, obs) in agents_obs]
            processed = policy.agent_connectors(acd_list)
            for d in processed:
                episode.add_init_obs(agent_id=d.agent_id, init_obs=d.data.raw_dict[SampleBatch.NEXT_OBS], init_infos=d.data.raw_dict[SampleBatch.INFOS], t=d.data.raw_dict[SampleBatch.T])
                to_eval[policy_id].append(d)

    def _handle_done_episode(self, env_id: EnvID, env_obs_or_exception: MultiAgentDict, is_done: bool, active_envs: Set[EnvID], to_eval: Dict[PolicyID, List[AgentConnectorDataType]], outputs: List[SampleBatchType]) -> None:
        if False:
            while True:
                i = 10
        'Handle an all-finished episode.\n\n        Add collected SampleBatch to batch builder. Reset corresponding env, etc.\n\n        Args:\n            env_id: Environment ID.\n            env_obs_or_exception: Last per-environment observation or Exception.\n            env_infos: Last per-environment infos.\n            is_done: If all agents are done.\n            active_envs: Set of active env ids.\n            to_eval: Output container for policy eval data.\n            outputs: Output container for collected sample batches.\n        '
        if isinstance(env_obs_or_exception, Exception):
            episode_or_exception: Exception = env_obs_or_exception
            outputs.append(RolloutMetrics(episode_faulty=True))
        else:
            episode_or_exception: EpisodeV2 = self._active_episodes[env_id]
            outputs.extend(self._get_rollout_metrics(episode_or_exception, policy_map=self._worker.policy_map))
            self._build_done_episode(env_id, is_done, outputs)
        self.end_episode(env_id, episode_or_exception)
        new_episode: EpisodeV2 = self.create_episode(env_id)
        while True:
            (resetted_obs, resetted_infos) = self._base_env.try_reset(env_id)
            if resetted_obs is None or resetted_obs == ASYNC_RESET_RETURN or (not isinstance(resetted_obs[env_id], Exception)):
                break
            else:
                outputs.append(RolloutMetrics(episode_faulty=True))
        for p in self._worker.policy_map.cache.values():
            p.agent_connectors.reset(env_id)
        if resetted_obs is not None and resetted_obs != ASYNC_RESET_RETURN:
            self._active_episodes[env_id] = new_episode
            self._call_on_episode_start(new_episode, env_id)
            self.__process_resetted_obs_for_eval(env_id, resetted_obs, resetted_infos, new_episode, to_eval)
            new_episode.step()
            active_envs.add(env_id)

    def create_episode(self, env_id: EnvID) -> EpisodeV2:
        if False:
            print('Hello World!')
        'Creates a new EpisodeV2 instance and returns it.\n\n        Calls `on_episode_created` callbacks, but does NOT reset the respective\n        sub-environment yet.\n\n        Args:\n            env_id: Env ID.\n\n        Returns:\n            The newly created EpisodeV2 instance.\n        '
        assert env_id not in self._active_episodes
        new_episode = EpisodeV2(env_id, self._worker.policy_map, self._worker.policy_mapping_fn, worker=self._worker, callbacks=self._callbacks)
        self._callbacks.on_episode_created(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, env_index=env_id, episode=new_episode)
        return new_episode

    def end_episode(self, env_id: EnvID, episode_or_exception: Union[EpisodeV2, Exception]):
        if False:
            for i in range(10):
                print('nop')
        'Cleans up an episode that has finished.\n\n        Args:\n            env_id: Env ID.\n            episode_or_exception: Instance of an episode if it finished successfully.\n                Otherwise, the exception that was thrown,\n        '
        self._callbacks.on_episode_end(worker=self._worker, base_env=self._base_env, policies=self._worker.policy_map, episode=episode_or_exception, env_index=env_id)
        for p in self._worker.policy_map.cache.values():
            if getattr(p, 'exploration', None) is not None:
                p.exploration.on_episode_end(policy=p, environment=self._base_env, episode=episode_or_exception, tf_sess=p.get_session())
        if isinstance(episode_or_exception, EpisodeV2):
            episode = episode_or_exception
            if episode.total_agent_steps == 0:
                msg = f'Data from episode {episode.episode_id} does not show any agent interactions. Hint: Make sure for at least one timestep in the episode, env.step() returns non-empty values.'
                raise ValueError(msg)
        if env_id in self._active_episodes:
            del self._active_episodes[env_id]

    def _try_build_truncated_episode_multi_agent_batch(self, batch_builder: _PolicyCollectorGroup, episode: EpisodeV2) -> Union[None, SampleBatch, MultiAgentBatch]:
        if False:
            return 10
        if self._count_steps_by == 'env_steps':
            built_steps = batch_builder.env_steps
            ongoing_steps = episode.active_env_steps
        else:
            built_steps = batch_builder.agent_steps
            ongoing_steps = episode.active_agent_steps
        if built_steps + ongoing_steps >= self._rollout_fragment_length:
            if self._count_steps_by != 'agent_steps':
                assert built_steps + ongoing_steps == self._rollout_fragment_length, f'built_steps ({built_steps}) + ongoing_steps ({ongoing_steps}) != rollout_fragment_length ({self._rollout_fragment_length}).'
            if built_steps < self._rollout_fragment_length:
                episode.postprocess_episode(batch_builder=batch_builder, is_done=False)
            if batch_builder.agent_steps > 0:
                return _build_multi_agent_batch(episode.episode_id, batch_builder, self._large_batch_threshold, self._multiple_episodes_in_batch)
            elif log_once('no_agent_steps'):
                logger.warning('Your environment seems to be stepping w/o ever emitting agent observations (agents are never requested to act)!')
        return None

    def _do_policy_eval(self, to_eval: Dict[PolicyID, List[AgentConnectorDataType]]) -> Dict[PolicyID, PolicyOutputType]:
        if False:
            while True:
                i = 10
        "Call compute_actions on collected episode data to get next action.\n\n        Args:\n            to_eval: Mapping of policy IDs to lists of AgentConnectorDataType objects\n                (items in these lists will be the batch's items for the model\n                forward pass).\n\n        Returns:\n            Dict mapping PolicyIDs to compute_actions_from_input_dict() outputs.\n        "
        policies = self._worker.policy_map

        def _try_find_policy_again(eval_data: AgentConnectorDataType):
            if False:
                print('Hello World!')
            policy_id = None
            for d in eval_data:
                episode = self._active_episodes[d.env_id]
                pid = episode.policy_for(d.agent_id, refresh=True)
                if policy_id is not None and pid != policy_id:
                    raise ValueError(f"Policy map changed. The list of eval data that was handled by a same policy is now handled by policy {pid} and {{policy_id}}. Please don't do this in the middle of an episode.")
                policy_id = pid
            return _get_or_raise(self._worker.policy_map, policy_id)
        eval_results: Dict[PolicyID, TensorStructType] = {}
        for (policy_id, eval_data) in to_eval.items():
            try:
                policy: Policy = _get_or_raise(policies, policy_id)
            except ValueError:
                policy: Policy = _try_find_policy_again(eval_data)
            if policy.config.get('_enable_new_api_stack', False):
                input_dict = concat_samples([d.data.sample_batch for d in eval_data])
            else:
                input_dict = _batch_inference_sample_batches([d.data.sample_batch for d in eval_data])
            eval_results[policy_id] = policy.compute_actions_from_input_dict(input_dict, timestep=policy.global_timestep, episodes=[self._active_episodes[t.env_id] for t in eval_data])
        return eval_results

    def _process_policy_eval_results(self, active_envs: Set[EnvID], to_eval: Dict[PolicyID, List[AgentConnectorDataType]], eval_results: Dict[PolicyID, PolicyOutputType], off_policy_actions: MultiEnvDict):
        if False:
            print('Hello World!')
        'Process the output of policy neural network evaluation.\n\n        Records policy evaluation results into agent connectors and\n        returns replies to send back to agents in the env.\n\n        Args:\n            active_envs: Set of env IDs that are still active.\n            to_eval: Mapping of policy IDs to lists of AgentConnectorDataType objects.\n            eval_results: Mapping of policy IDs to list of\n                actions, rnn-out states, extra-action-fetches dicts.\n            off_policy_actions: Doubly keyed dict of env-ids -> agent ids ->\n                off-policy-action, returned by a `BaseEnv.poll()` call.\n\n        Returns:\n            Nested dict of env id -> agent id -> actions to be sent to\n            Env (np.ndarrays).\n        '
        actions_to_send: Dict[EnvID, Dict[AgentID, EnvActionType]] = defaultdict(dict)
        for env_id in active_envs:
            actions_to_send[env_id] = {}
        for (policy_id, eval_data) in to_eval.items():
            actions: TensorStructType = eval_results[policy_id][0]
            actions = convert_to_numpy(actions)
            rnn_out: StateBatches = eval_results[policy_id][1]
            extra_action_out: dict = eval_results[policy_id][2]
            if isinstance(actions, list):
                actions = np.array(actions)
            actions: List[EnvActionType] = unbatch(actions)
            policy: Policy = _get_or_raise(self._worker.policy_map, policy_id)
            assert policy.agent_connectors and policy.action_connectors, 'EnvRunnerV2 requires action connectors to work.'
            for (i, action) in enumerate(actions):
                env_id: int = eval_data[i].env_id
                agent_id: AgentID = eval_data[i].agent_id
                input_dict: TensorStructType = eval_data[i].data.raw_dict
                rnn_states: List[StateBatches] = tree.map_structure(lambda x: x[i], rnn_out)
                fetches: Dict = tree.map_structure(lambda x: x[i], extra_action_out)
                ac_data = ActionConnectorDataType(env_id, agent_id, input_dict, (action, rnn_states, fetches))
                (action_to_send, rnn_states, fetches) = policy.action_connectors(ac_data).output
                action_to_buffer = action if env_id not in off_policy_actions or agent_id not in off_policy_actions[env_id] else off_policy_actions[env_id][agent_id]
                ac_data: ActionConnectorDataType = ActionConnectorDataType(env_id, agent_id, input_dict, (action_to_buffer, rnn_states, fetches))
                policy.agent_connectors.on_policy_output(ac_data)
                assert agent_id not in actions_to_send[env_id]
                actions_to_send[env_id][agent_id] = action_to_send
        return actions_to_send

    def _maybe_render(self):
        if False:
            i = 10
            return i + 15
        'Visualize environment.'
        if not self._render or not self._simple_image_viewer:
            return
        t5 = time.time()
        rendered = self._base_env.try_render()
        if isinstance(rendered, np.ndarray) and len(rendered.shape) == 3:
            self._simple_image_viewer.imshow(rendered)
        elif rendered not in [True, False, None]:
            raise ValueError(f"The env's ({self._base_env}) `try_render()` method returned an unsupported value! Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.")
        self._perf_stats.incr('env_render_time', time.time() - t5)

def _fetch_atari_metrics(base_env: BaseEnv) -> List[RolloutMetrics]:
    if False:
        for i in range(10):
            print('nop')
    'Atari games have multiple logical episodes, one per life.\n\n    However, for metrics reporting we count full episodes, all lives included.\n    '
    sub_environments = base_env.get_sub_environments()
    if not sub_environments:
        return None
    atari_out = []
    for sub_env in sub_environments:
        monitor = get_wrapper_by_cls(sub_env, MonitorEnv)
        if not monitor:
            return None
        for (eps_rew, eps_len) in monitor.next_episode_results():
            atari_out.append(RolloutMetrics(eps_len, eps_rew))
    return atari_out

def _get_or_raise(mapping: Dict[PolicyID, Union[Policy, Preprocessor, Filter]], policy_id: PolicyID) -> Union[Policy, Preprocessor, Filter]:
    if False:
        for i in range(10):
            print('nop')
    'Returns an object under key `policy_id` in `mapping`.\n\n    Args:\n        mapping (Dict[PolicyID, Union[Policy, Preprocessor, Filter]]): The\n            mapping dict from policy id (str) to actual object (Policy,\n            Preprocessor, etc.).\n        policy_id: The policy ID to lookup.\n\n    Returns:\n        Union[Policy, Preprocessor, Filter]: The found object.\n\n    Raises:\n        ValueError: If `policy_id` cannot be found in `mapping`.\n    '
    if policy_id not in mapping:
        raise ValueError('Could not find policy for agent: PolicyID `{}` not found in policy map, whose keys are `{}`.'.format(policy_id, mapping.keys()))
    return mapping[policy_id]