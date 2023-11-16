import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray, get_dummy_batch_for_space
from ray.rllib.utils.typing import EpisodeID, EnvID, TensorType, ViewRequirementsDict
from ray.util.annotations import PublicAPI
logger = logging.getLogger(__name__)
(_, tf, _) = try_import_tf()
(torch, _) = try_import_torch()

def _to_float_np_array(v: List[Any]) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    if torch and torch.is_tensor(v[0]):
        raise ValueError
    arr = np.array(v)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr

def _get_buffered_slice_with_paddings(d, inds):
    if False:
        while True:
            i = 10
    element_at_t = []
    for index in inds:
        if index < len(d):
            element_at_t.append(d[index])
        else:
            element_at_t.append(tree.map_structure(np.zeros_like, d[-1]))
    return element_at_t

@PublicAPI
class AgentCollector:
    """Collects samples for one agent in one trajectory (episode).

    The agent may be part of a multi-agent environment. Samples are stored in
    lists including some possible automatic "shift" buffer at the beginning to
    be able to save memory when storing things like NEXT_OBS, PREV_REWARDS,
    etc.., which are specified using the trajectory view API.
    """
    _next_unroll_id = 0

    def __init__(self, view_reqs: ViewRequirementsDict, *, max_seq_len: int=1, disable_action_flattening: bool=True, intial_states: Optional[List[TensorType]]=None, is_policy_recurrent: bool=False, is_training: bool=True, _enable_new_api_stack: bool=False):
        if False:
            i = 10
            return i + 15
        "Initialize an AgentCollector.\n\n        Args:\n            view_reqs: A dict of view requirements for the agent.\n            max_seq_len: The maximum sequence length to store.\n            disable_action_flattening: If True, don't flatten the action.\n            intial_states: The initial states from the policy.get_initial_states()\n            is_policy_recurrent: If True, the policy is recurrent.\n            is_training: Sets the is_training flag for the buffers. if True, all the\n                timesteps are stored in the buffers until explictly build_for_training\n                () is called. if False, only the content required for the last time\n                step is stored in the buffers. This will save memory during inference.\n                You can change the behavior at runtime by calling is_training(mode).\n        "
        self.max_seq_len = max_seq_len
        self.disable_action_flattening = disable_action_flattening
        self.view_requirements = view_reqs
        self.initial_states = intial_states if intial_states is not None else []
        self.is_policy_recurrent = is_policy_recurrent
        self._is_training = is_training
        self._enable_new_api_stack = _enable_new_api_stack
        view_req_shifts = [min(vr.shift_arr) - int((vr.data_col or k) in [SampleBatch.OBS, SampleBatch.INFOS]) for (k, vr) in view_reqs.items()]
        self.shift_before = -min(view_req_shifts)
        self.buffers: Dict[str, List[List[TensorType]]] = {}
        self.buffer_structs: Dict[str, Any] = {}
        self.episode_id = None
        self.unroll_id = None
        self.agent_steps = 0
        self.data_cols_with_dummy_values = set()

    @property
    def training(self) -> bool:
        if False:
            return 10
        return self._is_training

    def is_training(self, is_training: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._is_training = is_training

    def is_empty(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns True if this collector has no data.'
        return not self.buffers or all((len(item) == 0 for item in self.buffers.values()))

    def add_init_obs(self, episode_id: EpisodeID, agent_index: int, env_id: EnvID, init_obs: TensorType, init_infos: Optional[Dict[str, TensorType]]=None, t: int=-1) -> None:
        if False:
            i = 10
            return i + 15
        "Adds an initial observation (after reset) to the Agent's trajectory.\n\n        Args:\n            episode_id: Unique ID for the episode we are adding the\n                initial observation for.\n            agent_index: Unique int index (starting from 0) for the agent\n                within its episode. Not to be confused with AGENT_ID (Any).\n            env_id: The environment index (in a vectorized setup).\n            init_obs: The initial observation tensor (after `env.reset()`).\n            init_infos: The initial infos dict (after `env.reset()`).\n            t: The time step (episode length - 1). The initial obs has\n                ts=-1(!), then an action/reward/next-obs at t=0, etc..\n        "
        self.episode_id = episode_id
        if self.unroll_id is None:
            self.unroll_id = AgentCollector._next_unroll_id
            AgentCollector._next_unroll_id += 1
        if isinstance(init_obs, list):
            init_obs = np.array(init_obs)
        if SampleBatch.OBS not in self.buffers:
            single_row = {SampleBatch.OBS: init_obs, SampleBatch.INFOS: init_infos or {}, SampleBatch.AGENT_INDEX: agent_index, SampleBatch.ENV_ID: env_id, SampleBatch.T: t, SampleBatch.EPS_ID: self.episode_id, SampleBatch.UNROLL_ID: self.unroll_id}
            if SampleBatch.PREV_REWARDS in self.view_requirements:
                single_row[SampleBatch.REWARDS] = get_dummy_batch_for_space(space=self.view_requirements[SampleBatch.REWARDS].space, batch_size=0, fill_value=0.0)
            if SampleBatch.PREV_ACTIONS in self.view_requirements:
                potentially_flattened_batch = get_dummy_batch_for_space(space=self.view_requirements[SampleBatch.ACTIONS].space, batch_size=0, fill_value=0.0)
                if not self.disable_action_flattening:
                    potentially_flattened_batch = flatten_to_single_ndarray(potentially_flattened_batch)
                single_row[SampleBatch.ACTIONS] = potentially_flattened_batch
            self._build_buffers(single_row)
        flattened = tree.flatten(init_obs)
        for (i, sub_obs) in enumerate(flattened):
            self.buffers[SampleBatch.OBS][i].append(sub_obs)
        self.buffers[SampleBatch.INFOS][0].append(init_infos or {})
        self.buffers[SampleBatch.AGENT_INDEX][0].append(agent_index)
        self.buffers[SampleBatch.ENV_ID][0].append(env_id)
        self.buffers[SampleBatch.T][0].append(t)
        self.buffers[SampleBatch.EPS_ID][0].append(self.episode_id)
        self.buffers[SampleBatch.UNROLL_ID][0].append(self.unroll_id)

    def add_action_reward_next_obs(self, input_values: Dict[str, TensorType]) -> None:
        if False:
            while True:
                i = 10
        "Adds the given dictionary (row) of values to the Agent's trajectory.\n\n        Args:\n            values: Data dict (interpreted as a single row) to be added to buffer.\n                Must contain keys:\n                SampleBatch.ACTIONS, REWARDS, TERMINATEDS, TRUNCATEDS, and NEXT_OBS.\n        "
        if self.unroll_id is None:
            self.unroll_id = AgentCollector._next_unroll_id
            AgentCollector._next_unroll_id += 1
        values = copy.copy(input_values)
        assert SampleBatch.OBS not in values
        values[SampleBatch.OBS] = values[SampleBatch.NEXT_OBS]
        del values[SampleBatch.NEXT_OBS]
        if isinstance(values[SampleBatch.OBS], list):
            values[SampleBatch.OBS] = np.array(values[SampleBatch.OBS])
        if SampleBatch.T not in input_values:
            values[SampleBatch.T] = self.buffers[SampleBatch.T][0][-1] + 1
        if SampleBatch.EPS_ID in values:
            assert values[SampleBatch.EPS_ID] == self.episode_id
            del values[SampleBatch.EPS_ID]
        self.buffers[SampleBatch.EPS_ID][0].append(self.episode_id)
        if SampleBatch.UNROLL_ID in values:
            assert values[SampleBatch.UNROLL_ID] == self.unroll_id
            del values[SampleBatch.UNROLL_ID]
        self.buffers[SampleBatch.UNROLL_ID][0].append(self.unroll_id)
        for (k, v) in values.items():
            if k not in self.buffers:
                if self.training and k.startswith('state_out'):
                    vr = self.view_requirements[k]
                    data_col = vr.data_col or k
                    self._fill_buffer_with_initial_values(data_col, vr, build_for_inference=False)
                else:
                    self._build_buffers({k: v})
            should_flatten_action_key = k == SampleBatch.ACTIONS and (not self.disable_action_flattening)
            should_flatten_state_key = k.startswith('state_out') and (not self._enable_new_api_stack)
            if k == SampleBatch.INFOS or should_flatten_state_key or should_flatten_action_key:
                if should_flatten_action_key:
                    v = flatten_to_single_ndarray(v)
                if k in self.data_cols_with_dummy_values:
                    dummy = self.buffers[k][0].pop(-1)
                self.buffers[k][0].append(v)
                if k in self.data_cols_with_dummy_values:
                    self.buffers[k][0].append(dummy)
            else:
                flattened = tree.flatten(v)
                for (i, sub_list) in enumerate(self.buffers[k]):
                    if k in self.data_cols_with_dummy_values:
                        dummy = sub_list.pop(-1)
                    sub_list.append(flattened[i])
                    if k in self.data_cols_with_dummy_values:
                        sub_list.append(dummy)
        if not self.training:
            for k in self.buffers:
                for sub_list in self.buffers[k]:
                    if sub_list:
                        sub_list.pop(0)
        self.agent_steps += 1

    def build_for_inference(self) -> SampleBatch:
        if False:
            return 10
        'During inference, we will build a SampleBatch with a batch size of 1 that\n        can then be used to run the forward pass of a policy. This data will only\n        include the enviornment context for running the policy at the last timestep.\n\n        Returns:\n            A SampleBatch with a batch size of 1.\n        '
        batch_data = {}
        np_data = {}
        for (view_col, view_req) in self.view_requirements.items():
            data_col = view_req.data_col or view_col
            if not view_req.used_for_compute_actions:
                continue
            if np.any(view_req.shift_arr > 0):
                raise ValueError(f'During inference the agent can only use past observations to respect causality. However, view_col = {view_col} seems to depend on future indices {view_req.shift_arr}, while the used_for_compute_actions flag is set to True. Please fix the discrepancy. Hint: If you are using a custom model make sure the view_requirements are initialized properly and is point only refering to past timesteps during inference.')
            if data_col not in self.buffers:
                self._fill_buffer_with_initial_values(data_col, view_req, build_for_inference=True)
                self._prepare_for_data_cols_with_dummy_values(data_col)
            self._cache_in_np(np_data, data_col)
            data = []
            for d in np_data[data_col]:
                element_at_t = d[view_req.shift_arr + len(d) - 1]
                if element_at_t.shape[0] == 1:
                    data.append(element_at_t)
                    continue
                data.append(element_at_t[None])
            batch_data[view_col] = self._unflatten_as_buffer_struct(data, data_col)
        batch = self._get_sample_batch(batch_data)
        return batch

    def build_for_training(self, view_requirements: ViewRequirementsDict) -> SampleBatch:
        if False:
            i = 10
            return i + 15
        'Builds a SampleBatch from the thus-far collected agent data.\n\n        If the episode/trajectory has no TERMINATED|TRUNCATED=True at the end, will\n        copy the necessary n timesteps at the end of the trajectory back to the\n        beginning of the buffers and wait for new samples coming in.\n        SampleBatches created by this method will be ready for postprocessing\n        by a Policy.\n\n        Args:\n            view_requirements: The viewrequirements dict needed to build the\n            SampleBatch from the raw buffers (which may have data shifts as well as\n            mappings from view-col to data-col in them).\n\n        Returns:\n            SampleBatch: The built SampleBatch for this agent, ready to go into\n            postprocessing.\n        '
        batch_data = {}
        np_data = {}
        for (view_col, view_req) in view_requirements.items():
            data_col = view_req.data_col or view_col
            if data_col not in self.buffers:
                is_state = self._fill_buffer_with_initial_values(data_col, view_req, build_for_inference=False)
                if not is_state:
                    continue
            obs_shift = -1 if data_col in [SampleBatch.OBS, SampleBatch.INFOS] else 0
            self._cache_in_np(np_data, data_col)
            data = []
            for d in np_data[data_col]:
                shifted_data = []
                count = int(math.ceil((len(d) - int(data_col in self.data_cols_with_dummy_values) - self.shift_before) / view_req.batch_repeat_value))
                for i in range(count):
                    inds = self.shift_before + obs_shift + view_req.shift_arr + i * view_req.batch_repeat_value
                    if max(inds) < len(d):
                        element_at_t = d[inds]
                    else:
                        element_at_t = _get_buffered_slice_with_paddings(d, inds)
                        element_at_t = np.stack(element_at_t)
                    if element_at_t.shape[0] == 1:
                        element_at_t = element_at_t[0]
                    shifted_data.append(element_at_t)
                if shifted_data:
                    shifted_data_np = np.stack(shifted_data, 0)
                else:
                    shifted_data_np = np.array(shifted_data)
                data.append(shifted_data_np)
            batch_data[view_col] = self._unflatten_as_buffer_struct(data, data_col)
        batch = self._get_sample_batch(batch_data)
        if SampleBatch.TERMINATEDS in self.buffers and (not self.buffers[SampleBatch.TERMINATEDS][0][-1]) and (SampleBatch.TRUNCATEDS in self.buffers) and (not self.buffers[SampleBatch.TRUNCATEDS][0][-1]):
            if self.shift_before > 0:
                for (k, data) in self.buffers.items():
                    for i in range(len(data)):
                        self.buffers[k][i] = data[i][-self.shift_before:]
            self.agent_steps = 0
        self.unroll_id = None
        return batch

    def _build_buffers(self, single_row: Dict[str, TensorType]) -> None:
        if False:
            while True:
                i = 10
        'Builds the buffers for sample collection, given an example data row.\n\n        Args:\n            single_row (Dict[str, TensorType]): A single row (keys=column\n                names) of data to base the buffers on.\n        '
        for (col, data) in single_row.items():
            if col in self.buffers:
                continue
            shift = self.shift_before - (1 if col in [SampleBatch.OBS, SampleBatch.INFOS, SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.ENV_ID, SampleBatch.T, SampleBatch.UNROLL_ID] else 0)
            should_flatten_action_key = col == SampleBatch.ACTIONS and (not self.disable_action_flattening)
            should_flatten_state_key = col.startswith('state_out') and (not self._enable_new_api_stack)
            if col == SampleBatch.INFOS or should_flatten_state_key or should_flatten_action_key:
                if should_flatten_action_key:
                    data = flatten_to_single_ndarray(data)
                self.buffers[col] = [[data for _ in range(shift)]]
            else:
                self.buffers[col] = [[v for _ in range(shift)] for v in tree.flatten(data)]
                self.buffer_structs[col] = data

    def _get_sample_batch(self, batch_data: Dict[str, TensorType]) -> SampleBatch:
        if False:
            for i in range(10):
                print('nop')
        'Returns a SampleBatch from the given data dictionary. Also updates the\n        sequence information based on the max_seq_len.'
        batch = SampleBatch(batch_data, is_training=self.training)
        if self.is_policy_recurrent:
            seq_lens = []
            max_seq_len = self.max_seq_len
            count = batch.count
            while count > 0:
                seq_lens.append(min(count, max_seq_len))
                count -= max_seq_len
            batch['seq_lens'] = np.array(seq_lens)
            batch.max_seq_len = max_seq_len
        return batch

    def _cache_in_np(self, cache_dict: Dict[str, List[np.ndarray]], key: str) -> None:
        if False:
            print('Hello World!')
        'Caches the numpy version of the key in the buffer dict.'
        if key not in cache_dict:
            cache_dict[key] = [_to_float_np_array(d) for d in self.buffers[key]]

    def _unflatten_as_buffer_struct(self, data: List[np.ndarray], key: str) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Unflattens the given to match the buffer struct format for that key.'
        if key not in self.buffer_structs:
            return data[0]
        return tree.unflatten_as(self.buffer_structs[key], data)

    def _fill_buffer_with_initial_values(self, data_col: str, view_requirement: ViewRequirement, build_for_inference: bool=False) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Fills the buffer with the initial values for the given data column.\n        for dat_col starting with `state_out`, use the initial states of the policy,\n        but for other data columns, create a dummy value based on the view requirement\n        space.\n\n        Args:\n            data_col: The data column to fill the buffer with.\n            view_requirement: The view requirement for the view_col. Normally the view\n                requirement for the data column is used and if it does not exist for\n                some reason the view requirement for view column is used instead.\n            build_for_inference: Whether this is getting called for inference or not.\n\n        returns:\n            is_state: True if the data_col is an RNN state, False otherwise.\n        '
        try:
            space = self.view_requirements[data_col].space
        except KeyError:
            space = view_requirement.space
        is_state = True
        if data_col.startswith('state_out'):
            if self._enable_new_api_stack:
                self._build_buffers({data_col: self.initial_states})
            else:
                if not self.is_policy_recurrent:
                    raise ValueError(f'{data_col} is not available, because the given policy isnot recurrent according to the input model_inital_states.Have you forgotten to return non-empty lists inpolicy.get_initial_states()?')
                state_ind = int(data_col.split('_')[-1])
                self._build_buffers({data_col: self.initial_states[state_ind]})
        else:
            is_state = False
            if build_for_inference:
                if isinstance(space, Space):
                    fill_value = get_dummy_batch_for_space(space, batch_size=0)
                else:
                    fill_value = space
                self._build_buffers({data_col: fill_value})
        return is_state

    def _prepare_for_data_cols_with_dummy_values(self, data_col):
        if False:
            while True:
                i = 10
        self.data_cols_with_dummy_values.add(data_col)
        for b in self.buffers[data_col]:
            b.append(b[-1])