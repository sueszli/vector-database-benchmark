from collections import defaultdict
import logging
import pickle
from typing import Any
import numpy as np
from ray.rllib.utils.annotations import override
import tree
from ray.rllib.connectors.connector import AgentConnector, Connector, ConnectorContext
from ray import cloudpickle
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.models.base import STATE_OUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ActionConnectorDataType, AgentConnectorDataType
from ray.util.annotations import PublicAPI
logger = logging.getLogger(__name__)

@PublicAPI(stability='alpha')
class StateBufferConnector(AgentConnector):

    def __init__(self, ctx: ConnectorContext, states: Any=None):
        if False:
            print('Hello World!')
        super().__init__(ctx)
        self._initial_states = ctx.initial_states
        self._action_space_struct = get_base_struct_from_space(ctx.action_space)
        self._states = defaultdict(lambda : defaultdict(lambda : (None, None, None)))
        self._enable_new_api_stack = ctx.config.get('_enable_new_api_stack', False)
        if states:
            try:
                self._states = cloudpickle.loads(states)
            except pickle.UnpicklingError:
                logger.info('Can not restore StateBufferConnector states. This warning can usually be ignore, unless it is from restoring a stashed policy.')

    @override(Connector)
    def in_eval(self):
        if False:
            while True:
                i = 10
        super().in_eval()

    def reset(self, env_id: str):
        if False:
            while True:
                i = 10
        if env_id in self._states:
            del self._states[env_id]

    def on_policy_output(self, ac_data: ActionConnectorDataType):
        if False:
            return 10
        self._states[ac_data.env_id][ac_data.agent_id] = ac_data.output

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        if False:
            i = 10
            return i + 15
        d = ac_data.data
        assert type(d) == dict, 'Single agent data must be of type Dict[str, TensorStructType]'
        env_id = ac_data.env_id
        agent_id = ac_data.agent_id
        assert env_id is not None and agent_id is not None, f'StateBufferConnector requires env_id(f{env_id}) and agent_id(f{agent_id})'
        (action, states, fetches) = self._states[env_id][agent_id]
        if action is not None:
            d[SampleBatch.ACTIONS] = action
        else:
            d[SampleBatch.ACTIONS] = tree.map_structure(lambda s: np.zeros_like(s.sample(), s.dtype) if hasattr(s, 'dtype') else np.zeros_like(s.sample()), self._action_space_struct)
        if states is None:
            states = self._initial_states
        if self._enable_new_api_stack:
            if states:
                d[STATE_OUT] = states
        else:
            for (i, v) in enumerate(states):
                d['state_out_{}'.format(i)] = v
        if fetches:
            d.update(fetches)
        return ac_data

    def to_state(self):
        if False:
            i = 10
            return i + 15
        states = cloudpickle.dumps(self._states)
        return (StateBufferConnector.__name__, states)

    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        if False:
            return 10
        return StateBufferConnector(ctx, params)
register_connector(StateBufferConnector.__name__, StateBufferConnector)