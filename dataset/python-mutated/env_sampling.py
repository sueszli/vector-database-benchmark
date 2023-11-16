from typing import Any
from ray.rllib.connectors.connector import AgentConnector, ConnectorContext
from ray.rllib.connectors.registry import register_connector
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.util.annotations import PublicAPI

@PublicAPI(stability='alpha')
class EnvSamplingAgentConnector(AgentConnector):

    def __init__(self, ctx: ConnectorContext, sign=False, limit=None):
        if False:
            return 10
        super().__init__(ctx)
        self.observation_space = ctx.observation_space

    def transform(self, ac_data: AgentConnectorDataType) -> AgentConnectorDataType:
        if False:
            i = 10
            return i + 15
        return ac_data

    def to_state(self):
        if False:
            print('Hello World!')
        return (EnvSamplingAgentConnector.__name__, {})

    @staticmethod
    def from_state(ctx: ConnectorContext, params: Any):
        if False:
            return 10
        return EnvSamplingAgentConnector(ctx, **params)
register_connector(EnvSamplingAgentConnector.__name__, EnvSamplingAgentConnector)