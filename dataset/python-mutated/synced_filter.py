from ray.rllib.connectors.connector import AgentConnector, ConnectorContext
from ray.util.annotations import PublicAPI
from ray.rllib.utils.filter import Filter

@PublicAPI(stability='alpha')
class SyncedFilterAgentConnector(AgentConnector):
    """An agent connector that filters with synchronized parameters."""

    def __init__(self, ctx: ConnectorContext, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(ctx)
        if args or kwargs:
            raise ValueError('SyncedFilterAgentConnector does not take any additional arguments, but got args=`{}` and kwargs={}.'.format(args, kwargs))

    def apply_changes(self, other: 'Filter', *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Updates self with state from other filter.'
        return self.filter.apply_changes(other, *args, **kwargs)

    def copy(self) -> 'Filter':
        if False:
            i = 10
            return i + 15
        'Creates a new object with same state as self.\n\n        This is a legacy Filter method that we need to keep around for now\n\n        Returns:\n            A copy of self.\n        '
        return self.filter.copy()

    def sync(self, other: 'AgentConnector') -> None:
        if False:
            for i in range(10):
                print('nop')
        'Copies all state from other filter to self.'
        return self.filter.sync(other.filter)

    def reset_state(self) -> None:
        if False:
            while True:
                i = 10
        'Creates copy of current state and resets accumulated state'
        raise NotImplementedError

    def as_serializable(self) -> 'Filter':
        if False:
            return 10
        return self.filter.as_serializable()