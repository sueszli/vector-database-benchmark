import logging
from datetime import datetime
from manticore.core.plugin import IntrospectionAPIPlugin, StateDescriptor
from manticore.core.state import StateBase
from manticore.utils.enums import StateLists
logger = logging.getLogger(__name__)

class ManticoreServerIntrospectionPlugin(IntrospectionAPIPlugin):
    NAME = 'ManticoreServerIntrospectionPlugin'

    @property
    def name(self) -> str:
        if False:
            return 10
        return 'ManticoreServerIntrospectionPlugin'

    def create_state(self, state_id: int):
        if False:
            for i in range(10):
                print('nop')
        'Override create_state to force a state update right after creation.\n        This is helpful when retrieving info from a state yet to execute.'
        super().create_state(state_id)
        state = self.manticore._load(state_id)
        self._force_update_state_descriptor(state)

    def will_fork_state_callback(self, state: StateBase, expression, solutions, policy):
        if False:
            i = 10
            return i + 15
        self._force_update_state_descriptor(state)

    def will_transition_state_callback(self, state_id: int, from_list: StateLists, to_list: StateLists):
        if False:
            while True:
                i = 10
        state = self.manticore._load(state_id)
        self._force_update_state_descriptor(state)

    def _force_update_state_descriptor(self, state: StateBase):
        if False:
            for i in range(10):
                print('nop')
        'Force a given state to update its information, which can include the current PC, etc.\n        Calling _update_state_descriptor directly may become an issue if specific state implementations\n        start to require additional arguments for this method.'
        with self.locked_context('manticore_state', dict) as context:
            state._update_state_descriptor(context.setdefault(state.id, StateDescriptor(state_id=state.id)))
            context[state.id].last_intermittent_update = datetime.now()

    def did_terminate_worker_callback(self, worker_id: int):
        if False:
            for i in range(10):
                print('nop')
        logger.debug(f'worker exits (id: {worker_id})')