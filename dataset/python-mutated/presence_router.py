import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, TypeVar, Union
from typing_extensions import ParamSpec
from twisted.internet.defer import CancelledError
from synapse.api.presence import UserPresenceState
from synapse.util.async_helpers import delay_cancellation, maybe_awaitable
if TYPE_CHECKING:
    from synapse.server import HomeServer
GET_USERS_FOR_STATES_CALLBACK = Callable[[Iterable[UserPresenceState]], Awaitable[Dict[str, Set[UserPresenceState]]]]
GET_INTERESTED_USERS_CALLBACK = Callable[[str], Awaitable[Union[Set[str], str]]]
logger = logging.getLogger(__name__)
P = ParamSpec('P')
R = TypeVar('R')

def load_legacy_presence_router(hs: 'HomeServer') -> None:
    if False:
        i = 10
        return i + 15
    'Wrapper that loads a presence router module configured using the old\n    configuration, and registers the hooks they implement.\n    '
    if hs.config.server.presence_router_module_class is None:
        return
    module = hs.config.server.presence_router_module_class
    config = hs.config.server.presence_router_config
    api = hs.get_module_api()
    presence_router = module(config=config, module_api=api)
    presence_router_methods = {'get_users_for_states', 'get_interested_users'}

    def async_wrapper(f: Optional[Callable[P, R]]) -> Optional[Callable[P, Awaitable[R]]]:
        if False:
            return 10
        if f is None:
            return None

        def run(*args: P.args, **kwargs: P.kwargs) -> Awaitable[R]:
            if False:
                print('Hello World!')
            assert f is not None
            return maybe_awaitable(f(*args, **kwargs))
        return run
    hooks: Dict[str, Optional[Callable[..., Any]]] = {hook: async_wrapper(getattr(presence_router, hook, None)) for hook in presence_router_methods}
    api.register_presence_router_callbacks(**hooks)

class PresenceRouter:
    """
    A module that the homeserver will call upon to help route user presence updates to
    additional destinations.
    """
    ALL_USERS = 'ALL'

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        self._get_users_for_states_callbacks: List[GET_USERS_FOR_STATES_CALLBACK] = []
        self._get_interested_users_callbacks: List[GET_INTERESTED_USERS_CALLBACK] = []

    def register_presence_router_callbacks(self, get_users_for_states: Optional[GET_USERS_FOR_STATES_CALLBACK]=None, get_interested_users: Optional[GET_INTERESTED_USERS_CALLBACK]=None) -> None:
        if False:
            i = 10
            return i + 15
        paired_methods = [get_users_for_states, get_interested_users]
        if paired_methods.count(None) == 1:
            raise RuntimeError('PresenceRouter modules must register neither or both of the paired callbacks: [get_users_for_states, get_interested_users]')
        if get_users_for_states is not None:
            self._get_users_for_states_callbacks.append(get_users_for_states)
        if get_interested_users is not None:
            self._get_interested_users_callbacks.append(get_interested_users)

    async def get_users_for_states(self, state_updates: Iterable[UserPresenceState]) -> Dict[str, Set[UserPresenceState]]:
        """
        Given an iterable of user presence updates, determine where each one
        needs to go.

        Args:
            state_updates: An iterable of user presence state updates.

        Returns:
          A dictionary of user_id -> set of UserPresenceState, indicating which
          presence updates each user should receive.
        """
        if len(self._get_users_for_states_callbacks) == 0:
            return {}
        users_for_states: Dict[str, Set[UserPresenceState]] = {}
        for callback in self._get_users_for_states_callbacks:
            try:
                result: object = await delay_cancellation(callback(state_updates))
            except CancelledError:
                raise
            except Exception as e:
                logger.warning('Failed to run module API callback %s: %s', callback, e)
                continue
            if not isinstance(result, Dict):
                logger.warning('Wrong type returned by module API callback %s: %s, expected Dict', callback, result)
                continue
            for (key, new_entries) in result.items():
                if not isinstance(new_entries, Set):
                    logger.warning('Wrong type returned by module API callback %s: %s, expected Set', callback, new_entries)
                    break
                users_for_states.setdefault(key, set()).update(new_entries)
        return users_for_states

    async def get_interested_users(self, user_id: str) -> Union[Set[str], str]:
        """
        Retrieve a list of users that `user_id` is interested in receiving the
        presence of. This will be in addition to those they share a room with.
        Optionally, the object PresenceRouter.ALL_USERS can be returned to indicate
        that this user should receive all incoming local and remote presence updates.

        Note that this method will only be called for local users, but can return users
        that are local or remote.

        Args:
            user_id: A user requesting presence updates.

        Returns:
            A set of user IDs to return presence updates for, or ALL_USERS to return all
            known updates.
        """
        if len(self._get_interested_users_callbacks) == 0:
            return set()
        interested_users = set()
        for callback in self._get_interested_users_callbacks:
            try:
                result = await delay_cancellation(callback(user_id))
            except CancelledError:
                raise
            except Exception as e:
                logger.warning('Failed to run module API callback %s: %s', callback, e)
                continue
            if result == PresenceRouter.ALL_USERS:
                return PresenceRouter.ALL_USERS
            if not isinstance(result, Set):
                logger.warning('Wrong type returned by module API callback %s: %s, expected set', callback, result)
                continue
            interested_users.update(result)
        return interested_users