from typing import Any, Optional
import attr
from synapse.api.constants import PresenceState
from synapse.types import JsonDict

@attr.s(slots=True, auto_attribs=True)
class UserDevicePresenceState:
    """
    Represents the current presence state of a user's device.

    user_id: The user ID.
    device_id: The user's device ID.
    state: The presence state, see PresenceState.
    last_active_ts: Time in msec that the device last interacted with server.
    last_sync_ts: Time in msec that the device last *completed* a sync
        (or event stream).
    """
    user_id: str
    device_id: Optional[str]
    state: str
    last_active_ts: int
    last_sync_ts: int

    @classmethod
    def default(cls, user_id: str, device_id: Optional[str]) -> 'UserDevicePresenceState':
        if False:
            for i in range(10):
                print('nop')
        'Returns a default presence state.'
        return cls(user_id=user_id, device_id=device_id, state=PresenceState.OFFLINE, last_active_ts=0, last_sync_ts=0)

@attr.s(slots=True, frozen=True, auto_attribs=True)
class UserPresenceState:
    """Represents the current presence state of the user.

    user_id: The user ID.
    state: The presence state, see PresenceState.
    last_active_ts: Time in msec that the user last interacted with server.
    last_federation_update_ts: Time in msec since either a) we sent a presence
        update to other servers or b) we received a presence update, depending
        on if is a local user or not.
    last_user_sync_ts: Time in msec that the user last *completed* a sync
        (or event stream).
    status_msg: User set status message.
    currently_active: True if the user is currently syncing.
    """
    user_id: str
    state: str
    last_active_ts: int
    last_federation_update_ts: int
    last_user_sync_ts: int
    status_msg: Optional[str]
    currently_active: bool

    def as_dict(self) -> JsonDict:
        if False:
            return 10
        return attr.asdict(self)

    def copy_and_replace(self, **kwargs: Any) -> 'UserPresenceState':
        if False:
            for i in range(10):
                print('nop')
        return attr.evolve(self, **kwargs)

    @classmethod
    def default(cls, user_id: str) -> 'UserPresenceState':
        if False:
            print('Hello World!')
        'Returns a default presence state.'
        return cls(user_id=user_id, state=PresenceState.OFFLINE, last_active_ts=0, last_federation_update_ts=0, last_user_sync_ts=0, status_msg=None, currently_active=False)