import logging
from typing import TYPE_CHECKING, Optional
from synapse.api.errors import Codes, StoreError, SynapseError
from synapse.handlers.device import DeviceHandler
from synapse.types import Requester
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class SetPasswordHandler:
    """Handler which deals with changing user account passwords"""

    def __init__(self, hs: 'HomeServer'):
        if False:
            return 10
        self.store = hs.get_datastores().main
        self._auth_handler = hs.get_auth_handler()
        device_handler = hs.get_device_handler()
        assert isinstance(device_handler, DeviceHandler)
        self._device_handler = device_handler

    async def set_password(self, user_id: str, password_hash: str, logout_devices: bool, requester: Optional[Requester]=None) -> None:
        if not self._auth_handler.can_change_password():
            raise SynapseError(403, 'Password change disabled', errcode=Codes.FORBIDDEN)
        try:
            await self.store.user_set_password_hash(user_id, password_hash)
        except StoreError as e:
            if e.code == 404:
                raise SynapseError(404, 'Unknown user', Codes.NOT_FOUND)
            raise e
        if logout_devices:
            except_device_id = requester.device_id if requester else None
            except_access_token_id = requester.access_token_id if requester else None
            await self._device_handler.delete_all_devices_for_user(user_id, except_device_id=except_device_id)
            await self._auth_handler.delete_access_tokens_for_user(user_id, except_token_id=except_access_token_id)