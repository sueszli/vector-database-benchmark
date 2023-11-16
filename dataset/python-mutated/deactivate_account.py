import logging
from typing import TYPE_CHECKING, Optional
from synapse.api.errors import SynapseError
from synapse.handlers.device import DeviceHandler
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.types import Codes, Requester, UserID, create_requester
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class DeactivateAccountHandler:
    """Handler which deals with deactivating user accounts."""

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        self.store = hs.get_datastores().main
        self.hs = hs
        self._auth_handler = hs.get_auth_handler()
        self._device_handler = hs.get_device_handler()
        self._room_member_handler = hs.get_room_member_handler()
        self._identity_handler = hs.get_identity_handler()
        self._profile_handler = hs.get_profile_handler()
        self.user_directory_handler = hs.get_user_directory_handler()
        self._server_name = hs.hostname
        self._third_party_rules = hs.get_module_api_callbacks().third_party_event_rules
        self._user_parter_running = False
        self._third_party_rules = hs.get_module_api_callbacks().third_party_event_rules
        if hs.config.worker.run_background_tasks:
            hs.get_reactor().callWhenRunning(self._start_user_parting)
        self._account_validity_enabled = hs.config.account_validity.account_validity_enabled

    async def deactivate_account(self, user_id: str, erase_data: bool, requester: Requester, id_server: Optional[str]=None, by_admin: bool=False) -> bool:
        """Deactivate a user's account

        Args:
            user_id: ID of user to be deactivated
            erase_data: whether to GDPR-erase the user's data
            requester: The user attempting to make this change.
            id_server: Use the given identity server when unbinding
                any threepids. If None then will attempt to unbind using the
                identity server specified when binding (if known).
            by_admin: Whether this change was made by an administrator.

        Returns:
            True if identity server supports removing threepids, otherwise False.
        """
        assert isinstance(self._device_handler, DeviceHandler)
        if not await self._third_party_rules.check_can_deactivate_user(user_id, by_admin):
            raise SynapseError(403, 'Deactivation of this user is forbidden', Codes.FORBIDDEN)
        identity_server_supports_unbinding = True
        bound_threepids = await self.store.user_get_bound_threepids(user_id)
        for (medium, address) in bound_threepids:
            try:
                result = await self._identity_handler.try_unbind_threepid(user_id, medium, address, id_server)
            except Exception:
                logger.exception('Failed to remove threepid from ID server')
                raise SynapseError(400, 'Failed to remove threepid from ID server')
            identity_server_supports_unbinding &= result
        local_threepids = await self.store.user_get_threepids(user_id)
        for local_threepid in local_threepids:
            await self._auth_handler.delete_local_threepid(user_id, local_threepid.medium, local_threepid.address)
        await self._device_handler.delete_all_devices_for_user(user_id)
        await self._auth_handler.delete_access_tokens_for_user(user_id)
        await self.store.user_set_password_hash(user_id, None)
        await self.store.delete_all_pushers_for_user(user_id)
        await self.store.add_user_pending_deactivation(user_id)
        await self.user_directory_handler.handle_local_user_deactivated(user_id)
        if erase_data:
            user = UserID.from_string(user_id)
            await self._profile_handler.set_avatar_url(user, requester, '', by_admin, deactivation=True)
            await self._profile_handler.set_displayname(user, requester, '', by_admin, deactivation=True)
            logger.info('Marking %s as erased', user_id)
            await self.store.mark_user_erased(user_id)
        self._start_user_parting()
        await self._reject_pending_invites_for_user(user_id)
        if self._account_validity_enabled:
            await self.store.delete_account_validity_for_user(user_id)
        await self.store.set_user_deactivated_status(user_id, True)
        await self.store.purge_account_data_for_user(user_id)
        await self.store.bulk_delete_backup_keys_and_versions_for_user(user_id)
        await self._third_party_rules.on_user_deactivation_status_changed(user_id, True, by_admin)
        return identity_server_supports_unbinding

    async def _reject_pending_invites_for_user(self, user_id: str) -> None:
        """Reject pending invites addressed to a given user ID.

        Args:
            user_id: The user ID to reject pending invites for.
        """
        user = UserID.from_string(user_id)
        pending_invites = await self.store.get_invited_rooms_for_local_user(user_id)
        for room in pending_invites:
            try:
                await self._room_member_handler.update_membership(create_requester(user, authenticated_entity=self._server_name), user, room.room_id, 'leave', ratelimit=False, require_consent=False)
                logger.info('Rejected invite for deactivated user %r in room %r', user_id, room.room_id)
            except Exception:
                logger.exception('Failed to reject invite for user %r in room %r: ignoring and continuing', user_id, room.room_id)

    def _start_user_parting(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Start the process that goes through the table of users\n        pending deactivation, if it isn't already running.\n        "
        if not self._user_parter_running:
            run_as_background_process('user_parter_loop', self._user_parter_loop)

    async def _user_parter_loop(self) -> None:
        """Loop that parts deactivated users from rooms"""
        self._user_parter_running = True
        logger.info('Starting user parter')
        try:
            while True:
                user_id = await self.store.get_user_pending_deactivation()
                if user_id is None:
                    break
                logger.info('User parter parting %r', user_id)
                await self._part_user(user_id)
                await self.store.del_user_pending_deactivation(user_id)
                logger.info('User parter finished parting %r', user_id)
            logger.info('User parter finished: stopping')
        finally:
            self._user_parter_running = False

    async def _part_user(self, user_id: str) -> None:
        """Causes the given user_id to leave all the rooms they're joined to"""
        user = UserID.from_string(user_id)
        rooms_for_user = await self.store.get_rooms_for_user(user_id)
        for room_id in rooms_for_user:
            logger.info('User parter parting %r from %r', user_id, room_id)
            try:
                await self._room_member_handler.update_membership(create_requester(user, authenticated_entity=self._server_name), user, room_id, 'leave', ratelimit=False, require_consent=False)
            except Exception:
                logger.exception('Failed to part user %r from room %r: ignoring and continuing', user_id, room_id)

    async def activate_account(self, user_id: str) -> None:
        """
        Activate an account that was previously deactivated.

        This marks the user as active and not erased in the database, but does
        not attempt to rejoin rooms, re-add threepids, etc.

        If enabled, the user will be re-added to the user directory.

        The user will also need a password hash set to actually login.

        Args:
            user_id: ID of user to be re-activated
        """
        user = UserID.from_string(user_id)
        await self.store.mark_user_not_erased(user_id)
        await self.store.set_user_deactivated_status(user_id, False)
        await self._third_party_rules.on_user_deactivation_status_changed(user_id, False, True)
        profile = await self.store.get_profileinfo(user)
        await self.user_directory_handler.handle_local_profile_change(user_id, profile)