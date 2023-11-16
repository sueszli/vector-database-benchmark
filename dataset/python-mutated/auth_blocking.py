import logging
from typing import TYPE_CHECKING, Optional
from synapse.api.constants import LimitBlockingTypes, UserTypes
from synapse.api.errors import Codes, ResourceLimitError
from synapse.config.server import is_threepid_reserved
from synapse.types import Requester
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)

class AuthBlocking:

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main
        self._server_notices_mxid = hs.config.servernotices.server_notices_mxid
        self._hs_disabled = hs.config.server.hs_disabled
        self._hs_disabled_message = hs.config.server.hs_disabled_message
        self._admin_contact = hs.config.server.admin_contact
        self._max_mau_value = hs.config.server.max_mau_value
        self._limit_usage_by_mau = hs.config.server.limit_usage_by_mau
        self._mau_limits_reserved_threepids = hs.config.server.mau_limits_reserved_threepids
        self._is_mine_server_name = hs.is_mine_server_name
        self._track_appservice_user_ips = hs.config.appservice.track_appservice_user_ips

    async def check_auth_blocking(self, user_id: Optional[str]=None, threepid: Optional[dict]=None, user_type: Optional[str]=None, requester: Optional[Requester]=None) -> None:
        """Checks if the user should be rejected for some external reason,
        such as monthly active user limiting or global disable flag

        Args:
            user_id: If present, checks for presence against existing
                MAU cohort

            threepid: If present, checks for presence against configured
                reserved threepid. Used in cases where the user is trying register
                with a MAU blocked server, normally they would be rejected but their
                threepid is on the reserved list. user_id and
                threepid should never be set at the same time.

            user_type: If present, is used to decide whether to check against
                certain blocking reasons like MAU.

            requester: If present, and the authenticated entity is a user, checks for
                presence against existing MAU cohort. Passing in both a `user_id` and
                `requester` is an error.
        """
        if requester and user_id:
            raise Exception("Passed in both 'user_id' and 'requester' to 'check_auth_blocking'")
        if requester:
            if requester.authenticated_entity.startswith('@'):
                user_id = requester.authenticated_entity
            elif self._is_mine_server_name(requester.authenticated_entity):
                return
            if requester.app_service and (not self._track_appservice_user_ips):
                return
        if user_id is not None:
            if user_id == self._server_notices_mxid:
                return
            if await self.store.is_support_user(user_id):
                return
        if self._hs_disabled:
            raise ResourceLimitError(403, self._hs_disabled_message, errcode=Codes.RESOURCE_LIMIT_EXCEEDED, admin_contact=self._admin_contact, limit_type=LimitBlockingTypes.HS_DISABLED)
        if self._limit_usage_by_mau is True:
            assert not (user_id and threepid)
            if user_id:
                timestamp = await self.store.user_last_seen_monthly_active(user_id)
                if timestamp:
                    return
                is_trial = await self.store.is_trial_user(user_id)
                if is_trial:
                    return
            elif threepid:
                if is_threepid_reserved(self._mau_limits_reserved_threepids, threepid):
                    return
            elif user_type == UserTypes.SUPPORT:
                return
            current_mau = await self.store.get_monthly_active_count()
            if current_mau >= self._max_mau_value:
                raise ResourceLimitError(403, 'Monthly Active User Limit Exceeded', admin_contact=self._admin_contact, errcode=Codes.RESOURCE_LIMIT_EXCEEDED, limit_type=LimitBlockingTypes.MONTHLY_ACTIVE_USER)