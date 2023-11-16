from typing import TYPE_CHECKING, Iterable, Union
from synapse.server_notices.consent_server_notices import ConsentServerNotices
from synapse.server_notices.resource_limits_server_notices import ResourceLimitsServerNotices
from synapse.server_notices.worker_server_notices_sender import WorkerServerNoticesSender
if TYPE_CHECKING:
    from synapse.server import HomeServer

class ServerNoticesSender(WorkerServerNoticesSender):
    """A centralised place which sends server notices automatically when
    Certain Events take place
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(hs)
        self._server_notices: Iterable[Union[ConsentServerNotices, ResourceLimitsServerNotices]] = (ConsentServerNotices(hs), ResourceLimitsServerNotices(hs))

    async def on_user_syncing(self, user_id: str) -> None:
        """Called when the user performs a sync operation.

        Args:
            user_id: mxid of user who synced
        """
        for sn in self._server_notices:
            await sn.maybe_send_server_notice_to_user(user_id)

    async def on_user_ip(self, user_id: str) -> None:
        """Called on the master when a worker process saw a client request.

        Args:
            user_id: mxid
        """
        for sn in self._server_notices:
            await sn.maybe_send_server_notice_to_user(user_id)