from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from synapse.server import HomeServer

class WorkerServerNoticesSender:
    """Stub impl of ServerNoticesSender which does nothing"""

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        pass

    async def on_user_syncing(self, user_id: str) -> None:
        """Called when the user performs a sync operation.

        Args:
            user_id: mxid of user who synced
        """
        return None

    async def on_user_ip(self, user_id: str) -> None:
        """Called on the master when a worker process saw a client request.

        Args:
            user_id: mxid
        """
        raise AssertionError('on_user_ip unexpectedly called on worker')