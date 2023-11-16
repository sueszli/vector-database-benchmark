from typing import Any, Optional
from synapse.types import JsonDict, UserID
from ._base import Config

class ServerNoticesConfig(Config):
    """Configuration for the server notices room.

    Attributes:
        server_notices_mxid (str|None):
            The MXID to use for server notices.
            None if server notices are not enabled.

        server_notices_mxid_display_name (str|None):
            The display name to use for the server notices user.
            None if server notices are not enabled.

        server_notices_mxid_avatar_url (str|None):
            The MXC URL for the avatar of the server notices user.
            None if server notices are not enabled.

        server_notices_room_name (str|None):
            The name to use for the server notices room.
            None if server notices are not enabled.
    """
    section = 'servernotices'

    def __init__(self, *args: Any):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self.server_notices_mxid: Optional[str] = None
        self.server_notices_mxid_display_name: Optional[str] = None
        self.server_notices_mxid_avatar_url: Optional[str] = None
        self.server_notices_room_name: Optional[str] = None

    def read_config(self, config: JsonDict, **kwargs: Any) -> None:
        if False:
            return 10
        c = config.get('server_notices')
        if c is None:
            return
        mxid_localpart = c['system_mxid_localpart']
        self.server_notices_mxid = UserID(mxid_localpart, self.root.server.server_name).to_string()
        self.server_notices_mxid_display_name = c.get('system_mxid_display_name', None)
        self.server_notices_mxid_avatar_url = c.get('system_mxid_avatar_url', None)
        self.server_notices_room_name = c.get('room_name', 'Server Notices')