from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, TypedDict
from bokeh import __version__
from ...core.types import ID
from ..message import Message
__all__ = ('server_info_reply',)

class VersionInfo(TypedDict):
    bokeh: str
    server: str

class ServerInfo(TypedDict):
    version_info: VersionInfo

class server_info_reply(Message[ServerInfo]):
    """ Define the ``SERVER-INFO-REPLY`` message for replying to Server info
    requests from clients.

    The ``content`` fragment of for this message is has the form:

    .. code-block:: python

        {
            'version_info' : {
                'bokeh'  : <bokeh library version>
                'server' : <bokeh server version>
            }
        }

    """
    msgtype = 'SERVER-INFO-REPLY'

    @classmethod
    def create(cls, request_id: ID, **metadata: Any) -> server_info_reply:
        if False:
            while True:
                i = 10
        ' Create an ``SERVER-INFO-REPLY`` message\n\n        Args:\n            request_id (str) :\n                The message ID for the message that issues the info request\n\n        Any additional keyword arguments will be put into the message\n        ``metadata`` fragment as-is.\n\n        '
        header = cls.create_header(request_id=request_id)
        content = ServerInfo(version_info=_VERSION_INFO)
        return cls(header, metadata, content)
_VERSION_INFO = VersionInfo(bokeh=__version__, server=__version__)