from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ..message import Empty, Message
__all__ = ('server_info_req',)

class server_info_req(Message[Empty]):
    """ Define the ``SERVER-INFO-REQ`` message for requesting a Bokeh server
    provide information about itself.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'SERVER-INFO-REQ'

    @classmethod
    def create(cls, **metadata: Any) -> server_info_req:
        if False:
            for i in range(10):
                print('nop')
        ' Create an ``SERVER-INFO-REQ`` message\n\n        Any keyword arguments will be put into the message ``metadata``\n        fragment as-is.\n\n        '
        header = cls.create_header()
        content = Empty()
        return cls(header, metadata, content)