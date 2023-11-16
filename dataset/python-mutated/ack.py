from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ..message import Empty, Message
__all__ = ('ack',)

class ack(Message[Empty]):
    """ Define the ``ACK`` message for acknowledging successful client
    connection to a Bokeh server.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'ACK'

    @classmethod
    def create(cls, **metadata: Any) -> ack:
        if False:
            for i in range(10):
                print('nop')
        ' Create an ``ACK`` message\n\n        Any keyword arguments will be put into the message ``metadata``\n        fragment as-is.\n\n        '
        header = cls.create_header()
        content = Empty()
        return cls(header, metadata, content)