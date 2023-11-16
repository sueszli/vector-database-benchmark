from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ...core.types import ID
from ..message import Empty, Message
__all__ = ('ok',)

class ok(Message[Empty]):
    """ Define the ``OK`` message for acknowledging successful handling of a
    previous message.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'OK'

    @classmethod
    def create(cls, request_id: ID, **metadata: Any) -> ok:
        if False:
            return 10
        ' Create an ``OK`` message\n\n        Args:\n            request_id (str) :\n                The message ID for the message the precipitated the OK.\n\n        Any additional keyword arguments will be put into the message\n        ``metadata`` fragment as-is.\n\n        '
        header = cls.create_header(request_id=request_id)
        return cls(header, metadata, Empty())