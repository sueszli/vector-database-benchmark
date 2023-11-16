from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ..message import Empty, Message
__all__ = ('pull_doc_req',)

class pull_doc_req(Message[Empty]):
    """ Define the ``PULL-DOC-REQ`` message for requesting a Bokeh server reply
    with a new Bokeh Document.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'PULL-DOC-REQ'

    @classmethod
    def create(cls, **metadata: Any) -> pull_doc_req:
        if False:
            print('Hello World!')
        ' Create an ``PULL-DOC-REQ`` message\n\n        Any keyword arguments will be put into the message ``metadata``\n        fragment as-is.\n\n        '
        header = cls.create_header()
        return cls(header, metadata, Empty())