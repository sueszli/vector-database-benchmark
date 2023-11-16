from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, TypedDict
from ...core.types import ID
from ..exceptions import ProtocolError
from ..message import Message
if TYPE_CHECKING:
    from ...document.document import DocJson, Document
__all__ = ('pull_doc_reply',)

class PullDoc(TypedDict):
    doc: DocJson

class pull_doc_reply(Message[PullDoc]):
    """ Define the ``PULL-DOC-REPLY`` message for replying to Document pull
    requests from clients

    The ``content`` fragment of for this message is has the form:

    .. code-block:: python

        {
            'doc' : <Document JSON>
        }

    """
    msgtype = 'PULL-DOC-REPLY'

    @classmethod
    def create(cls, request_id: ID, document: Document, **metadata: Any) -> pull_doc_reply:
        if False:
            return 10
        ' Create an ``PULL-DOC-REPLY`` message\n\n        Args:\n            request_id (str) :\n                The message ID for the message that issues the pull request\n\n            document (Document) :\n                The Document to reply with\n\n        Any additional keyword arguments will be put into the message\n        ``metadata`` fragment as-is.\n\n        '
        header = cls.create_header(request_id=request_id)
        content = PullDoc(doc=document.to_json())
        msg = cls(header, metadata, content)
        return msg

    def push_to_document(self, doc: Document) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'doc' not in self.content:
            raise ProtocolError('No doc in PULL-DOC-REPLY')
        doc.replace_with_json(self.content['doc'])