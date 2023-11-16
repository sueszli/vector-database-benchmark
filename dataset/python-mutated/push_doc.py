from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, TypedDict
from ..exceptions import ProtocolError
from ..message import Message
if TYPE_CHECKING:
    from ...document.document import DocJson, Document
__all__ = ('push_doc',)

class PushDoc(TypedDict):
    doc: DocJson

class push_doc(Message[PushDoc]):
    """ Define the ``PUSH-DOC`` message for pushing Documents from clients to a
    Bokeh server.

    The ``content`` fragment of for this message is has the form:

    .. code-block:: python

        {
            'doc' : <Document JSON>
        }

    """
    msgtype = 'PUSH-DOC'

    @classmethod
    def create(cls, document: Document, **metadata: Any) -> push_doc:
        if False:
            while True:
                i = 10
        '\n\n        '
        header = cls.create_header()
        content = PushDoc(doc=document.to_json())
        msg = cls(header, metadata, content)
        return msg

    def push_to_document(self, doc: Document) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Raises:\n            ProtocolError\n\n        '
        if 'doc' not in self.content:
            raise ProtocolError('No doc in PUSH-DOC')
        doc.replace_with_json(self.content['doc'])