from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any
from ...core.serialization import Serializer
from ...document.callbacks import invoke_with_curdoc
from ...document.json import PatchJson
from ..message import Message
if TYPE_CHECKING:
    from ...core.has_props import Setter
    from ...document.document import Document
    from ...document.events import DocumentPatchedEvent
__all__ = ('patch_doc',)

class patch_doc(Message[PatchJson]):
    """ Define the ``PATCH-DOC`` message for sending Document patch events
    between remote documents.

    The ``content`` fragment of for this message is has the form:

    .. code-block:: python

        {
            'events'     : <protocol document events>
            'references' : <model references>
        }

    """
    msgtype = 'PATCH-DOC'

    @classmethod
    def create(cls, events: list[DocumentPatchedEvent], **metadata: Any) -> patch_doc:
        if False:
            return 10
        ' Create a ``PATCH-DOC`` message\n\n        Args:\n            events (list) :\n                A list of patch events to apply to a document\n\n        Any additional keyword arguments will be put into the message\n        ``metadata`` fragment as-is.\n\n        '
        header = cls.create_header()
        if not events:
            raise ValueError('PATCH-DOC message requires at least one event')
        docs = {event.document for event in events}
        if len(docs) != 1:
            raise ValueError('PATCH-DOC message configured with events for more than one document')
        [doc] = docs
        serializer = Serializer(references=doc.models.synced_references)
        patch_json = PatchJson(events=serializer.encode(events))
        doc.models.flush_synced(lambda model: not serializer.has_ref(model))
        msg = cls(header, metadata, patch_json)
        for buffer in serializer.buffers:
            msg.add_buffer(buffer)
        return msg

    def apply_to_document(self, doc: Document, setter: Setter | None=None) -> None:
        if False:
            return 10
        '\n\n        '
        invoke_with_curdoc(doc, lambda : doc.apply_json_patch(self.payload, setter=setter))