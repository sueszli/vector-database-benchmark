from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto
from google.cloud.support_v2.types import attachment
__protobuf__ = proto.module(package='google.cloud.support.v2', manifest={'ListAttachmentsRequest', 'ListAttachmentsResponse'})

class ListAttachmentsRequest(proto.Message):
    """The request message for the ListAttachments endpoint.

    Attributes:
        parent (str):
            Required. The resource name of Case object
            for which attachments should be listed.
        page_size (int):
            The maximum number of attachments fetched
            with each request. If not provided, the default
            is 10. The maximum page size that will be
            returned is 100.
        page_token (str):
            A token identifying the page of results to
            return. If unspecified, the first page is
            retrieved.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)

class ListAttachmentsResponse(proto.Message):
    """The response message for the ListAttachments endpoint.

    Attributes:
        attachments (MutableSequence[google.cloud.support_v2.types.Attachment]):
            The list of attachments associated with the
            given case.
        next_page_token (str):
            A token to retrieve the next page of results. This should be
            set in the ``page_token`` field of subsequent
            ``cases.attachments.list`` requests. If unspecified, there
            are no more results to retrieve.
    """

    @property
    def raw_page(self):
        if False:
            return 10
        return self
    attachments: MutableSequence[attachment.Attachment] = proto.RepeatedField(proto.MESSAGE, number=1, message=attachment.Attachment)
    next_page_token: str = proto.Field(proto.STRING, number=2)
__all__ = tuple(sorted(__protobuf__.manifest))