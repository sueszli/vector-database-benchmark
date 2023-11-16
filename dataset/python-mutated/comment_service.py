from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto
from google.cloud.support_v2.types import comment as gcs_comment
__protobuf__ = proto.module(package='google.cloud.support.v2', manifest={'ListCommentsRequest', 'ListCommentsResponse', 'CreateCommentRequest'})

class ListCommentsRequest(proto.Message):
    """The request message for the ListComments endpoint.

    Attributes:
        parent (str):
            Required. The resource name of Case object
            for which comments should be listed.
        page_size (int):
            The maximum number of comments fetched with
            each request. Defaults to 10.
        page_token (str):
            A token identifying the page of results to
            return. If unspecified, the first page is
            retrieved.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=4)
    page_token: str = proto.Field(proto.STRING, number=5)

class ListCommentsResponse(proto.Message):
    """The response message for the ListComments endpoint.

    Attributes:
        comments (MutableSequence[google.cloud.support_v2.types.Comment]):
            The list of Comments associated with the
            given Case.
        next_page_token (str):
            A token to retrieve the next page of results. This should be
            set in the ``page_token`` field of subsequent
            ``ListCommentsRequest`` message that is issued. If
            unspecified, there are no more results to retrieve.
    """

    @property
    def raw_page(self):
        if False:
            print('Hello World!')
        return self
    comments: MutableSequence[gcs_comment.Comment] = proto.RepeatedField(proto.MESSAGE, number=1, message=gcs_comment.Comment)
    next_page_token: str = proto.Field(proto.STRING, number=2)

class CreateCommentRequest(proto.Message):
    """The request message for CreateComment endpoint.

    Attributes:
        parent (str):
            Required. The resource name of Case to which
            this comment should be added.
        comment (google.cloud.support_v2.types.Comment):
            Required. The Comment object to be added to
            this Case.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    comment: gcs_comment.Comment = proto.Field(proto.MESSAGE, number=2, message=gcs_comment.Comment)
__all__ = tuple(sorted(__protobuf__.manifest))