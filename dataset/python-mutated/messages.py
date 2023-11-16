from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from typing_extensions import Literal
import httpx
from .files import Files, AsyncFiles, FilesWithRawResponse, AsyncFilesWithRawResponse
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import maybe_transform
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .....pagination import SyncCursorPage, AsyncCursorPage
from ....._base_client import AsyncPaginator, make_request_options
from .....types.beta.threads import ThreadMessage, message_list_params, message_create_params, message_update_params
if TYPE_CHECKING:
    from ....._client import OpenAI, AsyncOpenAI
__all__ = ['Messages', 'AsyncMessages']

class Messages(SyncAPIResource):
    files: Files
    with_raw_response: MessagesWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            return 10
        super().__init__(client)
        self.files = Files(client)
        self.with_raw_response = MessagesWithRawResponse(self)

    def create(self, thread_id: str, *, content: str, role: Literal['user'], file_ids: List[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        if False:
            print('Hello World!')
        '\n        Create a message.\n\n        Args:\n          content: The content of the message.\n\n          role: The role of the entity that is creating the message. Currently only `user` is\n              supported.\n\n          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that\n              the message should use. There can be a maximum of 10 files attached to a\n              message. Useful for tools like `retrieval` and `code_interpreter` that can\n              access and use files.\n\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/messages', body=maybe_transform({'content': content, 'role': role, 'file_ids': file_ids, 'metadata': metadata}, message_create_params.MessageCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    def retrieve(self, message_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        if False:
            return 10
        '\n        Retrieve a message.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get(f'/threads/{thread_id}/messages/{message_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    def update(self, message_id: str, *, thread_id: str, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        if False:
            for i in range(10):
                print('nop')
        '\n        Modifies a message.\n\n        Args:\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/messages/{message_id}', body=maybe_transform({'metadata': metadata}, message_update_params.MessageUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    def list(self, thread_id: str, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[ThreadMessage]:
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of messages for a given thread.\n\n        Args:\n          after: A cursor for use in pagination. `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list(f'/threads/{thread_id}/messages', page=SyncCursorPage[ThreadMessage], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, message_list_params.MessageListParams)), model=ThreadMessage)

class AsyncMessages(AsyncAPIResource):
    files: AsyncFiles
    with_raw_response: AsyncMessagesWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(client)
        self.files = AsyncFiles(client)
        self.with_raw_response = AsyncMessagesWithRawResponse(self)

    async def create(self, thread_id: str, *, content: str, role: Literal['user'], file_ids: List[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        """
        Create a message.

        Args:
          content: The content of the message.

          role: The role of the entity that is creating the message. Currently only `user` is
              supported.

          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that
              the message should use. There can be a maximum of 10 files attached to a
              message. Useful for tools like `retrieval` and `code_interpreter` that can
              access and use files.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/threads/{thread_id}/messages', body=maybe_transform({'content': content, 'role': role, 'file_ids': file_ids, 'metadata': metadata}, message_create_params.MessageCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    async def retrieve(self, message_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        """
        Retrieve a message.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._get(f'/threads/{thread_id}/messages/{message_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    async def update(self, message_id: str, *, thread_id: str, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ThreadMessage:
        """
        Modifies a message.

        Args:
          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/threads/{thread_id}/messages/{message_id}', body=maybe_transform({'metadata': metadata}, message_update_params.MessageUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ThreadMessage)

    def list(self, thread_id: str, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[ThreadMessage, AsyncCursorPage[ThreadMessage]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of messages for a given thread.\n\n        Args:\n          after: A cursor for use in pagination. `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list(f'/threads/{thread_id}/messages', page=AsyncCursorPage[ThreadMessage], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, message_list_params.MessageListParams)), model=ThreadMessage)

class MessagesWithRawResponse:

    def __init__(self, messages: Messages) -> None:
        if False:
            return 10
        self.files = FilesWithRawResponse(messages.files)
        self.create = to_raw_response_wrapper(messages.create)
        self.retrieve = to_raw_response_wrapper(messages.retrieve)
        self.update = to_raw_response_wrapper(messages.update)
        self.list = to_raw_response_wrapper(messages.list)

class AsyncMessagesWithRawResponse:

    def __init__(self, messages: AsyncMessages) -> None:
        if False:
            return 10
        self.files = AsyncFilesWithRawResponse(messages.files)
        self.create = async_to_raw_response_wrapper(messages.create)
        self.retrieve = async_to_raw_response_wrapper(messages.retrieve)
        self.update = async_to_raw_response_wrapper(messages.update)
        self.list = async_to_raw_response_wrapper(messages.list)