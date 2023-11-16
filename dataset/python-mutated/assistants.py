from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from typing_extensions import Literal
import httpx
from .files import Files, AsyncFiles, FilesWithRawResponse, AsyncFilesWithRawResponse
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import maybe_transform
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import Assistant, AssistantDeleted, assistant_list_params, assistant_create_params, assistant_update_params
from ...._base_client import AsyncPaginator, make_request_options
if TYPE_CHECKING:
    from ...._client import OpenAI, AsyncOpenAI
__all__ = ['Assistants', 'AsyncAssistants']

class Assistants(SyncAPIResource):
    files: Files
    with_raw_response: AssistantsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            print('Hello World!')
        super().__init__(client)
        self.files = Files(client)
        self.with_raw_response = AssistantsWithRawResponse(self)

    def create(self, *, model: str, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: List[assistant_create_params.Tool] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        if False:
            return 10
        '\n        Create an assistant with a model and instructions.\n\n        Args:\n          model: ID of the model to use. You can use the\n              [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n              see all of your available models, or see our\n              [Model overview](https://platform.openai.com/docs/models/overview) for\n              descriptions of them.\n\n          description: The description of the assistant. The maximum length is 512 characters.\n\n          file_ids: A list of [file](https://platform.openai.com/docs/api-reference/files) IDs\n              attached to this assistant. There can be a maximum of 20 files attached to the\n              assistant. Files are ordered by their creation date in ascending order.\n\n          instructions: The system instructions that the assistant uses. The maximum length is 32768\n              characters.\n\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          name: The name of the assistant. The maximum length is 256 characters.\n\n          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per\n              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post('/assistants', body=maybe_transform({'model': model, 'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'name': name, 'tools': tools}, assistant_create_params.AssistantCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    def retrieve(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        if False:
            print('Hello World!')
        '\n        Retrieves an assistant.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    def update(self, assistant_id: str, *, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: str | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: List[assistant_update_params.Tool] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        if False:
            print('Hello World!')
        'Modifies an assistant.\n\n        Args:\n          description: The description of the assistant.\n\n        The maximum length is 512 characters.\n\n          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs\n              attached to this assistant. There can be a maximum of 20 files attached to the\n              assistant. Files are ordered by their creation date in ascending order. If a\n              file was previosuly attached to the list but does not show up in the list, it\n              will be deleted from the assistant.\n\n          instructions: The system instructions that the assistant uses. The maximum length is 32768\n              characters.\n\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          model: ID of the model to use. You can use the\n              [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n              see all of your available models, or see our\n              [Model overview](https://platform.openai.com/docs/models/overview) for\n              descriptions of them.\n\n          name: The name of the assistant. The maximum length is 256 characters.\n\n          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per\n              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/assistants/{assistant_id}', body=maybe_transform({'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'model': model, 'name': name, 'tools': tools}, assistant_update_params.AssistantUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[Assistant]:
        if False:
            return 10
        'Returns a list of assistants.\n\n        Args:\n          after: A cursor for use in pagination.\n\n        `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list('/assistants', page=SyncCursorPage[Assistant], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, assistant_list_params.AssistantListParams)), model=Assistant)

    def delete(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AssistantDeleted:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete an assistant.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._delete(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=AssistantDeleted)

class AsyncAssistants(AsyncAPIResource):
    files: AsyncFiles
    with_raw_response: AsyncAssistantsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            return 10
        super().__init__(client)
        self.files = AsyncFiles(client)
        self.with_raw_response = AsyncAssistantsWithRawResponse(self)

    async def create(self, *, model: str, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: List[assistant_create_params.Tool] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """
        Create an assistant with a model and instructions.

        Args:
          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          description: The description of the assistant. The maximum length is 512 characters.

          file_ids: A list of [file](https://platform.openai.com/docs/api-reference/files) IDs
              attached to this assistant. There can be a maximum of 20 files attached to the
              assistant. Files are ordered by their creation date in ascending order.

          instructions: The system instructions that the assistant uses. The maximum length is 32768
              characters.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          name: The name of the assistant. The maximum length is 256 characters.

          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per
              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post('/assistants', body=maybe_transform({'model': model, 'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'name': name, 'tools': tools}, assistant_create_params.AssistantCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    async def retrieve(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """
        Retrieves an assistant.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._get(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    async def update(self, assistant_id: str, *, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: str | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: List[assistant_update_params.Tool] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """Modifies an assistant.

        Args:
          description: The description of the assistant.

        The maximum length is 512 characters.

          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs
              attached to this assistant. There can be a maximum of 20 files attached to the
              assistant. Files are ordered by their creation date in ascending order. If a
              file was previosuly attached to the list but does not show up in the list, it
              will be deleted from the assistant.

          instructions: The system instructions that the assistant uses. The maximum length is 32768
              characters.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          name: The name of the assistant. The maximum length is 256 characters.

          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per
              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/assistants/{assistant_id}', body=maybe_transform({'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'model': model, 'name': name, 'tools': tools}, assistant_update_params.AssistantUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[Assistant, AsyncCursorPage[Assistant]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of assistants.\n\n        Args:\n          after: A cursor for use in pagination.\n\n        `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list('/assistants', page=AsyncCursorPage[Assistant], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, assistant_list_params.AssistantListParams)), model=Assistant)

    async def delete(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AssistantDeleted:
        """
        Delete an assistant.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._delete(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=AssistantDeleted)

class AssistantsWithRawResponse:

    def __init__(self, assistants: Assistants) -> None:
        if False:
            i = 10
            return i + 15
        self.files = FilesWithRawResponse(assistants.files)
        self.create = to_raw_response_wrapper(assistants.create)
        self.retrieve = to_raw_response_wrapper(assistants.retrieve)
        self.update = to_raw_response_wrapper(assistants.update)
        self.list = to_raw_response_wrapper(assistants.list)
        self.delete = to_raw_response_wrapper(assistants.delete)

class AsyncAssistantsWithRawResponse:

    def __init__(self, assistants: AsyncAssistants) -> None:
        if False:
            return 10
        self.files = AsyncFilesWithRawResponse(assistants.files)
        self.create = async_to_raw_response_wrapper(assistants.create)
        self.retrieve = async_to_raw_response_wrapper(assistants.retrieve)
        self.update = async_to_raw_response_wrapper(assistants.update)
        self.list = async_to_raw_response_wrapper(assistants.list)
        self.delete = async_to_raw_response_wrapper(assistants.delete)