from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from typing_extensions import Literal
import httpx
from .steps import Steps, AsyncSteps, StepsWithRawResponse, AsyncStepsWithRawResponse
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import maybe_transform
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .....pagination import SyncCursorPage, AsyncCursorPage
from ....._base_client import AsyncPaginator, make_request_options
from .....types.beta.threads import Run, run_list_params, run_create_params, run_update_params, run_submit_tool_outputs_params
if TYPE_CHECKING:
    from ....._client import OpenAI, AsyncOpenAI
__all__ = ['Runs', 'AsyncRuns']

class Runs(SyncAPIResource):
    steps: Steps
    with_raw_response: RunsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            while True:
                i = 10
        super().__init__(client)
        self.steps = Steps(client)
        self.with_raw_response = RunsWithRawResponse(self)

    def create(self, thread_id: str, *, assistant_id: str, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Optional[str] | NotGiven=NOT_GIVEN, tools: Optional[List[run_create_params.Tool]] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        if False:
            i = 10
            return i + 15
        '\n        Create a run.\n\n        Args:\n          assistant_id: The ID of the\n              [assistant](https://platform.openai.com/docs/api-reference/assistants) to use to\n              execute this run.\n\n          instructions: Override the default system message of the assistant. This is useful for\n              modifying the behavior on a per-run basis.\n\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          model: The ID of the [Model](https://platform.openai.com/docs/api-reference/models) to\n              be used to execute this run. If a value is provided here, it will override the\n              model associated with the assistant. If not, the model associated with the\n              assistant will be used.\n\n          tools: Override the tools the assistant can use for this run. This is useful for\n              modifying the behavior on a per-run basis.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/runs', body=maybe_transform({'assistant_id': assistant_id, 'instructions': instructions, 'metadata': metadata, 'model': model, 'tools': tools}, run_create_params.RunCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    def retrieve(self, run_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        if False:
            return 10
        '\n        Retrieves a run.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get(f'/threads/{thread_id}/runs/{run_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    def update(self, run_id: str, *, thread_id: str, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        if False:
            i = 10
            return i + 15
        '\n        Modifies a run.\n\n        Args:\n          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful\n              for storing additional information about the object in a structured format. Keys\n              can be a maximum of 64 characters long and values can be a maxium of 512\n              characters long.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/runs/{run_id}', body=maybe_transform({'metadata': metadata}, run_update_params.RunUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    def list(self, thread_id: str, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[Run]:
        if False:
            return 10
        '\n        Returns a list of runs belonging to a thread.\n\n        Args:\n          after: A cursor for use in pagination. `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list(f'/threads/{thread_id}/runs', page=SyncCursorPage[Run], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, run_list_params.RunListParams)), model=Run)

    def cancel(self, run_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        if False:
            while True:
                i = 10
        '\n        Cancels a run that is `in_progress`.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/runs/{run_id}/cancel', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    def submit_tool_outputs(self, run_id: str, *, thread_id: str, tool_outputs: List[run_submit_tool_outputs_params.ToolOutput], extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        if False:
            return 10
        '\n        When a run has the `status: "requires_action"` and `required_action.type` is\n        `submit_tool_outputs`, this endpoint can be used to submit the outputs from the\n        tool calls once they\'re all completed. All outputs must be submitted in a single\n        request.\n\n        Args:\n          tool_outputs: A list of tools for which the outputs are being submitted.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._post(f'/threads/{thread_id}/runs/{run_id}/submit_tool_outputs', body=maybe_transform({'tool_outputs': tool_outputs}, run_submit_tool_outputs_params.RunSubmitToolOutputsParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

class AsyncRuns(AsyncAPIResource):
    steps: AsyncSteps
    with_raw_response: AsyncRunsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            return 10
        super().__init__(client)
        self.steps = AsyncSteps(client)
        self.with_raw_response = AsyncRunsWithRawResponse(self)

    async def create(self, thread_id: str, *, assistant_id: str, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Optional[str] | NotGiven=NOT_GIVEN, tools: Optional[List[run_create_params.Tool]] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        """
        Create a run.

        Args:
          assistant_id: The ID of the
              [assistant](https://platform.openai.com/docs/api-reference/assistants) to use to
              execute this run.

          instructions: Override the default system message of the assistant. This is useful for
              modifying the behavior on a per-run basis.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          model: The ID of the [Model](https://platform.openai.com/docs/api-reference/models) to
              be used to execute this run. If a value is provided here, it will override the
              model associated with the assistant. If not, the model associated with the
              assistant will be used.

          tools: Override the tools the assistant can use for this run. This is useful for
              modifying the behavior on a per-run basis.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/threads/{thread_id}/runs', body=maybe_transform({'assistant_id': assistant_id, 'instructions': instructions, 'metadata': metadata, 'model': model, 'tools': tools}, run_create_params.RunCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    async def retrieve(self, run_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        """
        Retrieves a run.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._get(f'/threads/{thread_id}/runs/{run_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    async def update(self, run_id: str, *, thread_id: str, metadata: Optional[object] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        """
        Modifies a run.

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
        return await self._post(f'/threads/{thread_id}/runs/{run_id}', body=maybe_transform({'metadata': metadata}, run_update_params.RunUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    def list(self, thread_id: str, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[Run, AsyncCursorPage[Run]]:
        if False:
            print('Hello World!')
        '\n        Returns a list of runs belonging to a thread.\n\n        Args:\n          after: A cursor for use in pagination. `after` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include after=obj_foo in order to\n              fetch the next page of the list.\n\n          before: A cursor for use in pagination. `before` is an object ID that defines your place\n              in the list. For instance, if you make a list request and receive 100 objects,\n              ending with obj_foo, your subsequent call can include before=obj_foo in order to\n              fetch the previous page of the list.\n\n          limit: A limit on the number of objects to be returned. Limit can range between 1 and\n              100, and the default is 20.\n\n          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending\n              order and `desc` for descending order.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list(f'/threads/{thread_id}/runs', page=AsyncCursorPage[Run], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, run_list_params.RunListParams)), model=Run)

    async def cancel(self, run_id: str, *, thread_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        """
        Cancels a run that is `in_progress`.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/threads/{thread_id}/runs/{run_id}/cancel', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

    async def submit_tool_outputs(self, run_id: str, *, thread_id: str, tool_outputs: List[run_submit_tool_outputs_params.ToolOutput], extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
        """
        When a run has the `status: "requires_action"` and `required_action.type` is
        `submit_tool_outputs`, this endpoint can be used to submit the outputs from the
        tool calls once they're all completed. All outputs must be submitted in a single
        request.

        Args:
          tool_outputs: A list of tools for which the outputs are being submitted.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/threads/{thread_id}/runs/{run_id}/submit_tool_outputs', body=maybe_transform({'tool_outputs': tool_outputs}, run_submit_tool_outputs_params.RunSubmitToolOutputsParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run)

class RunsWithRawResponse:

    def __init__(self, runs: Runs) -> None:
        if False:
            while True:
                i = 10
        self.steps = StepsWithRawResponse(runs.steps)
        self.create = to_raw_response_wrapper(runs.create)
        self.retrieve = to_raw_response_wrapper(runs.retrieve)
        self.update = to_raw_response_wrapper(runs.update)
        self.list = to_raw_response_wrapper(runs.list)
        self.cancel = to_raw_response_wrapper(runs.cancel)
        self.submit_tool_outputs = to_raw_response_wrapper(runs.submit_tool_outputs)

class AsyncRunsWithRawResponse:

    def __init__(self, runs: AsyncRuns) -> None:
        if False:
            i = 10
            return i + 15
        self.steps = AsyncStepsWithRawResponse(runs.steps)
        self.create = async_to_raw_response_wrapper(runs.create)
        self.retrieve = async_to_raw_response_wrapper(runs.retrieve)
        self.update = async_to_raw_response_wrapper(runs.update)
        self.list = async_to_raw_response_wrapper(runs.list)
        self.cancel = async_to_raw_response_wrapper(runs.cancel)
        self.submit_tool_outputs = async_to_raw_response_wrapper(runs.submit_tool_outputs)