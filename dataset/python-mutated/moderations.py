from __future__ import annotations
from typing import TYPE_CHECKING, List, Union
from typing_extensions import Literal
import httpx
from ..types import ModerationCreateResponse, moderation_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import maybe_transform
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .._base_client import make_request_options
if TYPE_CHECKING:
    from .._client import OpenAI, AsyncOpenAI
__all__ = ['Moderations', 'AsyncModerations']

class Moderations(SyncAPIResource):
    with_raw_response: ModerationsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            while True:
                i = 10
        super().__init__(client)
        self.with_raw_response = ModerationsWithRawResponse(self)

    def create(self, *, input: Union[str, List[str]], model: Union[str, Literal['text-moderation-latest', 'text-moderation-stable']] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ModerationCreateResponse:
        if False:
            return 10
        "\n        Classifies if text violates OpenAI's Content Policy\n\n        Args:\n          input: The input text to classify\n\n          model: Two content moderations models are available: `text-moderation-stable` and\n              `text-moderation-latest`.\n\n              The default is `text-moderation-latest` which will be automatically upgraded\n              over time. This ensures you are always using our most accurate model. If you use\n              `text-moderation-stable`, we will provide advanced notice before updating the\n              model. Accuracy of `text-moderation-stable` may be slightly lower than for\n              `text-moderation-latest`.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        "
        return self._post('/moderations', body=maybe_transform({'input': input, 'model': model}, moderation_create_params.ModerationCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ModerationCreateResponse)

class AsyncModerations(AsyncAPIResource):
    with_raw_response: AsyncModerationsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            while True:
                i = 10
        super().__init__(client)
        self.with_raw_response = AsyncModerationsWithRawResponse(self)

    async def create(self, *, input: Union[str, List[str]], model: Union[str, Literal['text-moderation-latest', 'text-moderation-stable']] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ModerationCreateResponse:
        """
        Classifies if text violates OpenAI's Content Policy

        Args:
          input: The input text to classify

          model: Two content moderations models are available: `text-moderation-stable` and
              `text-moderation-latest`.

              The default is `text-moderation-latest` which will be automatically upgraded
              over time. This ensures you are always using our most accurate model. If you use
              `text-moderation-stable`, we will provide advanced notice before updating the
              model. Accuracy of `text-moderation-stable` may be slightly lower than for
              `text-moderation-latest`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post('/moderations', body=maybe_transform({'input': input, 'model': model}, moderation_create_params.ModerationCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ModerationCreateResponse)

class ModerationsWithRawResponse:

    def __init__(self, moderations: Moderations) -> None:
        if False:
            i = 10
            return i + 15
        self.create = to_raw_response_wrapper(moderations.create)

class AsyncModerationsWithRawResponse:

    def __init__(self, moderations: AsyncModerations) -> None:
        if False:
            i = 10
            return i + 15
        self.create = async_to_raw_response_wrapper(moderations.create)