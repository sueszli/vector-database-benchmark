from __future__ import annotations
import base64
from typing import TYPE_CHECKING, List, Union, cast
from typing_extensions import Literal
import httpx
from ..types import CreateEmbeddingResponse, embedding_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import is_given, maybe_transform
from .._extras import numpy as np
from .._extras import has_numpy
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .._base_client import make_request_options
if TYPE_CHECKING:
    from .._client import OpenAI, AsyncOpenAI
__all__ = ['Embeddings', 'AsyncEmbeddings']

class Embeddings(SyncAPIResource):
    with_raw_response: EmbeddingsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(client)
        self.with_raw_response = EmbeddingsWithRawResponse(self)

    def create(self, *, input: Union[str, List[str], List[int], List[List[int]]], model: Union[str, Literal['text-embedding-ada-002']], encoding_format: Literal['float', 'base64'] | NotGiven=NOT_GIVEN, user: str | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> CreateEmbeddingResponse:
        if False:
            i = 10
            return i + 15
        '\n        Creates an embedding vector representing the input text.\n\n        Args:\n          input: Input text to embed, encoded as a string or array of tokens. To embed multiple\n              inputs in a single request, pass an array of strings or array of token arrays.\n              The input must not exceed the max input tokens for the model (8192 tokens for\n              `text-embedding-ada-002`) and cannot be an empty string.\n              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)\n              for counting tokens.\n\n          model: ID of the model to use. You can use the\n              [List models](https://platform.openai.com/docs/api-reference/models/list) API to\n              see all of your available models, or see our\n              [Model overview](https://platform.openai.com/docs/models/overview) for\n              descriptions of them.\n\n          encoding_format: The format to return the embeddings in. Can be either `float` or\n              [`base64`](https://pypi.org/project/pybase64/).\n\n          user: A unique identifier representing your end-user, which can help OpenAI to monitor\n              and detect abuse.\n              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        params = {'input': input, 'model': model, 'user': user, 'encoding_format': encoding_format}
        if not is_given(encoding_format) and has_numpy():
            params['encoding_format'] = 'base64'

        def parser(obj: CreateEmbeddingResponse) -> CreateEmbeddingResponse:
            if False:
                i = 10
                return i + 15
            if is_given(encoding_format):
                return obj
            for embedding in obj.data:
                data = cast(object, embedding.embedding)
                if not isinstance(data, str):
                    continue
                embedding.embedding = np.frombuffer(base64.b64decode(data), dtype='float32').tolist()
            return obj
        return self._post('/embeddings', body=maybe_transform(params, embedding_create_params.EmbeddingCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, post_parser=parser), cast_to=CreateEmbeddingResponse)

class AsyncEmbeddings(AsyncAPIResource):
    with_raw_response: AsyncEmbeddingsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            return 10
        super().__init__(client)
        self.with_raw_response = AsyncEmbeddingsWithRawResponse(self)

    async def create(self, *, input: Union[str, List[str], List[int], List[List[int]]], model: Union[str, Literal['text-embedding-ada-002']], encoding_format: Literal['float', 'base64'] | NotGiven=NOT_GIVEN, user: str | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> CreateEmbeddingResponse:
        """
        Creates an embedding vector representing the input text.

        Args:
          input: Input text to embed, encoded as a string or array of tokens. To embed multiple
              inputs in a single request, pass an array of strings or array of token arrays.
              The input must not exceed the max input tokens for the model (8192 tokens for
              `text-embedding-ada-002`) and cannot be an empty string.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          encoding_format: The format to return the embeddings in. Can be either `float` or
              [`base64`](https://pypi.org/project/pybase64/).

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        params = {'input': input, 'model': model, 'user': user, 'encoding_format': encoding_format}
        if not is_given(encoding_format) and has_numpy():
            params['encoding_format'] = 'base64'

        def parser(obj: CreateEmbeddingResponse) -> CreateEmbeddingResponse:
            if False:
                i = 10
                return i + 15
            if is_given(encoding_format):
                return obj
            for embedding in obj.data:
                data = cast(object, embedding.embedding)
                if not isinstance(data, str):
                    continue
                embedding.embedding = np.frombuffer(base64.b64decode(data), dtype='float32').tolist()
            return obj
        return await self._post('/embeddings', body=maybe_transform(params, embedding_create_params.EmbeddingCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, post_parser=parser), cast_to=CreateEmbeddingResponse)

class EmbeddingsWithRawResponse:

    def __init__(self, embeddings: Embeddings) -> None:
        if False:
            while True:
                i = 10
        self.create = to_raw_response_wrapper(embeddings.create)

class AsyncEmbeddingsWithRawResponse:

    def __init__(self, embeddings: AsyncEmbeddings) -> None:
        if False:
            return 10
        self.create = async_to_raw_response_wrapper(embeddings.create)