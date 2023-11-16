from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional
from typing_extensions import Literal
import httpx
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import maybe_transform
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from ...pagination import SyncCursorPage, AsyncCursorPage
from ..._base_client import AsyncPaginator, make_request_options
from ...types.fine_tuning import FineTuningJob, FineTuningJobEvent, job_list_params, job_create_params, job_list_events_params
if TYPE_CHECKING:
    from ..._client import OpenAI, AsyncOpenAI
__all__ = ['Jobs', 'AsyncJobs']

class Jobs(SyncAPIResource):
    with_raw_response: JobsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(client)
        self.with_raw_response = JobsWithRawResponse(self)

    def create(self, *, model: Union[str, Literal['babbage-002', 'davinci-002', 'gpt-3.5-turbo']], training_file: str, hyperparameters: job_create_params.Hyperparameters | NotGiven=NOT_GIVEN, suffix: Optional[str] | NotGiven=NOT_GIVEN, validation_file: Optional[str] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a job that fine-tunes a specified model from a given dataset.\n\n        Response includes details of the enqueued job including job status and the name\n        of the fine-tuned models once complete.\n\n        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)\n\n        Args:\n          model: The name of the model to fine-tune. You can select one of the\n              [supported models](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned).\n\n          training_file: The ID of an uploaded file that contains training data.\n\n              See [upload file](https://platform.openai.com/docs/api-reference/files/upload)\n              for how to upload a file.\n\n              Your dataset must be formatted as a JSONL file. Additionally, you must upload\n              your file with the purpose `fine-tune`.\n\n              See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)\n              for more details.\n\n          hyperparameters: The hyperparameters used for the fine-tuning job.\n\n          suffix: A string of up to 18 characters that will be added to your fine-tuned model\n              name.\n\n              For example, a `suffix` of "custom-model-name" would produce a model name like\n              `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`.\n\n          validation_file: The ID of an uploaded file that contains validation data.\n\n              If you provide this file, the data is used to generate validation metrics\n              periodically during fine-tuning. These metrics can be viewed in the fine-tuning\n              results file. The same data should not be present in both train and validation\n              files.\n\n              Your dataset must be formatted as a JSONL file. You must upload your file with\n              the purpose `fine-tune`.\n\n              See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)\n              for more details.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        return self._post('/fine_tuning/jobs', body=maybe_transform({'model': model, 'training_file': training_file, 'hyperparameters': hyperparameters, 'suffix': suffix, 'validation_file': validation_file}, job_create_params.JobCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    def retrieve(self, fine_tuning_job_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        if False:
            return 10
        '\n        Get info about a fine-tuning job.\n\n        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        return self._get(f'/fine_tuning/jobs/{fine_tuning_job_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[FineTuningJob]:
        if False:
            while True:
                i = 10
        "\n        List your organization's fine-tuning jobs\n\n        Args:\n          after: Identifier for the last job from the previous pagination request.\n\n          limit: Number of fine-tuning jobs to retrieve.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        "
        return self._get_api_list('/fine_tuning/jobs', page=SyncCursorPage[FineTuningJob], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, job_list_params.JobListParams)), model=FineTuningJob)

    def cancel(self, fine_tuning_job_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        if False:
            print('Hello World!')
        '\n        Immediately cancel a fine-tune job.\n\n        Args:\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        return self._post(f'/fine_tuning/jobs/{fine_tuning_job_id}/cancel', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    def list_events(self, fine_tuning_job_id: str, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[FineTuningJobEvent]:
        if False:
            while True:
                i = 10
        '\n        Get status updates for a fine-tuning job.\n\n        Args:\n          after: Identifier for the last event from the previous pagination request.\n\n          limit: Number of events to retrieve.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        return self._get_api_list(f'/fine_tuning/jobs/{fine_tuning_job_id}/events', page=SyncCursorPage[FineTuningJobEvent], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, job_list_events_params.JobListEventsParams)), model=FineTuningJobEvent)

class AsyncJobs(AsyncAPIResource):
    with_raw_response: AsyncJobsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            print('Hello World!')
        super().__init__(client)
        self.with_raw_response = AsyncJobsWithRawResponse(self)

    async def create(self, *, model: Union[str, Literal['babbage-002', 'davinci-002', 'gpt-3.5-turbo']], training_file: str, hyperparameters: job_create_params.Hyperparameters | NotGiven=NOT_GIVEN, suffix: Optional[str] | NotGiven=NOT_GIVEN, validation_file: Optional[str] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        """
        Creates a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name
        of the fine-tuned models once complete.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

        Args:
          model: The name of the model to fine-tune. You can select one of the
              [supported models](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned).

          training_file: The ID of an uploaded file that contains training data.

              See [upload file](https://platform.openai.com/docs/api-reference/files/upload)
              for how to upload a file.

              Your dataset must be formatted as a JSONL file. Additionally, you must upload
              your file with the purpose `fine-tune`.

              See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)
              for more details.

          hyperparameters: The hyperparameters used for the fine-tuning job.

          suffix: A string of up to 18 characters that will be added to your fine-tuned model
              name.

              For example, a `suffix` of "custom-model-name" would produce a model name like
              `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`.

          validation_file: The ID of an uploaded file that contains validation data.

              If you provide this file, the data is used to generate validation metrics
              periodically during fine-tuning. These metrics can be viewed in the fine-tuning
              results file. The same data should not be present in both train and validation
              files.

              Your dataset must be formatted as a JSONL file. You must upload your file with
              the purpose `fine-tune`.

              See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)
              for more details.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post('/fine_tuning/jobs', body=maybe_transform({'model': model, 'training_file': training_file, 'hyperparameters': hyperparameters, 'suffix': suffix, 'validation_file': validation_file}, job_create_params.JobCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    async def retrieve(self, fine_tuning_job_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        """
        Get info about a fine-tuning job.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(f'/fine_tuning/jobs/{fine_tuning_job_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[FineTuningJob, AsyncCursorPage[FineTuningJob]]:
        if False:
            while True:
                i = 10
        "\n        List your organization's fine-tuning jobs\n\n        Args:\n          after: Identifier for the last job from the previous pagination request.\n\n          limit: Number of fine-tuning jobs to retrieve.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        "
        return self._get_api_list('/fine_tuning/jobs', page=AsyncCursorPage[FineTuningJob], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, job_list_params.JobListParams)), model=FineTuningJob)

    async def cancel(self, fine_tuning_job_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> FineTuningJob:
        """
        Immediately cancel a fine-tune job.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(f'/fine_tuning/jobs/{fine_tuning_job_id}/cancel', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=FineTuningJob)

    def list_events(self, fine_tuning_job_id: str, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[FineTuningJobEvent, AsyncCursorPage[FineTuningJobEvent]]:
        if False:
            return 10
        '\n        Get status updates for a fine-tuning job.\n\n        Args:\n          after: Identifier for the last event from the previous pagination request.\n\n          limit: Number of events to retrieve.\n\n          extra_headers: Send extra headers\n\n          extra_query: Add additional query parameters to the request\n\n          extra_body: Add additional JSON properties to the request\n\n          timeout: Override the client-level default timeout for this request, in seconds\n        '
        return self._get_api_list(f'/fine_tuning/jobs/{fine_tuning_job_id}/events', page=AsyncCursorPage[FineTuningJobEvent], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, job_list_events_params.JobListEventsParams)), model=FineTuningJobEvent)

class JobsWithRawResponse:

    def __init__(self, jobs: Jobs) -> None:
        if False:
            return 10
        self.create = to_raw_response_wrapper(jobs.create)
        self.retrieve = to_raw_response_wrapper(jobs.retrieve)
        self.list = to_raw_response_wrapper(jobs.list)
        self.cancel = to_raw_response_wrapper(jobs.cancel)
        self.list_events = to_raw_response_wrapper(jobs.list_events)

class AsyncJobsWithRawResponse:

    def __init__(self, jobs: AsyncJobs) -> None:
        if False:
            while True:
                i = 10
        self.create = async_to_raw_response_wrapper(jobs.create)
        self.retrieve = async_to_raw_response_wrapper(jobs.retrieve)
        self.list = async_to_raw_response_wrapper(jobs.list)
        self.cancel = async_to_raw_response_wrapper(jobs.cancel)
        self.list_events = async_to_raw_response_wrapper(jobs.list_events)