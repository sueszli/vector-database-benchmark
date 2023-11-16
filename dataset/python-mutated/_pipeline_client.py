from __future__ import annotations
import logging
from collections.abc import Iterable
from typing import TypeVar, Generic, Optional, Any
from .configuration import Configuration
from .pipeline import Pipeline
from .pipeline.transport._base import PipelineClientBase
from .pipeline.transport import HttpTransport
from .pipeline.policies import ContentDecodePolicy, DistributedTracingPolicy, HttpLoggingPolicy, RequestIdPolicy, RetryPolicy, SensitiveHeaderCleanupPolicy
HTTPResponseType = TypeVar('HTTPResponseType')
HTTPRequestType = TypeVar('HTTPRequestType')
_LOGGER = logging.getLogger(__name__)

class PipelineClient(PipelineClientBase, Generic[HTTPRequestType, HTTPResponseType]):
    """Service client core methods.

    Builds a Pipeline client.

    :param str base_url: URL for the request.
    :keyword ~azure.core.configuration.Configuration config: If omitted, the standard configuration is used.
    :keyword Pipeline pipeline: If omitted, a Pipeline object is created and returned.
    :keyword list[HTTPPolicy] policies: If omitted, the standard policies of the configuration object is used.
    :keyword per_call_policies: If specified, the policies will be added into the policy list before RetryPolicy
    :paramtype per_call_policies: Union[HTTPPolicy, SansIOHTTPPolicy, list[HTTPPolicy], list[SansIOHTTPPolicy]]
    :keyword per_retry_policies: If specified, the policies will be added into the policy list after RetryPolicy
    :paramtype per_retry_policies: Union[HTTPPolicy, SansIOHTTPPolicy, list[HTTPPolicy], list[SansIOHTTPPolicy]]
    :keyword HttpTransport transport: If omitted, RequestsTransport is used for synchronous transport.
    :return: A pipeline object.
    :rtype: ~azure.core.pipeline.Pipeline

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sync.py
            :start-after: [START build_pipeline_client]
            :end-before: [END build_pipeline_client]
            :language: python
            :dedent: 4
            :caption: Builds the pipeline client.
    """

    def __init__(self, base_url: str, *, pipeline: Optional[Pipeline[HTTPRequestType, HTTPResponseType]]=None, config: Optional[Configuration[HTTPRequestType, HTTPResponseType]]=None, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        super(PipelineClient, self).__init__(base_url)
        self._config: Configuration[HTTPRequestType, HTTPResponseType] = config or Configuration(**kwargs)
        self._base_url = base_url
        self._pipeline = pipeline or self._build_pipeline(self._config, **kwargs)

    def __enter__(self) -> PipelineClient[HTTPRequestType, HTTPResponseType]:
        if False:
            print('Hello World!')
        self._pipeline.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._pipeline.__exit__(*exc_details)

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.__exit__()

    def _build_pipeline(self, config: Configuration[HTTPRequestType, HTTPResponseType], *, transport: Optional[HttpTransport[HTTPRequestType, HTTPResponseType]]=None, policies=None, per_call_policies=None, per_retry_policies=None, **kwargs) -> Pipeline[HTTPRequestType, HTTPResponseType]:
        if False:
            print('Hello World!')
        per_call_policies = per_call_policies or []
        per_retry_policies = per_retry_policies or []
        if policies is None:
            policies = [config.request_id_policy or RequestIdPolicy(**kwargs), config.headers_policy, config.user_agent_policy, config.proxy_policy, ContentDecodePolicy(**kwargs)]
            if isinstance(per_call_policies, Iterable):
                policies.extend(per_call_policies)
            else:
                policies.append(per_call_policies)
            policies.extend([config.redirect_policy, config.retry_policy, config.authentication_policy, config.custom_hook_policy])
            if isinstance(per_retry_policies, Iterable):
                policies.extend(per_retry_policies)
            else:
                policies.append(per_retry_policies)
            policies.extend([config.logging_policy, DistributedTracingPolicy(**kwargs), SensitiveHeaderCleanupPolicy(**kwargs) if config.redirect_policy else None, config.http_logging_policy or HttpLoggingPolicy(**kwargs)])
        else:
            if isinstance(per_call_policies, Iterable):
                per_call_policies_list = list(per_call_policies)
            else:
                per_call_policies_list = [per_call_policies]
            per_call_policies_list.extend(policies)
            policies = per_call_policies_list
            if isinstance(per_retry_policies, Iterable):
                per_retry_policies_list = list(per_retry_policies)
            else:
                per_retry_policies_list = [per_retry_policies]
            if len(per_retry_policies_list) > 0:
                index_of_retry = -1
                for (index, policy) in enumerate(policies):
                    if isinstance(policy, RetryPolicy):
                        index_of_retry = index
                if index_of_retry == -1:
                    raise ValueError('Failed to add per_retry_policies; no RetryPolicy found in the supplied list of policies. ')
                policies_1 = policies[:index_of_retry + 1]
                policies_2 = policies[index_of_retry + 1:]
                policies_1.extend(per_retry_policies_list)
                policies_1.extend(policies_2)
                policies = policies_1
        if transport is None:
            from .pipeline.transport._requests_basic import RequestsTransport
            transport = RequestsTransport(**kwargs)
        return Pipeline(transport, policies)

    def send_request(self, request: HTTPRequestType, **kwargs: Any) -> HTTPResponseType:
        if False:
            while True:
                i = 10
        "Method that runs the network request through the client's chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest('GET', 'http://www.example.com')\n        <HttpRequest [GET], url: 'http://www.example.com'>\n        >>> response = client.send_request(request)\n        <HttpResponse: 200 OK>\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        "
        stream = kwargs.pop('stream', False)
        return_pipeline_response = kwargs.pop('_return_pipeline_response', False)
        pipeline_response = self._pipeline.run(request, stream=stream, **kwargs)
        if return_pipeline_response:
            return pipeline_response
        return pipeline_response.http_response