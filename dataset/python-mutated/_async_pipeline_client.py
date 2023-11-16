from typing import TypeVar, AsyncContextManager, Any
from collections.abc import MutableSequence
from azure.core import AsyncPipelineClient
from .policies import AsyncARMAutoResourceProviderRegistrationPolicy, ARMHttpLoggingPolicy
HTTPRequestType = TypeVar('HTTPRequestType')
AsyncHTTPResponseType = TypeVar('AsyncHTTPResponseType', bound=AsyncContextManager)

class AsyncARMPipelineClient(AsyncPipelineClient[HTTPRequestType, AsyncHTTPResponseType]):
    """A pipeline client designed for ARM explicitly.

    :param str base_url: URL for the request.
    :keyword AsyncPipeline pipeline: If omitted, a Pipeline object is created and returned.
    :keyword list[AsyncHTTPPolicy] policies: If omitted, the standard policies of the configuration object is used.
    :keyword per_call_policies: If specified, the policies will be added into the policy list before RetryPolicy
    :paramtype per_call_policies: Union[AsyncHTTPPolicy, SansIOHTTPPolicy,
        list[AsyncHTTPPolicy], list[SansIOHTTPPolicy]]
    :keyword per_retry_policies: If specified, the policies will be added into the policy list after RetryPolicy
    :paramtype per_retry_policies: Union[AsyncHTTPPolicy, SansIOHTTPPolicy,
        list[AsyncHTTPPolicy], list[SansIOHTTPPolicy]]
    :keyword AsyncHttpTransport transport: If omitted, AioHttpTransport is used for asynchronous transport.
    """

    def __init__(self, base_url: str, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        if 'policies' not in kwargs:
            config = kwargs.get('config')
            if not config:
                raise ValueError("Current implementation requires to pass 'config' if you don't pass 'policies'")
            per_call_policies = kwargs.get('per_call_policies', [])
            if isinstance(per_call_policies, MutableSequence):
                per_call_policies.append(AsyncARMAutoResourceProviderRegistrationPolicy())
            else:
                per_call_policies = [per_call_policies, AsyncARMAutoResourceProviderRegistrationPolicy()]
            kwargs['per_call_policies'] = per_call_policies
            if not config.http_logging_policy:
                config.http_logging_policy = kwargs.get('http_logging_policy', ARMHttpLoggingPolicy(**kwargs))
            kwargs['config'] = config
        super(AsyncARMPipelineClient, self).__init__(base_url, **kwargs)